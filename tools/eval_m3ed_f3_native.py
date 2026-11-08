#!/usr/bin/env python3
"""Score M3ED day-car on the pinned F3 native LiDAR timeline (no RGB cache)."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_state.dense.protocol import load_protocol, sha256_file

REV = '833e2c65233f9787449ce0675ae41e3f6e9c193a'
PAPER_COUNT = 5387


def make_left_index(events, output, block=1<<20):
    """Exact left side of F3 generate_ts.py: lower_bound(events, 20*k).
    Stream native timestamps in blocks rather than loading the event recording.
    """
    import numpy as np
    bins = int(events[-1])//20
    indices = np.empty(bins, dtype=np.uint64)
    first = 0
    for offset in range(0,len(events),block):
        chunk = np.asarray(events[offset:offset+block], dtype=np.int64)
        last = min(bins, int(chunk[-1])//20+1)
        if last > first:
            indices[first:last] = offset + np.searchsorted(chunk, 20*np.arange(first,last))
            first = last
    if first != bins: raise ValueError('Incomplete event lookup table')
    np.save(output, {'left': indices})


def raw_path(root, seq, suffix):
    found = {p.resolve() for p in root.rglob(seq+suffix)}
    if len(found) != 1: raise ValueError(f'Expected one {seq}{suffix} under {root}; found {found}')
    return found.pop()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',required=True,type=Path,help='Raw validation *_data.h5 and *_depth_gt.h5, recursively located')
    p.add_argument('--f3-repo',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='Fresh output directory, including for audit')
    p.add_argument('--reference-support',type=Path,help='Require exact sample/time/GT-mask/target digest from another run')
    p.add_argument('--run',type=Path,help='EventState F3 experiment root')
    p.add_argument('--condition',choices=('baseline','alpha1'),default='alpha1')
    p.add_argument('--feature',choices=('h','z','concat'),default='z')
    p.add_argument('--teacher-weights',type=Path)
    p.add_argument('--audit-only',action='store_true',help='Generate sample/mask manifest without model weights')
    p.add_argument('--f3-model-config',type=Path,help='Instead score the actual metric stage-2 F3 model')
    p.add_argument('--f3-model-weights',type=Path)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if bool(a.f3_model_config) != bool(a.f3_model_weights) or (a.f3_model_config and a.run):
        p.error('Specify one EventState run or both metric F3 artifacts')
    if a.f3_model_weights and 'fullm3ed_pseudo' in str(a.f3_model_weights).lower():
        p.error('Known pseudo stage-1 weights cannot produce the stage-2 metric baseline')
    if not a.audit_only and not (a.run or a.f3_model_config): p.error('Missing predictor')
    repo=a.f3_repo.resolve()
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=REV or subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD','--','src','confs','scripts/generate_ts.py']).returncode:
        raise ValueError('Requires the pinned unchanged F3 sources/configs')
    import numpy as np
    import h5py
    import yaml
    # Pinned loader indexes NumPy arrays with arithmetic on uint64 scalars;
    # NumPy 1.x promotes those operations with Python int to float64.
    if not isinstance(np.uint64(40)//20, np.integer):
        raise RuntimeError('F3 loader requires NumPy 2.x (upstream pins 2.1.2); use a separate compatible env')
    cfg = yaml.safe_load((repo/'confs/monocular_depth/eval_m3ed_gt.yml').read_text())
    if cfg['eval'] != 'depth' or cfg['val']['mini_batch'] != 1:
        raise ValueError('Expected the batch-one native M3ED metric protocol')
    import hdf5plugin  # register M3ED compression filters
    sequences=load_protocol('m3ed_f3')['splits']['val']
    val_cfg = repo / cfg['val']['datasets'][0]
    official = yaml.safe_load(val_cfg.read_text())['datasets']
    if [Path(v['dataset_path']).name for v in official] != [s+'_data.h5' for s in sequences] or not all(
            v['camera']=='left' and v['dtype']=='m3ed_gt' and v['range']==[0,1,1] for v in official):
        raise ValueError('Sequence order/range differs from the official M3ED validation config')
    reference = json.loads(a.reference_support.read_text()) if a.reference_support else None
    if reference and (reference['frames'] != PAPER_COUNT or reference['revision'] != REV):
        raise ValueError('Reference manifest did not match the paper population')
    paths={s:(raw_path(a.raw_root,s,'_data.h5'),raw_path(a.raw_root,s,'_depth_gt.h5')) for s in sequences}
    if a.output.exists(): raise FileExistsError(a.output)
    identity = {'revision':REV,'files':{s:[str(f) for f in v] for s,v in paths.items()},
                'ground_truth':'native 720x1280; depth>0.1 and <80 after official fp32 disparity roundtrip',
                'population':'official centered 20ms, 50kHz indices, >=200k events, >=10 valid LiDAR pixels',
                'aggregation':'unweighted mean of per-frame F3 metrics','audit_only':a.audit_only}
    if not a.audit_only:
        import torch
        spec=importlib.util.spec_from_file_location('f3_eval_metrics',repo/'src/f3/tasks/depth/utils/utils.py')
        metrics=importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
        if a.run:
            from event_state.dense.benchmark_inference import DepthPredictor
            from event_state.data.event_representation import GEPEventFrame
            model=DepthPredictor(a.run,'m3ed_'+a.condition,a.feature,a.device,a.teacher_weights)
            gep=GEPEventFrame(720,1280,float(model.cfg.dataset.representation.percentile))
            identity.update(model=model.identity,adapter='GEP from identical sampled event window, native raw grid; resize full FOV to 352x640, bilinear prediction to native GT. No DAGR/crop; continuous across event-eligible windows within each sequence. Polarity used. Offline centered input includes 10ms lookahead.')
        else:
            sys.path.insert(0,str(repo/'src'))
            from f3.tasks.depth.utils import init_depth_model
            ref=init_depth_model(str(a.f3_model_config)).to(a.device).eval()
            weights=torch.load(a.f3_model_weights,map_location='cpu',weights_only=True)['model']
            # Torch-compiled checkpoints have the same learned tensors under _orig_mod.
            normalized={k.replace('_orig_mod.',''):v for k,v in weights.items()}
            if len(normalized)!=len(weights): raise ValueError('Ambiguous F3 checkpoint keys')
            ref.load_state_dict(normalized,strict=True)
            identity.update(f3_model_weights=sha256_file(a.f3_model_weights),f3_config=sha256_file(a.f3_model_config))
    sys.path.insert(0,str(repo/'src'))
    import torch
    import tempfile
    from f3.tasks.depth.utils.dataloader import TimeAlignedDepthAndEvents
    import inspect
    if not Path(inspect.getfile(TimeAlignedDepthAndEvents)).resolve().is_relative_to(repo):
        raise ValueError('Imported F3 outside pinned checkout')
    a.output.mkdir(parents=True)
    n=0; totals={}; support=hashlib.sha256(); contexts=hashlib.sha256(); per_seq={}
    with (a.output/'samples.csv').open('x',newline='') as out:
        rows=csv.writer(out); rows.writerow(['sequence','depth_index','depth_us','sampled_context_events','valid_pixels','mask_sha256'])
        for seq,(data_file,gt_file) in paths.items():
            if a.run and not a.audit_only: model.state=None
            before=n
            # Isolate upstream metadata_depths.json from stale metadata / original recordings.
            with tempfile.TemporaryDirectory(prefix='f3-left-',dir=a.output) as tmp:
                temp=Path(tmp)
                data=temp/data_file.name
                data.symlink_to(data_file.resolve())
                lookup=temp/'index.npy'
                with h5py.File(data) as raw:
                    make_left_index(raw['prophesee/left/t'],lookup)
                sample_set=TimeAlignedDepthAndEvents(str(data),str(lookup),str(gt_file),
                    min_numevents_ctx=cfg['min_numevents_ctx'],max_numevents_ctx=cfg['max_numevents_ctx'],time_ctx=cfg['time_ctx'],
                    bucket=cfg['bucket'],w=cfg['frame_sizes'][0],h=cfg['frame_sizes'][1],randomize_ctx=False,camera='left',dtype='m3ed_gt')
                for ordinal,i in enumerate(sample_set.valid_0_points):
                    t=int(sample_set.ts_map[i])
                    ctx,count,disparity,crop=sample_set[ordinal]
                    if tuple(crop.tolist())!=(0,0,720,1280) or disparity.shape!=(720,1280):
                        raise ValueError('Unexpected upstream depth coordinates')
                    contexts.update(f'{seq}:{t}:{count}\n'.encode())
                    contexts.update(memoryview(ctx.contiguous().numpy()).cast('B'))
                    gt_tensor=cfg['fb']/torch.as_tensor(disparity,device=a.device).float()
                    valid=(gt_tensor>cfg['min_depth'])&(gt_tensor<cfg['max_depth'])
                    pixels=int(valid.sum())
                    if not a.audit_only and (pixels>=10 or a.run):
                        with torch.inference_mode():
                            if a.run:
                                # Official loader has already gated, subsampled and encoded this window.
                                # GEP counts polarities; no further temporal interpolation or filtering.
                                x,y=(ctx[:,0]*1280).round(),(ctx[:,1]*720).round()
                                frame=gep(x,y,torch.ones_like(x),ctx[:,3],start_time=0,end_time=1)
                                frame=torch.nn.functional.interpolate(frame[None],size=(352,640),mode='bilinear',align_corners=False)[0]
                                pred=model.step(frame,pixels>=10)
                                if pred is not None:
                                    pred=torch.nn.functional.interpolate(pred[None,None],size=(720,1280),mode='bilinear',align_corners=False)[0,0]
                            else:
                                pred_disp,_=ref.infer_image(ctx[:,:3].to(a.device),torch.tensor([count],device=a.device),crop)
                                pred=cfg['fb']/pred_disp
                    mask=valid.cpu().numpy()
                    target=gt_tensor.cpu().numpy()
                    if pixels<10: continue
                    h=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                    support.update(f'{seq}:{i}:{t}:{pixels}:{h}\n'.encode()); support.update(target[mask].tobytes())
                    rows.writerow((seq,i,t,count,pixels,h)); n+=1
                    if not a.audit_only:
                        m=torch.from_numpy(mask).to(a.device); ytrue=torch.from_numpy(target).to(a.device)[m]
                        yp=pred[m]
                        if not torch.isfinite(yp).all() or (yp<=0).any(): raise ValueError('Invalid predictions; cannot silently drop pixels')
                        vals=metrics.eval_depth(yp,ytrue)
                        vals['mae']=float((yp-ytrue).abs().mean())
                        for k,v in vals.items():
                            if k not in totals: totals[k]=torch.zeros((),dtype=torch.float32,device=a.device)
                            totals[k]+=v
                    if n%250==0: print(f'{seq}: {n} scoreable frames',flush=True)
                sample_set.depths.close()
                sample_set.hdf5_file.close()
                del sample_set
            per_seq[seq]=n-before
    identity.update(frames=n,per_sequence=per_seq,support_sha256=support.hexdigest(),context_sha256=contexts.hexdigest(),paper_expected=PAPER_COUNT)
    (a.output/'support.json').write_text(json.dumps(identity,indent=2))
    if n!=PAPER_COUNT: raise RuntimeError(f'Population mismatch: {n} vs {PAPER_COUNT}. No validated score published; inspect samples.csv.')
    if reference is not None:
        if any(reference.get(k)!=identity[k] for k in ('frames','revision','support_sha256','context_sha256','per_sequence')):
            raise ValueError('Reference sample/GT mismatch; no validated score published')
    if not a.audit_only:
        identity['metrics']={k:(v/n).item() for k,v in totals.items()}
        (a.output/'metrics.json').write_text(json.dumps(identity,indent=2,allow_nan=False))
    print(json.dumps(identity,indent=2))


if __name__=='__main__': main()
