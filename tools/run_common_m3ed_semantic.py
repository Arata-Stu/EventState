#!/usr/bin/env python3
"""Frozen common Stage 1 -> causal semantic features -> 3-train/2-val heads."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from event_state.dense.protocol import sha256_file
from event_state.dense.common_protocol import load_common

MANIFEST=ROOT/'tools/manifests/m3ed_semantic_common_v1.yaml'


def score_groups(head_dir,cache,target_root,split,feature,device):
    import torch
    from torch.utils.data import DataLoader
    from event_state.segmentation import (EventStateSegmentationHead, M3EDSemanticFeatureDataset,
                                         SemanticSegmentationEvaluator, segmentation_collate)
    weights=torch.load(head_dir/'best.pt',map_location='cpu',weights_only=False)
    c=weights['config']
    model=EventStateSegmentationHead(in_channels=c['in_channels'],num_classes=c['num_classes'],
        width=c['head_width'],output_size=tuple(c['label_size']),head_type=c['head_type']).to(device).eval()
    model.load_state_dict(weights['model'])
    meters={g:SemanticSegmentationEvaluator(c['num_classes']) for g in
            ('all','low','other','low_500ms_plus',*split.validation)}
    support={g:0 for g in meters}
    with torch.inference_mode():
        for seq in split.validation:
            times=json.loads((cache/seq/'low_duration_us.json').read_text())
            ds=M3EDSemanticFeatureDataset(feature_cache_dir=cache,target_cache_dir=target_root,
                                          sequences=[seq],feature=feature,horizontal_flip_probability=0)
            for batch in DataLoader(ds,batch_size=c['batch_size'],collate_fn=segmentation_collate):
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,
                                    enabled=c['precision']=='fp16' and torch.device(device).type=='cuda'):
                    labels=model(batch['features'].to(device)).argmax(1).cpu()
                for j,i in enumerate(batch['frame_indices']):
                    duration=times[str(i)]
                    for g in ('all',seq,'low' if duration else 'other',*(('low_500ms_plus',) if duration>=500000 else ())):
                        meters[g].update(labels[j],batch['labels'][j]);support[g]+=1
    results={g:dict(frames=support[g],valid_pixels=int(m.confusion.sum()),
                    mIoU=m.compute()['mIoU'] if m.confusion.sum() else None) for g,m in meters.items()}
    (head_dir/'continuous_groups.json').write_text(json.dumps(results,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','prepared-root','target-root','cache','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--extra-target-root',type=Path,help='Prefer available semantic targets here (e.g. additionally prepared Rittenhouse)')
    p.add_argument('--features',nargs='+',choices=['z','h','concat'],default=['z','h','concat'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    import h5py
    from event_state.dense.temporal import LowActivityClock
    import hdf5plugin
    import numpy as np
    from event_state.segmentation import load_m3ed_semantic_split
    from event_state.dense.common_features import CommonFeatures
    from event_state.dense.source import prepared_frames,input_geometry
    from event_state.dense.runtime import atomic_save
    from event_state.training import load_checkpoint_config_metadata
    from event_state.dense.common_protocol import audit_common
    cfg=load_checkpoint_config_metadata(a.checkpoint).config
    audit_common(cfg)
    if cfg['dataset']['name']!='m3ed': raise ValueError('M3ED backbone required')
    split=load_m3ed_semantic_split(MANIFEST)
    common=load_common('m3ed')['splits']
    if list(split.train)!=common['train'] or list(split.validation)!=common['val']: raise ValueError('Split mismatch')
    keys=sorted(set(k for f in a.features for k in (('z','h') if f=='concat' else (f,))))
    threshold=float(cfg['dataset']['normalization_provenance']['event_rate_p10_hz'])
    targets={}; paths={}; sizes=0
    identity=dict(low_rate_p10_hz=threshold,checkpoint_sha256=sha256_file(a.checkpoint),features=keys,split_sha256=sha256_file(MANIFEST),sources={})
    for seq in (*split.train,*split.validation):
        raw=a.prepared_root/seq/'metadata.json'
        for base in (a.extra_target_root,a.target_root):
            if not base: continue
            path=base/seq/'targets.h5'
            if not path.is_file(): continue
            with h5py.File(path) as candidate:
                if 'semantics_11' in candidate and 'semantics_frame_valid' in candidate: break
        else: raise ValueError(f'{seq}: no semantic targets; depth-only targets cannot be used')
        paths[seq]=path.parent.resolve()
        meta=json.loads(raw.read_text()); timestamps=np.array(meta['timestamps'])
        with h5py.File(path) as data:
            if not np.array_equal(data['timestamps'][:],timestamps) or data['semantics_11'].shape!=(len(timestamps),352,640):
                raise ValueError(f'{seq}: semantic label geometry/timeline mismatch')
            valid=np.flatnonzero(data['semantics_frame_valid'][:]); valid=valid[valid>0]
        if not len(valid): raise ValueError(f'{seq}: no labels')
        targets[seq]={int(i):int(timestamps[i]) for i in valid}
        sizes+=len(valid)
        identity['sources'][seq]=[sha256_file(raw),sha256_file(path)]
    estimate=sizes*(len(keys)*384*22*40*2+4096)
    print(f'{sizes} labeled samples; approx {estimate/2**30:.1f} GiB feature storage',flush=True)
    if a.preflight: return
    if a.output.exists(): raise FileExistsError(a.output)
    if a.cache.exists():
        if not (a.cache/'_SUCCESS').exists() or json.loads((a.cache/'identity.json').read_text())!=identity:
            raise ValueError('Incomplete/different cache; preserved')
    else:
        a.cache.parent.mkdir(exist_ok=True,parents=True)
        if shutil.disk_usage(a.cache.parent).free < estimate*1.1: raise OSError('Insufficient feature cache space')
        net=CommonFeatures(a.checkpoint,'m3ed',a.device)
        a.cache.mkdir()
        for seq,expected in targets.items():
            dest=a.cache/seq;dest.mkdir();net.state=None; seen=[]; clock=LowActivityClock(); durations={}
            for frame in prepared_frames(a.prepared_root,seq,net.cfg,'m3ed'):
                i=frame['frame_index']
                lo=int(frame['previous_timestamp']); hi=int(frame['timestamp'])
                duration=clock.update(lo,hi,frame['event_count']*1e6/(hi-lo)<=threshold)
                maps=net.step(input_geometry(frame['events'],'m3ed'),keys if i in expected else ())
                if i not in expected: continue
                if expected[i]!=frame['timestamp']: raise ValueError('Misaligned semantic target')
                atomic_save(dict(task='m3ed_semantic',features=maps,sequence_name=seq,frame_index=i,
                                 timestamp=expected[i],state_policy='continuous',checkpoint_sha256=identity['checkpoint_sha256']),dest/f'{i:06d}.pt')
                seen.append(i); durations[i]=duration
            if seen!=list(expected): raise ValueError(f'{seq}: incomplete labels')
            (dest/'low_duration_us.json').write_text(json.dumps(durations))
            (dest/'metadata.json').write_text(json.dumps(dict(task='m3ed_semantic',features=keys,label_size=[352,640],patch_size=16,
                sequence_name=seq,checkpoint_sha256=identity['checkpoint_sha256'],state_policy='continuous',cached_frame_indices=seen)))
        del net
        (a.cache/'identity.json').write_text(json.dumps(identity))
        (a.cache/'_SUCCESS').touch()
    a.output.mkdir(parents=True)
    view=a.output/'targets'
    view.mkdir()
    for seq,path in paths.items(): (view/seq).symlink_to(path,target_is_directory=True)
    for feature in a.features:
        subprocess.run([sys.executable,str(ROOT/'tools/train_m3ed_semantic.py'),'--split-manifest',str(MANIFEST),
            '--feature-cache-dir',str(a.cache),'--target-cache-dir',str(view),'--feature',feature,
            '--output-dir',str(a.output/feature),'--device',a.device],check=True)
        score_groups(a.output/feature,a.cache,view,split,feature,a.device)
    (a.output/'_SUCCESS').touch()


if __name__=='__main__': main()
