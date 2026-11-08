#!/usr/bin/env python3
"""DSEC-Flow frozen common Stage 1 probe: paired cache, labeled train/val, hidden-test ZIP."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from event_state.flow.protocol import splits,labeled_pairs,test_pairs,read_gt,encode_png
from event_state.dense.protocol import sha256_file,digest


def data_plan(a,role):
    import numpy as np
    names=splits()[role]
    hidden=test_pairs(a.test_timestamps) if role=='test' else None
    result={}
    for seq in names:
        pairs=hidden[seq] if hidden else labeled_pairs(a.flow_root,seq)
        split='test' if hidden else 'train'
        ev=a.dsec/f'{split}_events'/seq/'events/left'
        for name in ('events.h5','rectify_map.h5'):
            if not (ev/name).is_file(): raise FileNotFoundError(ev/name)
        timeline=np.loadtxt(a.dsec/f'{split}_images'/seq/'images/timestamps.txt',dtype=np.int64).tolist()
        if timeline!=sorted(set(timeline)) or len(timeline)<2: raise ValueError('Invalid RGB timeline')
        # First frame has no preceding GEP interval. It can only be omitted for labeled data.
        if not hidden: pairs=[p for p in pairs if p[0]!=timeline[0]]
        if not pairs or not {t for u,v,_ in pairs for t in (u,v)}<=set(timeline[1:]):
            raise ValueError(f'{seq}: unmatched flow timestamps')
        result[seq]=(ev,timeline,pairs)
    return result


def extract(net, ev, timeline, pairs, feature, window):
    import torch
    from event_state.data.dsec import DSECEventReader
    from event_state.data.event_representation import GEPEventFrame
    from export_dsec_depth_submission import window_bounds
    ds=net.cfg.dataset
    if not ds.rectify_events: raise ValueError('Flow targets require rectified event input')
    rep=GEPEventFrame(480,640,float(ds.representation.percentile))
    keys=('z','h') if feature=='concat' else (feature,)
    requested={t for u,v,_ in pairs for t in (u,v)};last=max(requested)
    net.state=None
    reader=DSECEventReader(ev/'events.h5',rectify_map_path=ev/'rectify_map.h5',output_height=480,output_width=640)
    try:
        for prior,end in zip(timeline,timeline[1:]):
            if end>last: break
            start,_=window_bounds(prior,end,window)
            event=reader.slice(start,end)
            image=rep(*(torch.from_numpy(event[k]) for k in ('x','y','t','p')),start_time=start,end_time=end)
            maps=net.step(image,keys if end in requested else ())
            if end in requested: yield end,torch.cat([maps[k] for k in keys],dim=0)
    finally:
        reader.close()


def cache(a):
    from event_state.training import load_checkpoint_config_metadata
    from event_state.dense.common_protocol import audit_common,load_common
    cfg=load_checkpoint_config_metadata(a.checkpoint).config
    audit_common(cfg)
    if cfg['dataset']['name']!='dsec' or set(splits()['val'])&set(load_common('dsec')['splits']['train']):
        raise ValueError('Wrong backbone or flow holdout contamination')
    plans={r:data_plan(a,r) for r in ('train','val')}
    n=sum(len({t for u,v,_ in pairs for t in (u,v)}) for plan in plans.values() for ev,ts,pairs in plan.values())
    estimate=n*(384*(2 if a.feature=='concat' else 1)*30*40*2+4096)
    print(f'Flow 14 train / 4 val: {n} endpoint maps, approx {estimate/2**30:.1f} GiB',flush=True)
    if a.stage=='preflight':return
    from event_state.dense.runtime import atomic_save
    from event_state.dense.common_features import CommonFeatures
    if a.cache.exists():raise FileExistsError(a.cache)
    a.cache.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.cache.parent).free<estimate*1.1:raise OSError('Insufficient space for paired feature cache')
    net=CommonFeatures(a.checkpoint,'dsec',a.device);a.cache.mkdir()
    meta=dict(feature=a.feature,window_ms=a.window_ms,backbone=sha256_file(a.checkpoint),audit=net.audit,roles={})
    for role,plan in plans.items():
        meta['roles'][role]={}
        for seq,(ev,ts,pairs) in plan.items():
            dest=a.cache/seq;dest.mkdir()
            for t,x in extract(net,ev,ts,pairs,a.feature,a.window_ms): atomic_save(x,dest/f'{t}.pt')
            meta['roles'][role][seq]=pairs
            print(f'{role}: {seq} {len(pairs)} pairs',flush=True)
    (a.cache/'manifest.json').write_text(json.dumps(meta));(a.cache/'_SUCCESS').touch()


def fit(a):
    import torch
    from torch.utils.data import DataLoader
    from event_state.flow.probe import FlowProbe
    from event_state.dense.runtime import atomic_save
    if not (a.cache/'_SUCCESS').exists():raise ValueError('Incomplete cache')
    meta=json.loads((a.cache/'manifest.json').read_text())
    for role in ('train','val'):
        if set(meta['roles'][role])!=set(splits()[role]):raise ValueError('Wrong split in cache')
    if a.output.exists():raise FileExistsError(a.output)
    class Pairs:
        def __init__(self,role):
            self.rows=[(seq,u,v,path) for seq,rows in meta['roles'][role].items() for u,v,path in rows]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            seq,u,v,path=self.rows[i]
            x,y=[torch.load(a.cache/seq/f'{t}.pt',map_location='cpu',weights_only=True).float() for t in (u,v)]
            gt,valid=read_gt(path)
            return x,y,torch.from_numpy(gt).permute(2,0,1),torch.from_numpy(valid),seq
    torch.manual_seed(a.seed)
    train=Pairs('train');val=Pairs('val');channels=train[0][0].shape[0]
    model=FlowProbe(channels).to(a.device)
    loaders=[DataLoader(ds,batch_size=a.batch_size,shuffle=role=='train',num_workers=0) for role,ds in [('train',train),('val',val)]]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    config=dict(feature=meta['feature'],window_ms=meta['window_ms'],backbone=meta['backbone'],channels=channels,seed=a.seed,
                cache_manifest_sha=sha256_file(a.cache/'manifest.json'),splits=splits(),epochs=a.epochs,lr=a.lr,batch_size=a.batch_size)
    a.output.mkdir(parents=True)
    best=float('inf')
    for epoch in range(1,a.epochs+1):
        for training,loader in zip((True,False),loaders):
            model.train(training)
            per={}
            with torch.set_grad_enabled(training):
                for x,y,gt,mask,seq in loader:
                    pred=model(x.to(a.device),y.to(a.device))
                    err=torch.linalg.vector_norm(pred-gt.to(a.device),dim=1); mask=mask.to(a.device)
                    if not torch.isfinite(err[mask]).all():raise ValueError('Nonfinite EPE')
                    if not mask.any():continue
                    if training:
                        opt.zero_grad();err[mask].mean().backward();opt.step()
                    else:
                        for j,s in enumerate(seq):
                            v=err[j][mask[j]]
                            sums=per.setdefault(s,[0.,0,0,0,0,0.])
                            vpred=pred[j,:,mask[j]].double()
                            vgt=gt[j].to(a.device)[:,mask[j]].double()
                            cosine=((vpred*vgt).sum(0)+1)/((vpred.square().sum(0)+1)*(vgt.square().sum(0)+1)).sqrt()
                            sums[5]+=torch.rad2deg(torch.acos(cosine.clamp(-1,1))).sum().item()
                            sums[0]+=v.double().sum().item();sums[1]+=v.numel()
                            for k in (1,2,3):sums[k+1]+=(v>k).sum().item()
        if set(per)!=set(splits()['val']) or any(s[1]==0 for s in per.values()):raise ValueError('Missing validation coverage')
        scores={k:{'EPE':v[0]/v[1],'pixels':v[1],'AE_deg':v[5]/v[1],**{f'{i}PE_percent':100*v[i+1]/v[1] for i in (1,2,3)}} for k,v in per.items()}
        score=sum(s['EPE'] for s in scores.values())/len(scores)
        record=dict(epoch=epoch,val_mean_sequence_EPE=score,val=scores)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if score<best:
            best=score;atomic_save(dict(model=model.state_dict(),config=config,validation=record),a.output/'best.pt')
        print(f'{epoch}/{a.epochs} validation EPE={score:.4f}px',flush=True)
    (a.output/'_SUCCESS').touch()


def export(a):
    import torch
    from event_state.flow.probe import FlowProbe
    from event_state.dense.common_features import CommonFeatures
    if not (a.head.parent/'_SUCCESS').exists():raise ValueError('Use completed val-selected head')
    head=torch.load(a.head,map_location='cpu',weights_only=False);c=head['config']
    if c['backbone']!=sha256_file(a.checkpoint) or c['splits']!=splits():raise ValueError('Backbone/split mismatch')
    plan=data_plan(a,'test')
    net=CommonFeatures(a.checkpoint,'dsec',a.device)
    model=FlowProbe(c['channels']).to(a.device).eval();model.load_state_dict(head['model'])
    a.output.mkdir(parents=True,exist_ok=False);tmp=a.output/'submission.partial.zip';expected=set()
    with torch.inference_mode(),zipfile.ZipFile(tmp,'x') as z:
        for seq,(ev,ts,pairs) in plan.items():
            ending={};uses=Counter()
            for u,v,i in pairs:
                ending.setdefault(v,[]).append((u,i));uses[u]+=1;uses[v]+=1
                expected.add(f'{seq}/{i:06d}.png')
            held={}
            for stamp,x in extract(net,ev,ts,pairs,c['feature'],c['window_ms']):
                held[stamp]=x
                for u,i in ending.get(stamp,[]):
                    pred=model(held[u][None].to(a.device).float(),x[None].to(a.device).float())[0].permute(1,2,0).cpu().numpy()
                    z.writestr(f'{seq}/{i:06d}.png',encode_png(pred))
                    for key in (u,stamp):
                        uses[key]-=1
                        if not uses[key]:del held[key]
    with zipfile.ZipFile(tmp) as z:
        if len(expected)!=416 or set(z.namelist())!=expected or len(z.namelist())!=416 or z.testzip():raise ValueError('Incomplete flow ZIP')
    (a.output/'manifest.json').write_text(json.dumps(dict(head_sha256=sha256_file(a.head),config=c,frames=len(expected),zip_sha256=sha256_file(tmp)),indent=2))
    tmp.rename(a.output/'submission.zip')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','cache','train','export'])
    for k in ('dsec','flow-root','checkpoint','cache','output','head','test-timestamps'):p.add_argument('--'+k,type=Path)
    from export_dsec_depth_submission import parse_window_ms
    p.add_argument('--window-ms',type=parse_window_ms,default=None);p.add_argument('--feature',choices=['z','h','concat'],default='h')
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=.0002);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    required=dict(preflight=['checkpoint','dsec','flow_root'],cache=['checkpoint','dsec','flow_root','cache'],
                  train=['output','cache'],export=['output','checkpoint','dsec','head','test_timestamps'])
    if any(getattr(a,k) is None for k in required[a.stage]):p.error('required: '+', '.join(required[a.stage]))
    if min(a.epochs,a.batch_size,a.lr)<=0:p.error('Invalid training settings')
    if a.stage in ('preflight','cache'):cache(a)
    elif a.stage=='train':fit(a)
    else:export(a)


if __name__=='__main__':main()
