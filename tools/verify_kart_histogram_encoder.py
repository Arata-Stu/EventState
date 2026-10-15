#!/usr/bin/env python3
"""Server-only parity check: EventState transform/encoder versus kart adapter.

Uses synthetic VGA integer counts, not camera data. Does not validate calibration,
ROS timestamps, CUDA accumulation, TensorRT or driving behavior.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--kart-root',type=Path,required=True)
    p.add_argument('--dinov3-repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from omegaconf import OmegaConf
    from event_state.training.data import _build_transform
    from event_state.models.event_encoder import EventEncoder
    sys.path.insert(0,str(a.kart_root/'ros2_ws/src/kart_e2e'))
    from kart_e2e.evs_model import load_evs_encoder, encode_openeb_counts
    from kart_e2e.model import backbone
    bundle=torch.load(a.weights,map_location='cpu',weights_only=True)
    meta=bundle['metadata']
    kart=load_evs_encoder(a.weights,a.dinov3_repo,a.device)
    ref=EventEncoder(in_channels=20,patch_init='random',backbone=backbone(a.dinov3_repo))
    ref.backbone.load_state_dict(bundle['state_dict'],strict=True)
    ref=ref.to(a.device).eval()
    transform=_build_transform(OmegaConf.create(meta['dataset']),training=False,channels=20)
    torch.manual_seed(0)
    reports=[]
    with torch.inference_mode():
        for label in ['empty','counts']:
            counts=torch.zeros(1,20,480,640) if label=='empty' else torch.randint(0,3,(1,20,480,640)).float()
            normalized,_=transform(counts,None)
            h,w=kart.content_shape
            # Direct integer sum avoids using the training pooling implementation.
            low=counts.reshape(1,20,h,480//h,w,640//w).sum((3,5)).to(a.device)
            other=kart.prepare_content(low)
            torch.testing.assert_close(normalized.to(a.device),other,rtol=1e-6,atol=1e-6)
            expected=ref(normalized.to(a.device))
            actual=encode_openeb_counts(kart,low)
            torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-5)
            reports.append(dict(case=label,input_max_error=float((normalized.to(a.device)-other).abs().max()),
                token_max_error=float((expected-actual).abs().max()),shape=list(actual.shape)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='passed',scope='synthetic counts and real exported weights; not ROS',
        weights=str(a.weights.resolve()),cases=reports),indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()
