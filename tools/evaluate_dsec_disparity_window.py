#!/usr/bin/env python3
"""Choose accumulation duration using only the four labeled F3 public validation sequences."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_dsec_depth_submission import parse_window_ms, predict_sequence
from event_state.dense.protocol import load_protocol, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dsec', type=Path, required=True)
    p.add_argument('--window-ms', type=parse_window_ms, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    import numpy as np
    from event_state.data.event_representation import GEPEventFrame
    from event_state.dense.benchmark_inference import DepthPredictor
    from event_state.dense.targets import read_target
    protocol = load_protocol('dsec_f3')
    index = json.loads((a.run/'targets/dsec_val.json').read_text())
    names = protocol['splits']['val']
    if (index['protocol_sha256'] != digest(protocol) or index['role'] != 'val' or
            index['task'] != 'depth' or not index['labeled'] or
            {r['sequence'] for r in index['records']} != set(names)):
        raise ValueError('Expected labeled dsec_f3 validation depth index with native disparity targets')
    if a.output.exists(): raise FileExistsError(a.output)
    model = DepthPredictor(a.run, 'dsec_alpha1', 'concat', a.device)
    cfg = model.cfg.dataset
    if cfg.get('event_window_us') is not None or cfg.get('event_window_fraction',1) != 1 or not cfg.rectify_events:
        raise ValueError('Requires original rectified RGB-interval checkpoint')
    gep = GEPEventFrame(480,640,float(cfg.representation.percentile))
    data = {}
    for seq in names:
        ev = a.dsec/'train_events'/seq/'events/left'
        timeline = np.atleast_1d(np.loadtxt(a.dsec/'train_images'/seq/'images/timestamps.txt', dtype=np.int64))
        records = [r for r in index['records'] if r['sequence']==seq]
        if len(timeline)<2 or np.any(np.diff(timeline)<=0):
            raise ValueError('Invalid training sequence timestamp order')
        # The first RGB timestamp has no previous interval, for every candidate.
        rows = {r['timestamp']:r for r in records if r['timestamp']!=timeline[0]}
        if not rows or not set(rows)<=set(timeline[1:]): raise ValueError('Unmatched val timestamps')
        q = rows[min(rows)]['disparity_to_depth_q']
        err = 0.; pixels = 0; frames = 0
        for stamp, disp in predict_sequence(model,gep,ev,timeline,rows,q,a.window_ms):
            gt, valid = read_target(rows[stamp],'disparity')
            encoded = np.clip(disp*256,0,65535).astype(np.uint16)
            err += float(np.abs(encoded[valid]/256. - gt[valid]).sum())
            pixels += int(valid.sum()); frames += 1
        if frames != len(rows) or not pixels: raise ValueError('Incomplete val coverage')
        data[seq] = dict(frames=frames,valid_pixels=pixels,abs_error_sum=err,mae_px=err/pixels)
    result = dict(role='public_val',window_ms=a.window_ms,model=model.identity,sequences=data,
                  pooled_mae_px=sum(v['abs_error_sum'] for v in data.values())/sum(v['valid_pixels'] for v in data.values()),
                  mean_sequence_mae_px=sum(v['mae_px'] for v in data.values())/len(data))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
