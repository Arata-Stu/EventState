#!/usr/bin/env python3
"""One fixed public-val-selected EventState model -> official DSEC test disparity ZIP."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_state.dense.protocol import load_protocol, sha256_file
from prepare_dense_targets import parse_test_csv, unique_path


def parse_window_ms(value):
    if value == 'rgb-interval':
        return None
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise argparse.ArgumentTypeError('use rgb-interval or a positive integer in milliseconds')
    return int(value)


def window_bounds(previous, target, window_ms):
    # Longer windows may overlap across updates, always within one sequence.
    return (previous if window_ms is None else target - 1000 * window_ms), target


def predict_sequence(model, gep, ev, timeline, requested, q, window_ms):
    import numpy as np
    import torch
    from event_state.data.dsec import DSECEventReader
    from event_state.dense.targets import depth_to_disparity
    model.state = None
    last_target = max(requested)
    src = DSECEventReader(ev/'events.h5',rectify_map_path=ev/'rectify_map.h5',
                          output_height=480,output_width=640)
    try:
        for prev, end in zip(timeline,timeline[1:]):
            lo, hi = window_bounds(int(prev), int(end), window_ms)
            if hi > last_target: break
            events = src.slice(lo,hi)
            x = gep(*(torch.from_numpy(events[k]) for k in ('x','y','t','p')),start_time=lo,end_time=hi)
            pred = model.step(x, hi in requested)
            if pred is not None:
                disp = depth_to_disparity(pred.cpu().numpy(),q)
                if not np.isfinite(disp).all(): raise ValueError('Nonfinite disparity')
                yield hi, disp
    finally:
        src.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('run','dsec','timestamps','output'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--window-ms', type=parse_window_ms, default=None, metavar='MS|rgb-interval',
                   help='One causal accumulation window for all sequences. Default: previous RGB interval.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--teacher-weights', type=Path)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--name', default='dsec_alpha1')
    p.add_argument('--feature', default='concat')
    p.add_argument('--backbone', type=Path)
    a = p.parse_args()
    import numpy as np
    import yaml
    from event_state.dense.targets import depth_to_disparity
    official=json.loads((Path(__file__).parent/'manifests/dense/dsec_submission_csv.json').read_text())
    plan, identities = {}, {}
    for seq in load_protocol('dsec_f3')['splits']['test']:
        csv = unique_path(list(a.timestamps.rglob(seq+'.csv')) + [a.timestamps/seq/'timestamps.csv'])
        if sha256_file(csv) != official['csv_sha256'][seq]:
            raise ValueError(f'Official timestamp CSV hash mismatch: {csv}')
        rows = parse_test_csv(csv)
        ev = a.dsec/'test_events'/seq/'events/left'
        times = a.dsec/'test_images'/seq/'images/timestamps.txt'
        cal = unique_path([a.dsec/'test_calibration'/seq/d/'cam_to_cam.yaml' for d in ('calibration','')])
        q = yaml.safe_load(cal.read_text())['disparity_to_depth']['cams_03']
        depth_to_disparity(np.ones((1,1)), q)
        for f in (ev/'events.h5', ev/'rectify_map.h5', times):
            if not f.is_file(): raise FileNotFoundError(f)
        timeline = np.atleast_1d(np.loadtxt(times,dtype=np.int64))
        if timeline.ndim != 1 or np.any(np.diff(timeline) <= 0) or not {t for t,_ in rows} <= set(timeline[1:]):
            raise ValueError(f'{seq}: test timestamps not covered by causal RGB intervals')
        plan[seq] = ev, timeline, dict(rows), q
        identities[seq] = dict(csv_sha256=sha256_file(csv),q_sha256=sha256_file(cal), frames=len(rows))
    expected = {f'{s}/{idx:06d}.png' for s, (_,_,req,_) in plan.items() for idx in req.values()}
    print(f'{len(plan)} test sequences, {len(expected)} official images', flush=True)
    if len(expected) != 552:
        raise ValueError('Official test must have 552 images')
    print('Window:', 'previous RGB interval' if a.window_ms is None else f'{a.window_ms} ms trailing')
    if a.preflight: return
    from PIL import Image
    from event_state.data.event_representation import GEPEventFrame
    from event_state.dense.benchmark_inference import DepthPredictor
    model = DepthPredictor(a.run, a.name, a.feature, a.device, a.teacher_weights, backbone=a.backbone)
    cfg = model.cfg.dataset
    if cfg.get('event_window_us') is not None or cfg.get('event_window_fraction',1) != 1 or not cfg.rectify_events:
        raise ValueError('Expected original rectified RGB-interval GEP checkpoint')
    gep = GEPEventFrame(height=480,width=640,percentile=float(cfg.representation.percentile))
    a.output.mkdir(parents=True,exist_ok=False)
    tmp = a.output/'submission.partial.zip'
    clipped = 0
    with zipfile.ZipFile(tmp,'x') as archive:
        for seq,(ev,timeline,requested,q) in plan.items():
            for hi, disp in predict_sequence(model, gep, ev, timeline, requested, q, a.window_ms):
                scaled = disp*256
                clipped += int(np.count_nonzero((scaled<0)|(scaled>65535)))
                png = io.BytesIO()
                Image.fromarray(np.clip(scaled,0,65535).astype(np.uint16)).save(png,format='PNG')
                archive.writestr(f'{seq}/{requested[hi]:06d}.png',png.getvalue())
            print(f'{seq}: exported {len(requested)}',flush=True)
    with zipfile.ZipFile(tmp) as archive:
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected) or archive.testzip():
            raise ValueError('Incomplete ZIP; not ready to submit')
        for name in expected:
            content = archive.read(name)
            # IHDR: native width, height, 16 bit depth and grayscale color type.
            if content[16:26] != bytes.fromhex('00000280000001e01000'):
                raise ValueError(f'Wrong PNG encoding or dimensions: {name}')
    dest = a.output/'submission.zip'
    (a.output/'manifest.json').write_text(json.dumps(dict(model=model.identity, selected=f'{a.name}/{a.feature}, public val',
        window_ms=a.window_ms, window_mode='rgb_interval' if a.window_ms is None else 'fixed_trailing',
        pretraining_audit=model.pretraining_audit, training_window='rgb_interval',
        inputs='left events only; RGB timestamps; continuous state', sequences=identities, png_saturated_pixels=clipped,
        encoding='inverse calibrated event Q, clip to PNG representable range then uint16(d*256)',sha256=sha256_file(tmp)),indent=2))
    tmp.rename(dest)
    print(f'Prepared {dest}; run official checker before upload.')


if __name__ == '__main__': main()
