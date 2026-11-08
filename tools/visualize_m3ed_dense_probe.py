#!/usr/bin/env python3
"""Render frozen continuous M3ED depth readouts using existing evaluated caches."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--heads', type=Path, required=True)
    p.add_argument('--prepared-root', type=Path, required=True)
    p.add_argument('--sequence', default='car_urban_day_ucity_small_loop')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=600)
    p.add_argument('--fps', type=float, default=20, help='Playback frames/sec; one labeled sample per video frame')
    p.add_argument('--display-max', type=float, default=80)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.start_frame < 0 or a.max_frames < 1 or not 0 < a.fps <= 240 or not 0 < a.display_max < float('inf'):
        p.error('Invalid frame range, fps or depth color limit')
    if a.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    import numpy as np
    import torch
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from event_state.dense.data import DenseFeatureDataset
    from event_state.dense.head import build_dense_head
    from event_state.dense.protocol import sha256_file
    from compare_dense_probes import compare
    from visualize_m3ed_downstream import _crop_rgb, _event_image

    features = ('z', 'h', 'concat')
    models, datasets, reports, configs = {}, {}, {}, {}
    for feature in features:
        head = a.heads / feature / 'best.pt'
        report = json.loads((a.heads / feature / 'continuous.json').read_text())
        if report['head_sha256'] != sha256_file(head) or report['cache_metadata_sha256'] != sha256_file(a.cache / 'metadata.json'):
            raise ValueError('Head/cache differs from evaluated result')
        payload = torch.load(head, map_location='cpu', weights_only=False)
        cfg = payload['config']
        ds = DenseFeatureDataset(a.cache, feature, minimum=cfg['minimum'], maximum=cfg['maximum'])
        if ds.meta['dataset'] != 'm3ed' or ds.meta['role'] != 'val' or ds.meta['state_policy'] != 'continuous' or cfg['task'] != 'depth' or cfg['feature'] != feature:
            raise ValueError('Requires evaluated M3ED val continuous depth heads')
        for key, expected in [('checkpoint_sha256', cfg['backbone_sha256']), ('protocol_sha256', cfg['protocol_sha256']),
                              ('geometry', cfg['geometry']), ('normalization_audit', cfg['normalization_audit'])]:
            if ds.meta[key] != expected:
                raise ValueError(f'Head provenance mismatch: {key}')
        model = build_dense_head(cfg['feature'], cfg['in_channels'], cfg['width']).to(a.device)
        model.load_state_dict(payload['model'], strict=True)
        models[feature] = model.eval()
        datasets[feature], reports[feature], configs[feature] = ds, report, cfg
    for feature in ('h', 'concat'):
        compare(reports['z'], reports[feature], 'readout')
    ds = datasets['z']
    indices = sorted((i for i, r in enumerate(ds.records) if r['sequence'] == a.sequence), key=lambda i: ds.records[i]['timestamp'])
    indices = indices[a.start_frame:a.start_frame + a.max_frames]
    if not indices:
        raise ValueError('Sequence/frame range has no labeled samples')
    prepared = a.prepared_root / a.sequence
    source = ds.meta['source_identity'][a.sequence]
    if sha256_file(prepared / 'metadata.json') != source['metadata_sha256']:
        raise ValueError('Prepared source metadata changed')
    for i in indices:
        stamp = ds.records[i]['timestamp']
        for path in (prepared / 'aligned_rgb' / f'{stamp}.png', prepared / 'events' / f'{stamp}.pt'):
            if not path.is_file():
                raise FileNotFoundError(path)

    # Fixed linear near-red / far-blue ramp, identical for predictions and GT.
    def color(depth, valid=None):
        u = np.clip(depth / a.display_max, 0, 1)
        rgb = np.stack([1-u, 1-np.abs(2*u-1), u], axis=-1)
        result = np.round(rgb * 255).astype(np.uint8)
        if valid is not None:
            result[~valid] = 0
        return result

    a.output_dir.mkdir(parents=True)
    manifest = dict(sequence=a.sequence, cache=str(a.cache.resolve()), heads=str(a.heads.resolve()),
                    cache_sha256=sha256_file(a.cache / 'metadata.json'),
                    head_sha256={f: reports[f]['head_sha256'] for f in features},
                    start_frame=a.start_frame, samples=len(indices), fps=a.fps,
                    timing='One labeled sample per video frame; timestamp gaps are not resampled. Not necessarily real time.',
                    state_policy='continuous cached from sequence start', display_max_m=a.display_max,
                    colors='linear: red near, blue far; invalid GT black; predictions dense',
                    selection='User sequence/frame range, not selected by prediction quality', configs=configs)
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    partial = a.output_dir / 'depth.partial.mp4'
    writer = imageio_ffmpeg.write_frames(str(partial), size=(960, 440), fps=a.fps,
                                        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p')
    writer.send(None)
    try:
        with (a.output_dir / 'frames.csv').open('w', newline='') as stream, torch.inference_mode():
            table = csv.writer(stream)
            table.writerow(['timestamp_us', 'low_activity', 'low_duration_ms', 'valid_pixels', 'z_mae', 'h_mae', 'concat_mae'])
            for number, i in enumerate(indices):
                sample = ds[i]
                stamp = sample['timestamp']
                target, valid = sample['target'].numpy(), sample['valid'].numpy()
                rgb = _crop_rgb(prepared / 'aligned_rgb' / f'{stamp}.png')
                events, _ = _event_image(prepared / 'events' / f'{stamp}.pt')
                panels = [('RGB', rgb), ('GEP events', events), ('GT (invalid=black)', color(target, valid))]
                errors = []
                for feature in features:
                    item = sample if feature == 'z' else datasets[feature][i]
                    prediction = models[feature](item['feature'][None].to(a.device), target.shape)[0].cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite prediction')
                    mae = float(np.abs(prediction[valid]-target[valid]).mean()) if valid.any() else None
                    errors.append(mae)
                    label = f'{feature} MAE={mae:.3f}m' if mae is not None else f'{feature} MAE=N/A'
                    panels.append((label, color(prediction)))
                canvas = Image.new('RGB', (960, 440), '#181818')
                draw = ImageDraw.Draw(canvas)
                for j, (label, array) in enumerate(panels):
                    x, y = (j % 3)*320, (j // 3)*202
                    canvas.paste(Image.fromarray(array).resize((320, 176)), (x, y+24))
                    draw.text((x+5, y+5), label, fill='white')
                draw.text((5, 407), f't={stamp}us low={sample["low_activity"]} duration={sample["low_duration_us"]/1000:.0f}ms', fill='white')
                draw.text((5, 423), f'Fixed depth colors: red=0m, blue={a.display_max:g}m | sampled playback {a.fps:g}fps', fill='white')
                writer.send(np.asarray(canvas).tobytes())
                table.writerow([stamp, sample['low_activity'], sample['low_duration_us']/1000, int(valid.sum()), *errors])
                if number % 100 == 0:
                    print(f'[depth-video] {number+1}/{len(indices)}', flush=True)
    finally:
        writer.close()
    partial.rename(a.output_dir / 'depth.mp4')
    print(f'[depth-video] complete: {a.output_dir / "depth.mp4"}')


if __name__ == '__main__':
    main()
