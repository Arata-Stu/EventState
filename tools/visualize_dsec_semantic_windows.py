#!/usr/bin/env python3
"""Stream paired RGB-interval/20ms Semantic inference into a video, without feature caches."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def paired_paths(raw_paths, cached_paths):
    """Never silently intersect timelines: missing inputs would change recurrent history."""
    raw = [int(p.stem) for p in raw_paths]
    cached = [int(p.stem) for p in cached_paths]
    if not raw or raw != sorted(set(raw)) or raw != cached:
        raise ValueError('20ms/raw and legacy cache timelines must match exactly from sequence start')
    return list(zip(raw_paths, cached_paths))


def validate_windows(legacy, short):
    for cfg in (legacy, short):
        ds = cfg['dataset']
        if ds['name'] != 'dsec' or ds['representation']['type'] != 'gep_rgb':
            raise ValueError('This renderer requires DSEC GEP RGB checkpoints')
        if not ds.get('rectify_events', True) or ds.get('event_window_fraction', 1) != 1:
            raise ValueError('Expected rectified, full-fraction windows')
    if legacy['dataset'].get('event_window_us') is not None:
        raise ValueError('Legacy checkpoint must use RGB intervals, not a fixed window')
    if short['dataset'].get('event_window_us') != 20000:
        raise ValueError('Short checkpoint must use exactly 20000 us')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('backbone-50', 'backbone-20', 'head-50', 'head-20', 'dataset-root', 'event-cache-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--teacher-checkpoint', type=Path, required=True)
    p.add_argument('--sequence', default='zurich_city_07_a')
    p.add_argument('--feature', choices=('z', 'h', 'concat'), default='h')
    p.add_argument('--start-frame', type=int, default=0, help='Output offset; all preceding inputs still update state')
    p.add_argument('--max-frames', type=int, default=600)
    p.add_argument('--fps', type=float, default=20)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.start_frame < 0 or a.max_frames <= 0 or a.fps <= 0:
        p.error('start-frame >= 0, max-frames > 0, fps > 0 required')
    if a.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing videos are preserved')

    import numpy as np
    import torch
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from omegaconf import OmegaConf
    from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata
    from event_state.data.dsec_feature_source import RawGEPFeatureSource
    from event_state.data.dsec import event_representation_metadata, EVENT_WINDOW_NAME
    from event_state.training.data import build_event_representation
    from event_state.segmentation import EventStateSegmentationHead, load_dsec_semantic_split
    from cache_dsec_semantic_features import _semantic_transform, _sha256
    from visualize_dsec_semantic_sequences import _rgb_image, _semantic_image, PALETTE, DSEC_SEMANTIC_11_CLASSES

    split = load_dsec_semantic_split(Path(__file__).parent / 'manifests/dsec_semantic_split.yaml')
    if a.sequence not in split.val:
        raise ValueError('Use a development validation sequence; this tool does not select on test')
    configs = [load_checkpoint_config_metadata(path).config for path in (a.backbone_50, a.backbone_20)]
    validate_windows(*configs)
    configs = [OmegaConf.create(c) for c in configs]
    source = RawGEPFeatureSource(a.dataset_root, a.sequence, configs[1])
    writer = None
    try:
        cache = a.event_cache_dir / a.sequence
        meta = json.loads((cache / 'metadata.json').read_text())
        if meta.get('representation', {}).get('type') != 'gep_rgb' or meta.get('event_window_us') is not None:
            raise ValueError('Expected the legacy RGB-interval GEP event cache')
        if (meta.get('event_window') != EVENT_WINDOW_NAME
                or meta.get('coordinate_space') != 'rectified_event'
                or meta.get('representation') != event_representation_metadata(build_event_representation(configs[0].dataset))):
            raise ValueError('Legacy cache representation/window/coordinates do not match checkpoint')
        paths = sorted((p for p in cache.glob('*.pt') if p.stem.isdecimal()), key=lambda p: int(p.stem))
        pairs = paired_paths(source.paths, paths)
        if a.start_frame >= len(pairs):
            raise ValueError('start-frame outside sequence')
        models, heads, head_configs = [], [], []
        for cfg, backbone, head_path in zip(configs, (a.backbone_50, a.backbone_20), (a.head_50, a.head_20)):
            OmegaConf.update(cfg, 'teacher.checkpoint', str(a.teacher_checkpoint), force_add=True)
            OmegaConf.update(cfg, 'model.event_encoder.checkpoint', str(a.teacher_checkpoint), force_add=True)
            model = build_model(cfg, device=torch.device(a.device))
            load_checkpoint(backbone, model=model, device=torch.device(a.device), restore_rng=False)
            model.eval().requires_grad_(False)
            payload = torch.load(head_path, map_location='cpu', weights_only=False)
            hc = payload['config']
            if hc.get('feature') != a.feature or int(hc.get('num_classes', 11)) != 11:
                raise ValueError('Head feature/classes mismatch')
            if a.sequence in hc.get('training_sequences', []):
                raise ValueError('Selected video sequence was used to train this head')
            if hc.get('label_size', [440, 640]) != [440, 640]:
                raise ValueError('Head geometry differs from DSEC Semantic')
            # Input channels are verified by strict weight loading and first inference.
            channels = int(OmegaConf.select(cfg, 'model.temporal.output_dim')) if a.feature == 'h' else int(OmegaConf.select(cfg, 'model.temporal.input_dim'))
            if a.feature == 'concat':
                channels += int(OmegaConf.select(cfg, 'model.temporal.output_dim'))
            if int(hc.get('in_channels', channels)) != channels:
                raise ValueError('Backbone/head channel count mismatch')
            head = EventStateSegmentationHead(in_channels=channels, num_classes=11,
                width=int(hc.get('head_width', 192)), output_size=(440, 640),
                head_type=hc.get('head_type', 'linear')).to(a.device)
            head.load_state_dict(payload['model'], strict=True)
            head.eval().requires_grad_(False)
            models.append(model)
            heads.append(head)
            head_configs.append(hc)
        a.output_dir.mkdir(parents=True)
        manifest = dict(sequence=a.sequence, feature=a.feature, state_policy='continuous',
            rgb_usage='reference only', fps=a.fps, timing='constant-fps preview; original timestamps in frames.csv',
            legacy_window='previous RGB timestamp to current RGB timestamp, approximately 50ms', short_window_us=20000,
            start_frame=a.start_frame, max_frames=a.max_frames, warmup_frames=a.start_frame,
            head_backbone_pairing='user-specified; old head checkpoints do not embed a backbone hash',
            classes=list(DSEC_SEMANTIC_11_CLASSES), palette=PALETTE.tolist(),
            input_sources={'legacy': str(cache), 'legacy_metadata_sha256': _sha256(cache / 'metadata.json'),
                           'raw_20ms': source.metadata},
            head_configs=head_configs, checkpoints={str(p): _sha256(p) for p in
                (a.backbone_50, a.backbone_20, a.head_50, a.head_20)}, complete=False)
        manifest_path = a.output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
        partial = a.output_dir / 'comparison.partial.mp4'
        # Five equally sized native-aspect panels, with a header showing time and input windows.
        writer = imageio_ffmpeg.write_frames(str(partial), size=(1600, 260), fps=a.fps,
            codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', output_params=['-crf', '18'])
        writer.send(None)
        states = [None, None]
        count = 0
        with (a.output_dir / 'frames.csv').open('w') as stream, torch.inference_mode():
            table = csv.writer(stream)
            table.writerow(['input_index', 'frame_index', 'timestamp_us', 'legacy_window_us', 'short_window_us'])
            for i, (raw_path, cache_path) in enumerate(pairs[:a.start_frame + a.max_frames]):
                raw = source.load(raw_path)
                old = torch.load(cache_path, map_location='cpu', weights_only=False)
                record = source.records[raw_path.stem]
                if int(old['timestamp']) != record.timestamp or int(old['frame_index']) != record.frame_index:
                    raise ValueError('Frame identity differs between input windows')
                predictions, event_images = [], []
                for j, payload in enumerate((old, raw)):
                    events = payload['events'].float()
                    if not torch.isfinite(events).all() or events.min() < 0 or events.max() > 1:
                        raise ValueError('Expected finite unnormalized GEP RGB in [0,1]')
                    x = _semantic_transform(events, configs[j]).to(a.device)
                    out = models[j](x.unsqueeze(1), state=states[j])
                    states[j] = out['state']
                    if i < a.start_frame:
                        continue
                    names = ('z', 'h') if a.feature == 'concat' else (a.feature,)
                    grid = (x.shape[-2] // models[j].event_encoder.patch_size,
                            x.shape[-1] // models[j].event_encoder.patch_size)
                    maps = [out[n][0, 0].reshape(*grid, -1).permute(2, 0, 1) for n in names]
                    # Match the frozen head training's FP16 feature-cache rounding, without saving caches.
                    features = torch.cat(maps).half().float().unsqueeze(0)
                    predictions.append(_semantic_image(heads[j](features).argmax(1)[0].cpu()))
                    event_images.append(Image.fromarray((events[:, :440, :640].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)))
                if i < a.start_frame:
                    continue
                window = record.timestamp - record.previous_timestamp
                panels = [_rgb_image(a.dataset_root, a.sequence, record.timestamp), event_images[0],
                          predictions[0], event_images[1], predictions[1]]
                titles = ['RGB reference', f'Events {window / 1000:.1f}ms', f'Legacy prediction ({a.feature})',
                          'Events 20ms', f'20ms prediction ({a.feature})']
                canvas = Image.new('RGB', (1600, 260), (24, 24, 24))
                draw = ImageDraw.Draw(canvas)
                for col, (panel, title) in enumerate(zip(panels, titles)):
                    draw.text((320 * col + 5, 4), title, fill='white')
                    canvas.paste(panel.resize((320, 220), Image.Resampling.NEAREST), (320 * col, 20))
                draw.text((5, 243), f'{a.sequence} | timestamp={record.timestamp} us | input={i} | continuous', fill='white')
                writer.send(np.asarray(canvas).tobytes())
                table.writerow([i, record.frame_index, record.timestamp, window, 20000])
                count += 1
                if count % 100 == 0:
                    print(f'Rendered {count} frames', flush=True)
        writer.close()
        writer = None
        partial.rename(a.output_dir / 'comparison.mp4')
        manifest.update(complete=True, rendered_frames=count)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
        print(a.output_dir / 'comparison.mp4')
    finally:
        if writer is not None:
            writer.close()
        source.close()


if __name__ == '__main__':
    main()
