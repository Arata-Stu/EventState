#!/usr/bin/env python3
"""Render a portable encoder on chronological DSEC frames without feature caches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start', type=int, default=0, help='Zero-based valid frame offset')
    p.add_argument('--frames', type=int, default=400)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dinov3-repo', type=Path)
    a = p.parse_args()
    if a.start < 0 or a.frames < 2 or a.output.suffix != '.mp4':
        raise ValueError('Require start>=0, frames>=2 and .mp4 output')
    if a.output.exists() or a.output.with_suffix('.json').exists() or a.output.with_suffix('.pca.pt').exists():
        raise FileExistsError('Use a new output path')
    import numpy as np
    import torch
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from omegaconf import OmegaConf
    from deployment.event_encoder_model import load_encoder
    from event_state.data.dsec import DSECSequenceDataset
    from event_state.training.data import build_event_representation, _build_transform

    encoder = load_encoder(a.weights, device=a.device, dinov3_repo=a.dinov3_repo)
    if encoder.input_shape not in {(20, 240, 320), (20, 128, 160)}:
        raise ValueError('This renderer requires a supported 20ch preset')
    _, height, width = encoder.input_shape
    panel_height = round(320 * height / width)
    cfg = OmegaConf.create(encoder.metadata['dataset'])
    if cfg.representation.channels != 20 or cfg.representation.channel_layout != 'polarity_major':
        raise ValueError('Visualization expects polarity-major 20ch voxels')
    dataset = DSECSequenceDataset(a.root, split='train', sequences=[a.sequence], sequence_length=1,
        event_representation=build_event_representation(cfg),
        transform=_build_transform(cfg, training=False, channels=20),
        image_directory=cfg.image_directory, rectify_events=cfg.rectify_events,
        event_window_fraction=cfg.event_window_fraction)
    writer = None
    try:
        end = min(len(dataset), a.start + a.frames)
        if end - a.start < 2:
            raise ValueError('Need at least two available frames')
        indices = np.linspace(a.start, end - 1, min(32, end - a.start), dtype=int)
        samples = []
        with torch.inference_mode():
            for index in indices:
                sample = dataset[int(index)]
                feature = encoder(sample['events'].to(a.device)).float().cpu()[0]
                samples.append(torch.nn.functional.normalize(feature, dim=-1))
            merged = torch.cat(samples)
            mean = merged.mean(0)
            # Deterministic SVD, one basis and one color range for the entire video.
            _, _, vh = torch.linalg.svd(merged - mean, full_matrices=False)
            basis = vh[:3].T
            projected = (merged - mean) @ basis
            lower, upper = torch.quantile(projected, .01, dim=0), torch.quantile(projected, .99, dim=0)
            scale = (upper - lower).clamp_min(1e-6)
            timestamps = [int(dataset._frames_by_sequence[a.sequence][dataset.clip_records[i][1]].timestamp)
                          for i in range(a.start, end)]
            fps = 1e6 / float(np.median(np.diff(timestamps)))
            a.output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(str(a.output), (960, panel_height + 32), fps=fps,
                                                 codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p')
            writer.send(None)
            for index in range(a.start, end):
                sample = dataset[index]
                events = sample['events'].to(a.device)
                feature = torch.nn.functional.normalize(encoder(events).float().cpu()[0], dim=-1)
                colors = (((feature - mean) @ basis - lower) / scale).clamp(0, 1).reshape(height // 16, width // 16, 3)
                counts = (events * encoder.std + encoder.mean).clamp_min(0)[0].cpu()
                pos, neg = counts[:10].sum(0), counts[10:].sum(0)
                # Fixed log mapping, no per-frame min/max normalization.
                ev = torch.stack([torch.log1p(pos) / 5, torch.zeros_like(pos), torch.log1p(neg) / 5], -1).clamp(0, 1)
                rgb = sample['images'][0].permute(1, 2, 0).clamp(0, 1)
                panels = [rgb, ev, colors]
                canvas = Image.new('RGB', (960, panel_height + 32))
                draw = ImageDraw.Draw(canvas)
                for j, (panel, label) in enumerate(zip(panels, ['RGB', 'Events (+ red / - blue)', 'Encoder z / fixed PCA'])):
                    im = Image.fromarray((panel.numpy() * 255).astype('uint8'))
                    canvas.paste(im.resize((320, panel_height), Image.Resampling.NEAREST), (320*j, 32))
                    draw.text((320*j+5, 4), label, fill='white')
                draw.text((5, 18), f'{a.sequence}  t={int(sample["timestamps"][0])} us', fill='white')
                writer.send(np.asarray(canvas))
        writer.close()
        writer = None
        torch.save(dict(mean=mean, basis=basis, lower=lower, upper=upper, fit_indices=indices.tolist()),
                   a.output.with_suffix('.pca.pt'))
        a.output.with_suffix('.json').write_text(json.dumps(dict(
            weights=str(a.weights.resolve()), sequence=a.sequence, frame_offset=a.start,
            timestamps_us=timestamps, fps=fps, timing='constant fps from median timestamp interval',
            pca='L2 normalized z; fixed basis and 1/99 percentile range from up to 32 sampled frames',
            note='PCA colors are not semantic labels; no teacher feature comparison'), indent=2))
        print(a.output)
    finally:
        if writer is not None:
            writer.close()
        dataset.close()


if __name__ == '__main__':
    main()
