#!/usr/bin/env python3
"""GPU-host CPU job: train41 fixed moments without storing dense voxel caches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    import yaml
    from event_state.data.dsec import DSECSequenceDataset, event_representation_metadata, event_window_contract
    from event_state.data.event_representation import EventVoxelizer
    from event_state.data.transforms import PairedSequenceTransform

    split = yaml.safe_load((Path(__file__).parent / 'manifests/dsec_det_official_split.yaml').read_text())
    sequences = split['train']
    if len(set(sequences)) != 41 or set(sequences) & (set(split['val']) | set(split['test'])):
        raise ValueError('Expected disjoint official train41')
    voxel = EventVoxelizer(num_bins=10, height=480, width=640,
                           channel_layout='polarity_major', normalization='none')
    count = frames = 0
    mean = torch.zeros(20, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    sources = {}
    for sequence in sequences:
        dataset = DSECSequenceDataset(a.root, split='train', sequence_length=1,
            event_representation=voxel, transform=PairedSequenceTransform(height=480, width=640),
            sequences=[sequence], load_images=False, rectify_events=True)
        sequence_frames = 0
        try:
            for record in dataset._frames_by_sequence[sequence]:
                if record.frame_index == 0:
                    continue
                tensor, _ = dataset._load_event_frame(record)
                values = tensor.double().flatten(1)
                if not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Invalid voxel counts')
                variance, batch_mean = torch.var_mean(values, dim=1, correction=0)
                n = values.shape[1]
                delta = batch_mean - mean
                m2 += variance * n + delta.square() * (count * n / (count + n))
                mean += delta * (n / (count + n))
                count += n
                frames += 1
                sequence_frames += 1
        finally:
            dataset.close()
        if not sequence_frames:
            raise ValueError(f'Empty sequence: {sequence}')
        sources[sequence] = {'frames': sequence_frames, 'root': str(a.root.resolve())}
        print(f'[normalization] {sequence}: {sequence_frames} frames', flush=True)
    std = (m2 / count).clamp_min(0).sqrt()
    constant = std < 1e-6
    std[constant] = 1.0
    result = dict(format='eventstate_fixed_normalization_v1',
        representation=event_representation_metadata(voxel), sensor_size=[480, 640],
        event_window_boundary=event_window_contract(1.0)['event_window_boundary'],
        event_window_fraction=1.0, coordinate_space='rectified_event', spatial_quantization='nearest_integer_numpy_rint',
        includes_zero_pixels=True, statistics_stage='raw_counts_before_spatial_transform',
        split='dsec_det_train41', sequences=sequences, sources=sources, frames=frames,
        pixels_per_channel=count, mean=mean.tolist(), std=std.tolist(), std_floor=1e-6,
        constant_channels=constant.nonzero().flatten().tolist())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f'[complete] {a.output}')


if __name__ == '__main__':
    main()
