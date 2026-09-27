#!/usr/bin/env python3
"""Recompute GEP normalization from existing M3ED train4 caches, without raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')

    import torch
    import yaml
    from event_state.data import GEPEventFrame, M3EDSequenceDataset, PairedSequenceTransform

    torch.set_num_threads(1)
    split_path = Path(__file__).parent / 'manifests/m3ed_downstream_split.yaml'
    split = yaml.safe_load(split_path.read_text())['semantic']
    sequences = split['train']
    if len(sequences) != 4 or len(set(sequences)) != 4 or set(sequences) & set(split['validation'] + split['test']):
        raise ValueError('Expected disjoint M3ED train4 split')
    total = frames = 0
    sums = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    sources = {}
    for sequence in sequences:
        path = args.prepared_root / sequence / 'metadata.json'
        raw = path.read_bytes()
        metadata = json.loads(raw)
        rep = metadata.get('representation', {})
        if rep.get('type') != 'gep_rgb' or rep.get('channels') != 3 or rep.get('percentile', 90.0) != 90.0:
            raise ValueError(f'Expected GEP RGB percentile 90: {path}')
        dataset = M3EDSequenceDataset(
            root=args.prepared_root, prepared_root=args.prepared_root,
            split='train', sequences=[sequence], sequence_length=1,
            event_representation=GEPEventFrame(height=360, width=640),
            # Same deterministic crop as preparation and cached-teacher training.
            transform=PairedSequenceTransform(height=352, width=640),
            load_images=False,
        )
        print(f'[statistics] {sequence}: {len(dataset)} frames', flush=True)
        for index in range(len(dataset)):
            tensor = dataset[index]['events'][0]
            if not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
                raise ValueError(f'Expected unnormalized GEP in [0,1]: {sequence}, frame {index}')
            flat = tensor.double().flatten(1)
            sums += flat.sum(1)
            squares += flat.square().sum(1)
            total += flat.shape[1]
            frames += 1
            if (index + 1) % 1000 == 0:
                print(f'[statistics] {sequence}: {index + 1}/{len(dataset)}', flush=True)
        sources[sequence] = {'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'frames': len(dataset)}
    if total == 0:
        raise ValueError('No training pixels')
    mean = sums / total
    std = (squares / total - mean.square()).clamp_min(0).sqrt()
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError('Invalid or zero normalization standard deviation')
    result = dict(dataset='M3ED', representation='gep_rgb', sequences=sequences,
                  pixel_count=total, frame_count=frames, normalize_mean=mean.tolist(),
                  normalize_std=std.tolist(), input_size=[352, 640],
                  crop='rows [4:356], all columns', includes_zero_pixels=True,
                  prepared_root=str(args.prepared_root.resolve()), sources=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(f'[complete] {args.output}', flush=True)
    print(json.dumps({k: result[k] for k in ('normalize_mean', 'normalize_std')}))


if __name__ == '__main__':
    main()
