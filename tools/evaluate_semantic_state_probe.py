#!/usr/bin/env python3
"""Evaluate a FIXED semantic head with activity strata on a state-policy cache."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from event_state.segmentation import (DSECSemanticFeatureDataset, M3EDSemanticFeatureDataset,
    EventStateSegmentationHead, load_dsec_semantic_split, load_m3ed_semantic_split, segmentation_collate)
from event_state.segmentation.state_probe import ActivityEvaluator


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['dsec', 'm3ed'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--feature-cache-dir', type=Path, required=True)
    p.add_argument('--labels-root', type=Path, required=True, help='M3ED: downstream target cache root')
    p.add_argument('--feature', choices=['z', 'h', 'concat'], required=True)
    p.add_argument('--state-policy', choices=['continuous', 'frame'], required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--low-activity-threshold', type=float, default=0.1,
                   help='Predeclared fraction of active pixels; never fit to validation accuracy')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = a.split_manifest or Path(__file__).parent / 'manifests' / (
        'dsec_semantic_split.yaml' if a.dataset == 'dsec' else 'm3ed_downstream_split.yaml')
    split = (load_dsec_semantic_split if a.dataset == 'dsec' else load_m3ed_semantic_split)(manifest)
    role = 'val' if a.dataset == 'dsec' else 'validation'
    sequences = split.sequences(role)
    metadata = [json.loads((a.feature_cache_dir / s / 'metadata.json').read_text()) for s in sequences]
    if any(m['state_policy'] != a.state_policy for m in metadata):
        raise ValueError('Cache state policy mismatch')
    digests = {m['checkpoint_sha256'] for m in metadata}
    if len(digests) != 1:
        raise ValueError('Mixed backbone checkpoints')
    kwargs = dict(feature_cache_dir=a.feature_cache_dir, sequences=sequences, feature=a.feature, load_activity=True)
    dataset = (DSECSemanticFeatureDataset(**kwargs, labels_root=a.labels_root, role=role) if a.dataset == 'dsec'
               else M3EDSemanticFeatureDataset(**kwargs, target_cache_dir=a.labels_root))
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if config.get('feature') != a.feature:
        raise ValueError('Head feature mismatch')
    device = torch.device(a.device)
    head = EventStateSegmentationHead(in_channels=dataset[0]['feature'].shape[0],
        num_classes=int(config.get('num_classes', 11)), width=int(config.get('head_width', 192)),
        output_size=dataset.label_size, head_type=config.get('head_type', 'nonlinear')).to(device)
    head.load_state_dict(checkpoint['model']); head.eval()
    evaluator = ActivityEvaluator(int(config.get('num_classes', 11)), a.low_activity_threshold)
    samples = hashlib.sha256()
    masks = hashlib.sha256()
    z_features = hashlib.sha256()
    with torch.inference_mode():
        for batch in tqdm(DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
                                    collate_fn=segmentation_collate), desc=a.feature):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                prediction = head(batch['features'].to(device)).argmax(1).cpu()
            evaluator.update(prediction, batch['labels'], batch['event_activity'])
            for seq, frame, stamp in zip(batch['sequence_names'], batch['frame_indices'], batch['timestamps']):
                samples.update(f'{seq}:{frame}:{stamp}\n'.encode())
            masks.update(batch['event_activity'].numpy().tobytes())
            if a.feature == 'z':
                z_features.update(batch['features'].numpy().tobytes())
    result = dict(dataset=a.dataset, role=role, feature=a.feature, state_policy=a.state_policy,
        sequences=list(sequences), samples_sha256=samples.hexdigest(), activity_sha256=masks.hexdigest(),
        z_features_sha256=z_features.hexdigest() if a.feature == 'z' else None,
        backbone_sha256=next(iter(digests)), head_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        head_checkpoint=str(a.checkpoint.resolve()), low_activity_threshold=a.low_activity_threshold,
        threshold_source='predeclared_fixed_fraction', metrics=evaluator.compute())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
