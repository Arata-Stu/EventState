#!/usr/bin/env python3
"""Evaluate a frozen-feature segmentation head on DSEC-Semantic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.segmentation import (
    DSECSemanticFeatureDataset,
    EventStateSegmentationHead,
    SemanticSegmentationEvaluator,
    load_dsec_semantic_split,
    segmentation_collate,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_semantic_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSEC-Semantic mIoU")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("val", "test"), default="test")
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    expected_feature = config.get("feature")
    if expected_feature is not None and expected_feature != args.feature:
        raise ValueError(
            f"Head was trained for feature={expected_feature!r}, not {args.feature!r}"
        )
    split = load_dsec_semantic_split(args.split_manifest)
    dataset = DSECSemanticFeatureDataset(
        feature_cache_dir=args.feature_cache_dir,
        labels_root=args.labels_root,
        sequences=split.sequences(args.role),
        role=args.role,
        feature=args.feature,
        num_classes=int(config.get("num_classes", 11)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=segmentation_collate,
    )
    channels = int(dataset[0]["feature"].shape[0])
    model = EventStateSegmentationHead(
        in_channels=channels,
        num_classes=int(config.get("num_classes", 11)),
        width=int(config.get("head_width", 192)),
        output_size=dataset.label_size,
        head_type=str(config.get("head_type", "nonlinear")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    evaluator = SemanticSegmentationEvaluator(
        int(config.get("num_classes", 11)),
        ignore_index=int(config.get("ignore_index", 255)),
    )
    use_amp = args.precision == "fp16" and device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"DSEC-Semantic {args.role}", unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(features)
            evaluator.update(logits, labels)
    metrics = {
        "role": args.role,
        "feature": args.feature,
        "classes": config.get("classes"),
        **evaluator.compute(),
    }
    encoded = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
