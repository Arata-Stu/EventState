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
from event_state.activity_reporting import (
    add_activity_arguments, load_activity_options, write_activity_csv,
)
from event_state.segmentation.activity_evaluation import (
    ActivitySemanticEvaluator, validate_semantic_cache,
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
    add_activity_arguments(parser)
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
    activity_index = load_activity_options(args.activity_manifest, args.condition_label, dataset="dsec")
    activity_states = None
    if activity_index is not None:
        keys, activity_states = validate_semantic_cache(
            activity_index, dataset="dsec", feature_cache_dir=args.feature_cache_dir,
            sequences=split.sequences(args.role), samples=dataset.samples,
            load_payload=lambda path: torch.load(path, map_location="cpu", weights_only=True, mmap=True),
        )
        evaluator = ActivitySemanticEvaluator(
            activity_index, dataset="dsec", expected_keys=keys, condition_label=args.condition_label,
            num_classes=int(config.get("num_classes", 11)), ignore_index=int(config.get("ignore_index", 255)),
        )
    else:
        evaluator = SemanticSegmentationEvaluator(
            int(config.get("num_classes", 11)), ignore_index=int(config.get("ignore_index", 255)),
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
    use_amp = args.precision == "fp16" and device.type == "cuda"
    with torch.inference_mode():
        label = f"{args.condition_label} | " if activity_index is not None else ""
        for batch in tqdm(loader, desc=f"{label}DSEC-Semantic {args.role}", unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(features)
            if activity_index is None:
                evaluator.update(logits, labels)
            else:
                evaluator.update(logits, labels, sequence_names=batch["sequence_names"],
                                 timestamps=batch["timestamps"])
    metrics = {
        "role": args.role,
        "feature": args.feature,
        "classes": config.get("classes"),
        **evaluator.compute(),
    }
    if activity_index is not None:
        from event_state.dense.protocol import sha256_file
        metrics["activity_evaluation"]["state_policy_by_sequence"] = activity_states
        metrics["activity_evaluation"]["head_sha256"] = sha256_file(args.checkpoint)
    encoded = json.dumps(metrics, indent=2, sort_keys=True,
                         **({"allow_nan": False} if activity_index is not None else {})) + "\n"
    print(encoded, end="")
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        if activity_index is not None:
            write_activity_csv(output, metrics["activity_evaluation"])


if __name__ == "__main__":
    main()
