#!/usr/bin/env python3
"""Evaluate an end-to-end EventState+YOLOX checkpoint on DSEC-Detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from event_state.detection import (
    DSECDetectionEventDataset,
    EventStateYOLOX,
    detection_event_collate,
    load_dsec_detection_split,
)
from event_state.training import build_model
from train_dsec_detection_end_to_end import (
    DEFAULT_SPLIT,
    _freeze_disposable_projectors,
    _random_initialization_config,
    _value,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned or scratch EventState+YOLOX on DSEC-Detection"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("val", "test"), default="test")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid end-to-end checkpoint: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("reference_config"), dict):
        raise ValueError(f"Checkpoint lacks end-to-end reference_config: {path}")
    if "student" not in checkpoint or "detector" not in checkpoint:
        raise ValueError(f"Checkpoint lacks student/detector weights: {path}")
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    run_config = checkpoint["config"]
    feature = str(run_config.get("feature"))
    if feature not in {"z", "h", "concat"}:
        raise ValueError(f"Unsupported checkpoint feature: {feature!r}")
    mode = str(run_config.get("mode"))
    if mode not in {"finetune", "scratch"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode!r}")

    reference_config = OmegaConf.create(run_config["reference_config"])
    construction_config = _random_initialization_config(reference_config)
    device = torch.device(args.device)
    student = build_model(construction_config, device=device)
    student.load_state_dict(checkpoint["student"])
    _freeze_disposable_projectors(student)
    student.eval()

    feature_channels = int(student.event_dim)
    if feature == "h":
        feature_channels = int(student.temporal_dim)
    elif feature == "concat":
        feature_channels = int(student.event_dim) + int(student.temporal_dim)
    detector = EventStateYOLOX(
        in_channels=feature_channels,
        width=int(run_config.get("head_width", 192)),
        input_stride=int(student.event_encoder.patch_size),
        image_size=(430, 640),
    ).to(device)
    detector.load_state_dict(checkpoint["detector"])
    detector.eval()

    split = load_dsec_detection_split(args.split_manifest)
    event_mean = tuple(
        float(value)
        for value in _value(
            reference_config.dataset.representation, "normalize_mean"
        )
    )
    event_std = tuple(
        float(value)
        for value in _value(
            reference_config.dataset.representation, "normalize_std"
        )
    )
    dataset = DSECDetectionEventDataset(
        event_cache_dir=args.event_cache_dir,
        labels_root=args.labels_root,
        dataset_root=args.dataset_root,
        sequences=split.sequences(args.role),
        sequence_length=int(_value(reference_config.dataset, "sequence_length", 8)),
        input_size=(
            int(_value(reference_config.dataset, "input_height", 448)),
            int(_value(reference_config.dataset, "input_width", 640)),
        ),
        source_stride=int(student.event_encoder.patch_size),
        event_mean=event_mean,
        event_std=event_std,
        continuous=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_event_collate,
    )
    use_amp = args.precision == "fp16" and device.type == "cuda"
    metrics = evaluate(
        student,
        detector,
        loader,
        device,
        feature=feature,
        use_amp=use_amp,
    )
    result = {
        "role": args.role,
        "mode": mode,
        "feature": feature,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_best_validation_mAP": float(checkpoint.get("best_mAP", -1.0)),
        "state_policy": "continuous_reset_at_sequence_boundary",
        "stream_frames": len(dataset),
        "evaluated_frames": sum(
            sample["target"] is not None for sample in dataset.samples
        ),
        "official_split_counts": {"train": 41, "val": 6, "test": 13},
        **metrics,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="", flush=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
