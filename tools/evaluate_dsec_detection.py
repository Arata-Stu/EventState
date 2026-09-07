#!/usr/bin/env python3
"""Evaluate a trained EventState YOLOX probe on an official DSEC-Det role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.detection import (
    COCODetectionEvaluator,
    DSECDetectionFeatureDataset,
    EventStateYOLOX,
    detection_collate,
    load_dsec_detection_split,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_det_official_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSEC-Detection mAP")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument(
        "--protocol",
        choices=("probe", "dsec-det"),
        default=None,
        help="Defaults to the protocol recorded in the detector checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    split = load_dsec_detection_split(args.split_manifest)
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    checkpoint_protocol = str(config.get("protocol", "probe"))
    protocol = args.protocol or checkpoint_protocol
    if protocol != checkpoint_protocol:
        raise ValueError(
            f"Probe was trained with protocol={checkpoint_protocol!r}, not {protocol!r}"
        )
    dataset = DSECDetectionFeatureDataset(
        feature_cache_dir=args.feature_cache_dir,
        labels_root=args.labels_root,
        dataset_root=args.dataset_root,
        sequences=split.sequences(args.role),
        feature=args.feature,
        protocol=protocol,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_collate,
    )
    expected_feature = config.get("feature")
    if expected_feature is not None and expected_feature != args.feature:
        raise ValueError(
            f"Probe was trained for feature={expected_feature!r}, not {args.feature!r}"
        )
    channels = int(dataset[0]["feature"].shape[0])
    if dataset.patch_size is None:
        raise RuntimeError("Detection feature-cache patch size was not initialized")
    model = EventStateYOLOX(
        in_channels=channels,
        width=int(config.get("head_width", 192)),
        input_stride=int(config.get("input_stride", dataset.patch_size)),
        image_size=dataset.input_size,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    evaluator = COCODetectionEvaluator()
    use_amp = args.precision == "fp16" and device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"DSEC-Det {args.role}", unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predictions = model(features)
            evaluator.update(
                predictions,
                batch["targets"],
                sequence_names=batch["sequence_names"],
                timestamps=batch["timestamps"],
                image_size=dataset.input_size,
            )
    metrics = {
        "role": args.role,
        "feature": args.feature,
        "protocol": protocol,
        "coordinate_space": (
            "dsec_det_distorted" if protocol == "dsec-det" else "rectified_event"
        ),
        **evaluator.compute(),
    }
    encoded = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
