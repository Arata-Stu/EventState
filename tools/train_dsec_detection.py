#!/usr/bin/env python3
"""Train a frozen-feature YOLOX probe on the official DSEC-Detection split."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
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
    parser = argparse.ArgumentParser(
        description="Official-split DSEC-Detection frozen probe with a YOLOX-style head"
    )
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--head-width", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _move_targets(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


@torch.inference_mode()
def evaluate(
    model: EventStateYOLOX,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    use_amp: bool,
    image_size: tuple[int, int],
) -> dict[str, float]:
    model.eval()
    evaluator = COCODetectionEvaluator()
    for batch in tqdm(loader, desc="DSEC-Det validation", unit="batch"):
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(features)
        evaluator.update(
            predictions,
            batch["targets"],
            sequence_names=batch["sequence_names"],
            timestamps=batch["timestamps"],
            image_size=image_size,
        )
    return evaluator.compute()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_amp = args.precision == "fp16" and device.type == "cuda"

    split = load_dsec_detection_split(args.split_manifest)
    common = {
        "feature_cache_dir": args.feature_cache_dir,
        "labels_root": args.labels_root,
        "dataset_root": args.dataset_root,
        "feature": args.feature,
    }
    train_dataset = DSECDetectionFeatureDataset(
        sequences=split.train, horizontal_flip_probability=0.5, **common
    )
    validation_dataset = DSECDetectionFeatureDataset(sequences=split.val, **common)
    if (
        train_dataset.input_size != validation_dataset.input_size
        or train_dataset.patch_size != validation_dataset.patch_size
    ):
        raise ValueError("Train and validation feature-cache geometry differs")
    if train_dataset.patch_size is None:
        raise RuntimeError("Train feature-cache patch size was not initialized")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_collate,
    )
    sample_channels = int(train_dataset[0]["feature"].shape[0])
    model = EventStateYOLOX(
        in_channels=sample_channels,
        width=args.head_width,
        input_stride=train_dataset.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "official_split_counts": {"train": 41, "val": 6, "test": 13},
            "classes": ["car", "pedestrian"],
            "coordinate_space": "rectified_event",
            "train_frames": len(train_dataset),
            "validation_frames": len(validation_dataset),
            "input_stride": train_dataset.patch_size,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    start_epoch = 0
    best_map = -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("best_mAP", -1.0))

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        description = f"DSEC-Det train {epoch + 1}/{args.epochs}"
        for batch in tqdm(train_loader, desc=description, unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                losses = model(features, targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        scheduler.step()
        validation = evaluate(
            model,
            validation_loader,
            device,
            use_amp=use_amp,
            image_size=validation_dataset.input_size,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train/{key}": value / max(1, batches) for key, value in totals.items()},
            **{f"val/{key}": value for key, value in validation.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        payload = {
            "epoch": epoch,
            "best_mAP": max(best_map, validation["mAP"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        }
        _atomic_torch_save(payload, output_dir / "last.pt")
        if validation["mAP"] > best_map:
            best_map = validation["mAP"]
            _atomic_torch_save(payload, output_dir / "best.pt")


if __name__ == "__main__":
    main()
