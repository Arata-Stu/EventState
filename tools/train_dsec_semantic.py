#!/usr/bin/env python3
"""Train a frozen-feature segmentation head on DSEC-Semantic."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.segmentation import (
    DSEC_SEMANTIC_11_CLASSES,
    DSECSemanticFeatureDataset,
    EventStateSegmentationHead,
    SemanticSegmentationEvaluator,
    load_dsec_semantic_split,
    multiclass_dice_loss,
    segmentation_collate,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_semantic_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSEC-Semantic frozen EventState probe")
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--protocol",
        choices=("development", "official-final"),
        default="development",
    )
    parser.add_argument(
        "--selected-epochs-from",
        type=Path,
        default=None,
        help="Development best.pt; official-final trains from scratch for its selected epoch count",
    )
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--head-width", type=int, default=192)
    parser.add_argument(
        "--head-type", choices=("linear", "nonlinear", "gep_patch"), default="linear"
    )
    parser.add_argument(
        "--loss",
        choices=("ce", "ce-dice"),
        default="ce",
        help="ce-dice uses the equal-weight GEP semantic objective",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
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


@torch.inference_mode()
def evaluate(
    model: EventStateSegmentationHead,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    use_amp: bool,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float | list[float]]:
    model.eval()
    evaluator = SemanticSegmentationEvaluator(num_classes, ignore_index=ignore_index)
    for batch in tqdm(loader, desc="DSEC-Semantic validation", unit="batch"):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(features)
        evaluator.update(logits, labels)
    return evaluator.compute()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if not 0.0 <= args.horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal-flip-probability must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_amp = args.precision == "fp16" and device.type == "cuda"

    split = load_dsec_semantic_split(args.split_manifest)
    selected_development_epoch: int | None = None
    if args.selected_epochs_from is not None:
        if args.protocol != "official-final":
            raise ValueError("--selected-epochs-from is only valid with --protocol official-final")
        selection = torch.load(
            args.selected_epochs_from, map_location="cpu", weights_only=False
        )
        selected_development_epoch = int(selection["epoch"])
        args.epochs = selected_development_epoch + 1
    train_sequences = (
        split.train if args.protocol == "development" else split.official_train
    )
    common = {
        "feature_cache_dir": args.feature_cache_dir,
        "labels_root": args.labels_root,
        "feature": args.feature,
        "num_classes": len(DSEC_SEMANTIC_11_CLASSES),
    }
    train_dataset = DSECSemanticFeatureDataset(
        sequences=train_sequences,
        role="train",
        horizontal_flip_probability=args.horizontal_flip_probability,
        **common,
    )
    validation_dataset = None
    if args.protocol == "development":
        validation_dataset = DSECSemanticFeatureDataset(
            sequences=split.val,
            role="val",
            **common,
        )
        if train_dataset.label_size != validation_dataset.label_size:
            raise ValueError("Train and validation label sizes differ")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=segmentation_collate,
        generator=generator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=segmentation_collate,
        )
    sample_channels = int(train_dataset[0]["feature"].shape[0])
    model = EventStateSegmentationHead(
        in_channels=sample_channels,
        num_classes=len(DSEC_SEMANTIC_11_CLASSES),
        width=args.head_width,
        output_size=train_dataset.label_size,
        head_type=args.head_type,
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
            "classes": list(DSEC_SEMANTIC_11_CLASSES),
            "num_classes": len(DSEC_SEMANTIC_11_CLASSES),
            "split_counts": {"train": 6, "val": 2, "test": 3},
            "training_sequences": list(train_sequences),
            "train_frames": len(train_dataset),
            "validation_frames": (
                len(validation_dataset) if validation_dataset is not None else 0
            ),
            "selected_development_epoch": selected_development_epoch,
            "label_size": list(train_dataset.label_size),
            "in_channels": sample_channels,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    start_epoch = 0
    best_miou = -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        for key in ("protocol", "feature", "seed", "head_width", "head_type", "loss"):
            saved_value = saved_config.get(key, "ce" if key == "loss" else None)
            if saved_value != config.get(key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{saved_value!r} != {config.get(key)!r}"
                )
        if saved_config.get("selected_development_epoch") != config.get(
            "selected_development_epoch"
        ):
            raise ValueError("Resume selected development epoch changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_best = checkpoint.get("best_mIoU")
        best_miou = float(saved_best) if saved_best is not None else -1.0

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        batches = 0
        description = f"DSEC-Semantic train {epoch + 1}/{args.epochs}"
        for batch in tqdm(train_loader, desc=description, unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(features)
                ce_loss = F.cross_entropy(
                    logits, labels, ignore_index=args.ignore_index
                )
                if args.loss == "ce-dice":
                    dice_loss = multiclass_dice_loss(
                        logits, labels, ignore_index=args.ignore_index
                    )
                else:
                    dice_loss = logits.new_zeros(())
                loss = ce_loss + dice_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            total_ce_loss += float(ce_loss.detach())
            total_dice_loss += float(dice_loss.detach())
            batches += 1
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train/loss": total_loss / max(1, batches),
            "train/ce_loss": total_ce_loss / max(1, batches),
            "train/dice_loss": total_dice_loss / max(1, batches),
        }
        validation = None
        if validation_loader is not None:
            validation = evaluate(
                model,
                validation_loader,
                device,
                use_amp=use_amp,
                num_classes=len(DSEC_SEMANTIC_11_CLASSES),
                ignore_index=args.ignore_index,
            )
            record.update({f"val/{key}": value for key, value in validation.items()})
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        miou = float(validation["mIoU"]) if validation is not None else None
        payload = {
            "epoch": epoch,
            "best_mIoU": max(best_miou, miou) if miou is not None else None,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        }
        _atomic_torch_save(payload, output_dir / "last.pt")
        if miou is not None and miou > best_miou:
            best_miou = miou
            _atomic_torch_save(payload, output_dir / "best.pt")


if __name__ == "__main__":
    main()
