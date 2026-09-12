#!/usr/bin/env python3
"""Fine-tune or train EventState+YOLOX from scratch on DSEC-Detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.detection import (
    COCODetectionEvaluator,
    DSECDetectionEventDataset,
    EventStateYOLOX,
    detection_event_collate,
    load_dsec_detection_split,
)
from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_det_official_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official DSEC-Detection end-to-end fine-tuning or scratch training"
    )
    parser.add_argument("--mode", choices=("finetune", "scratch"), required=True)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        required=True,
        help="Defines the exact EventState architecture; weights load only in finetune mode",
    )
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--freeze-event-encoder",
        action="store_true",
        help=(
            "Keep the pretrained event ViT fixed while updating the temporal "
            "backbone and detection head"
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--head-width", type=int, default=192)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _value(config: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_initialization_config(config: Any) -> Any:
    """Keep the architecture but prevent any RGB/DINO weight loading."""

    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(result, "model.event_encoder.pretrained_init", False, merge=False)
    OmegaConf.update(result, "model.event_encoder.patch_init", "random", merge=False)
    OmegaConf.update(
        result, "model.event_encoder.checkpoint", None, merge=False, force_add=True
    )
    return result


def _freeze_disposable_projectors(model: nn.Module) -> None:
    for name in ("h_projector", "z_projector"):
        projector = getattr(model, name, None)
        if isinstance(projector, nn.Module):
            projector.eval()
            for parameter in projector.parameters():
                parameter.requires_grad_(False)


def _freeze_event_encoder(model: nn.Module) -> None:
    encoder = getattr(model, "event_encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("EventState model lacks an event_encoder module")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)


def _select_feature(outputs: Mapping[str, Any], feature: str) -> Tensor:
    z = outputs.get("z")
    h = outputs.get("h")
    if not isinstance(z, Tensor) or not isinstance(h, Tensor):
        raise ValueError("EventState must return z and h tensors")
    if z.ndim != 4 or h.ndim != 4:
        raise ValueError("EventState z/h must have shape [B, T, N, D]")
    if feature == "z":
        return z[:, -1]
    if feature == "h":
        return h[:, -1]
    return torch.cat((z[:, -1], h[:, -1]), dim=-1)


def _detector_features(
    student: nn.Module,
    events: Tensor,
    sampling_grids: Tensor,
    flip: Tensor,
    *,
    feature: str,
    state: Any = None,
) -> tuple[Tensor, Any]:
    outputs = student(events, state=state)
    tokens = _select_feature(outputs, feature)
    patch_size = int(student.event_encoder.patch_size)
    grid_height = int(events.shape[-2]) // patch_size
    grid_width = int(events.shape[-1]) // patch_size
    if tokens.shape[1] != grid_height * grid_width:
        raise ValueError("EventState token count does not match the input patch grid")
    maps = tokens.reshape(tokens.shape[0], grid_height, grid_width, -1).permute(0, 3, 1, 2)
    warped = F.grid_sample(
        maps,
        sampling_grids.to(device=maps.device, dtype=maps.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    warped = torch.stack(
        [value.flip(-1) if bool(do_flip) else value for value, do_flip in zip(warped, flip)]
    )
    return warped, outputs.get("state")


def _move_targets(
    targets: list[dict[str, Tensor]], device: torch.device
) -> list[dict[str, Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


@torch.inference_mode()
def evaluate(
    student: nn.Module,
    detector: EventStateYOLOX,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    feature: str,
    use_amp: bool,
) -> dict[str, float]:
    student.eval()
    detector.eval()
    evaluator = COCODetectionEvaluator()
    recurrent_state = None
    for batch in tqdm(loader, desc="DSEC-Det continuous validation", unit="frame"):
        if len(batch["sequence_names"]) != 1:
            raise ValueError("Continuous validation requires batch size 1")
        if bool(batch["is_sequence_start"][0]):
            recurrent_state = None
        events = batch["events"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            features, recurrent_state = _detector_features(
                student,
                events,
                batch["sampling_grids"],
                batch["flip"],
                feature=feature,
                state=recurrent_state,
            )
            if not bool(batch["evaluate"][0]):
                continue
            predictions = detector(features)
        evaluator.update(
            predictions,
            batch["targets"],
            sequence_names=batch["sequence_names"],
            timestamps=batch["timestamps"],
            image_size=(430, 640),
        )
    return evaluator.compute()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.validate_every <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, validate-every, and batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.learning_rate <= 0 or args.backbone_lr_mult <= 0:
        raise ValueError("learning rates must be positive")
    if not 0.0 <= args.horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal-flip-probability must be in [0, 1]")

    _seed_everything(args.seed)
    device = torch.device(args.device)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    reference = args.reference_checkpoint.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Reference checkpoint not found: {reference}")
    metadata = load_checkpoint_config_metadata(reference)
    saved_config = OmegaConf.create(metadata.config)
    # Construct the exact architecture without reading the original RGB/DINO
    # weights. Fine-tuning immediately replaces this random state with the
    # complete EventState checkpoint; scratch intentionally keeps it.
    construction_config = _random_initialization_config(saved_config)
    student = build_model(construction_config, device=device)
    if args.mode == "finetune":
        load_checkpoint(reference, model=student, device=device, restore_rng=False)
    elif args.freeze_event_encoder:
        raise ValueError("--freeze-event-encoder is only valid with --mode finetune")
    _freeze_disposable_projectors(student)
    if args.freeze_event_encoder:
        _freeze_event_encoder(student)

    split = load_dsec_detection_split(args.split_manifest)
    sequence_length = int(_value(saved_config, "dataset.sequence_length", 8))
    input_size = (
        int(_value(saved_config, "dataset.input_height", 448)),
        int(_value(saved_config, "dataset.input_width", 640)),
    )
    source_stride = int(student.event_encoder.patch_size)
    event_mean = tuple(
        float(value)
        for value in _value(saved_config, "dataset.representation.normalize_mean")
    )
    event_std = tuple(
        float(value)
        for value in _value(saved_config, "dataset.representation.normalize_std")
    )
    dataset_common = {
        "event_cache_dir": args.event_cache_dir,
        "labels_root": args.labels_root,
        "dataset_root": args.dataset_root,
        "sequence_length": sequence_length,
        "input_size": input_size,
        "source_stride": source_stride,
        "event_mean": event_mean,
        "event_std": event_std,
    }
    train_dataset = DSECDetectionEventDataset(
        sequences=split.train,
        continuous=False,
        horizontal_flip_probability=args.horizontal_flip_probability,
        **dataset_common,
    )
    validation_dataset = DSECDetectionEventDataset(
        sequences=split.val,
        continuous=True,
        **dataset_common,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": detection_event_collate,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        **loader_common,
    )

    feature_channels = int(student.event_dim)
    if args.feature == "h":
        feature_channels = int(student.temporal_dim)
    elif args.feature == "concat":
        feature_channels = int(student.event_dim) + int(student.temporal_dim)
    # Keep the detector initialization identical across E0/E2/E4 for a given
    # seed even though constructing their temporal backbones consumes a
    # different number of random values.
    torch.manual_seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
    detector = EventStateYOLOX(
        in_channels=feature_channels,
        width=args.head_width,
        input_stride=source_stride,
        image_size=(430, 640),
    ).to(device)
    student_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": student_parameters, "lr": args.learning_rate * args.backbone_lr_mult},
            {"params": detector.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.01
        + 0.99 * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_config.update(
        {
            "reference_checkpoint_sha256": _sha256(reference),
            "reference_config": OmegaConf.to_container(saved_config, resolve=True),
            "initialization": (
                "eventstate_checkpoint" if args.mode == "finetune" else "random"
            ),
            "trainable_components": (
                ["temporal_backbone", "detector_head"]
                if args.freeze_event_encoder
                else ["event_encoder", "temporal_backbone", "detector_head"]
            ),
            "sequence_length": sequence_length,
            "train_frames": len(train_dataset),
            "validation_stream_frames": len(validation_dataset),
            "official_split_counts": {"train": 41, "val": 6, "test": 13},
            "coordinate_space": "dsec_det_distorted",
            "detection_scale": 1,
            "classes": ["car", "pedestrian"],
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    start_epoch = 0
    best_map = -1.0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = resume.get("config", {})
        if (
            resume_config.get("mode") != args.mode
            or resume_config.get("feature") != args.feature
            or bool(resume_config.get("freeze_event_encoder", False))
            != args.freeze_event_encoder
        ):
            raise ValueError(
                "Resume checkpoint mode/feature/freeze policy does not match this run"
            )
        student.load_state_dict(resume["student"])
        detector.load_state_dict(resume["detector"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_map = float(resume.get("best_mAP", -1.0))

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        student.train()
        _freeze_disposable_projectors(student)
        if args.freeze_event_encoder:
            # requires_grad=False prevents weight updates; eval mode additionally
            # prevents any train-time stochastic behavior in the frozen ViT.
            _freeze_event_encoder(student)
        detector.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in tqdm(
            train_loader, desc=f"DSEC-Det {args.mode} {epoch + 1}/{args.epochs}", unit="batch"
        ):
            events = batch["events"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features, _ = _detector_features(
                    student,
                    events,
                    batch["sampling_grids"],
                    batch["flip"],
                    feature=args.feature,
                )
                losses = detector(features, targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*student_parameters, *detector.parameters()], 10.0
            )
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(gradient_norm)
        scheduler.step()

        should_validate = (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs
        validation = None
        if should_validate:
            validation = evaluate(
                student,
                detector,
                validation_loader,
                device,
                feature=args.feature,
                use_amp=use_amp,
            )
        record = {
            "epoch": epoch,
            "mode": args.mode,
            "feature": args.feature,
            "learning_rate/student": optimizer.param_groups[0]["lr"],
            "learning_rate/head": optimizer.param_groups[1]["lr"],
            **{f"train/{key}": value / max(1, batches) for key, value in totals.items()},
        }
        if validation is not None:
            record.update({f"val/{key}": value for key, value in validation.items()})
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        current_map = float(validation["mAP"]) if validation is not None else best_map
        payload = {
            "epoch": epoch,
            "best_mAP": max(best_map, current_map),
            "student": student.state_dict(),
            "detector": detector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": run_config,
        }
        _atomic_save(payload, output_dir / "last.pt")
        if validation is not None and current_map > best_map:
            best_map = current_map
            _atomic_save(payload, output_dir / "best.pt")


if __name__ == "__main__":
    main()
