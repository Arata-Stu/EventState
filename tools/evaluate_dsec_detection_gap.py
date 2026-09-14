#!/usr/bin/env python3
"""Evaluate DSEC-Detection robustness under deterministic event gaps."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import Tensor
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
from event_state.training.event_dropout import empty_event_input
from train_dsec_detection_end_to_end import (
    DEFAULT_SPLIT,
    _detector_features,
    _freeze_disposable_projectors,
    _random_initialization_config,
    _value,
)


RECOVERY_OFFSETS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Online DSEC-Detection evaluation with repeated deterministic event gaps. "
            "No feature cache is written."
        )
    )
    parser.add_argument("--mode", choices=("frozen", "end-to-end"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Frozen detector checkpoint or end-to-end checkpoint",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=None,
        help="Required in frozen mode: pretrained EventState checkpoint",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional local DINO checkpoint used only to construct a frozen student",
    )
    parser.add_argument("--feature", choices=("z", "h", "concat"), default=None)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("val", "test"), default="test")
    parser.add_argument("--drop-length", type=int, required=True)
    parser.add_argument("--drop-start", type=int, default=32)
    parser.add_argument("--drop-stride", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    """Replace non-finite metric values so strict JSON remains portable."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validate_gap(length: int, start: int, stride: int) -> None:
    if length < 0:
        raise ValueError("--drop-length must be non-negative")
    if start < 0:
        raise ValueError("--drop-start must be non-negative")
    if stride <= 0:
        raise ValueError("--drop-stride must be positive")
    if length > stride:
        raise ValueError("--drop-length must not exceed --drop-stride")


def _gap_phase(frame_index: int, *, start: int, stride: int) -> int | None:
    if frame_index < start:
        return None
    return (frame_index - start) % stride


def _subset_names(
    frame_index: int,
    *,
    length: int,
    start: int,
    stride: int,
) -> tuple[list[str], bool]:
    names = ["overall"]
    phase = _gap_phase(frame_index, start=start, stride=stride)
    dropped = length > 0 and phase is not None and phase < length
    names.append("dropped" if dropped else "observed")
    if length > 0 and (frame_index == start - 1 or phase == stride - 1):
        names.append("pre_gap_1")
    if length > 0 and phase is not None:
        recovery_offset = phase - length + 1
        if recovery_offset in RECOVERY_OFFSETS:
            names.append(f"recovery_{recovery_offset}")
    return names, dropped


def _checkpoint_dict(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return value


def _override_local_teacher(config: Any, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    resolved = str(checkpoint.expanduser().resolve())
    OmegaConf.update(config, "teacher.checkpoint", resolved, merge=False)
    OmegaConf.update(
        config,
        "model.event_encoder.checkpoint",
        resolved,
        merge=False,
        force_add=True,
    )


def _load_frozen(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, EventStateYOLOX, Any, str, dict[str, Any]]:
    if args.student_checkpoint is None:
        raise ValueError("--student-checkpoint is required in frozen mode")
    student_path = args.student_checkpoint.expanduser().resolve()
    detector_path = args.checkpoint.expanduser().resolve()
    if not student_path.is_file() or not detector_path.is_file():
        raise FileNotFoundError("Frozen student or detector checkpoint was not found")
    metadata = load_checkpoint_config_metadata(student_path)
    config = OmegaConf.create(metadata.config)
    _override_local_teacher(config, args.teacher_checkpoint)
    student = build_model(config, device=device)
    student_state = load_checkpoint(
        student_path, model=student, device=device, restore_rng=False
    )
    detector_checkpoint = _checkpoint_dict(detector_path)
    detector_config = detector_checkpoint.get("config", {})
    if str(detector_config.get("protocol")) != "dsec-det":
        raise ValueError("Frozen gap evaluation requires a dsec-det detector checkpoint")
    recorded_feature = str(detector_config.get("feature"))
    feature = args.feature or recorded_feature
    if feature != recorded_feature:
        raise ValueError(
            f"Frozen detector was trained with feature={recorded_feature!r}, not {feature!r}"
        )
    channels = int(student.event_dim)
    if feature == "h":
        channels = int(student.temporal_dim)
    elif feature == "concat":
        channels = int(student.event_dim) + int(student.temporal_dim)
    detector = EventStateYOLOX(
        in_channels=channels,
        width=int(detector_config.get("head_width", 192)),
        input_stride=int(detector_config.get("input_stride", student.event_encoder.patch_size)),
        image_size=(430, 640),
    ).to(device)
    detector.load_state_dict(detector_checkpoint["model"])
    identity = {
        "student_checkpoint": str(student_path),
        "student_checkpoint_step": int(student_state.global_step),
        "detector_checkpoint": str(detector_path),
        "detector_checkpoint_epoch": int(detector_checkpoint.get("epoch", -1)) + 1,
        "detector_best_validation_mAP": float(
            detector_checkpoint.get("best_mAP", -1.0)
        ),
    }
    return student, detector, config, feature, identity


def _load_end_to_end(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, EventStateYOLOX, Any, str, dict[str, Any]]:
    if args.student_checkpoint is not None:
        raise ValueError("--student-checkpoint is only valid in frozen mode")
    path = args.checkpoint.expanduser().resolve()
    checkpoint = _checkpoint_dict(path)
    run_config = checkpoint.get("config")
    if not isinstance(run_config, dict) or not isinstance(
        run_config.get("reference_config"), dict
    ):
        raise ValueError(f"End-to-end checkpoint lacks reference_config: {path}")
    if "student" not in checkpoint or "detector" not in checkpoint:
        raise ValueError(f"End-to-end checkpoint lacks model weights: {path}")
    feature = str(run_config.get("feature"))
    if args.feature is not None and args.feature != feature:
        raise ValueError(
            f"End-to-end checkpoint uses feature={feature!r}, not {args.feature!r}"
        )
    config = OmegaConf.create(run_config["reference_config"])
    construction_config = _random_initialization_config(config)
    student = build_model(construction_config, device=device)
    student.load_state_dict(checkpoint["student"])
    _freeze_disposable_projectors(student)
    channels = int(student.event_dim)
    if feature == "h":
        channels = int(student.temporal_dim)
    elif feature == "concat":
        channels = int(student.event_dim) + int(student.temporal_dim)
    detector = EventStateYOLOX(
        in_channels=channels,
        width=int(run_config.get("head_width", 192)),
        input_stride=int(student.event_encoder.patch_size),
        image_size=(430, 640),
    ).to(device)
    detector.load_state_dict(checkpoint["detector"])
    identity = {
        "checkpoint": str(path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_best_validation_mAP": float(checkpoint.get("best_mAP", -1.0)),
        "training_mode": str(run_config.get("mode")),
        "freeze_event_encoder": bool(run_config.get("freeze_event_encoder", False)),
    }
    return student, detector, config, feature, identity


def _safe_compute(
    evaluator: COCODetectionEvaluator, evaluated_frames: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluated_frames": evaluated_frames,
        "ground_truth_boxes": len(evaluator.annotations),
    }
    if evaluated_frames == 0 or not evaluator.annotations:
        result["metrics"] = None
    else:
        result["metrics"] = evaluator.compute()
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    _validate_gap(args.drop_length, args.drop_start, args.drop_stride)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    device = torch.device(args.device)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    if args.mode == "frozen":
        student, detector, config, feature, identity = _load_frozen(args, device)
    else:
        student, detector, config, feature, identity = _load_end_to_end(args, device)
    student.eval()
    detector.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    split = load_dsec_detection_split(args.split_manifest)
    representation_type = str(_value(config.dataset.representation, "type"))
    event_mean = tuple(
        float(value)
        for value in _value(config.dataset.representation, "normalize_mean")
    )
    event_std = tuple(
        float(value)
        for value in _value(config.dataset.representation, "normalize_std")
    )
    dataset = DSECDetectionEventDataset(
        event_cache_dir=args.event_cache_dir,
        labels_root=args.labels_root,
        dataset_root=args.dataset_root,
        sequences=split.sequences(args.role),
        sequence_length=int(_value(config.dataset, "sequence_length", 8)),
        input_size=(
            int(_value(config.dataset, "input_height", 448)),
            int(_value(config.dataset, "input_width", 640)),
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
    subset_keys = (
        "overall",
        "observed",
        "dropped",
        "pre_gap_1",
        *(f"recovery_{offset}" for offset in RECOVERY_OFFSETS),
    )
    evaluators = {name: COCODetectionEvaluator() for name in subset_keys}
    evaluated_counts = {name: 0 for name in subset_keys}
    stream_frames = 0
    dropped_stream_frames = 0
    recurrent_state = None
    sequence_frame_index = 0
    current_sequence: str | None = None
    description = f"DSEC-Det {args.role} {args.mode} gap={args.drop_length}"

    for batch in tqdm(loader, desc=description, unit="frame"):
        sequence = str(batch["sequence_names"][0])
        if bool(batch["is_sequence_start"][0]) or sequence != current_sequence:
            recurrent_state = None
            sequence_frame_index = 0
            current_sequence = sequence
        subset_names, dropped = _subset_names(
            sequence_frame_index,
            length=args.drop_length,
            start=args.drop_start,
            stride=args.drop_stride,
        )
        events = batch["events"].to(device, non_blocking=True)
        if dropped:
            empty = empty_event_input(
                events,
                representation_type=representation_type,
                normalize_mean=event_mean,
                normalize_std=event_std,
            )
            events = empty.expand_as(events)
            dropped_stream_frames += 1
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            features, recurrent_state = _detector_features(
                student,
                events,
                batch["sampling_grids"],
                batch["flip"],
                feature=feature,
                state=recurrent_state,
            )
            if bool(batch["evaluate"][0]):
                predictions = detector(features)
        if bool(batch["evaluate"][0]):
            for name in subset_names:
                evaluators[name].update(
                    predictions,
                    batch["targets"],
                    sequence_names=batch["sequence_names"],
                    timestamps=batch["timestamps"],
                    image_size=(430, 640),
                )
                evaluated_counts[name] += 1
        stream_frames += 1
        sequence_frame_index += 1

    subsets = {
        name: _safe_compute(evaluators[name], evaluated_counts[name])
        for name in subset_keys
    }
    payload = {
        "format_version": 1,
        "role": args.role,
        "mode": args.mode,
        "feature": feature,
        "state_policy": "continuous_reset_at_sequence_boundary",
        "gap": {
            "length": args.drop_length,
            "start": args.drop_start,
            "stride": args.drop_stride,
            "semantics": "empty event input is forwarded through the recurrent model",
        },
        "stream_frames": stream_frames,
        "dropped_stream_frames": dropped_stream_frames,
        "dropped_stream_fraction": dropped_stream_frames / max(1, stream_frames),
        "official_split_counts": {"train": 41, "val": 6, "test": 13},
        "identity": identity,
        "subsets": subsets,
    }
    payload = _json_safe(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(encoded, end="", flush=True)
    _atomic_write_json(payload, args.output)


if __name__ == "__main__":
    main()
