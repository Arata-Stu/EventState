#!/usr/bin/env python3
"""Stream one complete DSEC sequence and cache compact visualization features."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import torch

from event_state.training.checkpoint import atomic_torch_save, load_checkpoint
from event_state.training.factory import build_evaluation_runtime
from export_features import (
    _batch_bool,
    _event_preview,
    _move,
    _objective_metadata,
    _prepare_config,
    _rgb_preview,
)


SEQUENCE_FEATURE_FORMAT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpoint chronologically over a complete validation sequence. "
            "The recurrent state is preserved across every clip."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default=None,
        help="Evaluation split override; final train41 checkpoints should use test",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--event-window-fraction",
        type=float,
        default=1.0,
        help="Causal RGB-interval tail fraction used by the selected event cache",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-policy",
        choices=("continuous", "clip", "frame"),
        default="continuous",
        help=(
            "continuous keeps state for the complete sequence, clip resets at each "
            "validation clip, and frame resets before every frame"
        ),
    )
    parser.add_argument(
        "--drop-length",
        type=int,
        default=0,
        help="Number of consecutive event frames to replace with an empty event frame",
    )
    parser.add_argument(
        "--drop-start",
        type=int,
        default=32,
        help="Zero-based sequence frame at which the first drop window starts",
    )
    parser.add_argument(
        "--drop-stride",
        type=int,
        default=64,
        help="Distance in frames between repeated drop-window starts",
    )
    parser.add_argument(
        "--feature",
        action="append",
        choices=("Pz", "Ph"),
        default=None,
        help="Projected feature to export; repeat as needed (defaults to active objectives)",
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=None,
        help="Write shared RGB/event/teacher clips here (normally only for E0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing clip artifacts; otherwise valid files are preserved",
    )
    return parser.parse_args()


def _valid_existing(
    path: Path,
    *,
    sequence: str,
    clip_index: int,
    artifact_kind: str,
    checkpoint_step: int | None = None,
    state_policy: str | None = None,
    event_drop: dict[str, int] | None = None,
    event_window_fraction: float | None = None,
    required_features: list[str] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        return (
            isinstance(value, dict)
            and value.get("format_version") == SEQUENCE_FEATURE_FORMAT_VERSION
            and value.get("sequence_name") == sequence
            and int(value.get("clip_index", -1)) == clip_index
            and value.get("artifact_kind") == artifact_kind
            and (
                checkpoint_step is None
                or int(value.get("checkpoint_step", -1)) == checkpoint_step
            )
            and (state_policy is None or value.get("state_policy") == state_policy)
            and (event_drop is None or value.get("event_drop") == event_drop)
            and (
                event_window_fraction is None
                or float(value.get("event_window_fraction", -1.0))
                == event_window_fraction
            )
            and (
                required_features is None
                or (
                    isinstance(value.get("features"), dict)
                    and set(required_features) <= set(value["features"])
                )
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _write_metadata(
    directory: Path,
    *,
    sequence: str,
    experiment: str,
    checkpoint: Path,
    checkpoint_step: int,
    clip_count: int,
    frame_count: int,
    feature_names: list[str],
    state_policy: str,
    event_drop: dict[str, int],
    event_window_fraction: float,
) -> None:
    metadata = {
        "format_version": SEQUENCE_FEATURE_FORMAT_VERSION,
        "sequence_name": sequence,
        "experiment": experiment,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_step": checkpoint_step,
        "clip_count": clip_count,
        "frame_count": frame_count,
        "feature_names": feature_names,
        "state_policy": state_policy,
        "event_drop": event_drop,
        "event_window_fraction": event_window_fraction,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _drop_mask(
    *,
    sequence_offset: int,
    frame_count: int,
    start: int,
    stride: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(
        sequence_offset,
        sequence_offset + frame_count,
        device=device,
    )
    if length == 0:
        return torch.zeros(frame_count, dtype=torch.bool, device=device)
    relative = indices - start
    return (relative >= 0) & (torch.remainder(relative, stride) < length)


def _empty_event_value(events: torch.Tensor, config: Any) -> torch.Tensor:
    representation = config.dataset.representation
    if str(representation.type) == "gep_rgb":
        means = events.new_tensor(list(representation.normalize_mean))
        stds = events.new_tensor(list(representation.normalize_std))
        return ((1.0 - means) / stds).view(1, 1, -1, 1, 1)
    return events.new_zeros((1, 1, events.shape[2], 1, 1))


def _mask_events(
    events: torch.Tensor,
    mask: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    if not bool(mask.any()):
        return events
    expanded_mask = mask.view(1, -1, 1, 1, 1)
    return torch.where(expanded_mask, _empty_event_value(events, config), events)


def main() -> None:
    args = parse_args()
    if args.drop_length < 0:
        raise ValueError("--drop-length must be non-negative")
    if args.drop_start < 0:
        raise ValueError("--drop-start must be non-negative")
    if args.drop_stride <= 0:
        raise ValueError("--drop-stride must be positive")
    if args.drop_length > args.drop_stride:
        raise ValueError("--drop-length must not exceed --drop-stride")
    event_drop = {
        "length": int(args.drop_length),
        "start": int(args.drop_start),
        "stride": int(args.drop_stride),
    }
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.context_dir is not None:
        args.context_dir = args.context_dir.expanduser()
        args.context_dir.mkdir(parents=True, exist_ok=True)

    config, _ = _prepare_config(args)
    print(
        f"Building validation runtime for {args.sequence}; "
        f"state_policy={args.state_policy}, event_drop={event_drop}...",
        flush=True,
    )
    runtime = build_evaluation_runtime(config)
    state = load_checkpoint(
        args.checkpoint,
        model=runtime.model,
        device=runtime.device,
        restore_rng=False,
        expected_config=config,
        compatibility_mode="evaluation",
    )
    runtime.model.eval()
    objectives = _objective_metadata(config)
    active_features = {
        "Pz": objectives["z"],
        "Ph": objectives["h"],
    }
    feature_names = args.feature or [
        name for name, active in active_features.items() if active
    ]
    unavailable = [name for name in feature_names if not active_features[name]]
    if unavailable:
        raise ValueError(
            "Requested features do not have trained objectives: " + ", ".join(unavailable)
        )
    if not feature_names:
        raise RuntimeError("The checkpoint has no active projected objective")

    recurrent_state = None
    clip_count = 0
    frame_count = 0
    dataset = runtime.validation_loader.dataset
    grid_height = int(config.dataset.input_height) // int(config.teacher.patch_size)
    grid_width = int(config.dataset.input_width) // int(config.teacher.patch_size)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if runtime.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        for batch in runtime.validation_loader:
            sequence_name = str(batch["sequence_name"][0])
            if sequence_name != args.sequence:
                continue
            if _batch_bool(batch, "is_sequence_start") or args.state_policy == "clip":
                recurrent_state = None
            events = _move(batch["events"], runtime.device)
            event_dropped = _drop_mask(
                sequence_offset=frame_count,
                frame_count=int(events.shape[1]),
                start=args.drop_start,
                stride=args.drop_stride,
                length=args.drop_length,
                device=runtime.device,
            )
            events = _mask_events(events, event_dropped, config)
            if args.state_policy == "frame":
                frame_outputs = [
                    runtime.model(events[:, index : index + 1], state=None)
                    for index in range(int(events.shape[1]))
                ]
                outputs = {
                    "z": torch.cat([value["z"] for value in frame_outputs], dim=1),
                    "h": torch.cat([value["h"] for value in frame_outputs], dim=1),
                    "state": None,
                }
                recurrent_state = None
            else:
                outputs = runtime.model(events, state=recurrent_state)
                recurrent_state = outputs.get("state")
            timestamps = batch["timestamps"][0].detach().cpu()
            features: dict[str, torch.Tensor] = {}
            if "Pz" in feature_names:
                features["Pz"] = (
                    runtime.model.project_z(outputs["z"])[0].detach().cpu().to(torch.float16)
                )
            if "Ph" in feature_names:
                features["Ph"] = (
                    runtime.model.project_h(outputs["h"])[0].detach().cpu().to(torch.float16)
                )
            clip_path = args.output_dir / f"{clip_count:06d}.pt"
            payload = {
                "format_version": SEQUENCE_FEATURE_FORMAT_VERSION,
                "artifact_kind": "model_features",
                "sequence_name": sequence_name,
                "clip_index": clip_count,
                "checkpoint_step": int(state.global_step),
                "state_policy": args.state_policy,
                "event_drop": event_drop,
                "event_window_fraction": float(args.event_window_fraction),
                "event_dropped": event_dropped.detach().cpu(),
                "timestamps": timestamps,
                "frame_indices": batch["frame_indices"][0].detach().cpu(),
                "grid_size": [grid_height, grid_width],
                "features": features,
            }
            if args.overwrite or not _valid_existing(
                clip_path,
                sequence=sequence_name,
                clip_index=clip_count,
                artifact_kind="model_features",
                checkpoint_step=int(state.global_step),
                state_policy=args.state_policy,
                event_drop=event_drop,
                event_window_fraction=float(args.event_window_fraction),
                required_features=feature_names,
            ):
                atomic_torch_save(payload, clip_path)

            if args.context_dir is not None:
                teacher = batch.get("teacher_features")
                if not isinstance(teacher, torch.Tensor):
                    raise RuntimeError("Sequence visualization requires cached teacher features")
                context_path = args.context_dir / f"{clip_count:06d}.pt"
                context = {
                    "format_version": SEQUENCE_FEATURE_FORMAT_VERSION,
                    "artifact_kind": "shared_context",
                    "sequence_name": sequence_name,
                    "clip_index": clip_count,
                    "timestamps": timestamps,
                    "frame_indices": batch["frame_indices"][0].detach().cpu(),
                    "event_counts": torch.where(
                        event_dropped.detach().cpu(),
                        torch.zeros_like(batch["event_counts"][0]),
                        batch["event_counts"][0],
                    ),
                    "original_event_counts": batch["event_counts"][0].detach().cpu(),
                    "event_dropped": event_dropped.detach().cpu(),
                    "event_drop": event_drop,
                    "event_window_fraction": float(args.event_window_fraction),
                    "grid_size": [grid_height, grid_width],
                    "event_rgb": _event_preview(events, config),
                    "empty_event_rgb": _event_preview(
                        _empty_event_value(events, config).expand_as(events[:, :1]),
                        config,
                    )[0],
                    "rgb": _rgb_preview(dataset, sequence_name, timestamps),
                    "teacher": teacher[0].detach().cpu().to(torch.float16),
                }
                if args.overwrite or not _valid_existing(
                    context_path,
                    sequence=sequence_name,
                    clip_index=clip_count,
                    artifact_kind="shared_context",
                    event_drop=event_drop,
                    event_window_fraction=float(args.event_window_fraction),
                ):
                    atomic_torch_save(context, context_path)

            clip_frames = int(timestamps.numel())
            frame_count += clip_frames
            clip_count += 1
            print(
                f"[{sequence_name}] clip={clip_count} frames={frame_count}",
                flush=True,
            )

    if clip_count == 0:
        raise ValueError(f"Sequence not found in the validation split: {args.sequence}")
    _write_metadata(
        args.output_dir,
        sequence=args.sequence,
        experiment=str(config.experiment.name),
        checkpoint=args.checkpoint,
        checkpoint_step=int(state.global_step),
        clip_count=clip_count,
        frame_count=frame_count,
        feature_names=feature_names,
        state_policy=args.state_policy,
        event_drop=event_drop,
        event_window_fraction=float(args.event_window_fraction),
    )
    if args.context_dir is not None:
        _write_metadata(
            args.context_dir,
            sequence=args.sequence,
            experiment="shared_context",
            checkpoint=args.checkpoint,
            checkpoint_step=int(state.global_step),
            clip_count=clip_count,
            frame_count=frame_count,
            feature_names=["teacher", "event_rgb", "rgb"],
            state_policy="state_independent",
            event_drop=event_drop,
            event_window_fraction=float(args.event_window_fraction),
        )
    print(
        f"Exported {frame_count} frames in {clip_count} clips to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
