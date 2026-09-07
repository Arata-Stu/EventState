#!/usr/bin/env python3
"""Export one streaming DSEC clip for representation visualization."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from event_state.training.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_checkpoint_config_metadata,
)
from event_state.training.factory import build_evaluation_runtime


FEATURE_ARTIFACT_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export z/h, their teacher-space projections, DINOv3 tokens, and "
            "input previews for one validation clip."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Relocated DINOv3 checkpoint; omit when the saved path is still valid",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Validation sequence to export (defaults to the first configured sequence)",
    )
    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Zero-based, non-overlapping clip index within the selected sequence",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _set(config: Any, path: str, value: Any) -> None:
    OmegaConf.update(config, path, value, merge=False, force_add=True)


def _prepare_config(args: argparse.Namespace) -> tuple[Any, Any]:
    metadata = load_checkpoint_config_metadata(args.checkpoint)
    config = OmegaConf.create(metadata.config)
    _set(config, "dataset.root", str(args.root.expanduser().resolve()))
    _set(
        config,
        "dataset.event_cache_dir",
        str(args.event_cache_dir.expanduser().resolve()),
    )
    _set(
        config,
        "teacher.cache_dir",
        str(args.teacher_cache_dir.expanduser().resolve()),
    )
    if args.teacher_checkpoint is not None:
        _set(
            config,
            "teacher.checkpoint",
            str(args.teacher_checkpoint.expanduser().resolve()),
        )
    if args.sequence is not None:
        _set(config, "dataset.val_sequences", [args.sequence])
    _set(config, "device", args.device)
    _set(config, "training.num_workers", 0)
    _set(config, "training.pin_memory", False)
    _set(config, "training.persistent_workers", False)
    _set(config, "evaluation.checkpoint", str(args.checkpoint.expanduser().resolve()))
    return config, metadata


def _batch_bool(batch: dict[str, Any], name: str) -> bool:
    value = batch.get(name)
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    return value


def _event_preview(events: torch.Tensor, config: Any) -> torch.Tensor:
    """Convert transformed event input to a compact uint8 preview."""

    value = events.detach().cpu().float()
    representation = config.dataset.representation
    means = OmegaConf.select(representation, "normalize_mean")
    stds = OmegaConf.select(representation, "normalize_std")
    if means is not None and stds is not None and value.shape[2] == len(means):
        mean = value.new_tensor(list(means)).view(1, 1, -1, 1, 1)
        std = value.new_tensor(list(stds)).view(1, 1, -1, 1, 1)
        value = value * std + mean
    else:
        flat = value.flatten(-2)
        lower = torch.quantile(flat, 0.01, dim=-1, keepdim=True).unsqueeze(-1)
        upper = torch.quantile(flat, 0.99, dim=-1, keepdim=True).unsqueeze(-1)
        value = (value - lower) / (upper - lower).clamp_min(1e-6)
    return (value.clamp(0, 1) * 255).round().to(torch.uint8)[0]


def _rgb_preview(dataset: Any, sequence_name: str, timestamps: torch.Tensor) -> torch.Tensor:
    frames = dataset._frames_by_sequence[sequence_name]
    by_timestamp = {frame.timestamp: frame for frame in frames}
    selected = []
    for timestamp in timestamps.reshape(-1).tolist():
        frame = by_timestamp.get(int(timestamp))
        if frame is None:
            raise ValueError(f"Timestamp {timestamp} is absent from {sequence_name}")
        selected.append(dataset._load_image(frame.image_path))
    images = torch.stack(selected)
    _, transformed = dataset.transform(None, images)
    if transformed is None:
        raise RuntimeError("RGB preview transform returned no images")
    return (transformed.clamp(0, 1) * 255).round().to(torch.uint8)


def _objective_metadata(config: Any) -> dict[str, bool]:
    return {
        "z": str(config.loss.z_objective.type).lower() == "direct_dino"
        and float(config.loss.z_objective.weight) > 0,
        "h": bool(config.loss.h_distill.enabled),
    }


def main() -> None:
    args = parse_args()
    if args.clip_index < 0:
        raise ValueError("--clip-index must be non-negative")
    for name in ("checkpoint", "root", "event_cache_dir", "teacher_cache_dir"):
        path = getattr(args, name).expanduser()
        expected = path.is_file() if name == "checkpoint" else path.is_dir()
        if not expected:
            raise FileNotFoundError(f"{name.replace('_', '-')} not found: {path}")
    if args.teacher_checkpoint is not None and not args.teacher_checkpoint.expanduser().is_file():
        raise FileNotFoundError(
            f"teacher-checkpoint not found: {args.teacher_checkpoint.expanduser()}"
        )

    config, metadata = _prepare_config(args)
    print("Building validation runtime and checking the selected sequence caches...", flush=True)
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

    selected_batch: dict[str, Any] | None = None
    selected_outputs: dict[str, Any] | None = None
    recurrent_state = None
    current_sequence: str | None = None
    sequence_clip_index = -1
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if runtime.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast:
        for batch in runtime.validation_loader:
            sequence_name = str(batch["sequence_name"][0])
            if sequence_name != current_sequence or _batch_bool(batch, "is_sequence_start"):
                current_sequence = sequence_name
                sequence_clip_index = 0
                recurrent_state = None
            else:
                sequence_clip_index += 1
            events = _move(batch["events"], runtime.device)
            outputs = runtime.model(events, state=recurrent_state)
            recurrent_state = outputs.get("state")
            if args.sequence is not None and sequence_name != args.sequence:
                continue
            if sequence_clip_index == args.clip_index:
                selected_batch = dict(batch)
                selected_outputs = {
                    **outputs,
                    "z_projected": runtime.model.project_z(outputs["z"]),
                    "h_projected": runtime.model.project_h(outputs["h"]),
                }
                break

    if selected_batch is None or selected_outputs is None:
        sequence_label = args.sequence or "the first validation sequence"
        raise IndexError(f"Clip {args.clip_index} was not found in {sequence_label}")

    sequence_name = str(selected_batch["sequence_name"][0])
    teacher = selected_batch.get("teacher_features")
    if not isinstance(teacher, torch.Tensor):
        raise RuntimeError("Feature export currently requires cached teacher features")
    z = selected_outputs["z"]
    h = selected_outputs["h"]
    timestamps = selected_batch["timestamps"][0].detach().cpu()
    dataset = runtime.validation_loader.dataset
    grid_height = int(config.dataset.input_height) // int(config.teacher.patch_size)
    grid_width = int(config.dataset.input_width) // int(config.teacher.patch_size)
    payload = {
        "format_version": FEATURE_ARTIFACT_FORMAT_VERSION,
        "experiment": str(config.experiment.name),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_step": int(state.global_step),
        "sequence_name": sequence_name,
        "clip_index": int(args.clip_index),
        "timestamps": timestamps,
        "frame_indices": selected_batch["frame_indices"][0].detach().cpu(),
        "event_counts": selected_batch["event_counts"][0].detach().cpu(),
        "grid_size": [grid_height, grid_width],
        "input_size": [int(config.dataset.input_height), int(config.dataset.input_width)],
        "active_objectives": _objective_metadata(config),
        "event_rgb": _event_preview(selected_batch["events"], config),
        "rgb": _rgb_preview(dataset, sequence_name, timestamps),
        "z": z[0].detach().cpu(),
        "h": h[0].detach().cpu(),
        "z_projected": selected_outputs["z_projected"][0].detach().cpu(),
        "h_projected": selected_outputs["h_projected"][0].detach().cpu(),
        "teacher": teacher[0].detach().cpu(),
        "checkpoint_signature_format": metadata.signatures.get("format_version"),
    }
    destination = atomic_torch_save(payload, args.output.expanduser())
    print(
        f"Exported {sequence_name} clip={args.clip_index} step={state.global_step} "
        f"to {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
