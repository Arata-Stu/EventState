#!/usr/bin/env python3
"""Export one checkpoint over multiple complete DSEC sequences in one load."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from event_state.training.checkpoint import atomic_torch_save, load_checkpoint
from event_state.training.factory import build_evaluation_runtime
from export_feature_sequence import (
    SEQUENCE_FEATURE_FORMAT_VERSION,
    _valid_existing,
    _write_metadata,
)
from export_features import (
    _batch_bool,
    _event_preview,
    _move,
    _objective_metadata,
    _prepare_config,
    _rgb_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream one checkpoint over several complete DSEC sequences"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Per-sequence output subdirectory")
    parser.add_argument("--write-context", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature", action="append", choices=("Pz", "Ph"), required=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _set(config: Any, path: str, value: Any) -> None:
    OmegaConf.update(config, path, value, merge=False, force_add=True)


def main() -> None:
    args = parse_args()
    if len(args.sequences) != len(set(args.sequences)):
        raise ValueError("--sequences contains duplicates")
    for name in ("checkpoint", "root", "event_cache_dir", "teacher_cache_dir"):
        path = getattr(args, name).expanduser()
        expected = path.is_file() if name == "checkpoint" else path.is_dir()
        if not expected:
            raise FileNotFoundError(f"{name.replace('_', '-')} not found: {path}")
    if args.teacher_checkpoint is not None:
        if not args.teacher_checkpoint.expanduser().is_file():
            raise FileNotFoundError(
                f"teacher-checkpoint not found: {args.teacher_checkpoint.expanduser()}"
            )

    # Reuse the single-sequence evaluation contract, then select all requested
    # sequences so the model and manifests are initialized only once.
    args.sequence = None
    args.event_window_fraction = 1.0
    config, _ = _prepare_config(args)
    _set(config, "dataset.val_split", args.split)
    _set(config, "dataset.val_sequences", args.sequences)
    print(
        f"Building one {args.label} runtime for {len(args.sequences)} sequences...",
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
    active = _objective_metadata(config)
    unavailable = [
        name
        for name in args.feature
        if not active["z" if name == "Pz" else "h"]
    ]
    if unavailable:
        raise ValueError("Requested untrained projected features: " + ", ".join(unavailable))

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = runtime.validation_loader.dataset
    grid_height = int(config.dataset.input_height) // int(config.teacher.patch_size)
    grid_width = int(config.dataset.input_width) // int(config.teacher.patch_size)
    clip_counts = {sequence: 0 for sequence in args.sequences}
    frame_counts = {sequence: 0 for sequence in args.sequences}
    recurrent_state = None
    current_sequence: str | None = None
    event_drop = {"length": 0, "start": 0, "stride": 1}
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if runtime.device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.no_grad(), autocast:
        for batch in runtime.validation_loader:
            sequence = str(batch["sequence_name"][0])
            if sequence != current_sequence or _batch_bool(batch, "is_sequence_start"):
                current_sequence = sequence
                recurrent_state = None
            events = _move(batch["events"], runtime.device)
            outputs = runtime.model(events, state=recurrent_state)
            recurrent_state = outputs.get("state")
            features: dict[str, torch.Tensor] = {}
            if "Pz" in args.feature:
                features["Pz"] = (
                    runtime.model.project_z(outputs["z"])[0]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                )
            if "Ph" in args.feature:
                features["Ph"] = (
                    runtime.model.project_h(outputs["h"])[0]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                )

            clip_index = clip_counts[sequence]
            timestamps = batch["timestamps"][0].detach().cpu()
            destination = output_root / sequence / args.label
            destination.mkdir(parents=True, exist_ok=True)
            clip_path = destination / f"{clip_index:06d}.pt"
            dropped = torch.zeros(int(events.shape[1]), dtype=torch.bool)
            payload = {
                "format_version": SEQUENCE_FEATURE_FORMAT_VERSION,
                "artifact_kind": "model_features",
                "sequence_name": sequence,
                "clip_index": clip_index,
                "checkpoint_step": int(state.global_step),
                "state_policy": "continuous",
                "event_drop": event_drop,
                "event_window_fraction": 1.0,
                "event_dropped": dropped,
                "timestamps": timestamps,
                "frame_indices": batch["frame_indices"][0].detach().cpu(),
                "grid_size": [grid_height, grid_width],
                "features": features,
            }
            if args.overwrite or not _valid_existing(
                clip_path,
                sequence=sequence,
                clip_index=clip_index,
                artifact_kind="model_features",
                checkpoint_step=int(state.global_step),
                state_policy="continuous",
                event_drop=event_drop,
                event_window_fraction=1.0,
                required_features=args.feature,
            ):
                atomic_torch_save(payload, clip_path)

            if args.write_context:
                context_dir = output_root / sequence / "context"
                context_dir.mkdir(parents=True, exist_ok=True)
                context_path = context_dir / f"{clip_index:06d}.pt"
                teacher = batch.get("teacher_features")
                if not isinstance(teacher, torch.Tensor):
                    raise RuntimeError("Visualization requires cached teacher features")
                context = {
                    "format_version": SEQUENCE_FEATURE_FORMAT_VERSION,
                    "artifact_kind": "shared_context",
                    "sequence_name": sequence,
                    "clip_index": clip_index,
                    "timestamps": timestamps,
                    "frame_indices": batch["frame_indices"][0].detach().cpu(),
                    "event_counts": batch["event_counts"][0].detach().cpu(),
                    "original_event_counts": batch["event_counts"][0].detach().cpu(),
                    "event_dropped": dropped,
                    "event_drop": event_drop,
                    "event_window_fraction": 1.0,
                    "grid_size": [grid_height, grid_width],
                    "event_rgb": _event_preview(events, config),
                    "rgb": _rgb_preview(dataset, sequence, timestamps),
                    "teacher": teacher[0].detach().cpu().to(torch.float16),
                }
                if args.overwrite or not _valid_existing(
                    context_path,
                    sequence=sequence,
                    clip_index=clip_index,
                    artifact_kind="shared_context",
                    event_drop=event_drop,
                    event_window_fraction=1.0,
                ):
                    atomic_torch_save(context, context_path)

            clip_frames = int(timestamps.numel())
            clip_counts[sequence] += 1
            frame_counts[sequence] += clip_frames
            print(
                f"[{args.label}:{sequence}] clips={clip_counts[sequence]} "
                f"frames={frame_counts[sequence]}",
                flush=True,
            )

    for sequence in args.sequences:
        if clip_counts[sequence] == 0:
            raise RuntimeError(f"No clips exported for {sequence}")
        _write_metadata(
            output_root / sequence / args.label,
            sequence=sequence,
            experiment=str(config.experiment.name),
            checkpoint=args.checkpoint,
            checkpoint_step=int(state.global_step),
            clip_count=clip_counts[sequence],
            frame_count=frame_counts[sequence],
            feature_names=args.feature,
            state_policy="continuous",
            event_drop=event_drop,
            event_window_fraction=1.0,
        )
        if args.write_context:
            _write_metadata(
                output_root / sequence / "context",
                sequence=sequence,
                experiment="shared_context",
                checkpoint=args.checkpoint,
                checkpoint_step=int(state.global_step),
                clip_count=clip_counts[sequence],
                frame_count=frame_counts[sequence],
                feature_names=["teacher", "event_rgb", "rgb"],
                state_policy="state_independent",
                event_drop=event_drop,
                event_window_fraction=1.0,
            )


if __name__ == "__main__":
    main()
