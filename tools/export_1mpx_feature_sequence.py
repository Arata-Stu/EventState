#!/usr/bin/env python3
"""Run an EventState checkpoint over a DAGR-downsampled 1Mpx stream."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from event_state.data.event_representation import GEPEventFrame
from event_state.training.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_checkpoint_config_metadata,
)
from event_state.training.factory import build_model, resolve_device


FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a DAGR-downsampled 640x360 1Mpx HDF5 file through an "
            "EventState checkpoint using 88 pixels of bottom padding and a token mask."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--start", type=float, default=0.0, help="Seconds from stream start")
    parser.add_argument("--end", type=float, default=None, help="Seconds from stream start")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=("z", "h", "Pz", "Ph"),
        default=None,
        help="Features saved per frame; defaults to z/h and trained projections",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Relocated DINOv3 initialization checkpoint, when the saved path is unavailable",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _set(config: Any, path: str, value: Any) -> None:
    OmegaConf.update(config, path, value, merge=False, force_add=True)


def _read_metadata(handle: h5py.File) -> dict[str, Any]:
    raw = handle.attrs.get("event_state_metadata")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("input was not produced by tools/prepare_1mpx_dagr.py")
    metadata = json.loads(raw)
    expected = {
        "format_version": 1,
        "event_size": [360, 640],
        "model_input_size": [448, 640],
        "padding": {"top": 0, "bottom": 88, "left": 0, "right": 0},
        "patch_size": 16,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"input metadata {name}={metadata.get(name)!r}, expected {value!r}")
    return metadata


def _indices_for_window(
    timestamps: h5py.Dataset,
    ms_to_idx: h5py.Dataset,
    start_us: int,
    end_us: int,
) -> tuple[int, int]:
    start_ms = max(0, start_us // 1000)
    end_ms = max(0, end_us // 1000 + 1)
    count = len(timestamps)
    begin_hint = int(ms_to_idx[start_ms]) if start_ms < len(ms_to_idx) else count
    end_hint = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else count
    if begin_hint >= end_hint:
        return begin_hint, begin_hint
    local = np.asarray(timestamps[begin_hint:end_hint], dtype=np.uint64)
    begin = begin_hint + int(np.searchsorted(local, start_us, side="right"))
    end = begin_hint + int(np.searchsorted(local, end_us, side="right"))
    return begin, end


def _mean_token_cosine(current: torch.Tensor, previous: torch.Tensor) -> float:
    similarity = (
        F.normalize(current.float(), dim=-1)
        * F.normalize(previous.float(), dim=-1)
    ).sum(-1)
    return float(similarity.mean())


def _mean_token_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float(), dim=-1).mean())


def _active_projection_names(config: Any) -> list[str]:
    names = []
    if str(config.loss.z_objective.type).lower() == "direct_dino" and float(
        config.loss.z_objective.weight
    ) > 0:
        names.append("Pz")
    if bool(config.loss.h_distill.enabled):
        names.append("Ph")
    return names


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if args.window_ms <= 0 or args.start < 0:
        raise ValueError("--window-ms must be positive and --start non-negative")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if (
        args.backbone_checkpoint is not None
        and not args.backbone_checkpoint.expanduser().is_file()
    ):
        raise FileNotFoundError(
            f"backbone checkpoint not found: {args.backbone_checkpoint.expanduser()}"
        )
    frames_dir = output_dir / "frames"
    metrics_path = output_dir / "metrics.csv"
    metadata_path = output_dir / "metadata.json"
    output_exists = metrics_path.exists() or metadata_path.exists() or frames_dir.exists()
    if output_exists and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale_path in frames_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].pt"):
            stale_path.unlink()

    checkpoint_metadata = load_checkpoint_config_metadata(checkpoint_path)
    config = OmegaConf.create(checkpoint_metadata.config)
    if str(config.dataset.representation.type) != "gep_rgb":
        raise ValueError("this 1Mpx adapter currently requires a gep_rgb checkpoint")
    if int(config.dataset.input_height) != 448 or int(config.dataset.input_width) != 640:
        raise ValueError("checkpoint must use a 448x640 EventState input")
    if args.backbone_checkpoint is not None:
        relocated = str(args.backbone_checkpoint.expanduser().resolve())
        _set(config, "teacher.checkpoint", relocated)
        if OmegaConf.select(config, "model.event_encoder.checkpoint") is not None:
            _set(config, "model.event_encoder.checkpoint", relocated)
    device = resolve_device(args.device)
    model = build_model(config, device=device)
    checkpoint_state = load_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
        restore_rng=False,
        expected_config=config,
        compatibility_mode="evaluation",
    )
    model.eval()

    active_projections = _active_projection_names(config)
    requested_features = args.features or ["z", "h", *active_projections]
    unavailable = sorted(
        (set(requested_features) & {"Pz", "Ph"}) - set(active_projections)
    )
    if unavailable:
        raise ValueError(
            "requested projections do not have trained objectives: " + ", ".join(unavailable)
        )
    percentile = float(config.dataset.representation.percentile)
    representation = GEPEventFrame(height=448, width=640, percentile=percentile)
    means = torch.tensor(list(config.dataset.representation.normalize_mean)).view(3, 1, 1)
    stds = torch.tensor(list(config.dataset.representation.normalize_std)).view(3, 1, 1)
    window_us = int(round(args.window_ms * 1000.0))
    previous_features: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    recurrent_state = None
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with h5py.File(input_path, "r") as handle:
        input_metadata = _read_metadata(handle)
        timestamps = handle["events/t"]
        if len(timestamps) == 0:
            raise RuntimeError("downsampled input contains no events")
        ms_to_idx = handle["ms_to_idx"]
        timestamp_offset = int(handle["t_offset"][()])
        patch_mask_cpu = torch.from_numpy(
            np.asarray(handle["geometry/patch_mask"], dtype=np.bool_)
        )
        patch_fraction = torch.from_numpy(
            np.asarray(handle["geometry/patch_valid_fraction"], dtype=np.float32)
        )
        flat_mask = patch_mask_cpu.reshape(-1).to(device=device)
        token_mask = flat_mask.view(1, -1, 1)
        valid = flat_mask
        stream_end_us = int(timestamps[-1])
        start_us = int(round(args.start * 1_000_000.0))
        requested_end_us = (
            stream_end_us if args.end is None else int(round(args.end * 1_000_000.0))
        )
        end_us = min(stream_end_us, requested_end_us)
        if start_us >= end_us:
            raise ValueError("selected interval contains no stream duration")

        saved_frame_index = 0
        # Always warm the recurrent model from the stream origin. This preserves
        # the state that existed before a requested low-activity interval.
        with torch.inference_mode(), autocast:
            for stream_frame_index, frame_start in enumerate(range(0, end_us, window_us)):
                frame_end = min(end_us, frame_start + window_us)
                begin, end = _indices_for_window(timestamps, ms_to_idx, frame_start, frame_end)
                x = torch.from_numpy(np.asarray(handle["events/x"][begin:end], dtype=np.int64))
                y = torch.from_numpy(np.asarray(handle["events/y"][begin:end], dtype=np.int64))
                t = torch.from_numpy(np.asarray(handle["events/t"][begin:end], dtype=np.int64))
                p = torch.from_numpy(np.asarray(handle["events/p"][begin:end], dtype=np.int8))
                event_image = representation(
                    x, y, t, p, start_time=frame_start, end_time=frame_end
                )
                normalized = ((event_image - means) / stds).unsqueeze(0).to(device)
                z = model.encode_event(normalized) * token_mask
                h, recurrent_state = model.update_state(z, recurrent_state)
                h = h * token_mask
                all_features = {"z": z[0], "h": h[0]}
                if "Pz" in requested_features:
                    all_features["Pz"] = model.project_z(z)[0] * token_mask[0]
                if "Ph" in requested_features:
                    all_features["Ph"] = model.project_h(h)[0] * token_mask[0]
                current_features = {
                    name: value[valid].detach().float().cpu()
                    for name, value in all_features.items()
                }
                if frame_end <= start_us:
                    previous_features = current_features
                    print(
                        f"\rwarming state; t={frame_end / 1e6:.3f}s",
                        end="",
                        flush=True,
                    )
                    continue
                selected_features = {
                    name: all_features[name][valid].detach().cpu().to(torch.float16)
                    for name in requested_features
                }
                artifact = {
                    "format_version": FORMAT_VERSION,
                    "frame_index": saved_frame_index,
                    "stream_frame_index": stream_frame_index,
                    "start_timestamp": timestamp_offset + frame_start,
                    "end_timestamp": timestamp_offset + frame_end,
                    "event_count": end - begin,
                    "grid_size": [28, 40],
                    "patch_mask": patch_mask_cpu,
                    "patch_valid_fraction": patch_fraction,
                    "features_are_mask_compacted": True,
                    "features": selected_features,
                }
                atomic_torch_save(
                    artifact, frames_dir / f"{saved_frame_index:06d}.pt"
                )
                row: dict[str, Any] = {
                    "frame_index": saved_frame_index,
                    "stream_frame_index": stream_frame_index,
                    "time_s": frame_end / 1_000_000.0,
                    "event_count": end - begin,
                }
                for name, value in current_features.items():
                    row[f"{name}_mean_norm"] = _mean_token_norm(value)
                    row[f"{name}_previous_cosine"] = (
                        math.nan
                        if name not in previous_features
                        else _mean_token_cosine(value, previous_features[name])
                    )
                    previous_features[name] = value
                rows.append(row)
                print(
                    f"\rframe {saved_frame_index + 1}; "
                    f"t={frame_end / 1e6:.3f}s; events={end - begin:,}",
                    end="",
                    flush=True,
                )
                saved_frame_index += 1

    fieldnames = list(rows[0])
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_metadata = {
        "format_version": FORMAT_VERSION,
        "source": str(input_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_state.global_step),
        "window_ms": args.window_ms,
        "selected_interval_s": [args.start, args.end],
        "event_geometry": [360, 640],
        "model_input_geometry": [448, 640],
        "bottom_padding": 88,
        "grid_size": [28, 40],
        "valid_patch_rows": 23,
        "valid_tokens": int(patch_mask_cpu.sum()),
        "saved_features": requested_features,
        "input_metadata": input_metadata,
    }
    metadata_path.write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print()
    print(f"Saved {len(rows)} masked feature frames to {output_dir}")


if __name__ == "__main__":
    main()
