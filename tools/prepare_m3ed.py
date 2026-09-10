#!/usr/bin/env python3
"""Prepare half-scale, DAGR-filtered M3ED for EventState pretraining."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import h5py
import hdf5plugin  # noqa: F401 - registers common M3ED compression filters.
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from event_state.data.dagr_downsample import DAGRDownsampler
from event_state.data.event_representation import EventVoxelizer, GEPEventFrame
from event_state.data.m3ed import M3ED_PREPARED_FORMAT_VERSION


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare M3ED at 640x360. DAGR's stateful signed-event filter is "
            "applied to the raw 1280x720 stream before calibration and imaging."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="M3ED download root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Sequence names or *_data.h5 paths; defaults to every recording below root",
    )
    parser.add_argument("--camera", choices=("left",), default="left")
    parser.add_argument(
        "--representation", choices=("gep_rgb", "voxel_grid"), default="gep_rgb"
    )
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument(
        "--voxel-normalization",
        choices=("none", "nonzero_standardize", "log1p"),
        default="nonzero_standardize",
    )
    parser.add_argument("--chunk-events", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sequence_name(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_data")] if stem.endswith("_data") else stem


def discover_recordings(root: Path, requested: Sequence[str] | None) -> list[Path]:
    candidates = sorted(root.rglob("*_data.h5"))
    by_name: dict[str, Path] = {}
    for path in candidates:
        name = _sequence_name(path)
        if name in by_name:
            raise ValueError(f"Duplicate M3ED sequence name {name!r} below {root}")
        by_name[name] = path
    if requested is None:
        selected = list(by_name.values())
    else:
        selected = []
        for value in requested:
            direct = Path(value).expanduser()
            if not direct.is_absolute():
                direct = root / direct
            if direct.is_file():
                selected.append(direct.resolve())
                continue
            name = Path(value).name
            if name.endswith("_data.h5"):
                name = _sequence_name(Path(name))
            if name not in by_name:
                raise FileNotFoundError(f"M3ED sequence not found: {value}")
            selected.append(by_name[name])
    names = [_sequence_name(path) for path in selected]
    if len(names) != len(set(names)):
        raise ValueError("Requested M3ED sequences contain duplicates")
    if not selected:
        raise ValueError(f"No *_data.h5 recordings found below {root}")
    return selected


def _decode(value: Any) -> str:
    if isinstance(value, h5py.Dataset):
        value = value[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _camera_matrix(calibration: h5py.Group, scale: float = 1.0) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(calibration["intrinsics"], dtype=np.float64)
    return np.asarray(
        [[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _distortion(calibration: h5py.Group) -> tuple[str, np.ndarray]:
    return (
        _decode(calibration["distortion_model"]).lower(),
        np.asarray(calibration["distortion_coeffs"], dtype=np.float64),
    )


def _rgb_to_event_map(
    event_calibration: h5py.Group,
    rgb_calibration: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a rotation-only raw-RGB to rectified-event remap at half scale."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    output_size = (640, 360)
    target_k = _camera_matrix(event_calibration, scale=0.5)
    source_k = _camera_matrix(rgb_calibration)
    model, coefficients = _distortion(rgb_calibration)
    target_from_source = np.asarray(
        rgb_calibration["T_to_prophesee_left"], dtype=np.float64
    )
    rotation = target_from_source[:3, :3]
    if model == "equidistant":
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            source_k,
            coefficients.reshape(-1, 1),
            rotation,
            target_k,
            output_size,
            cv2.CV_32FC1,
        )
    elif model in {"radtan", "plumb_bob"}:
        map_x, map_y = cv2.initUndistortRectifyMap(
            source_k,
            coefficients,
            rotation,
            target_k,
            output_size,
            cv2.CV_32FC1,
        )
    else:
        raise ValueError(f"Unsupported M3ED RGB distortion model: {model}")
    return map_x, map_y, target_k


def _rectify_half_scale_events(
    x: np.ndarray,
    y: np.ndarray,
    event_calibration: h5py.Group,
    target_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    if len(x) == 0:
        return x.astype(np.float32), y.astype(np.float32), np.empty(0, dtype=np.bool_)
    raw_k = _camera_matrix(event_calibration, scale=0.5)
    model, coefficients = _distortion(event_calibration)
    points = np.stack((x, y), axis=-1).astype(np.float64).reshape(-1, 1, 2)
    if model == "equidistant":
        rectified = cv2.fisheye.undistortPoints(
            points,
            raw_k,
            coefficients.reshape(-1, 1),
            R=np.eye(3),
            P=target_k,
        )
    elif model in {"radtan", "plumb_bob"}:
        rectified = cv2.undistortPoints(
            points, raw_k, coefficients, R=np.eye(3), P=target_k
        )
    else:
        raise ValueError(f"Unsupported M3ED event distortion model: {model}")
    coordinates = rectified.reshape(-1, 2)
    rx, ry = coordinates[:, 0], coordinates[:, 1]
    valid = (
        np.isfinite(rx)
        & np.isfinite(ry)
        & (rx >= 0)
        & (rx < 640)
        & (ry >= 0)
        & (ry < 360)
    )
    return rx[valid].astype(np.float32), ry[valid].astype(np.float32), valid


def _downsample_range(
    event_group: h5py.Group,
    start: int,
    end: int,
    downsampler: DAGRDownsampler,
    chunk_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output: list[list[np.ndarray]] = [[], [], [], []]
    for chunk_start in range(start, end, chunk_events):
        chunk_end = min(end, chunk_start + chunk_events)
        x = np.asarray(event_group["x"][chunk_start:chunk_end])
        y = np.asarray(event_group["y"][chunk_start:chunk_end])
        p = np.asarray(event_group["p"][chunk_start:chunk_end])
        reduced_x, reduced_y, reduced_p, selected = downsampler(x, y, p)
        if len(selected):
            output[0].append(reduced_x)
            output[1].append(reduced_y)
            output[2].append(reduced_p)
            output[3].append(
                np.asarray(event_group["t"][chunk_start:chunk_end], dtype=np.int64)[selected]
            )
    dtypes = (np.uint16, np.uint16, np.int8, np.int64)
    return tuple(
        np.concatenate(values) if values else np.empty(0, dtype=dtype)
        for values, dtype in zip(output, dtypes)
    )  # type: ignore[return-value]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(array).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _representation(args: argparse.Namespace) -> GEPEventFrame | EventVoxelizer:
    if args.representation == "gep_rgb":
        return GEPEventFrame(height=360, width=640, percentile=args.percentile)
    return EventVoxelizer(
        num_bins=args.event_bins,
        height=360,
        width=640,
        polarity_split=True,
        normalization=args.voxel_normalization,
    )


def prepare_recording(
    source_path: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install with the prepare extra")
    sequence_name = _sequence_name(source_path)
    sequence_root = output_root / sequence_name
    image_root = sequence_root / "aligned_rgb"
    event_root = sequence_root / "events"
    image_root.mkdir(parents=True, exist_ok=True)
    event_root.mkdir(parents=True, exist_ok=True)
    representation = _representation(args)
    existing_metadata_path = sequence_root / "metadata.json"
    if existing_metadata_path.exists() and not args.overwrite:
        try:
            existing_metadata = json.loads(
                existing_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid existing M3ED preparation metadata: {existing_metadata_path}"
            ) from error
        expected_existing = {
            "dataset": "M3ED",
            "sequence_name": sequence_name,
            "source_size_bytes": source_path.stat().st_size,
            "source_mtime_ns": source_path.stat().st_mtime_ns,
            "event_size": [360, 640],
            "downsampling": "dagr_stateful_signed_event_downsampling",
        }
        for field, value in expected_existing.items():
            if existing_metadata.get(field) != value:
                raise RuntimeError(
                    f"Existing M3ED preparation differs at {field}; use --overwrite: "
                    f"{existing_metadata_path}"
                )
        if existing_metadata.get("representation", {}).get("type") != args.representation:
            raise RuntimeError(
                f"Existing M3ED representation differs; use --overwrite: "
                f"{existing_metadata_path}"
            )
    statistics_sum = np.zeros(representation.channels, dtype=np.float64)
    statistics_square_sum = np.zeros(representation.channels, dtype=np.float64)
    statistics_pixels = 0
    total_source_events = 0
    total_dagr_events = 0
    total_rectified_events = 0

    with h5py.File(source_path, "r") as source:
        required = (
            "/prophesee/left/x",
            "/prophesee/left/y",
            "/prophesee/left/t",
            "/prophesee/left/p",
            "/prophesee/left/calib",
            "/ovc/rgb/data",
            "/ovc/rgb/calib",
            "/ovc/ts",
            "/ovc/ts_map_prophesee_left_t",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{source_path} lacks M3ED datasets: {missing}")
        event_group = source["/prophesee/left"]
        rgb_group = source["/ovc/rgb"]
        event_resolution = tuple(
            int(value) for value in np.asarray(event_group["calib/resolution"])
        )
        if event_resolution != (1280, 720):
            raise ValueError(
                f"DAGR half-scale preparation requires 1280x720 M3ED events, got "
                f"{event_resolution}: {source_path}"
            )
        timestamps = np.asarray(source["/ovc/ts"], dtype=np.int64)
        event_indices = np.asarray(
            source["/ovc/ts_map_prophesee_left_t"], dtype=np.int64
        )
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError(f"M3ED recording has fewer than two RGB timestamps: {source_path}")
        if len(timestamps) != len(rgb_group["data"]) or len(timestamps) != len(event_indices):
            raise ValueError(f"M3ED RGB/timestamp/event-map lengths differ: {source_path}")
        if np.any(np.diff(timestamps) <= 0) or np.any(np.diff(event_indices) < 0):
            raise ValueError(f"M3ED timestamps or mapped event indices are unordered: {source_path}")
        event_total = len(event_group["t"])
        if event_indices[0] < 0 or event_indices[-1] > event_total:
            raise ValueError(f"M3ED mapped event index is out of bounds: {source_path}")

        map_x, map_y, target_k = _rgb_to_event_map(
            event_group["calib"], rgb_group["calib"]
        )
        downsampler = DAGRDownsampler()
        # DAGR's residual state belongs to the complete stream, not an RGB
        # interval. Consume the prefix even though it has no teacher frame.
        _downsample_range(
            event_group, 0, int(event_indices[0]), downsampler, args.chunk_events
        )

        progress = tqdm(range(len(timestamps)), desc=sequence_name, unit="frame")
        for frame_index in progress:
            timestamp = int(timestamps[frame_index])
            image_path = image_root / f"{timestamp}.png"
            if args.overwrite or not image_path.is_file():
                rgb = np.asarray(rgb_group["data"][frame_index])
                if rgb.ndim == 3 and rgb.shape[-1] == 1:
                    rgb = np.repeat(rgb, 3, axis=-1)
                if rgb.ndim != 3 or rgb.shape[-1] != 3:
                    raise ValueError(f"Unexpected M3ED RGB shape {rgb.shape}: {source_path}")
                aligned = cv2.remap(
                    rgb,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                _atomic_png(image_path, aligned.astype(np.uint8))
            if frame_index == 0:
                continue

            start = int(event_indices[frame_index - 1])
            end = int(event_indices[frame_index])
            x, y, p, t = _downsample_range(
                event_group, start, end, downsampler, args.chunk_events
            )
            total_source_events += end - start
            total_dagr_events += len(t)
            rx, ry, valid = _rectify_half_scale_events(
                x, y, event_group["calib"], target_k
            )
            p, t = p[valid], t[valid]
            boundary = (t > int(timestamps[frame_index - 1])) & (t <= timestamp)
            rx, ry, p, t = rx[boundary], ry[boundary], p[boundary], t[boundary]
            total_rectified_events += len(t)
            tensor = representation(
                torch.from_numpy(rx),
                torch.from_numpy(ry),
                torch.from_numpy(t),
                torch.from_numpy(p),
                start_time=int(timestamps[frame_index - 1]),
                end_time=timestamp,
            ).float()
            event_path = event_root / f"{timestamp}.pt"
            if args.overwrite or not event_path.is_file():
                _atomic_torch_save(
                    event_path,
                    {
                        "format_version": M3ED_PREPARED_FORMAT_VERSION,
                        "dataset": "M3ED",
                        "sequence_name": sequence_name,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "previous_timestamp": int(timestamps[frame_index - 1]),
                        "events": tensor,
                        "event_count": len(t),
                        "source_event_count": end - start,
                        "dagr_event_count": len(valid),
                        "representation": args.representation,
                    },
                )
            # Match the deterministic 640x352 training crop exactly. Statistics
            # over all 360 rows would include pixels the model never observes.
            flat = tensor[:, 4:356].double().flatten(1)
            statistics_sum += flat.sum(dim=1).numpy()
            statistics_square_sum += (flat * flat).sum(dim=1).numpy()
            statistics_pixels += flat.shape[1]

        source_relative = (
            str(source_path.relative_to(args.root.expanduser().resolve()))
            if source_path.is_relative_to(args.root.expanduser().resolve())
            else str(source_path)
        )
        metadata = {
            "format_version": M3ED_PREPARED_FORMAT_VERSION,
            "dataset": "M3ED",
            "sequence_name": sequence_name,
            "source_h5": source_relative,
            "source_size_bytes": source_path.stat().st_size,
            "source_mtime_ns": source_path.stat().st_mtime_ns,
            "frame_count": len(timestamps),
            "timestamps": [int(value) for value in timestamps],
            "native_event_size": [720, 1280],
            "event_size": [360, 640],
            "recommended_model_input_size": [352, 640],
            "downsampling": "dagr_stateful_signed_event_downsampling",
            "downsample_factor": [2, 2],
            "operation_order": [
                "dagr_downsample_raw_event_stream",
                "rectify_half_scale_events",
                "build_event_representation",
            ],
            "coordinate_space": "rectified_left_event",
            "rgb_alignment": "rotation_only_rgb_to_left_event",
            "event_window": "previous_rgb_timestamp < event_timestamp <= rgb_timestamp",
            "representation": {
                "type": args.representation,
                "channels": representation.channels,
                "percentile": args.percentile if args.representation == "gep_rgb" else None,
                "event_bins": args.event_bins if args.representation == "voxel_grid" else None,
                "voxel_normalization": (
                    args.voxel_normalization if args.representation == "voxel_grid" else None
                ),
            },
            "counts": {
                "source_events_in_rgb_intervals": total_source_events,
                "dagr_retained_events": total_dagr_events,
                "rectified_in_bounds_events": total_rectified_events,
            },
        }
        metadata_path = sequence_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    mean = statistics_sum / max(1, statistics_pixels)
    variance = statistics_square_sum / max(1, statistics_pixels) - mean * mean
    return {
        "sequence": sequence_name,
        "pixels": statistics_pixels,
        "sum": statistics_sum.tolist(),
        "square_sum": statistics_square_sum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.chunk_events <= 0:
        raise ValueError("--chunk-events must be positive")
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    args.root = args.root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    recordings = discover_recordings(args.root, args.sequences)
    results = [prepare_recording(path, args.output_root, args) for path in recordings]
    pixels = sum(int(item["pixels"]) for item in results)
    sums = np.sum([item["sum"] for item in results], axis=0, dtype=np.float64)
    squares = np.sum([item["square_sum"] for item in results], axis=0, dtype=np.float64)
    mean = sums / max(1, pixels)
    std = np.sqrt(np.maximum(squares / max(1, pixels) - mean * mean, 0.0))
    statistics = {
        "dataset": "M3ED",
        "representation": args.representation,
        "sequences": [item["sequence"] for item in results],
        "pixel_count": pixels,
        "normalize_mean": mean.tolist(),
        "normalize_std": std.tolist(),
    }
    statistics_path = args.output_root / "event_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2))
    print(f"Prepared M3ED data: {args.output_root}")


if __name__ == "__main__":
    main()
