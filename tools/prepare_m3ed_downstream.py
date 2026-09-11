#!/usr/bin/env python3
"""Align official M3ED depth, semantics, and pose to prepared EventState frames."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import h5py
import hdf5plugin  # noqa: F401 - register compression filters used by M3ED inputs.
import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from event_state.data.m3ed import (
    M3ED_PREPARED_FORMAT_VERSION,
    discover_prepared_m3ed_sequences,
)
from event_state.data.m3ed_downstream import (
    M3ED_DOWNSTREAM_FORMAT_VERSION,
    M3ED_IGNORE_LABEL,
    camera_relative_motion,
    map_cityscapes_19_to_dsec_11,
    match_nearest_timestamps,
)


TASKS = ("depth", "semantics", "pose")
NATIVE_SIZE = (720, 1280)
EVENT_SIZE = (360, 640)
MODEL_SIZE = (352, 640)
MODEL_CROP = (4, 356, 0, 640)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add official M3ED depth, semantic pseudo-label, and pose targets to an "
            "existing EventState M3ED preparation without changing event statistics"
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Official M3ED root")
    parser.add_argument(
        "--prepared-root", type=Path, required=True, help="Existing prepare_m3ed.py output"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Prepared sequence names; defaults to all sequences in prepared-root",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-depth-delta-us", type=int, default=50_000)
    parser.add_argument("--max-pose-delta-us", type=int, default=50_000)
    parser.add_argument("--max-semantics-delta-us", type=int, default=0)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=200.0)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail instead of skipping a requested target file that is not distributed",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _decode(value: Any) -> str:
    if isinstance(value, h5py.Dataset):
        value = value[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _camera_matrix(calibration: h5py.Group, scale: float) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(calibration["intrinsics"], dtype=np.float64)
    return np.asarray(
        [[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0, 0, 1]],
        dtype=np.float64,
    )


def _distortion(calibration: h5py.Group) -> tuple[str, np.ndarray]:
    return (
        _decode(calibration["distortion_model"]).lower(),
        np.asarray(calibration["distortion_coeffs"], dtype=np.float64),
    )


def _rectified_event_remap(calibration: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    source_k = _camera_matrix(calibration, 1.0)
    target_k = _camera_matrix(calibration, 0.5)
    model, coefficients = _distortion(calibration)
    if model == "equidistant":
        return cv2.fisheye.initUndistortRectifyMap(
            source_k,
            coefficients.reshape(-1, 1),
            np.eye(3),
            target_k,
            (EVENT_SIZE[1], EVENT_SIZE[0]),
            cv2.CV_32FC1,
        )
    if model in {"radtan", "plumb_bob"}:
        return cv2.initUndistortRectifyMap(
            source_k,
            coefficients,
            np.eye(3),
            target_k,
            (EVENT_SIZE[1], EVENT_SIZE[0]),
            cv2.CV_32FC1,
        )
    raise ValueError(f"Unsupported M3ED event distortion model: {model}")


def _rectify_sparse_depth(
    depth: np.ndarray,
    calibration: h5py.Group,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat sparse depth so half scaling does not discard three quarters of it."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    source = np.asarray(depth, dtype=np.float32).squeeze()
    if source.shape != NATIVE_SIZE:
        raise ValueError(f"Expected M3ED depth shape {NATIVE_SIZE}, got {source.shape}")
    valid = np.isfinite(source) & (source >= min_depth_m) & (source <= max_depth_m)
    y, x = np.nonzero(valid)
    output = np.full(EVENT_SIZE[0] * EVENT_SIZE[1], np.inf, dtype=np.float32)
    if len(x):
        points = np.stack((x, y), axis=-1).astype(np.float64).reshape(-1, 1, 2)
        source_k = _camera_matrix(calibration, 1.0)
        target_k = _camera_matrix(calibration, 0.5)
        model, coefficients = _distortion(calibration)
        if model == "equidistant":
            rectified = cv2.fisheye.undistortPoints(
                points,
                source_k,
                coefficients.reshape(-1, 1),
                R=np.eye(3),
                P=target_k,
            )
        elif model in {"radtan", "plumb_bob"}:
            rectified = cv2.undistortPoints(
                points, source_k, coefficients, R=np.eye(3), P=target_k
            )
        else:
            raise ValueError(f"Unsupported M3ED event distortion model: {model}")
        coordinates = rectified.reshape(-1, 2)
        finite = np.isfinite(coordinates).all(axis=1)
        rx = np.zeros(len(coordinates), dtype=np.int64)
        ry = np.zeros(len(coordinates), dtype=np.int64)
        rx[finite] = np.rint(coordinates[finite, 0]).astype(np.int64)
        ry[finite] = np.rint(coordinates[finite, 1]).astype(np.int64)
        in_bounds = (
            finite
            & (rx >= 0)
            & (rx < EVENT_SIZE[1])
            & (ry >= 0)
            & (ry < EVENT_SIZE[0])
        )
        flat_indices = ry[in_bounds] * EVENT_SIZE[1] + rx[in_bounds]
        # If multiple LiDAR samples land on one pixel, keep the visible surface.
        np.minimum.at(output, flat_indices, source[y[in_bounds], x[in_bounds]])
    output = output.reshape(EVENT_SIZE)
    valid_output = np.isfinite(output)
    output[~valid_output] = np.nan
    top, bottom, left, right = MODEL_CROP
    return output[top:bottom, left:right], valid_output[top:bottom, left:right]


def _rectify_semantics(
    labels: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    source = np.asarray(labels).squeeze()
    if source.shape != NATIVE_SIZE:
        raise ValueError(f"Expected M3ED semantics shape {NATIVE_SIZE}, got {source.shape}")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError("M3ED semantic predictions must have an integer dtype")
    valid_labels = (source == M3ED_IGNORE_LABEL) | ((source >= 0) & (source <= 18))
    if not np.all(valid_labels):
        unexpected = np.unique(source[~valid_labels])
        raise ValueError(
            f"M3ED semantic labels outside Cityscapes 0..18/255: {unexpected}"
        )
    rectified = cv2.remap(
        source.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=M3ED_IGNORE_LABEL,
    )
    top, bottom, left, right = MODEL_CROP
    return rectified[top:bottom, left:right]


def _source_identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _require_dataset(source: h5py.File, key: str, path: Path) -> h5py.Dataset:
    if key not in source:
        available: list[str] = []
        source.visititems(
            lambda name, value: available.append(name) if isinstance(value, h5py.Dataset) else None
        )
        raise KeyError(f"{path} lacks {key}; datasets={available}")
    value = source[key]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{key} is not a dataset in {path}")
    return value


def _create_spatial_dataset(
    target: h5py.File,
    name: str,
    frame_count: int,
    *,
    dtype: Any,
    fillvalue: Any,
) -> h5py.Dataset:
    return target.create_dataset(
        name,
        shape=(frame_count, *MODEL_SIZE),
        dtype=dtype,
        chunks=(1, *MODEL_SIZE),
        compression="lzf",
        fillvalue=fillvalue,
    )


def _available_sources(root: Path, sequence_name: str) -> dict[str, Path]:
    sequence_root = root / sequence_name
    return {
        "data": sequence_root / f"{sequence_name}_data.h5",
        "depth": sequence_root / f"{sequence_name}_depth_gt.h5",
        "semantics": sequence_root / f"{sequence_name}_semantics.h5",
        "pose": sequence_root / f"{sequence_name}_pose_gt.h5",
    }


def _validate_prepared_metadata(path: Path, sequence_name: str) -> dict[str, Any]:
    metadata = _read_json(path)
    expected = {
        "format_version": M3ED_PREPARED_FORMAT_VERSION,
        "dataset": "M3ED",
        "sequence_name": sequence_name,
        "event_size": list(EVENT_SIZE),
        "recommended_model_input_size": list(MODEL_SIZE),
        "coordinate_space": "rectified_left_event",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"Prepared M3ED {field}={metadata.get(field)!r} does not match "
                f"{value!r}: {path}"
            )
    timestamps = np.asarray(metadata.get("timestamps", []), dtype=np.int64)
    if len(timestamps) != int(metadata.get("frame_count", -1)) or len(timestamps) < 2:
        raise ValueError(f"Invalid prepared timestamps: {path}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Prepared timestamps are not strictly increasing: {path}")
    return metadata


def _reusable(
    metadata_path: Path,
    success_path: Path,
    targets_path: Path,
    *,
    sequence_name: str,
    tasks: list[str],
    timestamps: np.ndarray,
    sources: dict[str, Path],
    root: Path,
    generation: dict[str, Any],
) -> bool:
    if not metadata_path.is_file() or not success_path.is_file() or not targets_path.is_file():
        return False
    metadata = _read_json(metadata_path)
    expected = {
        "format_version": M3ED_DOWNSTREAM_FORMAT_VERSION,
        "dataset": "M3ED",
        "sequence_name": sequence_name,
        "frame_count": len(timestamps),
        "timestamps": timestamps.tolist(),
        "tasks": tasks,
        "generation": generation,
    }
    if any(metadata.get(field) != value for field, value in expected.items()):
        return False
    expected_sources = {
        name: _source_identity(path, root)
        for name, path in sources.items()
        if name == "data" or name in tasks
    }
    return metadata.get("sources") == expected_sources


def prepare_sequence(
    sequence_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prepared_sequence = args.prepared_root / sequence_name
    prepared_metadata_path = prepared_sequence / "metadata.json"
    prepared = _validate_prepared_metadata(prepared_metadata_path, sequence_name)
    timestamps = np.asarray(prepared["timestamps"], dtype=np.int64)
    frame_count = len(timestamps)
    source_paths = _available_sources(args.root, sequence_name)
    if not source_paths["data"].is_file():
        raise FileNotFoundError(f"M3ED data file missing: {source_paths['data']}")

    requested = list(dict.fromkeys(args.tasks))
    available = [task for task in requested if source_paths[task].is_file()]
    missing = [task for task in requested if task not in available]
    if missing and args.require_all:
        raise FileNotFoundError(
            f"M3ED targets missing for {sequence_name}: {', '.join(missing)}"
        )
    if not available:
        return {"sequence": sequence_name, "tasks": [], "missing": missing, "skipped": True}

    output_sequence = args.output_root / sequence_name
    output_sequence.mkdir(parents=True, exist_ok=True)
    targets_path = output_sequence / "targets.h5"
    metadata_path = output_sequence / "metadata.json"
    success_path = output_sequence / "_SUCCESS"
    relevant_sources = {
        "data": source_paths["data"],
        **{task: source_paths[task] for task in available},
    }
    generation = {
        "max_depth_delta_us": args.max_depth_delta_us,
        "max_pose_delta_us": args.max_pose_delta_us,
        "max_semantics_delta_us": args.max_semantics_delta_us,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
    }
    if not args.overwrite and _reusable(
        metadata_path,
        success_path,
        targets_path,
        sequence_name=sequence_name,
        tasks=available,
        timestamps=timestamps,
        sources=relevant_sources,
        root=args.root,
        generation=generation,
    ):
        return {"sequence": sequence_name, "tasks": available, "missing": missing, "reused": True}

    # Never leave a completion marker referring to a superseded payload.
    success_path.unlink(missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".targets.", suffix=".h5", dir=output_sequence
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    counts: dict[str, int] = {}
    task_metadata: dict[str, Any] = {}
    try:
        with h5py.File(source_paths["data"], "r") as data_source:
            calibration = data_source["/prophesee/left/calib"]
            if not isinstance(calibration, h5py.Group):
                raise TypeError("/prophesee/left/calib must be a group")
            map_x, map_y = _rectified_event_remap(calibration)
            with h5py.File(temporary_path, "w") as target:
                target.attrs["format_version"] = M3ED_DOWNSTREAM_FORMAT_VERSION
                target.attrs["dataset"] = "M3ED"
                target.attrs["sequence_name"] = sequence_name
                target.create_dataset("timestamps", data=timestamps)
                target.create_dataset("frame_indices", data=np.arange(frame_count, dtype=np.int64))

                if "depth" in available:
                    with h5py.File(source_paths["depth"], "r") as source:
                        depth_source = _require_dataset(
                            source, "depth/prophesee/left", source_paths["depth"]
                        )
                        source_ts = np.asarray(
                            _require_dataset(source, "ts", source_paths["depth"]),
                            dtype=np.int64,
                        )
                        if len(depth_source) != len(source_ts):
                            raise ValueError("M3ED depth and timestamp lengths differ")
                        matches = match_nearest_timestamps(
                            source_ts,
                            timestamps,
                            max_delta_us=args.max_depth_delta_us,
                        )
                        depth_out = _create_spatial_dataset(
                            target, "depth_m", frame_count, dtype=np.float32, fillvalue=np.nan
                        )
                        valid_out = _create_spatial_dataset(
                            target, "depth_valid", frame_count, dtype=np.bool_, fillvalue=False
                        )
                        valid_pixel_count = 0
                        depth_frame_valid = np.zeros(frame_count, dtype=np.bool_)
                        for frame_index in tqdm(
                            np.flatnonzero(matches.valid),
                            desc=f"{sequence_name} depth",
                            unit="frame",
                        ):
                            depth, valid = _rectify_sparse_depth(
                                depth_source[int(matches.indices[frame_index])],
                                calibration,
                                min_depth_m=args.min_depth_m,
                                max_depth_m=args.max_depth_m,
                            )
                            depth_out[frame_index] = depth
                            valid_out[frame_index] = valid
                            valid_pixel_count += int(valid.sum())
                            depth_frame_valid[frame_index] = bool(valid.any())
                        target.create_dataset("depth_source_indices", data=matches.indices)
                        target.create_dataset("depth_time_delta_us", data=matches.deltas_us)
                        target.create_dataset("depth_frame_valid", data=depth_frame_valid)
                        counts["depth_frames"] = int(depth_frame_valid.sum())
                        counts["depth_valid_pixels"] = valid_pixel_count
                        task_metadata["depth"] = {
                            "source_dataset": "depth/prophesee/left",
                            "unit": "meter",
                            "valid_range_m": [args.min_depth_m, args.max_depth_m],
                            "max_time_delta_us": args.max_depth_delta_us,
                            "resampling": "forward_rectification_with_nearest-pixel_z-buffer",
                        }

                if "semantics" in available:
                    with h5py.File(source_paths["semantics"], "r") as source:
                        predictions = _require_dataset(
                            source, "predictions", source_paths["semantics"]
                        )
                        source_ts = np.asarray(
                            _require_dataset(source, "ts", source_paths["semantics"]),
                            dtype=np.int64,
                        )
                        if len(predictions) != len(source_ts):
                            raise ValueError(
                                "M3ED semantic prediction and timestamp lengths differ"
                            )
                        matches = match_nearest_timestamps(
                            source_ts,
                            timestamps,
                            max_delta_us=args.max_semantics_delta_us,
                        )
                        labels_19 = _create_spatial_dataset(
                            target,
                            "semantics_19",
                            frame_count,
                            dtype=np.uint8,
                            fillvalue=M3ED_IGNORE_LABEL,
                        )
                        labels_11 = _create_spatial_dataset(
                            target,
                            "semantics_11",
                            frame_count,
                            dtype=np.uint8,
                            fillvalue=M3ED_IGNORE_LABEL,
                        )
                        for frame_index in tqdm(
                            np.flatnonzero(matches.valid),
                            desc=f"{sequence_name} semantics",
                            unit="frame",
                        ):
                            rectified = _rectify_semantics(
                                predictions[int(matches.indices[frame_index])], map_x, map_y
                            )
                            labels_19[frame_index] = rectified
                            labels_11[frame_index] = map_cityscapes_19_to_dsec_11(rectified)
                        target.create_dataset("semantics_source_indices", data=matches.indices)
                        target.create_dataset("semantics_time_delta_us", data=matches.deltas_us)
                        target.create_dataset("semantics_frame_valid", data=matches.valid)
                        counts["semantics_frames"] = int(matches.valid.sum())
                        task_metadata["semantics"] = {
                            "source_dataset": "predictions",
                            "source": "M3ED InternImage pseudo-labels",
                            "protocols": {
                                "semantics_19": "Cityscapes-19",
                                "semantics_11": "DSEC-11",
                            },
                            "ignore_label": M3ED_IGNORE_LABEL,
                            "max_time_delta_us": args.max_semantics_delta_us,
                            "resampling": "rectification_map_nearest_neighbor",
                        }

                if "pose" in available:
                    with h5py.File(source_paths["pose"], "r") as source:
                        poses_source = _require_dataset(source, "Cn_T_C0", source_paths["pose"])
                        source_ts = np.asarray(
                            _require_dataset(source, "ts", source_paths["pose"]), dtype=np.int64
                        )
                        if len(poses_source) != len(source_ts):
                            raise ValueError("M3ED pose and timestamp lengths differ")
                        source_pose_array = np.asarray(poses_source, dtype=np.float64)
                        if source_pose_array.shape != (len(source_ts), 4, 4):
                            raise ValueError(
                                "M3ED Cn_T_C0 must have shape [N,4,4], got "
                                f"{source_pose_array.shape}"
                            )
                        matches = match_nearest_timestamps(
                            source_ts, timestamps, max_delta_us=args.max_pose_delta_us
                        )
                        pose_match_valid = matches.valid.copy()
                        matched_rows = np.flatnonzero(pose_match_valid)
                        if len(matched_rows):
                            matched_source = matches.indices[matched_rows]
                            pose_match_valid[matched_rows] &= np.isfinite(
                                source_pose_array[matched_source]
                            ).all(axis=(1, 2))
                        poses = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
                        poses[pose_match_valid] = source_pose_array[
                            matches.indices[pose_match_valid]
                        ]
                        relative = np.full_like(poses, np.nan)
                        linear = np.full((frame_count, 3), np.nan, dtype=np.float64)
                        angular = np.full((frame_count, 3), np.nan, dtype=np.float64)
                        delta_us = np.full(frame_count, -1, dtype=np.int64)
                        relative_valid = np.zeros(frame_count, dtype=np.bool_)
                        for frame_index in range(1, frame_count):
                            if not (
                                pose_match_valid[frame_index - 1]
                                and pose_match_valid[frame_index]
                            ):
                                continue
                            previous_index = int(matches.indices[frame_index - 1])
                            current_index = int(matches.indices[frame_index])
                            source_delta = int(source_ts[current_index] - source_ts[previous_index])
                            if source_delta <= 0:
                                continue
                            rel, velocity, omega = camera_relative_motion(
                                poses[frame_index - 1],
                                poses[frame_index],
                                delta_seconds=source_delta / 1_000_000.0,
                            )
                            relative[frame_index] = rel
                            linear[frame_index] = velocity
                            angular[frame_index] = omega
                            delta_us[frame_index] = source_delta
                            relative_valid[frame_index] = True
                        target.create_dataset("pose_Cn_T_C0", data=poses)
                        target.create_dataset("pose_frame_valid", data=pose_match_valid)
                        target.create_dataset("pose_source_indices", data=matches.indices)
                        target.create_dataset("pose_time_delta_us", data=matches.deltas_us)
                        target.create_dataset("relative_pose_prev_T_current", data=relative)
                        target.create_dataset("relative_pose_valid", data=relative_valid)
                        target.create_dataset("relative_pose_delta_us", data=delta_us)
                        target.create_dataset("linear_velocity_mps", data=linear)
                        target.create_dataset("angular_velocity_radps", data=angular)
                        counts["pose_frames"] = int(pose_match_valid.sum())
                        counts["relative_pose_frames"] = int(relative_valid.sum())
                        task_metadata["pose"] = {
                            "source_dataset": "Cn_T_C0",
                            "max_time_delta_us": args.max_pose_delta_us,
                            "matching": "nearest_timestamp",
                            "relative_convention": "previous_Cn_T_C0 @ inv(current_Cn_T_C0)",
                        }
                target.flush()
        os.replace(temporary_path, targets_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    sources_metadata = {
        name: _source_identity(path, args.root) for name, path in relevant_sources.items()
    }
    metadata = {
        "format_version": M3ED_DOWNSTREAM_FORMAT_VERSION,
        "dataset": "M3ED",
        "sequence_name": sequence_name,
        "prepared_format_version": M3ED_PREPARED_FORMAT_VERSION,
        "prepared_metadata": str(prepared_metadata_path),
        "frame_count": frame_count,
        "timestamps": timestamps.tolist(),
        "tasks": available,
        "generation": generation,
        "missing_requested_tasks": missing,
        "coordinate_space": "rectified_left_event",
        "native_size": list(NATIVE_SIZE),
        "event_size": list(EVENT_SIZE),
        "target_size": list(MODEL_SIZE),
        "crop_top_bottom_left_right": list(MODEL_CROP),
        "sources": sources_metadata,
        "task_metadata": task_metadata,
        "counts": counts,
    }
    _atomic_json(metadata_path, metadata)
    success_path.write_text("complete\n", encoding="utf-8")
    return {"sequence": sequence_name, "tasks": available, "missing": missing, "counts": counts}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install the prepare extra")
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("Depth range must satisfy 0 < min-depth-m < max-depth-m")
    for name in ("max_depth_delta_us", "max_pose_delta_us", "max_semantics_delta_us"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    args.root = args.root.expanduser().resolve()
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sequence_names = discover_prepared_m3ed_sequences(args.prepared_root, args.sequences)
    results = [prepare_sequence(sequence_name, args) for sequence_name in sequence_names]
    summary = {
        "dataset": "M3ED",
        "format_version": M3ED_DOWNSTREAM_FORMAT_VERSION,
        "root": str(args.root),
        "prepared_root": str(args.prepared_root),
        "output_root": str(args.output_root),
        "requested_tasks": list(dict.fromkeys(args.tasks)),
        "sequences": results,
    }
    _atomic_json(args.output_root / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
