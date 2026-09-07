#!/usr/bin/env python3
"""Prepare spatially aligned DSEC RGB frames and optional event caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_state.data.calibration import load_dsec_camera_alignment  # noqa: E402
from event_state.data.cache_metadata import (  # noqa: E402
    ALIGNMENT_FORMAT_VERSION,
    EVENT_CACHE_FORMAT_VERSION,
    SUCCESS_MARKER_NAME,
    build_input_fingerprint,
    canonical_json_sha256,
    frame_manifest_sha256,
    reject_untracked_outputs,
    relative_file_id,
    success_marker_payload,
)
from event_state.data.dsec import (  # noqa: E402
    DSECEventReader,
    discover_dsec_sequences,
    event_representation_metadata,
    event_window_contract,
    event_window_start_timestamp,
    validate_event_cache_payload,
)
from event_state.data.event_representation import EventVoxelizer, GEPEventFrame  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warp DSEC RGB frames into the rectified event-camera view and optionally "
            "cache event representations."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="DSEC dataset root")
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="train",
        help="Dataset split to prepare; all processes train followed by test",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit sequence names; defaults to every sequence in the split",
    )
    parser.add_argument(
        "--image-output-subdir",
        default="aligned_event",
        help="Output directory under <split>_images/<sequence>/images/left",
    )
    parser.add_argument(
        "--event-cache-dir",
        type=Path,
        default=None,
        help="Optional cache root; tensors are saved as <sequence>/<timestamp>.pt",
    )
    parser.add_argument(
        "--representation",
        choices=("gep_rgb", "voxel_grid"),
        default="gep_rgb",
        help="Event representation to cache (default: GEP-compatible 3-channel RGB)",
    )
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument(
        "--event-window-fraction",
        type=float,
        default=1.0,
        help=(
            "Causal tail fraction of each RGB interval used for events; "
            "for example 0.25 keeps only the final quarter (default: 1.0)"
        ),
    )
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument(
        "--voxel-normalization",
        choices=("none", "nonzero_standardize", "log1p"),
        default="nonzero_standardize",
    )
    parser.add_argument(
        "--rectify-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the official per-event rectify map before voxelization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated files; existing files are otherwise preserved",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    root = args.root.expanduser().resolve()
    cache_root = args.event_cache_dir.expanduser().resolve() if args.event_cache_dir else None
    totals = {"images_written": 0, "images_skipped": 0, "events_written": 0, "events_skipped": 0}
    splits = ("train", "test") if args.split == "all" else (args.split,)
    for split in splits:
        sequences = discover_dsec_sequences(root, split, args.sequences)
        print(f"Preparing DSEC {split}: {len(sequences)} sequences")
        for sequence_name in sequences:
            result = prepare_sequence(
                root=root,
                split=split,
                sequence_name=sequence_name,
                image_output_subdir=args.image_output_subdir,
                event_cache_dir=cache_root,
                representation_name=args.representation,
                percentile=args.percentile,
                event_bins=args.event_bins,
                voxel_normalization=args.voxel_normalization,
                event_window_fraction=args.event_window_fraction,
                rectify_events=args.rectify_events,
                overwrite=args.overwrite,
            )
            for key, value in result.items():
                totals[key] += value

    print(
        "DSEC preparation complete: "
        f"images={totals['images_written']} written/{totals['images_skipped']} existing, "
        f"events={totals['events_written']} written/{totals['events_skipped']} existing"
    )


def prepare_sequence(
    *,
    root: Path,
    split: str,
    sequence_name: str,
    image_output_subdir: str,
    event_cache_dir: Path | None,
    representation_name: str,
    percentile: float,
    event_bins: int,
    voxel_normalization: str,
    event_window_fraction: float,
    rectify_events: bool,
    overwrite: bool,
) -> dict[str, int]:
    validate_preparation_options(
        image_output_subdir=image_output_subdir,
        representation_name=representation_name,
        percentile=percentile,
        event_bins=event_bins,
        voxel_normalization=voxel_normalization,
        event_window_fraction=event_window_fraction,
    )
    require_opencv()
    image_sequence_root = root / f"{split}_images" / sequence_name / "images"
    image_dir = image_sequence_root / "left" / "rectified"
    timestamp_path = image_sequence_root / "timestamps.txt"
    calibration_path = (
        root / f"{split}_calibration" / sequence_name / "calibration" / "cam_to_cam.yaml"
    )
    event_dir = root / f"{split}_events" / sequence_name / "events" / "left"

    frame_pairs = load_frame_pairs(image_dir, timestamp_path)
    alignment = load_dsec_camera_alignment(calibration_path)
    output_dir = image_sequence_root / "left" / image_output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_frame_manifest = frame_manifest_sha256(
        (
            frame_index,
            frame_pairs[frame_index - 1][1] if frame_index else timestamp,
            timestamp,
        )
        for frame_index, (_, timestamp) in enumerate(frame_pairs)
    )
    alignment_input_fingerprint = build_input_fingerprint(
        root,
        files={
            "rgb_timestamps": timestamp_path,
            "camera_calibration": calibration_path,
        },
        file_sets={"rectified_rgb_frames": [path for path, _ in frame_pairs]},
    )
    alignment_metadata = {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": split,
        "sequence_name": sequence_name,
        "source_image_directory": relative_file_id(root, image_dir),
        "frame_count": len(frame_pairs),
        "timestamp_manifest_sha256": alignment_frame_manifest,
        "input_fingerprint": alignment_input_fingerprint,
        "interpolation": "opencv_inter_linear",
        "border_mode": "constant_zero",
        "calibration": alignment.metadata(),
    }
    alignment_marker = output_dir / SUCCESS_MARKER_NAME
    if overwrite:
        alignment_marker.unlink(missing_ok=True)
    alignment_metadata_path = output_dir / "metadata.json"
    alignment_output_paths = [
        output_dir / f"{timestamp}.png" for _, timestamp in frame_pairs
    ]
    reject_untracked_outputs(
        alignment_metadata_path,
        alignment_output_paths,
        overwrite=overwrite,
        description=f"Aligned RGB cache for {sequence_name}",
    )
    validate_existing_metadata_base(
        alignment_metadata_path,
        alignment_metadata,
        ignored_fields={"output_fingerprint"},
        overwrite=overwrite,
    )

    counts = {"images_written": 0, "images_skipped": 0, "events_written": 0, "events_skipped": 0}
    for source_path, timestamp in tqdm(frame_pairs, desc=f"{sequence_name}: RGB", unit="frame"):
        output_path = output_dir / f"{timestamp}.png"
        if output_path.exists() and not overwrite:
            existing_image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            if existing_image is None or existing_image.shape[:2] != (
                alignment.height,
                alignment.width,
            ):
                raise RuntimeError(
                    f"Existing aligned RGB frame is invalid; use --overwrite: {output_path}"
                )
            counts["images_skipped"] += 1
            continue
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to read DSEC RGB frame: {source_path}")
        warped = cv2.warpPerspective(
            image,
            alignment.image_to_event_homography,
            (alignment.width, alignment.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        atomic_imwrite(output_path, warped, overwrite=overwrite)
        counts["images_written"] += 1
    alignment_metadata["output_fingerprint"] = build_input_fingerprint(
        root,
        files={},
        file_sets={"aligned_rgb_frames": alignment_output_paths},
    )
    ensure_json_metadata(
        alignment_metadata_path,
        alignment_metadata,
        overwrite=overwrite,
    )
    ensure_json_metadata(
        alignment_marker,
        success_marker_payload(alignment_metadata),
        overwrite=overwrite,
    )

    if event_cache_dir is None:
        return counts

    if representation_name == "gep_rgb":
        representation = GEPEventFrame(
            height=alignment.height,
            width=alignment.width,
            percentile=percentile,
        )
    else:
        representation = EventVoxelizer(
            num_bins=event_bins,
            height=alignment.height,
            width=alignment.width,
            polarity_split=True,
            normalization=voxel_normalization,
        )
    representation_metadata = event_representation_metadata(representation)

    sequence_cache_dir = event_cache_dir / sequence_name
    sequence_cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_files = {
        "rgb_timestamps": timestamp_path,
        "events": event_dir / "events.h5",
        "camera_calibration": calibration_path,
    }
    if rectify_events:
        fingerprint_files["rectify_map"] = event_dir / "rectify_map.h5"
    input_fingerprint = build_input_fingerprint(root, files=fingerprint_files)
    event_frames = [
        (frame_index, frame_pairs[frame_index - 1][1], timestamp)
        for frame_index, (_, timestamp) in enumerate(frame_pairs)
        if frame_index > 0
    ]
    window_contract = event_window_contract(event_window_fraction)
    cache_metadata = {
        "format_version": EVENT_CACHE_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": split,
        "sequence_name": sequence_name,
        "coordinate_space": "rectified_event" if rectify_events else "raw_event",
        "spatial_quantization": (
            "nearest_integer_numpy_rint" if rectify_events else "raw_integer_coordinates"
        ),
        "height": alignment.height,
        "width": alignment.width,
        "first_frame_cached": False,
        "frame_count": len(event_frames),
        "timestamp_manifest_sha256": frame_manifest_sha256(event_frames),
        "input_fingerprint": input_fingerprint,
        "representation": representation_metadata,
    }
    cache_metadata.update(window_contract)
    event_marker = sequence_cache_dir / SUCCESS_MARKER_NAME
    if overwrite:
        event_marker.unlink(missing_ok=True)
    event_metadata_path = sequence_cache_dir / "metadata.json"
    event_cache_paths = [
        sequence_cache_dir / f"{timestamp}.pt" for _, _, timestamp in event_frames
    ]
    reject_untracked_outputs(
        event_metadata_path,
        event_cache_paths,
        overwrite=overwrite,
        description=f"Event cache for {sequence_name}",
    )
    ensure_json_metadata(
        event_metadata_path,
        cache_metadata,
        overwrite=overwrite,
    )

    reader = DSECEventReader(
        event_dir / "events.h5",
        rectify_map_path=(event_dir / "rectify_map.h5") if rectify_events else None,
        output_height=alignment.height,
        output_width=alignment.width,
    )
    try:
        iterator = enumerate(zip(frame_pairs[:-1], frame_pairs[1:]), start=1)
        for frame_index, ((__, previous_timestamp), (_, timestamp)) in tqdm(
            iterator,
            total=max(0, len(frame_pairs) - 1),
            desc=f"{sequence_name}: events",
            unit="frame",
        ):
            output_path = sequence_cache_dir / f"{timestamp}.pt"
            if output_path.exists() and not overwrite:
                validate_event_cache_payload(
                    safe_torch_load(output_path),
                    cache_path=output_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    previous_timestamp=previous_timestamp,
                    sequence_name=sequence_name,
                    split=split,
                    height=alignment.height,
                    width=alignment.width,
                    representation=representation_metadata,
                    rectified=rectify_events,
                    input_fingerprint_digest=input_fingerprint["digest"],
                    manifest_digest=canonical_json_sha256(cache_metadata),
                    event_window_fraction=event_window_fraction,
                )
                counts["events_skipped"] += 1
                continue
            window_start = event_window_start_timestamp(
                previous_timestamp,
                timestamp,
                event_window_fraction,
            )
            selected_event_count = 0
            source_event_count = 0
            if event_window_fraction != 1.0:
                selected_event_count = reader.count(window_start, timestamp)
                source_event_count = reader.count(previous_timestamp, timestamp)
            events = reader.slice(window_start, timestamp)
            tensor = representation(
                *(torch.from_numpy(events[key]) for key in ("x", "y", "t", "p")),
                start_time=window_start,
                end_time=timestamp,
            )
            payload = {
                "format_version": EVENT_CACHE_FORMAT_VERSION,
                "dataset": "DSEC",
                "events": tensor.cpu().contiguous(),
                "frame_index": frame_index,
                "timestamp": int(timestamp),
                "previous_timestamp": int(previous_timestamp),
                "event_count": int(len(events["t"])),
                "sequence_name": sequence_name,
                "split": split,
                "height": alignment.height,
                "width": alignment.width,
                "representation": representation_metadata,
                "event_window_boundary": window_contract["event_window_boundary"],
                "rectified": rectify_events,
                "input_fingerprint_digest": input_fingerprint["digest"],
                "manifest_digest": canonical_json_sha256(cache_metadata),
            }
            if event_window_fraction != 1.0:
                payload.update(
                    {
                        "event_window": window_contract["event_window"],
                        "event_window_fraction": float(event_window_fraction),
                        "window_start_timestamp": int(window_start),
                        "selected_event_count_before_rectification": int(
                            selected_event_count
                        ),
                        "source_event_count": int(source_event_count),
                    }
                )
            atomic_torch_save(output_path, payload, overwrite=overwrite)
            counts["events_written"] += 1
    finally:
        reader.close()
    ensure_json_metadata(
        event_marker,
        success_marker_payload(
            cache_metadata,
            output_fingerprint=build_input_fingerprint(
                sequence_cache_dir,
                files={},
                file_sets={"cache_items": event_cache_paths},
            ),
        ),
        overwrite=overwrite,
    )
    return counts


def validate_args(args: argparse.Namespace) -> None:
    if args.split == "all" and args.sequences:
        raise ValueError("--sequences cannot be combined with --split all")
    validate_preparation_options(
        image_output_subdir=args.image_output_subdir,
        representation_name=args.representation,
        percentile=args.percentile,
        event_bins=args.event_bins,
        voxel_normalization=args.voxel_normalization,
        event_window_fraction=args.event_window_fraction,
    )


def validate_preparation_options(
    *,
    image_output_subdir: str,
    representation_name: str,
    percentile: float,
    event_bins: int,
    voxel_normalization: str,
    event_window_fraction: float,
) -> None:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if event_bins <= 0:
        raise ValueError("event bins must be positive")
    event_window_contract(event_window_fraction)
    if representation_name not in {"gep_rgb", "voxel_grid"}:
        raise ValueError(f"Unknown event representation: {representation_name}")
    if voxel_normalization not in {"none", "nonzero_standardize", "log1p"}:
        raise ValueError(f"Unknown voxel normalization: {voxel_normalization}")
    output_subdir = Path(image_output_subdir)
    if (
        output_subdir.is_absolute()
        or len(output_subdir.parts) != 1
        or output_subdir.parts[0] in {".", "..", "rectified"}
    ):
        raise ValueError("image output subdir must be one relative directory name")


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for DSEC image alignment. "
            "Install the preparation dependencies with `pip install -e '.[prepare]'`."
        )


def load_frame_pairs(image_dir: Path, timestamp_path: Path) -> list[tuple[Path, int]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"DSEC rectified RGB directory not found: {image_dir}")
    if not timestamp_path.is_file():
        raise FileNotFoundError(f"DSEC RGB timestamp file not found: {timestamp_path}")

    images = [path for path in image_dir.iterdir() if path.suffix.lower() == ".png"]
    if any(not path.stem.isdecimal() for path in images):
        raise ValueError(f"DSEC RGB filenames must use numeric frame indices: {image_dir}")
    images.sort(key=lambda path: int(path.stem))
    image_indices = [int(path.stem) for path in images]
    if image_indices and image_indices != list(
        range(image_indices[0], image_indices[0] + len(image_indices))
    ):
        raise ValueError(f"DSEC RGB frame indices are not contiguous: {image_dir}")
    timestamps = np.atleast_1d(np.loadtxt(timestamp_path, dtype=np.int64))
    if timestamps.ndim != 1:
        raise ValueError(f"DSEC RGB timestamps must be one-dimensional: {timestamp_path}")
    if len(images) != len(timestamps):
        raise ValueError(
            f"DSEC image/timestamp count mismatch for {image_dir}: "
            f"{len(images)} images versus {len(timestamps)} timestamps"
        )
    if len(timestamps) < 2:
        raise ValueError(f"At least two DSEC RGB frames are required: {image_dir}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"DSEC RGB timestamps are not strictly increasing: {timestamp_path}")
    return [(path, int(timestamp)) for path, timestamp in zip(images, timestamps)]


def ensure_json_metadata(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Existing metadata is unreadable; use --overwrite: {path}"
            ) from error
        if existing != payload:
            raise RuntimeError(f"Existing metadata is incompatible; use --overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_bytes_write(path, encoded.encode("utf-8"), overwrite=overwrite)


def validate_existing_metadata_base(
    path: Path,
    expected: dict[str, Any],
    *,
    ignored_fields: set[str],
    overwrite: bool,
) -> None:
    if overwrite or not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Existing metadata is unreadable; use --overwrite: {path}") from error
    comparable = {
        key: value for key, value in existing.items() if key not in ignored_fields
    }
    if comparable != expected or not ignored_fields.issubset(existing):
        raise RuntimeError(f"Existing metadata is incompatible; use --overwrite: {path}")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for existing environments.
        return torch.load(path, map_location="cpu")


def atomic_imwrite(path: Path, image: np.ndarray, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"OpenCV failed to write image: {path}")
        _commit_temporary_file(temporary_path, path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        _commit_temporary_file(temporary_path, path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_bytes_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _commit_temporary_file(temporary_path, path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)


def _commit_temporary_file(temporary_path: Path, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    os.replace(temporary_path, path)


if __name__ == "__main__":
    main()
