"""Sequence-level DSEC loading with strict temporal and spatial alignment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # Registers DSEC's HDF5 compression filters.
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .cache_metadata import (
    ALIGNMENT_FORMAT_VERSION,
    EVENT_CACHE_FORMAT_VERSION,
    TEACHER_CACHE_FORMAT_VERSION,
    build_input_fingerprint,
    canonical_json_sha256,
    frame_manifest_sha256,
    validate_input_fingerprint,
    validate_success_marker,
)
from .event_representation import EventVoxelizer, GEPEventFrame
from .transforms import PairedSequenceTransform


EVENT_WINDOW_NAME = "rgb_interval"
EVENT_WINDOW_BOUNDARY = "previous_timestamp < event_timestamp <= timestamp"


@dataclass(frozen=True)
class DSECFrame:
    sequence_name: str
    frame_index: int
    timestamp: int
    previous_timestamp: int
    image_path: Path


def event_representation_metadata(event_representation: Any) -> dict[str, Any]:
    """Return the exact algorithm identity stored in an event-cache manifest."""

    if isinstance(event_representation, GEPEventFrame):
        return {
            "type": "gep_rgb",
            "channels": 3,
            "percentile": float(event_representation.percentile),
            "polarity_colors_rgb": {"negative": "red", "positive": "blue"},
        }
    if isinstance(event_representation, EventVoxelizer):
        return {
            "type": "voxel_grid",
            "channels": int(event_representation.channels),
            "event_bins": int(event_representation.num_bins),
            "polarity_split": bool(event_representation.polarity_split),
            "channel_order": "negative_0,positive_0,...",
            "normalization": str(event_representation.normalization),
        }
    raise TypeError(
        "Cached DSEC events require GEPEventFrame or EventVoxelizer so their "
        "representation metadata can be verified"
    )


def validate_event_cache_payload(
    payload: Any,
    *,
    cache_path: str | Path,
    frame_index: int,
    timestamp: int,
    previous_timestamp: int,
    sequence_name: str,
    split: str,
    height: int,
    width: int,
    representation: Mapping[str, Any],
    rectified: bool,
    input_fingerprint_digest: str,
    manifest_digest: str,
) -> tuple[Tensor, int]:
    """Validate one event-cache item against its frame and sequence manifest."""

    path = Path(cache_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Event cache payload must be a mapping: {path}")
    expected_fields = {
        "format_version": EVENT_CACHE_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": split,
        "sequence_name": sequence_name,
        "frame_index": int(frame_index),
        "timestamp": int(timestamp),
        "previous_timestamp": int(previous_timestamp),
        "height": int(height),
        "width": int(width),
        "representation": dict(representation),
        "event_window_boundary": EVENT_WINDOW_BOUNDARY,
        "rectified": bool(rectified),
        "input_fingerprint_digest": input_fingerprint_digest,
        "manifest_digest": manifest_digest,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Event cache {field}={payload.get(field)!r} does not match "
                f"{expected!r}: {path}"
            )

    event_count = payload.get("event_count")
    if (
        not isinstance(event_count, int)
        or isinstance(event_count, bool)
        or event_count < 0
    ):
        raise ValueError(f"Event cache event_count must be a non-negative integer: {path}")
    tensor = payload.get("events")
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Event cache does not contain an events tensor: {path}")
    expected_shape = (int(representation["channels"]), int(height), int(width))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Event cache shape {tuple(tensor.shape)} does not match {expected_shape}: {path}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(f"Event cache tensor must use float32: {path}")
    return tensor, event_count


def discover_dsec_sequences(
    root: str | Path,
    split: str,
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Return sequences that have both image and event data.

    A caller-provided sequence manifest is kept in its given order. This makes
    train/validation membership explicit and prevents frame-level leakage.
    """

    root = Path(root).expanduser()
    image_root = root / f"{split}_images"
    event_root = root / f"{split}_events"
    if not image_root.is_dir():
        raise FileNotFoundError(f"DSEC image split not found: {image_root}")
    if not event_root.is_dir():
        raise FileNotFoundError(f"DSEC event split not found: {event_root}")

    available = {
        path.name
        for path in image_root.iterdir()
        if path.is_dir() and (event_root / path.name).is_dir()
    }
    if requested is None:
        sequences = sorted(available)
    else:
        sequences = list(requested)
        if len(sequences) != len(set(sequences)):
            raise ValueError("The DSEC sequence manifest contains duplicates")
        missing = sorted(set(sequences) - available)
        if missing:
            raise FileNotFoundError(
                f"Sequences are absent from {split!r}: {', '.join(missing)}"
            )
    if not sequences:
        raise ValueError(f"No DSEC sequences selected for split {split!r}")
    return sequences


class DSECEventReader:
    """Lazily slice one DSEC HDF5 event stream in absolute microseconds."""

    def __init__(
        self,
        events_path: str | Path,
        *,
        rectify_map_path: str | Path | None,
        output_height: int,
        output_width: int,
    ) -> None:
        self.events_path = Path(events_path)
        self.rectify_map_path = Path(rectify_map_path) if rectify_map_path else None
        self.output_height = output_height
        self.output_width = output_width
        self._h5: h5py.File | None = None
        self._rectify_map: np.ndarray | None = None

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.events_path, "r")
            required = ("events/x", "events/y", "events/t", "events/p", "ms_to_idx")
            missing = [key for key in required if key not in self._h5]
            if missing:
                self.close()
                raise KeyError(f"{self.events_path} lacks datasets: {missing}")
        return self._h5

    @property
    def timestamp_offset(self) -> int:
        h5_file = self._open()
        return int(h5_file["t_offset"][()]) if "t_offset" in h5_file else 0

    def slice(self, start_time: int, end_time: int) -> dict[str, np.ndarray]:
        """Return events satisfying ``start_time < t <= end_time``."""

        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        h5_file = self._open()
        local_start = int(start_time) - self.timestamp_offset
        local_end = int(end_time) - self.timestamp_offset
        timestamps = h5_file["events/t"]
        ms_to_idx = h5_file["ms_to_idx"]

        start_ms = max(0, math.floor(local_start / 1000))
        end_ms = max(0, math.floor(local_end / 1000) + 1)
        start_index = int(ms_to_idx[start_ms]) if start_ms < len(ms_to_idx) else len(timestamps)
        end_index = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else len(timestamps)
        if start_index >= end_index:
            return self._empty()

        local_t = np.asarray(timestamps[start_index:end_index], dtype=np.int64)
        relative_start = int(np.searchsorted(local_t, local_start, side="right"))
        relative_end = int(np.searchsorted(local_t, local_end, side="right"))
        absolute_start = start_index + relative_start
        absolute_end = start_index + relative_end
        if absolute_start >= absolute_end:
            return self._empty()

        result = {
            "x": np.asarray(h5_file["events/x"][absolute_start:absolute_end]),
            "y": np.asarray(h5_file["events/y"][absolute_start:absolute_end]),
            "t": local_t[relative_start:relative_end] + self.timestamp_offset,
            "p": np.asarray(h5_file["events/p"][absolute_start:absolute_end]),
        }
        if self.rectify_map_path is not None:
            result = self._rectify(result)
        return result

    def _rectify(self, events: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._rectify_map is None:
            if not self.rectify_map_path or not self.rectify_map_path.is_file():
                raise FileNotFoundError(f"DSEC rectify map not found: {self.rectify_map_path}")
            with h5py.File(self.rectify_map_path, "r") as rectify_file:
                if "rectify_map" not in rectify_file:
                    raise KeyError(f"rectify_map dataset not found: {self.rectify_map_path}")
                self._rectify_map = np.asarray(rectify_file["rectify_map"], dtype=np.float32)
            if self._rectify_map.ndim != 3 or self._rectify_map.shape[-1] != 2:
                raise ValueError(
                    f"Unexpected rectify map shape {self._rectify_map.shape}; expected [H, W, 2]"
                )

        raw_x = events["x"].astype(np.int64, copy=False)
        raw_y = events["y"].astype(np.int64, copy=False)
        map_height, map_width = self._rectify_map.shape[:2]
        valid_raw = (
            (raw_x >= 0) & (raw_x < map_width) & (raw_y >= 0) & (raw_y < map_height)
        )
        coordinates = self._rectify_map[raw_y[valid_raw], raw_x[valid_raw]]
        # Both supported representations use a discrete pixel lattice. Apply
        # one shared nearest-pixel policy here so their bounds and event_count
        # semantics cannot diverge at the image border.
        continuous_x = coordinates[:, 0]
        continuous_y = coordinates[:, 1]
        finite = np.isfinite(continuous_x) & np.isfinite(continuous_y)
        rectified_x = np.full(continuous_x.shape, -1, dtype=np.int64)
        rectified_y = np.full(continuous_y.shape, -1, dtype=np.int64)
        rectified_x[finite] = np.rint(continuous_x[finite]).astype(np.int64)
        rectified_y[finite] = np.rint(continuous_y[finite]).astype(np.int64)
        valid_rectified = (
            finite
            & (rectified_x >= 0)
            & (rectified_x < self.output_width)
            & (rectified_y >= 0)
            & (rectified_y < self.output_height)
        )
        source_indices = np.flatnonzero(valid_raw)[valid_rectified]
        return {
            "x": rectified_x[valid_rectified],
            "y": rectified_y[valid_rectified],
            "t": events["t"][source_indices],
            "p": events["p"][source_indices],
        }

    @staticmethod
    def _empty() -> dict[str, np.ndarray]:
        return {
            "x": np.empty(0, dtype=np.float32),
            "y": np.empty(0, dtype=np.float32),
            "t": np.empty(0, dtype=np.int64),
            "p": np.empty(0, dtype=np.int8),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


class DSECSequenceDataset(Dataset[dict[str, Any]]):
    """Return contiguous DSEC clips without ever crossing a sequence boundary."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        sequence_length: int,
        event_representation: Any,
        transform: PairedSequenceTransform,
        sequences: Sequence[str] | None = None,
        clip_stride: int = 1,
        include_incomplete_clips: bool = False,
        image_directory: str = "aligned_event",
        rectify_events: bool = True,
        event_cache_dir: str | Path | None = None,
        feature_cache_dir: str | Path | None = None,
        load_events: bool = True,
        load_images: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if clip_stride <= 0:
            raise ValueError("clip_stride must be positive")
        if not load_events and not load_images:
            raise ValueError("At least one input modality must be loaded")
        if not rectify_events and (load_images or feature_cache_dir is not None):
            raise ValueError(
                "Raw event coordinates cannot be paired with rectified RGB or teacher "
                "features; set rectify_events=True"
            )
        if feature_cache_dir is not None and transform.stochastic:
            raise ValueError(
                "Cached teacher features cannot be paired with stochastic spatial transforms; "
                "disable augmentation or use the online teacher"
            )

        self.root = Path(root).expanduser()
        self.split = split
        self.sequence_length = sequence_length
        self.include_incomplete_clips = bool(include_incomplete_clips)
        self.event_representation = event_representation
        self.transform = transform
        self.image_directory = image_directory
        self.rectify_events = rectify_events
        self.event_cache_dir = Path(event_cache_dir).expanduser() if event_cache_dir else None
        self.feature_cache_dir = (
            Path(feature_cache_dir).expanduser() if feature_cache_dir else None
        )
        self.load_events = load_events
        self.load_images = load_images
        self.sequence_names = discover_dsec_sequences(self.root, split, sequences)
        self._readers: dict[str, DSECEventReader] = {}
        self._frames_by_sequence: dict[str, list[DSECFrame]] = {}
        self._event_cache_manifests: dict[str, dict[str, Any]] = {}
        self._teacher_cache_manifests: dict[str, dict[str, Any]] = {}
        self._clips: list[tuple[str, int, int]] = []

        for sequence_name in self.sequence_names:
            frames = self._load_frame_manifest(sequence_name)
            self._frames_by_sequence[sequence_name] = frames
            if self.event_cache_dir is not None:
                self._event_cache_manifests[sequence_name] = (
                    self._validate_event_cache_manifest(sequence_name, frames)
                )
            if self.feature_cache_dir is not None:
                self._teacher_cache_manifests[sequence_name] = (
                    self._validate_teacher_cache_structure(sequence_name, frames)
                )
            # Frame zero has no preceding RGB interval and is intentionally excluded.
            if self.include_incomplete_clips:
                self._clips.extend(
                    (
                        sequence_name,
                        start,
                        min(self.sequence_length, len(frames) - start),
                    )
                    for start in range(1, len(frames), clip_stride)
                )
            else:
                stop = len(frames) - self.sequence_length + 1
                self._clips.extend(
                    (sequence_name, start, self.sequence_length)
                    for start in range(1, max(1, stop), clip_stride)
                    if start + self.sequence_length <= len(frames)
                )
        if not self._clips:
            raise ValueError(
                f"No clips of length {sequence_length} are available in DSEC split {split!r}"
            )

    def _validate_event_cache_manifest(
        self,
        sequence_name: str,
        frames: Sequence[DSECFrame],
    ) -> dict[str, Any]:
        if self.event_cache_dir is None:
            raise RuntimeError("event_cache_dir is not configured")
        metadata_path = self.event_cache_dir / sequence_name / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Event cache metadata not found: {metadata_path}. "
                "Regenerate it with tools/prepare_dsec.py."
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid event cache metadata: {metadata_path}") from error
        expected_common = {
            "format_version": EVENT_CACHE_FORMAT_VERSION,
            "dataset": "DSEC",
            "split": self.split,
            "sequence_name": sequence_name,
            "coordinate_space": "rectified_event" if self.rectify_events else "raw_event",
            "spatial_quantization": (
                "nearest_integer_numpy_rint"
                if self.rectify_events
                else "raw_integer_coordinates"
            ),
            "height": int(self.event_representation.height),
            "width": int(self.event_representation.width),
            "event_window": EVENT_WINDOW_NAME,
            "event_window_boundary": EVENT_WINDOW_BOUNDARY,
            "first_frame_cached": False,
            "frame_count": len(frames) - 1,
            "timestamp_manifest_sha256": frame_manifest_sha256(
                (frame.frame_index, frame.previous_timestamp, frame.timestamp)
                for frame in frames[1:]
            ),
        }
        for field, expected in expected_common.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"Event cache {field}={metadata.get(field)!r} does not match "
                    f"{expected!r}: {metadata_path}"
                )

        cached_representation = metadata.get("representation", {})
        expected_representation = event_representation_metadata(self.event_representation)
        if cached_representation != expected_representation:
            raise ValueError(
                f"Event cache representation differs from the configured representation: "
                f"{metadata_path}"
            )

        required_roles = {"rgb_timestamps", "events", "camera_calibration"}
        if self.rectify_events:
            required_roles.add("rectify_map")
        fingerprint_digest = validate_input_fingerprint(
            metadata.get("input_fingerprint"),
            required_file_roles=required_roles,
        )
        expected_paths = {
            "rgb_timestamps": (
                f"{self.split}_images/{sequence_name}/images/timestamps.txt"
            ),
            "events": f"{self.split}_events/{sequence_name}/events/left/events.h5",
            "camera_calibration": (
                f"{self.split}_calibration/{sequence_name}/calibration/cam_to_cam.yaml"
            ),
        }
        if self.rectify_events:
            expected_paths["rectify_map"] = (
                f"{self.split}_events/{sequence_name}/events/left/rectify_map.h5"
            )
        actual_paths = {
            entry["role"]: entry["path"]
            for entry in metadata["input_fingerprint"]["files"]
        }
        for role, expected_path in expected_paths.items():
            if actual_paths.get(role) != expected_path:
                raise ValueError(
                    f"Event cache fingerprint path for {role!r} does not match "
                    f"{expected_path!r}: {metadata_path}"
                )
        if metadata["input_fingerprint"].get("digest") != fingerprint_digest:
            raise ValueError(f"Event cache fingerprint is inconsistent: {metadata_path}")
        current_files: dict[str, Path] = {
            "rgb_timestamps": (
                self.root
                / f"{self.split}_images"
                / sequence_name
                / "images"
                / "timestamps.txt"
            ),
            "events": (
                self.root
                / f"{self.split}_events"
                / sequence_name
                / "events"
                / "left"
                / "events.h5"
            ),
            "camera_calibration": (
                self.root
                / f"{self.split}_calibration"
                / sequence_name
                / "calibration"
                / "cam_to_cam.yaml"
            ),
        }
        if self.rectify_events:
            current_files["rectify_map"] = (
                self.root
                / f"{self.split}_events"
                / sequence_name
                / "events"
                / "left"
                / "rectify_map.h5"
            )
        current_fingerprint = build_input_fingerprint(self.root, files=current_files)
        if current_fingerprint != metadata["input_fingerprint"]:
            raise ValueError(
                f"DSEC inputs changed after the event cache was built: {metadata_path}"
            )

        sequence_cache_dir = self.event_cache_dir / sequence_name
        expected_cache_files = [
            sequence_cache_dir / f"{frame.timestamp}.pt"
            for frame in frames[1:]
        ]
        missing_cache_files = [
            path
            for path in expected_cache_files
            if not path.is_file()
        ]
        if missing_cache_files:
            preview = ", ".join(path.name for path in missing_cache_files[:5])
            raise FileNotFoundError(
                f"Event cache is missing {len(missing_cache_files)} frames in "
                f"{sequence_cache_dir}; first files: {preview}"
            )
        validate_success_marker(
            sequence_cache_dir,
            metadata,
            output_paths=expected_cache_files,
        )
        return metadata

    def _validate_teacher_cache_structure(
        self,
        sequence_name: str,
        frames: Sequence[DSECFrame],
    ) -> dict[str, Any]:
        if self.feature_cache_dir is None:
            raise RuntimeError("feature_cache_dir is not configured")
        sequence_cache_dir = self.feature_cache_dir / sequence_name
        metadata_path = sequence_cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Teacher cache metadata not found: {metadata_path}. "
                "Regenerate it with tools/cache_dinov3_features.py."
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid teacher cache metadata: {metadata_path}") from error
        expected_common = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "DSEC",
            "split": self.split,
            "sequence_name": sequence_name,
            "frame_count": len(frames),
            "timestamp_manifest_sha256": frame_manifest_sha256(
                (frame.frame_index, frame.previous_timestamp, frame.timestamp)
                for frame in frames
            ),
        }
        for field, expected in expected_common.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"Teacher cache {field}={metadata.get(field)!r} does not match "
                    f"{expected!r}: {metadata_path}"
                )
        validate_input_fingerprint(
            metadata.get("input_fingerprint"),
            required_file_roles={"rgb_timestamps", "alignment_metadata"},
            required_file_set_roles={"aligned_rgb_frames"},
        )
        image_root = self.root / f"{self.split}_images" / sequence_name / "images"
        current_fingerprint = build_input_fingerprint(
            self.root,
            files={
                "rgb_timestamps": image_root / "timestamps.txt",
                "alignment_metadata": (
                    image_root / "left" / self.image_directory / "metadata.json"
                ),
            },
            file_sets={"aligned_rgb_frames": [frame.image_path for frame in frames]},
        )
        if current_fingerprint != metadata["input_fingerprint"]:
            raise ValueError(
                f"Aligned RGB inputs changed after the teacher cache was built: {metadata_path}"
            )
        expected_cache_files = [
            sequence_cache_dir / f"{frame.timestamp}.pt"
            for frame in frames
        ]
        missing_cache_files = [path for path in expected_cache_files if not path.is_file()]
        if missing_cache_files:
            preview = ", ".join(path.name for path in missing_cache_files[:5])
            raise FileNotFoundError(
                f"Teacher cache is missing {len(missing_cache_files)} frames in "
                f"{sequence_cache_dir}; first files: {preview}"
            )
        validate_success_marker(
            sequence_cache_dir,
            metadata,
            output_paths=expected_cache_files,
        )
        return metadata

    def _load_frame_manifest(self, sequence_name: str) -> list[DSECFrame]:
        sequence_root = self.root / f"{self.split}_images" / sequence_name / "images"
        timestamp_path = sequence_root / "timestamps.txt"
        if not timestamp_path.is_file():
            raise FileNotFoundError(f"DSEC RGB timestamps not found: {timestamp_path}")
        timestamps = np.atleast_1d(np.loadtxt(timestamp_path, dtype=np.int64))
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError(f"At least two timestamps are required: {timestamp_path}")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"RGB timestamps are not strictly increasing: {timestamp_path}")

        image_dir = sequence_root / "left" / self.image_directory
        if not image_dir.is_dir():
            hint = (
                " Run `python tools/prepare_dsec.py` first."
                if self.image_directory == "aligned_event"
                else ""
            )
            raise FileNotFoundError(f"DSEC image directory not found: {image_dir}.{hint}")
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        by_stem = {path.stem: path for path in image_paths}
        if len(by_stem) != len(image_paths):
            raise ValueError(f"Duplicate image stems found in {image_dir}")
        if image_dir.name == "rectified":
            if len(image_paths) != len(timestamps):
                raise ValueError(
                    f"DSEC rectified RGB count differs from timestamps: {image_dir}"
                )
            if any(not path.stem.isdecimal() for path in image_paths):
                raise ValueError(f"DSEC RGB filenames must use numeric indices: {image_dir}")
            ordered_paths = sorted(image_paths, key=lambda path: int(path.stem))
            indices = [int(path.stem) for path in ordered_paths]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError(f"DSEC RGB frame indices are not contiguous: {image_dir}")
        else:
            missing_timestamps = [
                int(timestamp)
                for timestamp in timestamps
                if str(int(timestamp)) not in by_stem
            ]
            if missing_timestamps:
                preview = ", ".join(str(value) for value in missing_timestamps[:5])
                raise ValueError(
                    f"Cannot pair {len(missing_timestamps)} RGB timestamps by filename in "
                    f"{image_dir}; first timestamps: {preview}"
                )
            ordered_paths = [by_stem[str(int(timestamp))] for timestamp in timestamps]

        frames: list[DSECFrame] = []
        for index, (timestamp, image_path) in enumerate(zip(timestamps, ordered_paths)):
            previous = int(timestamps[index - 1]) if index else int(timestamp)
            frames.append(
                DSECFrame(
                    sequence_name=sequence_name,
                    frame_index=index,
                    timestamp=int(timestamp),
                    previous_timestamp=previous,
                    image_path=image_path,
                )
            )
        if image_dir.name != "rectified":
            self._validate_alignment_manifest(sequence_name, image_dir, frames)
        return frames

    def _validate_alignment_manifest(
        self,
        sequence_name: str,
        image_dir: Path,
        frames: Sequence[DSECFrame],
    ) -> None:
        metadata_path = image_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Aligned RGB metadata not found: {metadata_path}. "
                "Regenerate it with tools/prepare_dsec.py."
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid aligned RGB metadata: {metadata_path}") from error
        expected = {
            "format_version": ALIGNMENT_FORMAT_VERSION,
            "dataset": "DSEC",
            "split": self.split,
            "sequence_name": sequence_name,
            "frame_count": len(frames),
            "timestamp_manifest_sha256": frame_manifest_sha256(
                (frame.frame_index, frame.previous_timestamp, frame.timestamp)
                for frame in frames
            ),
        }
        for field, expected_value in expected.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"Aligned RGB {field}={metadata.get(field)!r} does not match "
                    f"{expected_value!r}: {metadata_path}"
                )
        calibration_metadata = metadata.get("calibration")
        expected_sensor_size = (
            int(self.event_representation.height),
            int(self.event_representation.width),
        )
        if not isinstance(calibration_metadata, dict) or (
            calibration_metadata.get("height"),
            calibration_metadata.get("width"),
        ) != expected_sensor_size:
            raise ValueError(
                "Aligned RGB event-camera size does not match the configured event "
                f"representation {expected_sensor_size}: {metadata_path}"
            )
        validate_input_fingerprint(
            metadata.get("input_fingerprint"),
            required_file_roles={"rgb_timestamps", "camera_calibration"},
            required_file_set_roles={"rectified_rgb_frames"},
        )
        source_directory_id = metadata.get("source_image_directory")
        if (
            not isinstance(source_directory_id, str)
            or Path(source_directory_id).is_absolute()
            or ".." in Path(source_directory_id).parts
        ):
            raise ValueError(
                f"Aligned RGB source_image_directory must be dataset-relative: {metadata_path}"
            )
        source_directory = self.root / source_directory_id
        source_images = sorted(
            path
            for path in source_directory.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        current_fingerprint = build_input_fingerprint(
            self.root,
            files={
                "rgb_timestamps": (
                    self.root
                    / f"{self.split}_images"
                    / sequence_name
                    / "images"
                    / "timestamps.txt"
                ),
                "camera_calibration": (
                    self.root
                    / f"{self.split}_calibration"
                    / sequence_name
                    / "calibration"
                    / "cam_to_cam.yaml"
                ),
            },
            file_sets={"rectified_rgb_frames": source_images},
        )
        if current_fingerprint != metadata["input_fingerprint"]:
            raise ValueError(
                f"DSEC RGB/calibration inputs changed after alignment: {metadata_path}"
            )
        validate_input_fingerprint(
            metadata.get("output_fingerprint"),
            required_file_set_roles={"aligned_rgb_frames"},
        )
        current_output_fingerprint = build_input_fingerprint(
            self.root,
            files={},
            file_sets={"aligned_rgb_frames": [frame.image_path for frame in frames]},
        )
        if current_output_fingerprint != metadata["output_fingerprint"]:
            raise ValueError(f"Aligned RGB outputs changed after preparation: {metadata_path}")
        validate_success_marker(image_dir, metadata)

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_name, start, clip_length = self._clips[index]
        records = self._frames_by_sequence[sequence_name][start : start + clip_length]

        event_sequence: Tensor | None = None
        event_counts: list[int] = []
        if self.load_events:
            event_tensors = []
            for record in records:
                event_tensor, event_count = self._load_event_frame(record)
                event_tensors.append(event_tensor)
                event_counts.append(event_count)
            event_sequence = torch.stack(event_tensors)

        image_sequence: Tensor | None = None
        if self.load_images:
            image_sequence = torch.stack(
                [self._load_image(record.image_path) for record in records]
            )

        event_sequence, image_sequence = self.transform(event_sequence, image_sequence)
        sample: dict[str, Any] = {
            "timestamps": torch.tensor([record.timestamp for record in records], dtype=torch.int64),
            "frame_indices": torch.tensor(
                [record.frame_index for record in records], dtype=torch.int64
            ),
            "sequence_name": sequence_name,
            "is_sequence_start": start == 1,
            "is_sequence_end": start + clip_length == len(
                self._frames_by_sequence[sequence_name]
            ),
        }
        if event_sequence is not None:
            sample["events"] = event_sequence
            sample["event_counts"] = torch.tensor(event_counts, dtype=torch.int64)
        if image_sequence is not None:
            sample["images"] = image_sequence
        if self.feature_cache_dir is not None:
            sample["teacher_features"] = torch.stack(
                [self._load_teacher_feature(record) for record in records]
            )
        return sample

    def _reader(self, sequence_name: str) -> DSECEventReader:
        if sequence_name not in self._readers:
            event_base = self.root / f"{self.split}_events" / sequence_name / "events" / "left"
            self._readers[sequence_name] = DSECEventReader(
                event_base / "events.h5",
                rectify_map_path=(event_base / "rectify_map.h5") if self.rectify_events else None,
                output_height=self.event_representation.height,
                output_width=self.event_representation.width,
            )
        return self._readers[sequence_name]

    def _load_event_frame(self, record: DSECFrame) -> tuple[Tensor, int]:
        if self.event_cache_dir is not None:
            cache_path = self.event_cache_dir / record.sequence_name / f"{record.timestamp}.pt"
            payload = _safe_torch_load(cache_path)
            manifest = self._event_cache_manifests[record.sequence_name]
            return validate_event_cache_payload(
                payload,
                cache_path=cache_path,
                frame_index=record.frame_index,
                timestamp=record.timestamp,
                previous_timestamp=record.previous_timestamp,
                sequence_name=record.sequence_name,
                split=self.split,
                height=int(self.event_representation.height),
                width=int(self.event_representation.width),
                representation=event_representation_metadata(self.event_representation),
                rectified=self.rectify_events,
                input_fingerprint_digest=manifest["input_fingerprint"]["digest"],
                manifest_digest=canonical_json_sha256(manifest),
            )

        events = self._reader(record.sequence_name).slice(
            record.previous_timestamp,
            record.timestamp,
        )
        tensor = self.event_representation(
            *(torch.from_numpy(events[key]) for key in ("x", "y", "t", "p")),
            start_time=record.previous_timestamp,
            end_time=record.timestamp,
        )
        return tensor, len(events["t"])

    def _load_teacher_feature(self, record: DSECFrame) -> Tensor:
        cache_path = self.feature_cache_dir / record.sequence_name / f"{record.timestamp}.pt"
        payload = _safe_torch_load(cache_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Teacher cache payload must be a mapping: {cache_path}")
        manifest = self._teacher_cache_manifests[record.sequence_name]
        expected_fields = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "DSEC",
            "split": self.split,
            "sequence_name": record.sequence_name,
            "frame_index": record.frame_index,
            "timestamp": record.timestamp,
            "source_image": self._relative_dataset_path(record.image_path),
            "input_fingerprint_digest": manifest["input_fingerprint"]["digest"],
            "manifest_digest": canonical_json_sha256(manifest),
            "grid_size": [manifest["grid"]["height"], manifest["grid"]["width"]],
            "input_size": [manifest["input"]["height"], manifest["input"]["width"]],
            "model_identifier": manifest["model"]["identifier"],
            "checkpoint_identifier": manifest["model"]["checkpoint"],
            "cache_dtype": manifest["cache_dtype"],
        }
        for field, expected in expected_fields.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"Teacher cache {field}={payload.get(field)!r} does not match "
                    f"{expected!r}: {cache_path}"
                )
        tokens = payload.get("patch_tokens")
        if not isinstance(tokens, Tensor) or tokens.ndim != 2:
            raise ValueError(f"patch_tokens must have shape [N, D]: {cache_path}")
        expected_input_size = [self.transform.height, self.transform.width]
        if payload["input_size"] != expected_input_size:
            raise ValueError(
                f"Teacher cache input_size {payload['input_size']} does not match "
                f"{expected_input_size}: {cache_path}"
            )
        cached_grid = payload["grid_size"]
        if int(cached_grid[0]) * int(cached_grid[1]) != tokens.shape[0]:
            raise ValueError(f"Teacher cache grid_size is inconsistent: {cache_path}")
        if tokens.shape[1] != int(manifest["model"]["embedding_dim"]):
            raise ValueError(f"Teacher cache embedding dimension is inconsistent: {cache_path}")
        expected_dtype = getattr(torch, str(manifest["cache_dtype"]), None)
        if expected_dtype is None or tokens.dtype != expected_dtype:
            raise ValueError(f"Teacher cache tensor dtype is inconsistent: {cache_path}")
        return tokens

    def _relative_dataset_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"DSEC path is outside the dataset root: {path}") from error

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    def close(self) -> None:
        readers = getattr(self, "_readers", {})
        for reader in readers.values():
            reader.close()
        readers.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def __del__(self) -> None:
        self.close()


def _safe_torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Cache file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for existing environments.
        return torch.load(path, map_location="cpu")
