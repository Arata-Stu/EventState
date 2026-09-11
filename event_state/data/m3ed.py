"""Prepared M3ED clips for EventState pretraining.

Raw M3ED recordings are converted by :mod:`tools.prepare_m3ed`.  Keeping the
expensive calibration and DAGR downsampling out of DataLoader workers makes
training deterministic and preserves the downsampler state across the complete
event stream.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .cache_metadata import TEACHER_CACHE_FORMAT_VERSION
from .dsec import normalize_event_window_fraction
from .m3ed_downstream import M3ED_DOWNSTREAM_FORMAT_VERSION
from .transforms import PairedSequenceTransform


M3ED_PREPARED_FORMAT_VERSION = 1


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_metadata(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description} metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description} metadata: {path}")
    return value


@dataclass(frozen=True)
class M3EDFrame:
    sequence_name: str
    frame_index: int
    timestamp: int
    previous_timestamp: int
    image_path: Path
    event_path: Path | None


def discover_prepared_m3ed_sequences(
    prepared_root: str | Path,
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Return prepared sequence names, preserving an explicit manifest order."""

    root = Path(prepared_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Prepared M3ED root not found: {root}. Run tools/prepare_m3ed.py first."
        )
    available = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    }
    if requested is None:
        sequences = sorted(available)
    else:
        sequences = [str(value) for value in requested]
        if len(sequences) != len(set(sequences)):
            raise ValueError("The M3ED sequence manifest contains duplicates")
        missing = sorted(set(sequences) - available)
        if missing:
            raise FileNotFoundError(
                "Prepared M3ED sequences are missing: " + ", ".join(missing)
            )
    if not sequences:
        raise ValueError(f"No prepared M3ED sequences found under {root}")
    return sequences


class M3EDSequenceDataset(Dataset[dict[str, Any]]):
    """Return contiguous prepared M3ED clips with the DSEC training contract."""

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
        image_directory: str = "aligned_rgb",
        rectify_events: bool = True,
        event_cache_dir: str | Path | None = None,
        feature_cache_dir: str | Path | None = None,
        load_events: bool = True,
        load_images: bool = True,
        event_window_fraction: float = 1.0,
        prepared_root: str | Path | None = None,
        target_cache_dir: str | Path | None = None,
        target_tasks: Sequence[str] | None = None,
    ) -> None:
        if sequence_length <= 0 or clip_stride <= 0:
            raise ValueError("sequence_length and clip_stride must be positive")
        if not load_events and not load_images:
            raise ValueError("At least one input modality must be loaded")
        if not rectify_events:
            raise ValueError("Prepared M3ED pretraining requires rectified events")
        if normalize_event_window_fraction(event_window_fraction) != 1.0:
            raise ValueError(
                "Prepared M3ED currently uses complete consecutive RGB intervals"
            )
        if image_directory != "aligned_rgb":
            raise ValueError("M3ED image_directory must be aligned_rgb")

        self.root = Path(root).expanduser()
        self.prepared_root = Path(prepared_root or event_cache_dir or root).expanduser()
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.include_incomplete_clips = bool(include_incomplete_clips)
        self.event_representation = event_representation
        self.transform = transform
        self.event_cache_dir = self.prepared_root
        self.feature_cache_dir = (
            Path(feature_cache_dir).expanduser() if feature_cache_dir else None
        )
        self.target_cache_dir = (
            Path(target_cache_dir).expanduser() if target_cache_dir else None
        )
        self.target_tasks = tuple(
            dict.fromkeys(str(value) for value in (target_tasks or ()))
        )
        unsupported_tasks = sorted(set(self.target_tasks) - {"depth", "semantics", "pose"})
        if unsupported_tasks:
            raise ValueError("Unsupported M3ED downstream tasks: " + ", ".join(unsupported_tasks))
        if self.target_tasks and self.target_cache_dir is None:
            raise ValueError("target_cache_dir is required when target_tasks are requested")
        if self.target_tasks and self.transform.stochastic:
            raise ValueError("Prepared M3ED downstream targets require deterministic geometry")
        self.load_events = bool(load_events)
        self.load_images = bool(load_images)
        self.sequence_names = discover_prepared_m3ed_sequences(
            self.prepared_root, sequences
        )
        self._frames_by_sequence: dict[str, list[M3EDFrame]] = {}
        self._metadata_by_sequence: dict[str, dict[str, Any]] = {}
        self._teacher_metadata: dict[str, dict[str, Any]] = {}
        self._target_metadata: dict[str, dict[str, Any]] = {}
        self._clips: list[tuple[str, int, int]] = []

        for sequence_name in self.sequence_names:
            metadata, frames = self._load_sequence(sequence_name)
            self._metadata_by_sequence[sequence_name] = metadata
            self._frames_by_sequence[sequence_name] = frames
            if self.feature_cache_dir is not None:
                self._teacher_metadata[sequence_name] = self._load_teacher_metadata(
                    sequence_name, len(frames)
                )
            if self.target_tasks:
                self._target_metadata[sequence_name] = self._load_target_metadata(
                    sequence_name, metadata
                )
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
                self._clips.extend(
                    (sequence_name, start, self.sequence_length)
                    for start in range(1, len(frames) - self.sequence_length + 1, clip_stride)
                )
        if not self._clips:
            raise ValueError("No M3ED clips are available for the configured sequence length")

    def _load_sequence(
        self, sequence_name: str
    ) -> tuple[dict[str, Any], list[M3EDFrame]]:
        sequence_root = self.prepared_root / sequence_name
        metadata_path = sequence_root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid prepared M3ED metadata: {metadata_path}") from error
        expected = {
            "format_version": M3ED_PREPARED_FORMAT_VERSION,
            "dataset": "M3ED",
            "sequence_name": sequence_name,
            "event_size": [
                int(self.event_representation.height),
                int(self.event_representation.width),
            ],
            "downsampling": "dagr_stateful_signed_event_downsampling",
            "coordinate_space": "rectified_left_event",
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"Prepared M3ED {field}={metadata.get(field)!r} does not match "
                    f"{value!r}: {metadata_path}"
                )
        representation = metadata.get("representation", {})
        if representation.get("channels") != int(self.event_representation.channels):
            raise ValueError(f"Prepared M3ED representation mismatch: {metadata_path}")
        timestamps = np.asarray(metadata.get("timestamps", []), dtype=np.int64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError(f"Prepared M3ED requires at least two frames: {metadata_path}")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"M3ED timestamps are not strictly increasing: {metadata_path}")
        if int(metadata.get("frame_count", -1)) != len(timestamps):
            raise ValueError(f"M3ED frame_count is inconsistent: {metadata_path}")

        frames: list[M3EDFrame] = []
        for index, timestamp_value in enumerate(timestamps):
            timestamp = int(timestamp_value)
            image_path = sequence_root / "aligned_rgb" / f"{timestamp}.png"
            event_path = (
                None if index == 0 else sequence_root / "events" / f"{timestamp}.pt"
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"Prepared M3ED RGB frame missing: {image_path}")
            if event_path is not None and not event_path.is_file():
                raise FileNotFoundError(f"Prepared M3ED event frame missing: {event_path}")
            frames.append(
                M3EDFrame(
                    sequence_name=sequence_name,
                    frame_index=index,
                    timestamp=timestamp,
                    previous_timestamp=(int(timestamps[index - 1]) if index else timestamp),
                    image_path=image_path,
                    event_path=event_path,
                )
            )
        return metadata, frames

    def _load_teacher_metadata(
        self, sequence_name: str, frame_count: int
    ) -> dict[str, Any]:
        if self.feature_cache_dir is None:
            raise RuntimeError("feature_cache_dir is not configured")
        path = self.feature_cache_dir / sequence_name / "metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid M3ED teacher cache metadata: {path}") from error
        expected = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "M3ED",
            "sequence_name": sequence_name,
            "frame_count": frame_count,
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"M3ED teacher cache {field}={metadata.get(field)!r} does not "
                    f"match {value!r}: {path}"
                )
        return metadata

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_name, start, clip_length = self._clips[index]
        records = self._frames_by_sequence[sequence_name][start : start + clip_length]

        event_sequence: Tensor | None = None
        event_counts: list[int] = []
        if self.load_events:
            tensors = []
            for record in records:
                tensor, count = self._load_event(record)
                tensors.append(tensor)
                event_counts.append(count)
            event_sequence = torch.stack(tensors)
        image_sequence = (
            torch.stack([self._load_image(record.image_path) for record in records])
            if self.load_images
            else None
        )
        event_sequence, image_sequence = self.transform(event_sequence, image_sequence)

        sample: dict[str, Any] = {
            "timestamps": torch.tensor([item.timestamp for item in records], dtype=torch.int64),
            "frame_indices": torch.tensor(
                [item.frame_index for item in records], dtype=torch.int64
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
        if self.target_tasks:
            sample.update(self._load_targets(sequence_name, sample["frame_indices"]))
        return sample

    def _load_target_metadata(
        self, sequence_name: str, prepared_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        if self.target_cache_dir is None:
            raise RuntimeError("target_cache_dir is not configured")
        path = self.target_cache_dir / sequence_name / "metadata.json"
        metadata = _read_metadata(path, "M3ED downstream target")
        expected = {
            "format_version": M3ED_DOWNSTREAM_FORMAT_VERSION,
            "dataset": "M3ED",
            "sequence_name": sequence_name,
            "prepared_format_version": M3ED_PREPARED_FORMAT_VERSION,
            "frame_count": int(prepared_metadata["frame_count"]),
            "timestamps": prepared_metadata["timestamps"],
            "coordinate_space": "rectified_left_event",
            "target_size": [self.transform.height, self.transform.width],
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"M3ED downstream {field}={metadata.get(field)!r} does not match "
                    f"{value!r}: {path}"
                )
        missing = sorted(set(self.target_tasks) - set(metadata.get("tasks", [])))
        if missing:
            raise ValueError(
                f"M3ED downstream targets unavailable for {sequence_name}: {', '.join(missing)}"
            )
        targets_path = self.target_cache_dir / sequence_name / "targets.h5"
        success_path = self.target_cache_dir / sequence_name / "_SUCCESS"
        if not targets_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(f"Incomplete M3ED downstream cache: {targets_path}")
        required_by_task = {
            "depth": {"depth_m", "depth_valid", "depth_frame_valid"},
            "semantics": {
                "semantics_19",
                "semantics_11",
                "semantics_frame_valid",
            },
            "pose": {
                "pose_Cn_T_C0",
                "pose_frame_valid",
                "relative_pose_prev_T_current",
                "relative_pose_valid",
                "relative_pose_delta_us",
                "linear_velocity_mps",
                "angular_velocity_radps",
            },
        }
        with h5py.File(targets_path, "r") as source:
            if "timestamps" not in source:
                raise ValueError(f"M3ED downstream HDF5 lacks timestamps: {targets_path}")
            cached_timestamps = np.asarray(source["timestamps"], dtype=np.int64)
            if not np.array_equal(cached_timestamps, prepared_metadata["timestamps"]):
                raise ValueError(f"M3ED downstream HDF5 timestamps mismatch: {targets_path}")
            required = set().union(*(required_by_task[task] for task in self.target_tasks))
            missing_datasets = sorted(required - set(source.keys()))
            if missing_datasets:
                raise ValueError(
                    f"M3ED downstream HDF5 lacks datasets {missing_datasets}: {targets_path}"
                )
        return metadata

    def _load_targets(self, sequence_name: str, frame_indices: Tensor) -> dict[str, Tensor]:
        if self.target_cache_dir is None:
            raise RuntimeError("target_cache_dir is not configured")
        indices = frame_indices.detach().cpu().numpy().astype(np.int64)
        path = self.target_cache_dir / sequence_name / "targets.h5"
        output: dict[str, Tensor] = {}
        with h5py.File(path, "r") as source:
            if "depth" in self.target_tasks:
                output["depth_m"] = torch.from_numpy(np.asarray(source["depth_m"][indices]))
                output["depth_valid"] = torch.from_numpy(
                    np.asarray(source["depth_valid"][indices], dtype=np.bool_)
                )
                output["depth_frame_valid"] = torch.from_numpy(
                    np.asarray(source["depth_frame_valid"][indices], dtype=np.bool_)
                )
            if "semantics" in self.target_tasks:
                output["semantics_19"] = torch.from_numpy(
                    np.asarray(source["semantics_19"][indices], dtype=np.uint8)
                )
                output["semantics_11"] = torch.from_numpy(
                    np.asarray(source["semantics_11"][indices], dtype=np.uint8)
                )
                output["semantics_frame_valid"] = torch.from_numpy(
                    np.asarray(source["semantics_frame_valid"][indices], dtype=np.bool_)
                )
            if "pose" in self.target_tasks:
                for key in (
                    "pose_Cn_T_C0",
                    "pose_frame_valid",
                    "relative_pose_prev_T_current",
                    "relative_pose_valid",
                    "relative_pose_delta_us",
                    "linear_velocity_mps",
                    "angular_velocity_radps",
                ):
                    output[key] = torch.from_numpy(np.asarray(source[key][indices]))
        return output

    def _load_event(self, record: M3EDFrame) -> tuple[Tensor, int]:
        if record.event_path is None:
            raise RuntimeError("The first M3ED frame has no causal event interval")
        payload = _safe_torch_load(record.event_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid M3ED event cache: {record.event_path}")
        expected = {
            "dataset": "M3ED",
            "sequence_name": record.sequence_name,
            "frame_index": record.frame_index,
            "timestamp": record.timestamp,
            "previous_timestamp": record.previous_timestamp,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(f"M3ED event cache {field} mismatch: {record.event_path}")
        tensor = payload.get("events")
        count = payload.get("event_count")
        expected_shape = (
            int(self.event_representation.channels),
            int(self.event_representation.height),
            int(self.event_representation.width),
        )
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape:
            raise ValueError(f"M3ED event tensor shape mismatch: {record.event_path}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"M3ED event_count is invalid: {record.event_path}")
        return tensor.float(), count

    def _load_teacher_feature(self, record: M3EDFrame) -> Tensor:
        if self.feature_cache_dir is None:
            raise RuntimeError("feature_cache_dir is not configured")
        path = self.feature_cache_dir / record.sequence_name / f"{record.timestamp}.pt"
        payload = _safe_torch_load(path)
        if not isinstance(payload, dict) or payload.get("dataset") != "M3ED":
            raise ValueError(f"Invalid M3ED teacher cache: {path}")
        if payload.get("sequence_name") != record.sequence_name:
            raise ValueError(f"M3ED teacher cache sequence mismatch: {path}")
        if payload.get("frame_index") != record.frame_index or payload.get(
            "timestamp"
        ) != record.timestamp:
            raise ValueError(f"M3ED teacher cache frame mismatch: {path}")
        tokens = payload.get("patch_tokens")
        if not isinstance(tokens, Tensor) or tokens.ndim != 2:
            raise ValueError(f"M3ED teacher patch_tokens must have shape [N,D]: {path}")
        expected_patches = (self.transform.height // 16) * (self.transform.width // 16)
        if tokens.shape[0] != expected_patches:
            raise ValueError(f"M3ED teacher patch grid mismatch: {path}")
        return tokens.float()

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0


__all__ = [
    "M3EDFrame",
    "M3EDSequenceDataset",
    "M3ED_PREPARED_FORMAT_VERSION",
    "discover_prepared_m3ed_sequences",
]
