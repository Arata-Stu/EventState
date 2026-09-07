"""Feature-cache and label loading for DSEC-Detection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


DSEC_DETECTION_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "train",
)
DAGR_CLASSES = ("car", "pedestrian")
DAGR_CLASS_ID_MAP = {0: 1, 2: 0, 3: 0, 4: 0}


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_tracks_file(labels_root: str | Path, sequence: str) -> Path:
    root = Path(labels_root).expanduser()
    candidates = []
    for physical_split in ("train", "test"):
        candidates.extend(
            [
                root / physical_split / sequence / "object_detections" / "left" / "tracks.npy",
                root
                / f"{physical_split}_object_detections"
                / sequence
                / "object_detections"
                / "left"
                / "tracks.npy",
            ]
        )
    candidates.append(root / sequence / "object_detections" / "left" / "tracks.npy")
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Expected exactly one tracks.npy for {sequence!r} under {root}; found {detail}"
        )
    return matches[0]


def find_rectify_map(dataset_root: str | Path, sequence: str) -> Path:
    root = Path(dataset_root).expanduser()
    candidates = [
        root / "train_events" / sequence / "events" / "left" / "rectify_map.h5",
        root / "test_events" / sequence / "events" / "left" / "rectify_map.h5",
        root
        / "dsec_det_extra"
        / "train"
        / sequence
        / "events"
        / "left"
        / "rectify_map.h5",
        root
        / "dsec_det_extra"
        / "test"
        / sequence
        / "events"
        / "left"
        / "rectify_map.h5",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Expected exactly one rectify_map.h5 for {sequence!r}; found {detail}"
        )
    return matches[0]


def find_image_timestamps(dataset_root: str | Path, sequence: str) -> Path:
    root = Path(dataset_root).expanduser()
    candidates = [
        root / "train_images" / sequence / "images" / "timestamps.txt",
        root / "test_images" / sequence / "images" / "timestamps.txt",
        root / "dsec_det_extra" / "train" / sequence / "images" / "timestamps.txt",
        root / "dsec_det_extra" / "test" / sequence / "images" / "timestamps.txt",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Expected exactly one timestamps.txt for {sequence!r}; found {detail}"
        )
    return matches[0]


def _sample_box_perimeter(
    box: np.ndarray, samples_per_edge: int = 17
) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = (float(value) for value in box)
    x2 = x + width
    y2 = y + height
    horizontal = np.linspace(x, x2, samples_per_edge, dtype=np.float32)
    vertical = np.linspace(y, y2, samples_per_edge, dtype=np.float32)
    xs = np.concatenate(
        (horizontal, horizontal, np.full_like(vertical, x), np.full_like(vertical, x2))
    )
    ys = np.concatenate(
        (np.full_like(horizontal, y), np.full_like(horizontal, y2), vertical, vertical)
    )
    return xs, ys


def rectify_xywh_boxes(boxes: np.ndarray, rectify_map: np.ndarray) -> np.ndarray:
    """Map distorted DSEC-Det boxes to rectified EventState coordinates."""

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4] in xywh format")
    if rectify_map.ndim != 3 or rectify_map.shape[2] != 2:
        raise ValueError("rectify_map must have shape [H, W, 2]")
    height, width = rectify_map.shape[:2]
    output = np.empty_like(boxes, dtype=np.float32)
    for index, box in enumerate(boxes):
        xs, ys = _sample_box_perimeter(box)
        xi = np.clip(np.rint(xs).astype(np.int64), 0, width - 1)
        yi = np.clip(np.rint(ys).astype(np.int64), 0, height - 1)
        mapped = rectify_map[yi, xi]
        mapped = mapped[np.isfinite(mapped).all(axis=1)]
        if len(mapped) == 0:
            output[index] = np.array([math.nan] * 4, dtype=np.float32)
            continue
        x1, y1 = mapped.min(axis=0)
        x2, y2 = mapped.max(axis=0)
        output[index] = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    return output


def transform_xywh_boxes_to_input(
    boxes: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Apply the deterministic center crop/resize used by EventState validation."""

    source_height, source_width = source_size
    target_height, target_width = target_size
    target_ratio = target_width / target_height
    crop_width = min(
        source_width,
        max(1, round(math.sqrt(source_height * source_width * target_ratio))),
    )
    crop_height = min(source_height, max(1, round(crop_width / target_ratio)))
    if crop_height > source_height:
        crop_height = source_height
        crop_width = min(source_width, max(1, round(crop_height * target_ratio)))
    top = (source_height - crop_height) // 2
    left = (source_width - crop_width) // 2
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height
    output = boxes.astype(np.float32, copy=True)
    output[:, 0] = (output[:, 0] - left) * scale_x
    output[:, 1] = (output[:, 1] - top) * scale_y
    output[:, 2] *= scale_x
    output[:, 3] *= scale_y
    return output


def load_sequence_targets(
    *,
    labels_root: str | Path,
    dataset_root: str | Path,
    sequence: str,
    target_size: tuple[int, int] = (448, 640),
    min_box_side: float = 20.0,
    min_box_diagonal: float = 30.0,
) -> dict[int, dict[str, Tensor]]:
    """Load DAGR-compatible two-class targets indexed by timestamp."""

    tracks_path = find_tracks_file(labels_root, sequence)
    tracks = np.load(tracks_path, allow_pickle=False)
    required = {"t", "x", "y", "w", "h", "class_id"}
    fields = set(tracks.dtype.names or ())
    if not required <= fields:
        raise ValueError(f"{tracks_path} lacks track fields: {sorted(required - fields)}")
    with h5py.File(find_rectify_map(dataset_root, sequence), "r") as handle:
        if "rectify_map" not in handle:
            raise KeyError("rectify_map dataset is missing")
        rectify_map = np.asarray(handle["rectify_map"], dtype=np.float32)
    result: dict[int, dict[str, Tensor]] = {}
    for timestamp in np.unique(tracks["t"]):
        selected = tracks[tracks["t"] == timestamp]
        mapped_ids = np.array(
            [DAGR_CLASS_ID_MAP.get(int(value), -1) for value in selected["class_id"]],
            dtype=np.int64,
        )
        keep_class = mapped_ids >= 0
        if not np.any(keep_class):
            continue
        selected = selected[keep_class]
        mapped_ids = mapped_ids[keep_class]
        xywh = np.stack(
            [selected["x"], selected["y"], selected["w"], selected["h"]], axis=1
        ).astype(np.float32)
        xywh = rectify_xywh_boxes(xywh, rectify_map)
        xywh = transform_xywh_boxes_to_input(
            xywh,
            source_size=rectify_map.shape[:2],
            target_size=target_size,
        )
        target_height, target_width = target_size
        x1 = np.clip(xywh[:, 0], 0, target_width - 1)
        y1 = np.clip(xywh[:, 1], 0, target_height - 1)
        x2 = np.clip(xywh[:, 0] + xywh[:, 2], 0, target_width - 1)
        y2 = np.clip(xywh[:, 1] + xywh[:, 3], 0, target_height - 1)
        widths = x2 - x1
        heights = y2 - y1
        diagonals = np.sqrt(widths**2 + heights**2)
        keep = (
            np.isfinite(x1 + y1 + x2 + y2)
            & (widths > min_box_side)
            & (heights > min_box_side)
            & (diagonals > min_box_diagonal)
        )
        if not np.any(keep):
            continue
        boxes = np.stack((x1[keep], y1[keep], x2[keep], y2[keep]), axis=1)
        result[int(timestamp)] = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.from_numpy(mapped_ids[keep]),
        }
    return result


def load_dagr_sequence_targets(
    *,
    labels_root: str | Path,
    dataset_root: str | Path,
    sequence: str,
    scale: int = 2,
    cropped_height: int = 430,
    sensor_width: int = 640,
    min_box_side: float = 10.0,
    min_box_diagonal: float = 15.0,
) -> dict[int, dict[str, Tensor]]:
    """Load the exact frame subset and distorted coordinates used by DAGR."""

    if scale <= 0 or cropped_height <= 0 or sensor_width <= 0:
        raise ValueError("DAGR geometry values must be positive")
    tracks_path = find_tracks_file(labels_root, sequence)
    tracks = np.load(tracks_path, allow_pickle=False)
    required = {"t", "x", "y", "w", "h", "class_id"}
    fields = set(tracks.dtype.names or ())
    if not required <= fields:
        raise ValueError(f"{tracks_path} lacks track fields: {sorted(required - fields)}")

    width = sensor_width // scale
    height = cropped_height // scale
    mapped_ids = np.array(
        [DAGR_CLASS_ID_MAP.get(int(value), -1) for value in tracks["class_id"]],
        dtype=np.int64,
    )
    x1 = np.clip(tracks["x"].astype(np.float32) / scale, 0, width - 1)
    y1 = np.clip(tracks["y"].astype(np.float32) / scale, 0, height - 1)
    x2 = np.clip(
        (tracks["x"] + tracks["w"]).astype(np.float32) / scale,
        0,
        width - 1,
    )
    y2 = np.clip(
        (tracks["y"] + tracks["h"]).astype(np.float32) / scale,
        0,
        height - 1,
    )
    widths = x2 - x1
    heights = y2 - y1
    track_timestamps = tracks["t"].astype(np.int64)
    keep = (
        (mapped_ids >= 0)
        & np.isfinite(x1 + y1 + x2 + y2)
        & (widths > min_box_side)
        & (heights > min_box_side)
        & (np.sqrt(widths**2 + heights**2) > min_box_diagonal)
    )

    image_timestamps = np.atleast_1d(
        np.loadtxt(find_image_timestamps(dataset_root, sequence), dtype=np.int64)
    )
    if image_timestamps.ndim != 1 or len(image_timestamps) < 2:
        raise ValueError(f"Invalid DSEC image timestamps for {sequence}")
    valid_timestamps = np.unique(track_timestamps[keep])
    valid_indices = np.nonzero(np.isin(image_timestamps, valid_timestamps))[0]
    # DAGR constructs pairs only where two valid image indices are consecutive,
    # consumes events after the first image, and evaluates the second label.
    evaluated_indices = valid_indices[1:][np.diff(valid_indices) == 1]

    result: dict[int, dict[str, Tensor]] = {}
    for image_index in evaluated_indices:
        timestamp = int(image_timestamps[image_index])
        selected = keep & (track_timestamps == timestamp)
        boxes = np.stack((x1[selected], y1[selected], x2[selected], y2[selected]), axis=1)
        result[timestamp] = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.from_numpy(mapped_ids[selected]),
        }
    return result


def build_dagr_sampling_grid(
    rectify_map: np.ndarray,
    *,
    source_input_size: tuple[int, int],
    source_stride: int,
    scale: int = 2,
    cropped_height: int = 430,
    sensor_width: int = 640,
) -> Tensor:
    """Map a rectified EventState token map onto the DAGR distorted grid."""

    if rectify_map.ndim != 3 or rectify_map.shape[2] != 2:
        raise ValueError("rectify_map must have shape [H, W, 2]")
    if source_stride <= 0 or source_stride % scale:
        raise ValueError("source_stride must be positive and divisible by scale")
    source_height, source_width = rectify_map.shape[:2]
    target_height, target_width = source_input_size
    target_ratio = target_width / target_height
    crop_width = min(
        source_width,
        max(1, round(math.sqrt(source_height * source_width * target_ratio))),
    )
    crop_height = min(source_height, max(1, round(crop_width / target_ratio)))
    if crop_height > source_height:
        crop_height = source_height
        crop_width = min(source_width, max(1, round(crop_height * target_ratio)))
    top = (source_height - crop_height) // 2
    left = (source_width - crop_width) // 2
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height

    detector_stride = source_stride // scale
    detector_height = cropped_height // scale
    detector_width = sensor_width // scale
    grid_height = math.ceil(detector_height / detector_stride)
    grid_width = math.ceil(detector_width / detector_stride)
    ys = (np.arange(grid_height, dtype=np.float32) + 0.5) * detector_stride * scale
    xs = (np.arange(grid_width, dtype=np.float32) + 0.5) * detector_stride * scale
    raw_y = np.clip(np.rint(ys).astype(np.int64), 0, source_height - 1)
    raw_x = np.clip(np.rint(xs).astype(np.int64), 0, source_width - 1)
    mapped = rectify_map[raw_y[:, None], raw_x[None, :]].astype(np.float32)
    transformed_x = (mapped[..., 0] - left) * scale_x
    transformed_y = (mapped[..., 1] - top) * scale_y
    normalized_x = 2.0 * transformed_x / target_width - 1.0
    normalized_y = 2.0 * transformed_y / target_height - 1.0
    return torch.from_numpy(np.stack((normalized_x, normalized_y), axis=-1))


class DSECDetectionFeatureDataset(Dataset[dict[str, Any]]):
    """Frames with DSEC-Det labels backed by frozen EventState feature caches."""

    def __init__(
        self,
        *,
        feature_cache_dir: str | Path,
        labels_root: str | Path,
        dataset_root: str | Path,
        sequences: Sequence[str],
        feature: str,
        protocol: str = "probe",
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        if feature not in {"z", "h", "concat"}:
            raise ValueError("feature must be z, h, or concat")
        if protocol not in {"probe", "dsec-det"}:
            raise ValueError("protocol must be probe or dsec-det")
        self.feature_cache_dir = Path(feature_cache_dir).expanduser()
        self.feature = feature
        self.protocol = protocol
        self.input_size: tuple[int, int] | None = None
        self.patch_size: int | None = None
        self.sampling_grids: dict[str, Tensor] = {}
        source_input_size: tuple[int, int] | None = None
        source_patch_size: int | None = None
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.samples: list[tuple[str, int, Path, dict[str, Tensor]]] = []
        for sequence in sequences:
            directory = self.feature_cache_dir / sequence
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Detection feature metadata not found: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            input_size = metadata.get("input_size")
            if (
                not isinstance(input_size, list)
                or len(input_size) != 2
                or any(not isinstance(value, int) or value <= 0 for value in input_size)
            ):
                raise ValueError(f"Invalid detection feature input_size: {metadata_path}")
            current_input_size = (input_size[0], input_size[1])
            if source_input_size is None:
                source_input_size = current_input_size
            elif source_input_size != current_input_size:
                raise ValueError("Detection feature sequences use different input sizes")
            patch_size = metadata.get("patch_size")
            if not isinstance(patch_size, int) or patch_size <= 0:
                raise ValueError(f"Invalid detection feature patch_size: {metadata_path}")
            if source_patch_size is None:
                source_patch_size = patch_size
            elif source_patch_size != patch_size:
                raise ValueError("Detection feature sequences use different patch sizes")
            available = set(metadata.get("features", []))
            required_features = {"z", "h"} if feature == "concat" else {feature}
            if not required_features <= available:
                raise ValueError(
                    f"{metadata_path} does not contain {sorted(required_features)}"
                )
            if protocol == "dsec-det":
                targets = load_dagr_sequence_targets(
                    labels_root=labels_root,
                    dataset_root=dataset_root,
                    sequence=sequence,
                )
                coordinate_space = metadata.get("coordinate_space", "rectified_event")
                if coordinate_space == "dsec_det_distorted":
                    if current_input_size != (215, 320) or patch_size != 8:
                        raise ValueError(
                            f"Invalid prewarped DSEC-Det geometry: {metadata_path}"
                        )
                elif coordinate_space == "rectified_event":
                    with h5py.File(find_rectify_map(dataset_root, sequence), "r") as handle:
                        rectify_map = np.asarray(handle["rectify_map"], dtype=np.float32)
                    self.sampling_grids[sequence] = build_dagr_sampling_grid(
                        rectify_map,
                        source_input_size=current_input_size,
                        source_stride=patch_size,
                    )
                else:
                    raise ValueError(
                        f"Unsupported feature coordinate_space={coordinate_space!r}: "
                        f"{metadata_path}"
                    )
            else:
                targets = load_sequence_targets(
                    labels_root=labels_root,
                    dataset_root=dataset_root,
                    sequence=sequence,
                    target_size=current_input_size,
                )
            for path in sorted(directory.glob("*.pt")):
                if not path.stem.isdecimal():
                    continue
                timestamp = int(path.stem)
                if timestamp in targets:
                    self.samples.append((sequence, timestamp, path, targets[timestamp]))
        if not self.samples:
            raise ValueError("No labeled DSEC-Detection feature frames were found")
        if source_input_size is None or source_patch_size is None:
            raise RuntimeError("Detection feature geometry was not initialized")
        if protocol == "dsec-det":
            self.input_size = (430 // 2, 640 // 2)
            self.patch_size = 8
        else:
            self.input_size = source_input_size
            self.patch_size = source_patch_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, timestamp, path, target = self.samples[index]
        payload = _safe_torch_load(path)
        if not isinstance(payload, dict) or payload.get("sequence_name") != sequence:
            raise ValueError(f"Invalid detection feature payload: {path}")
        if int(payload.get("timestamp", -1)) != timestamp:
            raise ValueError(f"Detection feature timestamp mismatch: {path}")
        features = payload.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Detection feature payload lacks features: {path}")
        if self.feature == "concat":
            value = torch.cat((features["z"], features["h"]), dim=0)
        else:
            value = features[self.feature]
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Feature must have shape [C, H, W]: {path}")
        if self.protocol == "dsec-det" and sequence in self.sampling_grids:
            value = value.float()
            grid = self.sampling_grids[sequence]
            value = F.grid_sample(
                value.unsqueeze(0),
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0]
        expected_grid = (
            math.ceil(self.input_size[0] / self.patch_size),
            math.ceil(self.input_size[1] / self.patch_size),
        )
        if tuple(value.shape[-2:]) != expected_grid:
            raise ValueError(
                f"Feature grid {tuple(value.shape[-2:])} does not match metadata "
                f"{expected_grid}: {path}"
            )
        output_target = {key: tensor.clone() for key, tensor in target.items()}
        if torch.rand(()) < self.horizontal_flip_probability:
            value = value.flip(-1)
            image_width = float(self.input_size[1])
            boxes = output_target["boxes"]
            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()
            boxes[:, 0] = image_width - old_x2
            boxes[:, 2] = image_width - old_x1
        return {
            "feature": value.float(),
            "target": output_target,
            "sequence_name": sequence,
            "timestamp": timestamp,
        }


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([sample["feature"] for sample in batch]),
        "targets": [sample["target"] for sample in batch],
        "sequence_names": [sample["sequence_name"] for sample in batch],
        "timestamps": [sample["timestamp"] for sample in batch],
    }


__all__ = [
    "DAGR_CLASSES",
    "DAGR_CLASS_ID_MAP",
    "DSEC_DETECTION_CLASSES",
    "DSECDetectionFeatureDataset",
    "detection_collate",
    "find_rectify_map",
    "find_image_timestamps",
    "find_tracks_file",
    "build_dagr_sampling_grid",
    "load_dagr_sequence_targets",
    "load_sequence_targets",
    "rectify_xywh_boxes",
    "transform_xywh_boxes_to_input",
]
