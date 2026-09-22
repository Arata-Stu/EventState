"""Frozen-feature loading for DSEC-Semantic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


DSEC_SEMANTIC_11_CLASSES = (
    "background",
    "building",
    "fence",
    "person",
    "pole",
    "road",
    "sidewalk",
    "vegetation",
    "car",
    "wall",
    "traffic_sign",
)


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def semantic_physical_split(role: str) -> str:
    if role in {"train", "val"}:
        return "train"
    if role == "test":
        return "test"
    raise ValueError("role must be train, val, or test")


def find_semantic_label_dir(
    labels_root: str | Path,
    *,
    role: str,
    sequence: str,
    num_classes: int = 11,
) -> Path:
    root = Path(labels_root).expanduser()
    split = semantic_physical_split(role)
    candidates = (
        root / split / sequence / f"{num_classes}classes",
        root / split / sequence / "semantic" / "left" / f"{num_classes}classes" / "data",
        root / split / sequence / "semantic" / "left" / f"{num_classes}classes",
    )
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Expected one {num_classes}-class label directory for {sequence}; found {detail}"
        )
    return matches[0]


def load_semantic_label(path: str | Path, *, num_classes: int = 11) -> Tensor:
    label_path = Path(path)
    with Image.open(label_path) as image:
        value = np.asarray(image, dtype=np.uint8).copy()
    if value.ndim != 2:
        raise ValueError(f"Semantic label must be grayscale: {label_path}")
    invalid = (value != 255) & (value >= num_classes)
    if np.any(invalid):
        unexpected = np.unique(value[invalid]).tolist()
        raise ValueError(f"Unexpected semantic IDs {unexpected}: {label_path}")
    return torch.from_numpy(value.astype(np.int64, copy=False))


class DSECSemanticFeatureDataset(Dataset[dict[str, Any]]):
    """Sparse labeled frames backed by frozen z/h feature maps."""

    def __init__(
        self,
        *,
        feature_cache_dir: str | Path,
        labels_root: str | Path,
        sequences: Sequence[str],
        role: str,
        feature: str,
        num_classes: int = 11,
        horizontal_flip_probability: float = 0.0,
        load_activity: bool = False,
    ) -> None:
        if feature not in {"z", "h", "concat"}:
            raise ValueError("feature must be z, h, or concat")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.feature_cache_dir = Path(feature_cache_dir).expanduser()
        self.feature = feature
        self.load_activity = load_activity
        self.num_classes = int(num_classes)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.label_size: tuple[int, int] | None = None
        self.patch_size: int | None = None
        self.samples: list[tuple[str, int, Path, Path]] = []

        for sequence in sequences:
            directory = self.feature_cache_dir / sequence
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Semantic feature metadata not found: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("task") != "dsec_semantic":
                raise ValueError(f"Not a DSEC-Semantic cache: {metadata_path}")
            available = set(metadata.get("features", []))
            required = {"z", "h"} if feature == "concat" else {feature}
            if not required <= available:
                raise ValueError(f"{metadata_path} does not contain {sorted(required)}")
            current_size = metadata.get("label_size")
            if not (
                isinstance(current_size, list)
                and len(current_size) == 2
                and all(isinstance(value, int) and value > 0 for value in current_size)
            ):
                raise ValueError(f"Invalid label_size: {metadata_path}")
            size = (current_size[0], current_size[1])
            if self.label_size is None:
                self.label_size = size
            elif self.label_size != size:
                raise ValueError("Semantic caches use different label sizes")
            patch_size = metadata.get("patch_size")
            if not isinstance(patch_size, int) or patch_size <= 0:
                raise ValueError(f"Invalid patch_size: {metadata_path}")
            if self.patch_size is None:
                self.patch_size = patch_size
            elif self.patch_size != patch_size:
                raise ValueError("Semantic caches use different patch sizes")
            label_dir = find_semantic_label_dir(
                labels_root,
                role=role,
                sequence=sequence,
                num_classes=num_classes,
            )
            for feature_path in sorted(directory.glob("*.pt")):
                if not feature_path.stem.isdecimal():
                    continue
                frame_index = int(feature_path.stem)
                label_path = label_dir / f"{frame_index:06d}.png"
                if label_path.is_file():
                    self.samples.append((sequence, frame_index, feature_path, label_path))
        if not self.samples:
            raise ValueError("No labeled DSEC-Semantic feature frames were found")
        if self.label_size is None or self.patch_size is None:
            raise RuntimeError("Semantic feature geometry was not initialized")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, frame_index, feature_path, label_path = self.samples[index]
        payload = _safe_torch_load(feature_path)
        if not isinstance(payload, dict) or payload.get("sequence_name") != sequence:
            raise ValueError(f"Invalid semantic feature payload: {feature_path}")
        if int(payload.get("frame_index", -1)) != frame_index:
            raise ValueError(f"Semantic feature frame mismatch: {feature_path}")
        features = payload.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Semantic payload lacks features: {feature_path}")
        if self.feature == "concat":
            value = torch.cat((features["z"], features["h"]), dim=0)
        else:
            value = features[self.feature]
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Feature must have shape [C, H, W]: {feature_path}")
        label = load_semantic_label(label_path, num_classes=self.num_classes)
        if tuple(label.shape) != self.label_size:
            raise ValueError(f"Label size mismatch: {label_path}")
        activity = None
        if self.load_activity:
            from event_state.losses.activity_weighting import require_activity

            activity = require_activity(payload, value.shape[-2:])
            # Expand at native patch stride, then crop padding; never stretch 448 to 440.
            activity = F.interpolate(activity[None, None].float(), scale_factor=self.patch_size,
                                     mode="nearest")[0, 0].bool()
            activity = activity[:label.shape[0], :label.shape[1]]
            if activity.shape != label.shape:
                raise ValueError("Activity mask does not cover semantic labels")
        if torch.rand(()) < self.horizontal_flip_probability:
            value = value.flip(-1)
            label = label.flip(-1)
            if activity is not None:
                activity = activity.flip(-1)
        return {
            **({"event_activity": activity} if activity is not None else {}),
            "feature": value.float(),
            "label": label,
            "sequence_name": sequence,
            "frame_index": frame_index,
            "timestamp": int(payload.get("timestamp", -1)),
        }


def segmentation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **({"event_activity": torch.stack([sample["event_activity"] for sample in batch])}
           if "event_activity" in batch[0] else {}),
        "features": torch.stack([sample["feature"] for sample in batch]),
        "labels": torch.stack([sample["label"] for sample in batch]),
        "sequence_names": [sample["sequence_name"] for sample in batch],
        "frame_indices": [sample["frame_index"] for sample in batch],
        "timestamps": [sample["timestamp"] for sample in batch],
    }


__all__ = [
    "DSEC_SEMANTIC_11_CLASSES",
    "DSECSemanticFeatureDataset",
    "find_semantic_label_dir",
    "load_semantic_label",
    "segmentation_collate",
    "semantic_physical_split",
]
