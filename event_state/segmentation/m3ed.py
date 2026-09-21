"""Frozen-feature semantic evaluation data for prepared M3ED recordings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml


@dataclass(frozen=True)
class M3EDSemanticSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]

    def sequences(self, role: str) -> tuple[str, ...]:
        if role not in {"train", "validation"}:
            raise ValueError("role must be train or validation")
        return getattr(self, role)


def load_m3ed_semantic_split(path: str | Path) -> M3EDSemanticSplit:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"M3ED downstream split not found: {manifest_path}")
    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    semantic = value.get("semantic") if isinstance(value, dict) else None
    if not isinstance(semantic, dict):
        raise ValueError("M3ED downstream manifest lacks a semantic section")
    sections: dict[str, tuple[str, ...]] = {}
    for role in ("train", "validation"):
        entries = semantic.get(role)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"M3ED semantic {role} split must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in entries):
            raise ValueError(f"M3ED semantic {role} contains an invalid sequence")
        if len(entries) != len(set(entries)):
            raise ValueError(f"M3ED semantic {role} contains duplicates")
        sections[role] = tuple(entries)
    if set(sections["train"]) & set(sections["validation"]):
        raise ValueError("M3ED semantic train/validation sequences overlap")
    return M3EDSemanticSplit(**sections)


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class M3EDSemanticFeatureDataset(Dataset[dict[str, Any]]):
    """Labeled M3ED frames backed by sparse frozen z/h feature maps."""

    def __init__(
        self,
        *,
        feature_cache_dir: str | Path,
        target_cache_dir: str | Path,
        sequences: Sequence[str],
        feature: str,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        if feature not in {"z", "h", "concat"}:
            raise ValueError("feature must be z, h, or concat")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal-flip-probability must be in [0, 1]")
        self.feature = feature
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.label_size: tuple[int, int] | None = None
        self.patch_size: int | None = None
        self.samples: list[tuple[str, int, Path, Path]] = []
        feature_root = Path(feature_cache_dir).expanduser()
        target_root = Path(target_cache_dir).expanduser()

        for sequence in sequences:
            feature_dir = feature_root / sequence
            metadata_path = feature_dir / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid M3ED semantic feature metadata: {metadata_path}") from error
            if metadata.get("task") != "m3ed_semantic":
                raise ValueError(f"Not an M3ED semantic feature cache: {metadata_path}")
            required = {"z", "h"} if feature == "concat" else {feature}
            if not required <= set(metadata.get("features", [])):
                raise ValueError(f"{metadata_path} does not contain {sorted(required)}")
            size_value = metadata.get("label_size")
            if not (
                isinstance(size_value, list)
                and len(size_value) == 2
                and all(isinstance(value, int) and value > 0 for value in size_value)
            ):
                raise ValueError(f"Invalid label_size: {metadata_path}")
            size = (int(size_value[0]), int(size_value[1]))
            if self.label_size is None:
                self.label_size = size
            elif self.label_size != size:
                raise ValueError("M3ED semantic caches use different label sizes")
            patch_size = int(metadata.get("patch_size", 0))
            if patch_size <= 0:
                raise ValueError(f"Invalid patch_size: {metadata_path}")
            if self.patch_size is None:
                self.patch_size = patch_size
            elif self.patch_size != patch_size:
                raise ValueError("M3ED semantic caches use different patch sizes")

            targets_path = target_root / sequence / "targets.h5"
            if not targets_path.is_file():
                raise FileNotFoundError(f"M3ED semantic targets not found: {targets_path}")
            with h5py.File(targets_path, "r") as source:
                for key in ("semantics_11", "semantics_frame_valid", "timestamps"):
                    if key not in source:
                        raise ValueError(f"{targets_path} lacks {key}")
                labels = source["semantics_11"]
                valid = np.asarray(source["semantics_frame_valid"], dtype=np.bool_)
                if labels.ndim != 3 or len(labels) != len(valid):
                    raise ValueError(f"Invalid semantic target geometry: {targets_path}")
                if tuple(labels.shape[1:]) != size:
                    raise ValueError(f"Semantic label size mismatch: {targets_path}")
            for frame_index in metadata.get("cached_frame_indices", []):
                index = int(frame_index)
                if index < 0 or index >= len(valid) or not bool(valid[index]):
                    continue
                feature_path = feature_dir / f"{index:06d}.pt"
                if not feature_path.is_file():
                    raise FileNotFoundError(f"Cached feature missing: {feature_path}")
                self.samples.append((sequence, index, feature_path, targets_path))

        if not self.samples:
            raise ValueError("No labeled M3ED semantic feature frames were found")
        if self.label_size is None or self.patch_size is None:
            raise RuntimeError("M3ED semantic feature geometry was not initialized")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, frame_index, feature_path, targets_path = self.samples[index]
        payload = _safe_torch_load(feature_path)
        if not isinstance(payload, dict) or payload.get("sequence_name") != sequence:
            raise ValueError(f"Invalid M3ED semantic feature payload: {feature_path}")
        if int(payload.get("frame_index", -1)) != frame_index:
            raise ValueError(f"M3ED semantic frame mismatch: {feature_path}")
        features = payload.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"M3ED semantic payload lacks features: {feature_path}")
        if self.feature == "concat":
            value = torch.cat((features["z"], features["h"]), dim=0)
        else:
            value = features[self.feature]
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"Feature must have shape [C,H,W]: {feature_path}")
        with h5py.File(targets_path, "r") as source:
            label = torch.from_numpy(
                np.asarray(source["semantics_11"][frame_index], dtype=np.int64)
            )
            timestamp = int(source["timestamps"][frame_index])
        if tuple(label.shape) != self.label_size:
            raise ValueError(f"Semantic label size mismatch: {targets_path}")
        if torch.rand(()) < self.horizontal_flip_probability:
            value = value.flip(-1)
            label = label.flip(-1)
        return {
            "feature": value.float(),
            "label": label,
            "sequence_name": sequence,
            "frame_index": frame_index,
            "timestamp": timestamp,
        }


__all__ = [
    "M3EDSemanticFeatureDataset",
    "M3EDSemanticSplit",
    "load_m3ed_semantic_split",
]
