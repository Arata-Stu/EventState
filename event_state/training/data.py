"""Dataloader construction and resumable iteration."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from event_state.data import (
    DSECSequenceDataset,
    EventVoxelizer,
    GEPEventFrame,
    PairedSequenceTransform,
)
from event_state.data.cache_metadata import (
    canonical_json_sha256,
    dinov3_checkpoint_identity,
    dinov3_repository_identity,
)


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_optional_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_event_representation(dataset_config: Any) -> Any:
    representation_config = _value(dataset_config, "representation")
    representation_type = str(_value(representation_config, "type", "gep_rgb"))
    height = int(_value(dataset_config, "sensor_height", 480))
    width = int(_value(dataset_config, "sensor_width", 640))
    if representation_type == "gep_rgb":
        return GEPEventFrame(
            height=height,
            width=width,
            percentile=float(_value(representation_config, "percentile", 90.0)),
        )
    if representation_type == "voxel_grid":
        return EventVoxelizer(
            num_bins=int(_value(representation_config, "event_bins", 10)),
            height=height,
            width=width,
            polarity_split=bool(_value(representation_config, "polarity_split", True)),
            normalization=str(
                _value(representation_config, "voxel_normalization", "nonzero_standardize")
            ),
        )
    raise ValueError(
        f"Unsupported dataset.representation.type={representation_type!r}; "
        "expected 'gep_rgb' or 'voxel_grid'"
    )


def _build_transform(dataset_config: Any, *, training: bool, channels: int) -> Any:
    representation_config = _value(dataset_config, "representation")
    augmentation = _value(dataset_config, "augmentation")
    augmentation_enabled = training and bool(_value(augmentation, "enabled", False))
    representation_type = str(_value(representation_config, "type", "gep_rgb"))

    event_mean: tuple[float, ...] | None = None
    event_std: tuple[float, ...] | None = None
    if representation_type == "gep_rgb":
        event_mean = tuple(float(item) for item in _value(representation_config, "normalize_mean"))
        event_std = tuple(float(item) for item in _value(representation_config, "normalize_std"))
        if len(event_mean) != channels or len(event_std) != channels:
            raise ValueError(
                "GEP event normalization length must match the three-channel representation"
            )

    return PairedSequenceTransform(
        height=int(_value(dataset_config, "input_height")),
        width=int(_value(dataset_config, "input_width")),
        training=augmentation_enabled,
        scale=tuple(float(item) for item in _value(augmentation, "scale", (1.0, 1.0))),
        horizontal_flip_probability=(
            float(_value(augmentation, "horizontal_flip_probability", 0.0))
            if augmentation_enabled
            else 0.0
        ),
        event_mean=event_mean,
        event_std=event_std,
    )


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    generator: torch.Generator


def _expected_teacher_identity(teacher_config: Any) -> dict[str, Any]:
    backend = str(_value(teacher_config, "backend", "torch_hub")).lower()
    source = str(_value(teacher_config, "source", "github")).lower()
    configured_repository = str(_value(teacher_config, "repository"))
    if backend == "torch_hub" and source == "local":
        repository = Path(configured_repository).expanduser().resolve().name
    elif backend == "package":
        repository = "dinov3"
    else:
        repository = configured_repository
    repository_identity = dinov3_repository_identity(
        backend=backend,
        source=source,
        repository=configured_repository,
    )
    model_name = str(_value(teacher_config, "type", "dinov3_vits16"))
    checkpoint = _value(teacher_config, "checkpoint", None)
    pretrained = bool(_value(teacher_config, "pretrained", True))
    checkpoint_identity = dinov3_checkpoint_identity(
        checkpoint,
        pretrained=pretrained,
    )
    return {
        "identifier": (
            f"{backend}:{canonical_json_sha256(repository_identity)}:{model_name}"
        ),
        "name": model_name,
        "backend": backend,
        "repository": repository,
        "repository_identity": repository_identity,
        "source": source if backend == "torch_hub" else "installed_package",
        "pretrained": pretrained,
        "checkpoint": checkpoint_identity,
    }


def _validate_teacher_cache(
    dataset: DSECSequenceDataset,
    *,
    teacher_config: Any,
    dataset_config: Any,
) -> None:
    cache_root = dataset.feature_cache_dir
    if cache_root is None:
        return
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"Teacher feature cache not found: {cache_root}. "
            "Run tools/cache_dinov3_features.py first or disable teacher.cache_features."
        )
    expected_height = int(_value(dataset_config, "input_height"))
    expected_width = int(_value(dataset_config, "input_width"))
    expected_patch_size = int(_value(teacher_config, "patch_size", 16))
    expected_patches = (expected_height // expected_patch_size) * (
        expected_width // expected_patch_size
    )
    expected_grid_height = expected_height // expected_patch_size
    expected_grid_width = expected_width // expected_patch_size
    expected_model = str(_value(teacher_config, "type", "dinov3_vits16"))
    expected_embedding_dim = int(_value(teacher_config, "embedding_dim", 384))
    expected_mean = [float(item) for item in _value(teacher_config, "image_mean")]
    expected_std = [float(item) for item in _value(teacher_config, "image_std")]
    expected_identity = _expected_teacher_identity(teacher_config)
    sequence_count = len(dataset.sequence_names)
    print(
        f"[cache-contract] split={dataset.split}: validating {sequence_count} teacher manifests",
        flush=True,
    )
    for sequence_index, sequence_name in enumerate(dataset.sequence_names, start=1):
        print(
            f"[cache-contract] split={dataset.split} sequence={sequence_name} "
            f"({sequence_index}/{sequence_count})",
            flush=True,
        )
        metadata_path = cache_root / sequence_name / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Teacher cache metadata not found: {metadata_path}. Regenerate the cache."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        input_metadata = metadata.get("input", {})
        grid_metadata = metadata.get("grid", {})
        model_metadata = metadata.get("model", {})
        actual_size = (input_metadata.get("height"), input_metadata.get("width"))
        if actual_size != (expected_height, expected_width):
            raise ValueError(
                f"Teacher cache input size {actual_size} does not match "
                f"{(expected_height, expected_width)}: {metadata_path}"
            )
        if int(grid_metadata.get("num_patches", -1)) != expected_patches:
            raise ValueError(f"Teacher cache patch grid is stale: {metadata_path}")
        if int(grid_metadata.get("patch_size", -1)) != expected_patch_size:
            raise ValueError(f"Teacher cache patch size is stale: {metadata_path}")
        if (
            int(grid_metadata.get("height", -1)) != expected_grid_height
            or int(grid_metadata.get("width", -1)) != expected_grid_width
        ):
            raise ValueError(f"Teacher cache grid geometry is stale: {metadata_path}")
        expected_input_contract = {
            "geometry": "deterministic_center_crop_to_aspect_ratio_then_bilinear_resize",
            "color_space": "RGB",
            "value_range": [0.0, 1.0],
        }
        for field, expected_value in expected_input_contract.items():
            if input_metadata.get(field) != expected_value:
                raise ValueError(
                    f"Teacher cache input.{field} differs from the configured pipeline: "
                    f"{metadata_path}"
                )
        if model_metadata.get("name") != expected_model:
            raise ValueError(
                f"Teacher cache model {model_metadata.get('name')!r} does not match "
                f"{expected_model!r}: {metadata_path}"
            )
        if model_metadata.get("embedding_dim") != expected_embedding_dim:
            raise ValueError(f"Teacher cache embedding dimension is stale: {metadata_path}")
        if model_metadata.get("feature") != "x_norm_patchtokens":
            raise ValueError(
                f"Teacher cache does not contain normalized patch tokens: {metadata_path}"
            )
        if model_metadata.get("frozen") is not True:
            raise ValueError(f"Teacher cache was not produced by a frozen teacher: {metadata_path}")
        for field in (
            "identifier",
            "backend",
            "repository",
            "repository_identity",
            "source",
            "pretrained",
        ):
            if model_metadata.get(field) != expected_identity[field]:
                raise ValueError(
                    f"Teacher cache model.{field}={model_metadata.get(field)!r} does not "
                    f"match config value {expected_identity[field]!r}: {metadata_path}"
                )
        cached_checkpoint = model_metadata.get("checkpoint", {})
        expected_checkpoint = expected_identity["checkpoint"]
        if cached_checkpoint != expected_checkpoint:
            raise ValueError(
                f"Teacher cache checkpoint identity differs from the config: {metadata_path}"
            )
        if input_metadata.get("normalization_mean") != expected_mean:
            raise ValueError(f"Teacher cache image mean differs from the config: {metadata_path}")
        if input_metadata.get("normalization_std") != expected_std:
            raise ValueError(f"Teacher cache image std differs from the config: {metadata_path}")
        if metadata.get("cache_dtype") != str(_value(teacher_config, "cache_dtype", "float16")):
            raise ValueError(f"Teacher cache dtype differs from the config: {metadata_path}")
    print(f"[cache-contract] split={dataset.split} complete", flush=True)


def _validate_dataset_location(dataset_config: Any) -> None:
    if str(_value(dataset_config, "name", "dsec")).lower() != "dsec":
        raise ValueError("The Phase 0/1 runtime currently supports only DSEC")
    if _value(dataset_config, "root") in (None, ""):
        raise ValueError("dataset.root must point to the DSEC directory")


def _dataset_options(
    dataset_config: Any,
    teacher_config: Any,
    event_representation: Any,
) -> dict[str, Any]:
    cache_features = bool(_value(teacher_config, "cache_features", True))
    return {
        "root": _value(dataset_config, "root"),
        "sequence_length": int(_value(dataset_config, "sequence_length")),
        "event_representation": event_representation,
        "image_directory": str(_value(dataset_config, "image_directory", "aligned_event")),
        "rectify_events": bool(_value(dataset_config, "rectify_events", True)),
        "event_cache_dir": _value(dataset_config, "event_cache_dir"),
        "feature_cache_dir": (
            _value(teacher_config, "cache_dir") if cache_features else None
        ),
        "load_events": True,
        "load_images": not cache_features,
    }


def _build_validation_dataset(
    config: Any,
    event_representation: Any,
) -> DSECSequenceDataset:
    dataset_config = _value(config, "dataset")
    teacher_config = _value(config, "teacher")
    return DSECSequenceDataset(
        split=str(_value(dataset_config, "val_split", "test")),
        sequences=_as_optional_list(_value(dataset_config, "val_sequences")),
        clip_stride=int(_value(dataset_config, "sequence_length")),
        include_incomplete_clips=True,
        transform=_build_transform(
            dataset_config, training=False, channels=event_representation.channels
        ),
        **_dataset_options(dataset_config, teacher_config, event_representation),
    )


def _loader_options(training_config: Any) -> dict[str, Any]:
    num_workers = int(_value(training_config, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    return {
        "num_workers": num_workers,
        "pin_memory": bool(_value(training_config, "pin_memory", False)),
        "persistent_workers": (
            bool(_value(training_config, "persistent_workers", False)) and num_workers > 0
        ),
        "worker_init_fn": _seed_worker,
    }


def build_validation_dataloader(config: Any) -> DataLoader:
    """Build only chronological validation data for checkpoint evaluation."""

    dataset_config = _value(config, "dataset")
    teacher_config = _value(config, "teacher")
    training_config = _value(config, "training")
    _validate_dataset_location(dataset_config)
    event_representation = build_event_representation(dataset_config)
    validation_dataset = _build_validation_dataset(config, event_representation)
    if bool(_value(teacher_config, "cache_features", True)):
        _validate_teacher_cache(
            validation_dataset,
            teacher_config=teacher_config,
            dataset_config=dataset_config,
        )
    return DataLoader(
        validation_dataset,
        # Chronological state must never be mixed across sequences. A batch of
        # one also lets the final clip of each sequence be shorter than T.
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **_loader_options(training_config),
    )


def build_dataloaders(config: Any) -> DataLoaders:
    dataset_config = _value(config, "dataset")
    teacher_config = _value(config, "teacher")
    training_config = _value(config, "training")
    _validate_dataset_location(dataset_config)

    train_split = str(_value(dataset_config, "train_split", "train"))
    validation_split = str(_value(dataset_config, "val_split", "test"))
    train_sequences = _as_optional_list(_value(dataset_config, "train_sequences"))
    validation_sequences = _as_optional_list(_value(dataset_config, "val_sequences"))
    if train_split == validation_split:
        if train_sequences is None or validation_sequences is None:
            raise ValueError(
                "When train_split and val_split are identical, both sequence manifests "
                "must be explicit to prevent temporal leakage"
            )
        overlap = sorted(set(train_sequences) & set(validation_sequences))
        if overlap:
            raise ValueError(
                "Train/validation sequence manifests overlap: " + ", ".join(overlap)
            )

    event_representation = build_event_representation(dataset_config)
    cache_features = bool(_value(teacher_config, "cache_features", True))
    common = _dataset_options(dataset_config, teacher_config, event_representation)
    train_dataset = DSECSequenceDataset(
        split=train_split,
        sequences=train_sequences,
        clip_stride=int(_value(dataset_config, "clip_stride", 1)),
        include_incomplete_clips=False,
        transform=_build_transform(
            dataset_config, training=True, channels=event_representation.channels
        ),
        **common,
    )
    validation_dataset = _build_validation_dataset(config, event_representation)
    if cache_features:
        _validate_teacher_cache(
            train_dataset,
            teacher_config=teacher_config,
            dataset_config=dataset_config,
        )
        _validate_teacher_cache(
            validation_dataset,
            teacher_config=teacher_config,
            dataset_config=dataset_config,
        )

    batch_size = int(_value(training_config, "batch_size", 1))
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    generator = torch.Generator().manual_seed(int(_value(config, "seed", 0)))
    loader_options = _loader_options(training_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        # Chronological state must never be mixed across sequences. A batch of
        # one also lets the final clip of each sequence be shorter than T.
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    return DataLoaders(train=train_loader, validation=validation_loader, generator=generator)


class CyclingDataIterator:
    """Cycle a shuffled loader while exposing enough position for resume."""

    def __init__(self, loader: DataLoader, *, seed: int, state: Mapping[str, Any] | None = None):
        self.loader = loader
        self.seed = int(seed)
        self.epoch = int((state or {}).get("epoch", 0))
        self.batch_in_epoch = int((state or {}).get("batch_in_epoch", 0))
        self._iterator = None

    def _start_epoch(self) -> None:
        if self.loader.generator is not None:
            self.loader.generator.manual_seed(self.seed + self.epoch)
        self._iterator = iter(self.loader)
        skipped = 0
        while skipped < self.batch_in_epoch:
            try:
                next(self._iterator)
            except StopIteration as error:
                raise ValueError(
                    "Checkpoint dataloader position exceeds the current dataset length"
                ) from error
            skipped += 1

    def __next__(self) -> Any:
        if self._iterator is None:
            self._start_epoch()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_in_epoch = 0
            self._start_epoch()
            batch = next(self._iterator)
        self.batch_in_epoch += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch}
