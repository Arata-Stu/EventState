"""Atomic, complete checkpoints for exact-enough training continuation."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .optim import WarmupCosineScheduler


CHECKPOINT_SCHEMA_VERSION = 1
CONFIG_SIGNATURE_FORMAT_VERSION = 1

_RESUME_DATASET_RUNTIME_FIELDS = {"root", "event_cache_dir"}
_EVALUATION_DATASET_RUNTIME_FIELDS = {
    *_RESUME_DATASET_RUNTIME_FIELDS,
    "train_split",
    "val_split",
    "train_sequences",
    "val_sequences",
    "augmentation",
    "event_window_fraction",
}
_TEACHER_RUNTIME_FIELDS = {"cache_dir"}
_TRAINING_RUNTIME_FIELDS = {
    "resume",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "log_every",
    "validate_every",
    "checkpoint_every",
    "validation_batches",
}


@dataclass(frozen=True)
class CheckpointState:
    global_step: int
    best_validation_loss: float
    data_state: dict[str, Any]
    config: Any = None


@dataclass(frozen=True)
class CheckpointConfigMetadata:
    config: dict[str, Any]
    signatures: dict[str, Any]


def _plain_value(value: Any) -> Any:
    """Convert config containers into deterministic Python values."""

    if isinstance(value, MappingABC):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    return value


def _filtered_section(config: Mapping[str, Any], name: str, excluded: set[str]) -> Any:
    section = config.get(name)
    if not isinstance(section, MappingABC):
        return _plain_value(section)
    return {
        str(key): _plain_value(value)
        for key, value in section.items()
        if str(key) not in excluded
    }


@lru_cache(maxsize=None)
def _repository_identity(backend: str, source: str, repository: str) -> dict[str, Any]:
    # Imported lazily so checkpoint unit tests without DSEC data do not pay the
    # source-tree hashing/import cost unless a DINO source is actually configured.
    from event_state.data.cache_metadata import dinov3_repository_identity

    return dinov3_repository_identity(
        backend=backend,
        source=source,
        repository=repository,
    )


@lru_cache(maxsize=None)
def _checkpoint_identity(checkpoint: str | None, pretrained: bool) -> dict[str, Any]:
    from event_state.data.cache_metadata import dinov3_checkpoint_identity

    return dinov3_checkpoint_identity(checkpoint, pretrained=pretrained)


def _canonicalize_dino_source(
    section: Any,
    *,
    pretrained_key: str,
) -> Any:
    if not isinstance(section, dict):
        return section
    result = dict(section)
    backend = result.get("backend")
    repository = result.get("repository")
    source = result.get("source")
    if backend is not None and repository is not None:
        normalized_backend = str(backend).lower()
        normalized_source = str(source or "github").lower()
        result["repository_identity"] = _repository_identity(
            normalized_backend,
            normalized_source,
            str(repository),
        )
        result.pop("repository", None)
        result["backend"] = normalized_backend
        result["source"] = (
            normalized_source if normalized_backend == "torch_hub" else "installed_package"
        )
    if "checkpoint" in result:
        pretrained = bool(result.get(pretrained_key, True))
        checkpoint = result.get("checkpoint")
        result["checkpoint"] = _checkpoint_identity(
            None if checkpoint is None else str(checkpoint),
            pretrained,
        )
    return result


def critical_config(config: Any, *, mode: str = "resume") -> dict[str, Any]:
    """Return the checkpoint fields that define a comparable experiment.

    Dataset/cache locations, logging cadence, worker settings, and output paths
    may move between machines. Model, objective, representation, teacher
    identity, and optimization semantics may not silently change on resume.
    Evaluation additionally permits choosing another split/sequence manifest.
    """

    plain = _plain_value(config)
    if not isinstance(plain, MappingABC):
        raise ValueError("Checkpoint config must be a mapping")
    if mode not in {"resume", "evaluation"}:
        raise ValueError("Checkpoint config compatibility mode must be resume or evaluation")

    dataset_excluded = (
        _RESUME_DATASET_RUNTIME_FIELDS
        if mode == "resume"
        else _EVALUATION_DATASET_RUNTIME_FIELDS
    )
    teacher_section = _filtered_section(plain, "teacher", _TEACHER_RUNTIME_FIELDS)
    model = _plain_value(plain.get("model"))
    if isinstance(model, dict) and isinstance(model.get("event_encoder"), dict):
        model = dict(model)
        event_encoder = dict(model["event_encoder"])
        if isinstance(teacher_section, dict):
            for field in ("backend", "source", "repository", "checkpoint"):
                if field not in event_encoder and field in teacher_section:
                    event_encoder[field] = teacher_section[field]
        model["event_encoder"] = _canonicalize_dino_source(
            event_encoder, pretrained_key="pretrained_init"
        )
    teacher = _canonicalize_dino_source(
        teacher_section,
        pretrained_key="pretrained",
    )
    signature = {
        "model": model,
        "teacher": teacher,
        "dataset": _filtered_section(plain, "dataset", dataset_excluded),
        "loss": _plain_value(plain.get("loss")),
    }
    if mode == "resume":
        signature.update(
            {
                "seed": _plain_value(plain.get("seed")),
                "optimizer": _plain_value(plain.get("optimizer")),
                "scheduler": _plain_value(plain.get("scheduler")),
                "training": _filtered_section(
                    plain, "training", _TRAINING_RUNTIME_FIELDS
                ),
            }
        )
    return signature


def _config_differences(saved: Any, current: Any, *, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved:
                differences.append(f"{path}: absent in checkpoint")
            elif key not in current:
                differences.append(f"{path}: absent in current config")
            else:
                differences.extend(
                    _config_differences(saved[key], current[key], prefix=path)
                )
        return differences
    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            return [f"{prefix}: length {len(saved)} != {len(current)}"]
        differences = []
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            differences.extend(
                _config_differences(
                    saved_item,
                    current_item,
                    prefix=f"{prefix}[{index}]",
                )
            )
        return differences
    if saved != current:
        return [f"{prefix}: checkpoint={saved!r}, current={current!r}"]
    return []


def assert_checkpoint_config_compatible(
    saved_config: Any,
    current_config: Any,
    *,
    mode: str = "resume",
) -> None:
    """Fail before state restoration when experiment-defining config changed."""

    if saved_config is None:
        raise ValueError(
            "Checkpoint does not contain a resolved config; experiment compatibility "
            "cannot be verified"
        )
    saved = critical_config(saved_config, mode=mode)
    current = critical_config(current_config, mode=mode)
    differences = _config_differences(saved, current)
    if differences:
        preview = "; ".join(differences[:8])
        remainder = len(differences) - 8
        if remainder > 0:
            preview += f"; and {remainder} more difference(s)"
        raise ValueError(f"Checkpoint critical config mismatch ({mode}): {preview}")


def assert_checkpoint_signature_compatible(
    saved_signature: Any,
    current_config: Any,
    *,
    mode: str,
) -> None:
    """Compare a stored content-aware signature with a current config."""

    saved = _plain_value(saved_signature)
    if not isinstance(saved, MappingABC):
        raise ValueError(f"Checkpoint does not contain a valid {mode} config signature")
    current = critical_config(current_config, mode=mode)
    differences = _config_differences(saved, current)
    if differences:
        preview = "; ".join(differences[:8])
        remainder = len(differences) - 8
        if remainder > 0:
            preview += f"; and {remainder} more difference(s)"
        raise ValueError(f"Checkpoint critical config mismatch ({mode}): {preview}")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a checkpoint beside its destination, then atomically replace it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Any,
    global_step: int,
    best_validation_loss: float,
    data_state: Mapping[str, Any] | None,
    config: Any,
) -> Path:
    """Save all trainable/runtime state; the frozen teacher is intentionally absent."""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "best_validation_loss": float(best_validation_loss),
        "data_state": dict(data_state or {}),
        "rng_state": _rng_state(),
        "config": config,
        "config_signatures": {
            "format_version": CONFIG_SIGNATURE_FORMAT_VERSION,
            "resume": critical_config(config, mode="resume"),
            "evaluation": critical_config(config, mode="evaluation"),
        },
    }
    return atomic_torch_save(payload, path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device)


def _load_checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was available.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, MappingABC):
        raise TypeError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    if int(checkpoint.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {checkpoint.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    return checkpoint


def _assert_expected_config(
    checkpoint: Mapping[str, Any],
    expected_config: Any,
    *,
    compatibility_mode: str,
) -> None:
    signatures = checkpoint.get("config_signatures")
    if isinstance(signatures, MappingABC) and compatibility_mode in signatures:
        if signatures.get("format_version") != CONFIG_SIGNATURE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported checkpoint config signature format "
                f"{signatures.get('format_version')!r}"
            )
        assert_checkpoint_signature_compatible(
            signatures[compatibility_mode],
            expected_config,
            mode=compatibility_mode,
        )
        return
    assert_checkpoint_config_compatible(
        checkpoint.get("config"),
        expected_config,
        mode=compatibility_mode,
    )


def load_checkpoint_config(
    path: str | Path,
    *,
    expected_config: Any = None,
    compatibility_mode: str = "resume",
) -> Any:
    """Read and validate the resolved config before constructing a runtime."""

    checkpoint = _load_checkpoint_payload(path)
    if expected_config is not None:
        _assert_expected_config(
            checkpoint,
            expected_config,
            compatibility_mode=compatibility_mode,
        )
    config = checkpoint.get("config")
    if not isinstance(config, MappingABC):
        raise ValueError("Checkpoint does not contain a resolved mapping config")
    return _plain_value(config)


def load_checkpoint_config_metadata(path: str | Path) -> CheckpointConfigMetadata:
    """Read resolved config plus content-aware signatures in one checkpoint pass."""

    checkpoint = _load_checkpoint_payload(path)
    config = checkpoint.get("config")
    if not isinstance(config, MappingABC):
        raise ValueError("Checkpoint does not contain a resolved mapping config")
    signatures = checkpoint.get("config_signatures", {})
    if not isinstance(signatures, MappingABC):
        raise ValueError("Checkpoint config_signatures must be a mapping")
    if signatures and signatures.get("format_version") != CONFIG_SIGNATURE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint config signature format "
            f"{signatures.get('format_version')!r}"
        )
    return CheckpointConfigMetadata(
        config=_plain_value(config),
        signatures=_plain_value(signatures),
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    scaler: Any = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
    expected_config: Any = None,
    compatibility_mode: str = "resume",
) -> CheckpointState:
    checkpoint = _load_checkpoint_payload(path)
    if expected_config is not None:
        _assert_expected_config(
            checkpoint,
            expected_config,
            compatibility_mode=compatibility_mode,
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to_device(optimizer, torch.device(device))
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"])
    return CheckpointState(
        global_step=int(checkpoint.get("global_step", 0)),
        best_validation_loss=float(checkpoint.get("best_validation_loss", float("inf"))),
        data_state=dict(checkpoint.get("data_state", {})),
        config=checkpoint.get("config"),
    )


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    checkpoint_directory = Path(directory)
    candidates = sorted(checkpoint_directory.glob("step_*.pt"))
    return candidates[-1] if candidates else None
