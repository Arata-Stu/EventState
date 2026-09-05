"""Hydra-configured factories for the Phase 0/1 runtime."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import DataLoaders, build_dataloaders, build_validation_dataloader
from .optim import WarmupCosineScheduler, build_optimizer


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_device(requested: str | None) -> torch.device:
    requested = "auto" if requested is None else str(requested)
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: Any) -> None:
    dataset = _value(config, "dataset")
    model = _value(config, "model")
    teacher = _value(config, "teacher")
    loss = _value(config, "loss")
    training = _value(config, "training")
    scheduler = _value(config, "scheduler")
    representation = _value(dataset, "representation")
    event_encoder = _value(model, "event_encoder")
    temporal = _value(model, "temporal")
    projector = _value(model, "projector")

    representation_type = str(_value(representation, "type", "gep_rgb"))
    if representation_type == "gep_rgb":
        expected_channels = 3
    elif representation_type == "voxel_grid":
        if not bool(_value(representation, "polarity_split", True)):
            raise ValueError("Phase 0/1 voxel_grid requires polarity_split=true")
        expected_channels = 2 * int(_value(representation, "event_bins", 10))
    else:
        raise ValueError("dataset.representation.type must be gep_rgb or voxel_grid")
    declared_channels = int(_value(representation, "channels", expected_channels))
    if declared_channels != expected_channels:
        raise ValueError(
            f"dataset.representation.channels={declared_channels} but "
            f"{representation_type} produces {expected_channels} channels"
        )
    encoder_channels = int(_value(event_encoder, "in_channels"))
    if encoder_channels != expected_channels:
        raise ValueError(
            f"model.event_encoder.in_channels={encoder_channels} but "
            f"{representation_type} produces {expected_channels} channels"
        )
    patch_init = str(_value(event_encoder, "patch_init", "pretrained"))
    if encoder_channels != 3 and patch_init == "pretrained":
        raise ValueError(
            "A non-RGB event encoder cannot copy the 3-channel patch embedding; "
            "set model.event_encoder.patch_init=random or rgb_mean"
        )

    height = int(_value(dataset, "input_height"))
    width = int(_value(dataset, "input_width"))
    patch_size = int(_value(teacher, "patch_size", 16))
    if height % patch_size or width % patch_size:
        raise ValueError("DSEC input height and width must be divisible by the teacher patch size")
    teacher_dim = int(_value(teacher, "embedding_dim", 384))
    if int(_value(projector, "output_dim")) != teacher_dim:
        raise ValueError("model.projector.output_dim must match teacher.embedding_dim")
    if int(_value(temporal, "output_dim")) != int(_value(projector, "input_dim")):
        raise ValueError("temporal.output_dim must match projector.input_dim")
    if not bool(_value(teacher, "frozen", True)):
        raise ValueError("Phase 0/1 requires teacher.frozen=true")
    if not bool(_value(teacher, "pretrained", True)):
        raise ValueError("A randomly initialized RGB teacher is not a valid distillation target")
    if str(_value(teacher, "feature", "x_norm_patchtokens")) != "x_norm_patchtokens":
        raise ValueError("Phase 0/1 alignment requires DINOv3 x_norm_patchtokens")
    if str(_value(dataset, "event_window", "rgb_interval")) != "rgb_interval":
        raise ValueError("Phase 0/1 requires event_window=rgb_interval")
    if not bool(_value(dataset, "rectify_events", True)):
        raise ValueError("Phase 0/1 requires rectified events for RGB patch alignment")
    if str(_value(dataset, "image_directory", "aligned_event")) == "rectified":
        raise ValueError(
            "DSEC's native rectified RGB camera is not in event-camera coordinates; "
            "run tools/prepare_dsec.py and use image_directory=aligned_event"
        )
    image_mean = tuple(_value(teacher, "image_mean"))
    image_std = tuple(_value(teacher, "image_std"))
    if len(image_mean) != 3 or len(image_std) != 3 or any(float(item) <= 0 for item in image_std):
        raise ValueError("teacher image_mean/image_std must contain three values and positive std")

    augmentation = _value(dataset, "augmentation")
    cache_features = bool(_value(teacher, "cache_features", True))
    if cache_features and _value(teacher, "cache_dir") in (None, ""):
        raise ValueError(
            "teacher.cache_dir must be set when teacher.cache_features=true"
        )
    if cache_features and bool(_value(augmentation, "enabled", False)):
        raise ValueError(
            "Cached DINO features are incompatible with stochastic spatial augmentation"
        )

    h_config = _value(loss, "h_distill")
    z_config = _value(loss, "z_objective")
    h_enabled = bool(_value(h_config, "enabled", False))
    z_type = str(_value(z_config, "type", "none")).lower()
    if z_type not in {"none", "direct_dino"}:
        raise NotImplementedError(
            "Phase 0/1 supports z_objective.type=none or direct_dino"
        )
    if not h_enabled and z_type == "none":
        raise ValueError("At least one of h distillation or direct z distillation must be enabled")
    h_weights = (
        float(_value(h_config, "cosine_weight", 0.0)),
        float(_value(h_config, "mse_weight", 0.0)),
    )
    z_weights = (
        float(_value(z_config, "cosine_weight", 0.0)),
        float(_value(z_config, "mse_weight", 0.0)),
    )
    z_scale = float(_value(z_config, "weight", 1.0))
    if any(weight < 0 for weight in (*h_weights, *z_weights, z_scale)):
        raise ValueError("Distillation weights must be non-negative")
    if h_enabled and not any(h_weights):
        raise ValueError("Enabled h distillation requires a positive cosine or MSE weight")
    if z_type == "direct_dino" and (not any(z_weights) or z_scale == 0):
        raise ValueError("Direct z distillation requires positive loss and objective weights")

    positive_fields = {
        "training.max_steps": int(_value(training, "max_steps")),
        "training.batch_size": int(_value(training, "batch_size")),
        "training.gradient_accumulation": int(
            _value(training, "gradient_accumulation", 1)
        ),
        "training.log_every": int(_value(training, "log_every", 1)),
        "training.validate_every": int(_value(training, "validate_every", 1)),
        "training.checkpoint_every": int(_value(training, "checkpoint_every", 1)),
        "dataset.sequence_length": int(_value(dataset, "sequence_length")),
    }
    invalid = [name for name, value in positive_fields.items() if value <= 0]
    if invalid:
        raise ValueError("These values must be positive: " + ", ".join(invalid))
    warmup_steps = int(_value(scheduler, "warmup_steps", 0))
    if warmup_steps < 0 or warmup_steps >= positive_fields["training.max_steps"]:
        raise ValueError("scheduler.warmup_steps must be in [0, training.max_steps)")
    if float(_value(training, "gradient_clip", 0.0)) < 0:
        raise ValueError("training.gradient_clip must be non-negative")
    validation_batches = _value(training, "validation_batches", None)
    if validation_batches is not None and int(validation_batches) <= 0:
        raise ValueError("training.validation_batches must be positive or null")
    evaluation = _value(config, "evaluation", {})
    evaluation_batches = _value(evaluation, "max_batches", None)
    if evaluation_batches is not None and int(evaluation_batches) <= 0:
        raise ValueError("evaluation.max_batches must be positive or null")
    thresholds = _value(evaluation, "event_count_thresholds", None)
    if thresholds is not None:
        if len(thresholds) != 2 or float(thresholds[0]) > float(thresholds[1]):
            raise ValueError(
                "evaluation.event_count_thresholds must be [low_max, high_min] in order"
            )


@dataclass(frozen=True)
class RuntimeComponents:
    model: nn.Module
    teacher: nn.Module | None
    dataloaders: DataLoaders
    optimizer: torch.optim.Optimizer
    scheduler: WarmupCosineScheduler
    h_distillation_loss: nn.Module
    z_distillation_loss: nn.Module
    device: torch.device


@dataclass(frozen=True)
class EvaluationComponents:
    model: nn.Module
    teacher: nn.Module | None
    validation_loader: Any
    h_distillation_loss: nn.Module
    z_distillation_loss: nn.Module
    device: torch.device


def _build_model(config: Any) -> nn.Module:
    from event_state.models import (
        EventStateModel,
        TeacherProjection,
        build_event_encoder,
        build_temporal_backbone,
    )

    model = _value(config, "model")
    event_encoder_config = _value(model, "event_encoder")
    temporal_config = _value(model, "temporal")
    projector_config = _value(model, "projector")
    teacher_config = _value(config, "teacher")
    event_encoder = build_event_encoder(
        in_channels=int(_value(event_encoder_config, "in_channels")),
        patch_init=str(_value(event_encoder_config, "patch_init", "pretrained")),
        pretrained=bool(_value(event_encoder_config, "pretrained_init", True)),
        checkpoint=_value(
            event_encoder_config,
            "checkpoint",
            _value(teacher_config, "checkpoint", None),
        ),
        source=str(_value(event_encoder_config, "backend", _value(teacher_config, "backend"))),
        repository=str(
            _value(event_encoder_config, "repository", _value(teacher_config, "repository"))
        ),
        hub_source=str(_value(event_encoder_config, "source", _value(teacher_config, "source"))),
        backbone=str(_value(event_encoder_config, "architecture", "dinov3_vits16")),
    )
    temporal_model = build_temporal_backbone(
        kind=str(_value(temporal_config, "type")),
        input_dim=int(_value(temporal_config, "input_dim")),
        hidden_dim=int(_value(temporal_config, "hidden_dim", _value(temporal_config, "input_dim"))),
        output_dim=int(_value(temporal_config, "output_dim")),
        num_layers=int(_value(temporal_config, "num_layers", 1)),
        dropout=float(_value(temporal_config, "dropout", 0.0)),
    )
    projector_kwargs = {
        "input_dim": int(_value(projector_config, "input_dim")),
        "hidden_dim": int(_value(projector_config, "hidden_dim")),
        "output_dim": int(_value(projector_config, "output_dim")),
    }
    return EventStateModel(
        event_encoder=event_encoder,
        temporal_model=temporal_model,
        h_projector=TeacherProjection(**projector_kwargs),
        z_projector=TeacherProjection(**projector_kwargs),
    )


def _build_teacher(config: Any) -> nn.Module | None:
    teacher_config = _value(config, "teacher")
    if bool(_value(teacher_config, "cache_features", True)):
        return None
    from event_state.models import build_dinov3_teacher

    return build_dinov3_teacher(
        checkpoint=_value(teacher_config, "checkpoint", None),
        pretrained=bool(_value(teacher_config, "pretrained", True)),
        source=str(_value(teacher_config, "backend", "torch_hub")),
        repository=str(_value(teacher_config, "repository", "facebookresearch/dinov3")),
        hub_source=str(_value(teacher_config, "source", "github")),
        backbone=str(_value(teacher_config, "type", "dinov3_vits16")),
        image_mean=tuple(float(item) for item in _value(teacher_config, "image_mean")),
        image_std=tuple(float(item) for item in _value(teacher_config, "image_std")),
    )


def _build_losses(config: Any) -> tuple[nn.Module, nn.Module]:
    from event_state.models import DistillationLoss

    h_config = _value(_value(config, "loss"), "h_distill")
    z_config = _value(_value(config, "loss"), "z_objective")
    return (
        DistillationLoss(
            cosine_weight=float(_value(h_config, "cosine_weight", 1.0)),
            mse_weight=float(_value(h_config, "mse_weight", 0.0)),
        ),
        DistillationLoss(
            cosine_weight=float(_value(z_config, "cosine_weight", 1.0)),
            mse_weight=float(_value(z_config, "mse_weight", 0.0)),
        ),
    )


def build_runtime(config: Any) -> RuntimeComponents:
    validate_config(config)
    seed = int(_value(config, "seed", 0))
    seed_everything(seed)
    device = resolve_device(_value(config, "device", "auto"))
    dataloaders = build_dataloaders(config)
    model = _build_model(config).to(device)
    teacher = _build_teacher(config)
    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()
    optimizer = build_optimizer(model, _value(config, "optimizer"))
    training = _value(config, "training")
    scheduler_config = _value(config, "scheduler")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(_value(scheduler_config, "warmup_steps", 0)),
        total_steps=int(_value(training, "max_steps")),
        min_lr_ratio=float(_value(scheduler_config, "min_lr_ratio", 0.0)),
    )
    h_distillation_loss, z_distillation_loss = _build_losses(config)
    return RuntimeComponents(
        model=model,
        teacher=teacher,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        h_distillation_loss=h_distillation_loss,
        z_distillation_loss=z_distillation_loss,
        device=device,
    )


def build_evaluation_runtime(config: Any) -> EvaluationComponents:
    """Build a validation-only runtime without touching the training split."""

    validate_config(config)
    seed_everything(int(_value(config, "seed", 0)))
    device = resolve_device(_value(config, "device", "auto"))
    validation_loader = build_validation_dataloader(config)
    model = _build_model(config).to(device)
    teacher = _build_teacher(config)
    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()
    h_distillation_loss, z_distillation_loss = _build_losses(config)
    return EvaluationComponents(
        model=model,
        teacher=teacher,
        validation_loader=validation_loader,
        h_distillation_loss=h_distillation_loss,
        z_distillation_loss=z_distillation_loss,
        device=device,
    )
