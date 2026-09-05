"""Optimizer parameter groups and a resumable warmup-cosine schedule."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _component_for_parameter(name: str) -> str:
    if name.startswith("event_encoder.") or ".event_encoder." in name:
        return "event_encoder"
    if (
        name.startswith("temporal_model.")
        or name.startswith("temporal.")
        or ".temporal_model." in name
    ):
        return "temporal"
    if "projector" in name:
        return "projector"
    return "other"


def _uses_weight_decay(name: str, parameter: Tensor) -> bool:
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return False
    leaf_name = name.rsplit(".", 1)[-1]
    return leaf_name not in {
        "cls_token",
        "mask_token",
        "pos_embed",
        "storage_tokens",
    }


def build_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    """Build AdamW groups with component LR multipliers and proper no-decay sets."""

    optimizer_type = str(_value(config, "type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_type}; the MVP supports adamw")
    base_lr = float(_value(config, "lr"))
    weight_decay = float(_value(config, "weight_decay", 0.0))
    if base_lr <= 0 or weight_decay < 0:
        raise ValueError("optimizer.lr must be positive and weight_decay must be non-negative")

    multipliers = {
        "event_encoder": float(_value(config, "event_encoder_lr_mult", 1.0)),
        "temporal": float(_value(config, "temporal_lr_mult", 1.0)),
        "projector": float(_value(config, "projector_lr_mult", 1.0)),
        "other": 1.0,
    }
    if any(multiplier < 0 for multiplier in multipliers.values()):
        raise ValueError("Optimizer LR multipliers must be non-negative")

    grouped: dict[tuple[str, bool], list[Tensor]] = defaultdict(list)
    parameter_names: dict[int, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in parameter_names:
            raise ValueError(
                f"Trainable parameter is registered twice: {parameter_names[identifier]} and {name}"
            )
        parameter_names[identifier] = name
        component = _component_for_parameter(name)
        grouped[(component, _uses_weight_decay(name, parameter))].append(parameter)
    if not grouped:
        raise ValueError("The model has no trainable parameters")

    parameter_groups = []
    for (component, decay), parameters in sorted(grouped.items()):
        multiplier = multipliers[component]
        parameter_groups.append(
            {
                "params": parameters,
                "lr": base_lr * multiplier,
                "initial_lr": base_lr * multiplier,
                "weight_decay": weight_decay if decay else 0.0,
                "lr_mult": multiplier,
                "group_name": f"{component}/{'decay' if decay else 'no_decay'}",
            }
        )
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        betas=(
            float(_value(config, "beta1", 0.9)),
            float(_value(config, "beta2", 0.999)),
        ),
        eps=float(_value(config, "eps", 1e-8)),
    )


class WarmupCosineScheduler:
    """Step-based schedule whose state is independent of epoch length."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.num_updates = 0
        self.base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
        self._apply(self.num_updates)

    def _multiplier(self, update: int) -> float:
        if self.warmup_steps and update < self.warmup_steps:
            return (update + 1) / self.warmup_steps
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (update - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, update: int) -> None:
        multiplier = self._multiplier(update)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        self.num_updates += 1
        self._apply(self.num_updates)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "num_updates": self.num_updates,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        for name in ("warmup_steps", "total_steps", "min_lr_ratio"):
            expected = getattr(self, name)
            restored = state_dict[name]
            if restored != expected:
                raise ValueError(
                    f"Scheduler {name} mismatch: checkpoint has {restored}, config has {expected}"
                )
        restored_lrs = [float(value) for value in state_dict["base_lrs"]]
        if len(restored_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler parameter-group count differs from the checkpoint")
        self.base_lrs = restored_lrs
        self.num_updates = int(state_dict["num_updates"])
        self._apply(self.num_updates)
