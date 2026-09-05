"""Common temporal-backbone interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from torch import Tensor, nn


TemporalState: TypeAlias = Any


def validate_token_sequence(z: Tensor, *, feature_dim: int, name: str = "z") -> None:
    if not isinstance(z, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if z.ndim != 4:
        raise ValueError(f"{name} must have shape [B, T, N, D], got {tuple(z.shape)}")
    if any(size <= 0 for size in z.shape[:3]):
        raise ValueError(f"{name} batch, time, and token dimensions must be non-empty")
    if z.shape[-1] != feature_dim:
        raise ValueError(f"{name} feature dim must be {feature_dim}, got {z.shape[-1]}")


def detach_temporal_state(state: TemporalState) -> TemporalState:
    """Detach nested tensor state for truncated backpropagation through time."""

    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_temporal_state(item) for item in state)
    if isinstance(state, list):
        return [detach_temporal_state(item) for item in state]
    if isinstance(state, dict):
        return {key: detach_temporal_state(value) for key, value in state.items()}
    raise TypeError(f"Unsupported temporal state type: {type(state).__name__}")


class TemporalBackbone(nn.Module, ABC):
    """Interface preserving the dense patch dimension across time."""

    input_dim: int
    output_dim: int

    @abstractmethod
    def forward(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, TemporalState]:
        raise NotImplementedError

    def forward_step(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, TemporalState]:
        if not isinstance(z, Tensor) or z.ndim != 3:
            shape = tuple(z.shape) if isinstance(z, Tensor) else type(z).__name__
            raise ValueError(f"z must have shape [B, N, D] for a step, got {shape}")
        output, new_state = self(z.unsqueeze(1), state)
        return output[:, 0], new_state

    @staticmethod
    def detach_state(state: TemporalState) -> TemporalState:
        return detach_temporal_state(state)


__all__ = [
    "TemporalBackbone",
    "TemporalState",
    "detach_temporal_state",
    "validate_token_sequence",
]
