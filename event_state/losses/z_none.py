"""No direct objective on instantaneous event features."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class ZObjective(nn.Module):
    """Base interface for research-time objectives applied to ``z``."""

    def forward(
        self,
        z: Tensor,
        teacher_features: Tensor | None = None,
        *,
        events: Tensor | None = None,
        metadata: Any = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError


class NoZObjective(ZObjective):
    """Return a differentiable zero; z learns only through downstream h loss."""

    def forward(
        self,
        z: Tensor,
        teacher_features: Tensor | None = None,
        *,
        events: Tensor | None = None,
        metadata: Any = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        del teacher_features, events, metadata, mask
        if not isinstance(z, Tensor):
            raise TypeError("z must be a torch.Tensor")
        if z.ndim < 2:
            raise ValueError("z must have shape [..., N, D]")
        return z.sum() * 0.0


NoneZObjective = NoZObjective
ZNoneObjective = NoZObjective


__all__ = ["NoZObjective", "NoneZObjective", "ZNoneObjective", "ZObjective"]
