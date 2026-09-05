"""Small metric accumulators used by both training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


class WeightedMean:
    """Accumulate a scalar mean without bias from short final batches."""

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: float | Tensor, weight: float | int = 1.0) -> None:
        scalar = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
        self.total += scalar * float(weight)
        self.weight += float(weight)

    @property
    def mean(self) -> float:
        return self.total / self.weight if self.weight else float("nan")


def token_cosine(student: Tensor, target: Tensor) -> Tensor:
    """Return one cosine-similarity value per leading token."""

    if student.shape != target.shape:
        raise ValueError(
            f"Alignment tensors must have the same shape, got {student.shape} and {target.shape}"
        )
    if student.ndim < 2:
        raise ValueError("Alignment tensors must include a feature dimension")
    student = F.normalize(student.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    return (student * target).sum(dim=-1)


def frame_cosine(student: Tensor, target: Tensor) -> Tensor:
    """Average dense-token cosine similarity into ``[B, T]`` frame scores."""

    similarities = token_cosine(student, target)
    if similarities.ndim != 3:
        raise ValueError(
            "Sequence alignment expects tensors [B, T, N, D], "
            f"got token scores {similarities.shape}"
        )
    return similarities.mean(dim=-1)


@dataclass
class EventCountAnalysis:
    """Collect frame-level alignment scores and report event-count tertiles."""

    counts: list[Tensor] = field(default_factory=list)
    z_cosines: list[Tensor] = field(default_factory=list)
    h_cosines: list[Tensor] = field(default_factory=list)
    z_projected_cosines: list[Tensor] = field(default_factory=list)
    h_projected_cosines: list[Tensor] = field(default_factory=list)

    def update(
        self,
        event_counts: Tensor,
        z_cosine: Tensor,
        h_cosine: Tensor,
        z_projected_cosine: Tensor | None = None,
        h_projected_cosine: Tensor | None = None,
    ) -> None:
        if event_counts.shape != z_cosine.shape or event_counts.shape != h_cosine.shape:
            raise ValueError(
                "event_counts, z_cosine, and h_cosine must all have shape [B, T]"
            )
        if (z_projected_cosine is None) != (h_projected_cosine is None):
            raise ValueError("Projected z/h cosine values must be provided together")
        if z_projected_cosine is not None and (
            z_projected_cosine.shape != event_counts.shape
            or h_projected_cosine.shape != event_counts.shape
        ):
            raise ValueError("Projected cosine values must have shape [B, T]")
        self.counts.append(event_counts.detach().reshape(-1).cpu().to(torch.float64))
        self.z_cosines.append(z_cosine.detach().reshape(-1).cpu().to(torch.float64))
        self.h_cosines.append(h_cosine.detach().reshape(-1).cpu().to(torch.float64))
        if z_projected_cosine is not None and h_projected_cosine is not None:
            self.z_projected_cosines.append(
                z_projected_cosine.detach().reshape(-1).cpu().to(torch.float64)
            )
            self.h_projected_cosines.append(
                h_projected_cosine.detach().reshape(-1).cpu().to(torch.float64)
            )

    def compute(self, thresholds: tuple[float, float] | None = None) -> dict[str, float]:
        if not self.counts:
            return {}
        counts = torch.cat(self.counts)
        z_cosines = torch.cat(self.z_cosines)
        h_cosines = torch.cat(self.h_cosines)
        z_projected_cosines = (
            torch.cat(self.z_projected_cosines) if self.z_projected_cosines else None
        )
        h_projected_cosines = (
            torch.cat(self.h_projected_cosines) if self.h_projected_cosines else None
        )
        valid = torch.isfinite(counts) & (counts >= 0)
        counts, z_cosines, h_cosines = counts[valid], z_cosines[valid], h_cosines[valid]
        if z_projected_cosines is not None and h_projected_cosines is not None:
            z_projected_cosines = z_projected_cosines[valid]
            h_projected_cosines = h_projected_cosines[valid]
        if counts.numel() == 0:
            return {}

        if thresholds is None:
            quantiles = torch.quantile(counts, counts.new_tensor([1.0 / 3.0, 2.0 / 3.0]))
            low_max, high_min = float(quantiles[0]), float(quantiles[1])
        else:
            low_max, high_min = map(float, thresholds)
            if low_max > high_min:
                raise ValueError("The low event-count threshold must not exceed the high threshold")

        masks = {
            "low": counts <= low_max,
            "medium": (counts > low_max) & (counts <= high_min),
            "high": counts > high_min,
        }
        result = {
            "event_count/low_max": low_max,
            "event_count/high_min": high_min,
            "event_count/mean": float(counts.mean()),
        }
        for label, mask in masks.items():
            result[f"event_count/{label}_frames"] = float(mask.sum())
            if torch.any(mask):
                result[f"alignment/{label}_z_cosine"] = float(z_cosines[mask].mean())
                result[f"alignment/{label}_h_cosine"] = float(h_cosines[mask].mean())
                if z_projected_cosines is not None and h_projected_cosines is not None:
                    result[f"alignment/{label}_z_projected_cosine"] = float(
                        z_projected_cosines[mask].mean()
                    )
                    result[f"alignment/{label}_h_projected_cosine"] = float(
                        h_projected_cosines[mask].mean()
                    )
        return result


def global_gradient_norm(parameters: Iterable[Tensor]) -> float:
    """Compute an L2 gradient norm without modifying gradients."""

    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2)
        squared_norm += float(grad_norm * grad_norm)
    return squared_norm**0.5
