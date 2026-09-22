"""Native implementation of ScaleEvent's released CrossGram objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distillation import _validate_pair


class ScaleEventLoss(nn.Module):
    """Masked L1 + 10 intra-modal MSE + 4 cross-modal MSE.

    Means include masked-out elements, exactly as in the released criterion.
    Each frame is independent; time is never treated as an extra token axis.
    Row chunks limit temporary Gram matrix size without sampling token pairs.
    """

    def __init__(self, row_chunk_size: int = 128) -> None:
        super().__init__()
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")
        self.row_chunk_size = row_chunk_size

    def components(self, prediction: Tensor, target: Tensor,
                   mask: Tensor | None = None) -> dict[str, Tensor]:
        _validate_pair(prediction, target)
        prediction, target = prediction.float(), target.detach().float()
        if mask is not None:
            weights = torch.broadcast_to(mask, prediction.shape[:-1]).unsqueeze(-1)
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("ScaleEvent mask weights must be finite and non-negative")
            prediction, target = prediction * weights, target * weights
        l1 = F.l1_loss(prediction, target)
        student = F.normalize(prediction, dim=-1, eps=1e-12)
        teacher = F.normalize(target, dim=-1, eps=1e-12)
        n = student.shape[-2]
        if n == 0:
            raise ValueError("ScaleEvent requires non-empty tokens")
        intra, cross = student.sum() * 0.0, student.sum() * 0.0
        # Reference orientation: teacher @ student.T, gated by teacher Gram > .1.
        for start in range(0, n, self.row_chunk_size):
            q = teacher[..., start:start + self.row_chunk_size, :]
            k = student[..., start:start + self.row_chunk_size, :]
            reference = (q @ teacher.transpose(-1, -2)).clamp_min(0)
            keep = reference > 0.1
            reference = reference * keep
            event_gram = (k @ student.transpose(-1, -2)).clamp_min(0) * keep
            cross_gram = (q @ student.transpose(-1, -2)).clamp_min(0) * keep
            fraction = q.shape[-2] / n
            intra = intra + F.mse_loss(event_gram, reference) * fraction
            cross = cross + F.mse_loss(cross_gram, reference) * fraction
        return {"loss": l1 + 10 * intra + 4 * cross,
                "l1": l1, "intra": intra, "cross": cross}

    def forward(self, prediction: Tensor, target: Tensor,
                mask: Tensor | None = None) -> Tensor:
        return self.components(prediction, target, mask)["loss"]
