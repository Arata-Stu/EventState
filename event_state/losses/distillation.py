"""Dense feature distillation losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_pair(prediction: Tensor, target: Tensor) -> None:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.ndim < 2:
        raise ValueError("prediction and target must have shape [..., N, D]")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[-1] <= 0:
        raise ValueError("feature dimension must be non-empty")
    if prediction.device != target.device:
        raise ValueError(
            f"prediction and target must share a device, got "
            f"{prediction.device} and {target.device}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point tensors")


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a torch.Tensor or None")
    if mask.device != values.device:
        raise ValueError("mask must be on the same device as prediction and target")
    try:
        broadcast_mask = torch.broadcast_to(mask, values.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(values.shape)}"
        ) from error
    weights = broadcast_mask.to(dtype=values.dtype)
    denominator = weights.sum()
    if not bool((denominator > 0).item()):
        raise ValueError("mask must select at least one token")
    return (values * weights).sum() / denominator


def cosine_distillation_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Mean patch-wise cosine distance ``1 - cosine``."""

    _validate_pair(prediction, target)
    if eps <= 0:
        raise ValueError("eps must be positive")
    prediction = F.normalize(prediction, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    distance = 1.0 - (prediction * target).sum(dim=-1)
    return _masked_mean(distance, mask)


def normalized_mse_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Mean patch-wise squared L2 distance after per-token normalization."""

    _validate_pair(prediction, target)
    if eps <= 0:
        raise ValueError("eps must be positive")
    prediction = F.normalize(prediction, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    # The blueprint defines a squared L2 norm for each patch, followed by a
    # mean over patches.  Do not divide this term by the embedding dimension.
    per_token = (prediction - target).square().sum(dim=-1)
    return _masked_mean(per_token, mask)


class DistillationLoss(nn.Module):
    """Weighted cosine and normalized-MSE dense-token alignment."""

    def __init__(
        self,
        cosine_weight: float = 1.0,
        mse_weight: float = 1.0,
        eps: float = 1e-6,
        detach_target: bool = True,
    ) -> None:
        super().__init__()
        if cosine_weight < 0 or mse_weight < 0:
            raise ValueError("distillation weights must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.cosine_weight = float(cosine_weight)
        self.mse_weight = float(mse_weight)
        self.eps = float(eps)
        self.detach_target = bool(detach_target)

    def components(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        if self.detach_target:
            target = target.detach()
        _validate_pair(prediction, target)
        cosine = cosine_distillation_loss(
            prediction, target, mask=mask, eps=self.eps
        )
        mse = normalized_mse_loss(prediction, target, mask=mask, eps=self.eps)
        loss = self.cosine_weight * cosine + self.mse_weight * mse
        # Keep a differentiable scalar even when both ablation weights are zero.
        if self.cosine_weight == 0.0 and self.mse_weight == 0.0:
            loss = prediction.sum() * 0.0
        return {"loss": loss, "cosine": cosine, "mse": mse}

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.components(prediction, target, mask)["loss"]


def distillation_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    cosine_weight: float = 1.0,
    mse_weight: float = 1.0,
    mask: Tensor | None = None,
    eps: float = 1e-6,
    detach_target: bool = True,
) -> Tensor:
    criterion = DistillationLoss(
        cosine_weight=cosine_weight,
        mse_weight=mse_weight,
        eps=eps,
        detach_target=detach_target,
    )
    return criterion(prediction, target, mask)


__all__ = [
    "DistillationLoss",
    "cosine_distillation_loss",
    "distillation_loss",
    "normalized_mse_loss",
]
