"""Losses used by frozen semantic-segmentation probes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def multiclass_dice_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int | None = None,
    smooth: float = 1e-6,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    """Return the mean soft Dice loss over all classes.

    This matches GEP's semantic objective: ignored pixels are removed before
    computing a per-class Dice score, and the class losses are averaged with
    equal weight.
    """

    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected logits [B,C,H,W] and targets [B,H,W]")
    if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("Logits and targets must share batch and spatial dimensions")
    if smooth <= 0:
        raise ValueError("smooth must be positive")

    class_count = int(logits.shape[1])
    flattened_targets = targets.long().reshape(-1)
    valid = torch.ones_like(flattened_targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = flattened_targets != int(ignore_index)
    if not bool(valid.any()):
        raise ValueError("All semantic pixels are ignored; Dice loss is undefined")

    probabilities = torch.softmax(logits, dim=1)
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, class_count)[valid]
    valid_targets = flattened_targets[valid]
    one_hot = F.one_hot(valid_targets, num_classes=class_count).to(probabilities.dtype)
    if pixel_weights is None:
        weights = 1.0
    else:
        if pixel_weights.shape != targets.shape:
            raise ValueError("Dice pixel weights must match targets")
        weights = pixel_weights.reshape(-1)[valid, None].to(probabilities.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("Dice weights must be finite and non-negative")
    intersection = (probabilities * one_hot * weights).sum(dim=0)
    denominator = (probabilities * weights).sum(dim=0) + (one_hot * weights).sum(dim=0)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def activity_cross_entropy(logits: Tensor, targets: Tensor, weights: Tensor,
                           *, ignore_index: int = 255) -> Tensor:
    if weights.shape != targets.shape:
        raise ValueError("CE weights must match targets")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("CE weights must be finite and non-negative")
    valid_weights = weights * (targets != ignore_index)
    values = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
    return (values * valid_weights).sum() / valid_weights.sum().clamp_min(1e-12)


__all__ = ["multiclass_dice_loss"]
