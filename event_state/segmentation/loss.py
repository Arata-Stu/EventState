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
    intersection = (probabilities * one_hot).sum(dim=0)
    denominator = probabilities.sum(dim=0) + one_hot.sum(dim=0)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


__all__ = ["multiclass_dice_loss"]
