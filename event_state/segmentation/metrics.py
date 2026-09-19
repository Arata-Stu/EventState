"""Streaming semantic-segmentation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


class SemanticSegmentationEvaluator:
    def __init__(self, num_classes: int, *, ignore_index: int = 255) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits_or_labels: Tensor, targets: Tensor) -> None:
        predictions = (
            logits_or_labels.argmax(dim=1)
            if logits_or_labels.ndim == targets.ndim + 1
            else logits_or_labels
        )
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have matching spatial shape")
        predictions = predictions.detach().to("cpu", torch.int64).reshape(-1)
        targets = targets.detach().to("cpu", torch.int64).reshape(-1)
        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )
        predictions = predictions[valid]
        targets = targets[valid]
        valid_prediction = (predictions >= 0) & (predictions < self.num_classes)
        predictions = predictions[valid_prediction]
        targets = targets[valid_prediction]
        encoded = targets * self.num_classes + predictions
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | list[float]]:
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        ground_truth = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        union = ground_truth + predicted - true_positive
        valid = union > 0
        iou = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        iou[valid] = true_positive[valid] / union[valid]
        accuracy = true_positive.sum() / confusion.sum().clamp_min(1)
        class_accuracy = torch.full_like(iou, float("nan"))
        present = ground_truth > 0
        class_accuracy[present] = true_positive[present] / ground_truth[present]
        return {
            "mIoU": float(iou[valid].mean()) if valid.any() else float("nan"),
            "pixel_accuracy": float(accuracy),
            "mean_class_accuracy": (
                float(class_accuracy[present].mean()) if present.any() else float("nan")
            ),
            "class_IoU": [float(value) for value in iou],
            "evaluated_pixels": int(confusion.sum()),
        }


__all__ = ["SemanticSegmentationEvaluator"]

