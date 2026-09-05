"""Spatially coupled transforms for event/RGB sequences."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PairedSequenceTransform:
    """Apply one sampled geometry to every event and RGB frame in a clip."""

    height: int
    width: int
    training: bool = False
    scale: tuple[float, float] = (1.0, 1.0)
    horizontal_flip_probability: float = 0.0
    event_mean: tuple[float, ...] | None = None
    event_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Target height and width must be positive")
        if not (0 < self.scale[0] <= self.scale[1] <= 1.0):
            raise ValueError("scale must satisfy 0 < min <= max <= 1")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if (self.event_mean is None) != (self.event_std is None):
            raise ValueError("event_mean and event_std must be set together")
        if self.event_mean is not None:
            if len(self.event_mean) != len(self.event_std):
                raise ValueError("event_mean and event_std lengths differ")
            if any(value <= 0 for value in self.event_std):
                raise ValueError("event_std values must be positive")

    @property
    def stochastic(self) -> bool:
        return self.training and (
            self.scale != (1.0, 1.0) or self.horizontal_flip_probability > 0
        )

    def __call__(
        self,
        events: Tensor | None,
        images: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        reference = events if events is not None else images
        if reference is None:
            raise ValueError("At least one modality must be provided")
        if reference.ndim != 4:
            raise ValueError("Sequence tensors must have shape [T, C, H, W]")
        source_h, source_w = reference.shape[-2:]
        for name, tensor in (("events", events), ("images", images)):
            if tensor is not None and tensor.shape[-2:] != (source_h, source_w):
                raise ValueError(f"{name} is not spatially aligned with the other modality")

        top, left, crop_h, crop_w = self._sample_crop(source_h, source_w)
        flip = self.training and random.random() < self.horizontal_flip_probability
        events = self._apply(events, top, left, crop_h, crop_w, flip)
        images = self._apply(images, top, left, crop_h, crop_w, flip)
        if events is not None and self.event_mean is not None:
            if events.shape[1] != len(self.event_mean):
                raise ValueError(
                    f"Event tensor has {events.shape[1]} channels but normalization has "
                    f"{len(self.event_mean)} values"
                )
            mean = events.new_tensor(self.event_mean).view(1, -1, 1, 1)
            std = events.new_tensor(self.event_std).view(1, -1, 1, 1)
            events = (events - mean) / std
        return events, images

    def _sample_crop(self, source_h: int, source_w: int) -> tuple[int, int, int, int]:
        target_ratio = self.width / self.height
        scale = random.uniform(*self.scale) if self.training else 1.0
        desired_area = source_h * source_w * scale
        crop_w = min(source_w, max(1, round(math.sqrt(desired_area * target_ratio))))
        crop_h = min(source_h, max(1, round(crop_w / target_ratio)))
        if crop_h > source_h:
            crop_h = source_h
            crop_w = min(source_w, max(1, round(crop_h * target_ratio)))

        if self.training:
            top = random.randint(0, source_h - crop_h)
            left = random.randint(0, source_w - crop_w)
        else:
            top = (source_h - crop_h) // 2
            left = (source_w - crop_w) // 2
        return top, left, crop_h, crop_w

    def _apply(
        self,
        tensor: Tensor | None,
        top: int,
        left: int,
        crop_h: int,
        crop_w: int,
        flip: bool,
    ) -> Tensor | None:
        if tensor is None:
            return None
        tensor = tensor[..., top : top + crop_h, left : left + crop_w]
        if tensor.shape[-2:] != (self.height, self.width):
            tensor = F.interpolate(
                tensor,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
        if flip:
            tensor = tensor.flip(-1)
        return tensor.contiguous()
