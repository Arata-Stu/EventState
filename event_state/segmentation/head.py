"""Linear and compact nonlinear probes for frozen EventState features."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class EventStateSegmentationHead(nn.Module):
    """Decode one stride-16 z/h map into a dense semantic prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int = 11,
        width: int = 192,
        output_size: tuple[int, int] = (440, 640),
        head_type: str = "linear",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 1 or width <= 0:
            raise ValueError("in_channels/width must be positive and num_classes > 1")
        if any(value <= 0 for value in output_size):
            raise ValueError("output_size values must be positive")
        if head_type not in {"linear", "nonlinear"}:
            raise ValueError("head_type must be linear or nonlinear")
        self.head_type = head_type
        self.output_size = tuple(int(value) for value in output_size)
        if head_type == "linear":
            # One shared affine classifier per spatial token. The following
            # bilinear interpolation has no learned parameters.
            self.classifier = nn.Conv2d(in_channels, num_classes, 1)
            return
        mid = max(32, width // 2)
        low = max(32, width // 4)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False),
            nn.GroupNorm(16 if width % 16 == 0 else 1, width),
            nn.GELU(),
            ConvNormAct(width, width),
        )
        self.up1 = ConvNormAct(width, mid)
        self.up2 = ConvNormAct(mid, low)
        self.classifier = nn.Conv2d(low, num_classes, 1)

    def forward(self, features: Tensor) -> Tensor:
        if self.head_type == "linear":
            value = self.classifier(features)
            return F.interpolate(
                value, size=self.output_size, mode="bilinear", align_corners=False
            )
        value = self.project(features)
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        value = self.up1(value)
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        value = self.up2(value)
        value = self.classifier(value)
        return F.interpolate(value, size=self.output_size, mode="bilinear", align_corners=False)


__all__ = ["EventStateSegmentationHead"]
