"""Event-stream representations.

The implementation is deliberately independent from ``reference_repo``.  It
uses temporal bilinear voting, retains separate polarities, and makes the
timestamp boundary convention explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


Normalization = Literal["none", "nonzero_standardize", "log1p"]


@dataclass(frozen=True)
class GEPEventFrame:
    """Build GEP's effective three-channel event image.

    GEP constructs an RGB-like array and saves it through OpenCV as BGR. Once
    that PNG is read as RGB during training, negative events are red and
    positive events are blue. This method directly reproduces that effective
    tensor without an intermediate lossy PNG. At pixels containing both
    polarities, the larger normalized count wins (positive wins ties). The
    result is a float tensor in ``[0, 1]`` with shape ``[3, H, W]``.
    """

    height: int = 480
    width: int = 640
    percentile: float = 90.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")

    @property
    def channels(self) -> int:
        return 3

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor,
        *,
        start_time: int | float,
        end_time: int | float,
    ) -> Tensor:
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        if not (x.ndim == y.ndim == t.ndim == p.ndim == 1):
            raise ValueError("x, y, t, and p must be one-dimensional")
        if not (x.numel() == y.numel() == t.numel() == p.numel()):
            raise ValueError("x, y, t, and p must contain the same number of events")

        x = x.round().to(torch.int64)
        y = y.round().to(torch.int64)
        t = t.to(torch.float64)
        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
            & (t > float(start_time))
            & (t <= float(end_time))
        )
        x, y, p = x[valid], y[valid], p[valid]

        plane_size = self.height * self.width
        positive = torch.zeros(plane_size, dtype=torch.float32, device=x.device)
        negative = torch.zeros_like(positive)
        if x.numel() > 0:
            flat = y * self.width + x
            is_positive = p > 0
            positive.scatter_add_(
                0,
                flat[is_positive],
                torch.ones_like(flat[is_positive], dtype=torch.float32),
            )
            negative.scatter_add_(
                0,
                flat[~is_positive],
                torch.ones_like(flat[~is_positive], dtype=torch.float32),
            )
        positive = self._percentile_normalize(positive.view(self.height, self.width))
        negative = self._percentile_normalize(negative.view(self.height, self.width))

        positive_dominates = positive >= negative
        blue_intensity = positive * positive_dominates
        red_intensity = negative * ~positive_dominates
        red = 1.0 - blue_intensity
        green = 1.0 - red_intensity - blue_intensity
        blue = 1.0 - red_intensity
        return torch.stack((red, green, blue)).clamp_(0.0, 1.0)

    def _percentile_normalize(self, counts: Tensor) -> Tensor:
        nonzero = counts[counts > 0]
        if nonzero.numel() == 0:
            return counts
        threshold = torch.quantile(nonzero, self.percentile / 100.0)
        if threshold <= 0:
            threshold = nonzero.max()
        return counts.clamp(max=threshold) / threshold


@dataclass(frozen=True)
class EventVoxelizer:
    """Convert ``x, y, t, p`` events to a polarity-separated voxel grid.

    Channels are interleaved by temporal bin: ``negative_0, positive_0,
    negative_1, positive_1, ...``. Events vote linearly between adjacent time
    bins, which avoids a discontinuity at bin boundaries.
    """

    num_bins: int = 10
    height: int = 480
    width: int = 640
    polarity_split: bool = True
    normalization: Normalization = "nonzero_standardize"

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if not self.polarity_split:
            raise NotImplementedError("The MVP requires polarity_split=true")
        if self.normalization not in {"none", "nonzero_standardize", "log1p"}:
            raise ValueError(f"Unknown event normalization: {self.normalization}")

    @property
    def channels(self) -> int:
        return self.num_bins * 2

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor,
        *,
        start_time: int | float,
        end_time: int | float,
    ) -> Tensor:
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        if not (x.ndim == y.ndim == t.ndim == p.ndim == 1):
            raise ValueError("x, y, t, and p must be one-dimensional")
        if not (x.numel() == y.numel() == t.numel() == p.numel()):
            raise ValueError("x, y, t, and p must contain the same number of events")

        output = torch.zeros(
            self.channels * self.height * self.width,
            dtype=torch.float32,
            device=x.device,
        )
        if x.numel() == 0:
            return output.view(self.channels, self.height, self.width)

        x = x.to(torch.int64)
        y = y.to(torch.int64)
        t = t.to(torch.float64)
        positive = p > 0
        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
            & (t > float(start_time))
            & (t <= float(end_time))
        )
        if not torch.any(valid):
            return output.view(self.channels, self.height, self.width)

        x, y, t, positive = x[valid], y[valid], t[valid], positive[valid]
        temporal_position = (t - float(start_time)) * (self.num_bins - 1)
        temporal_position /= float(end_time - start_time)
        lower = temporal_position.floor().to(torch.int64).clamp_(0, self.num_bins - 1)
        upper = (lower + 1).clamp_(0, self.num_bins - 1)
        upper_weight = (temporal_position - lower).to(torch.float32)
        lower_weight = 1.0 - upper_weight
        polarity = positive.to(torch.int64)

        spatial_index = y * self.width + x
        lower_channel = lower * 2 + polarity
        upper_channel = upper * 2 + polarity
        plane_size = self.height * self.width
        output.scatter_add_(0, lower_channel * plane_size + spatial_index, lower_weight)
        output.scatter_add_(0, upper_channel * plane_size + spatial_index, upper_weight)
        volume = output.view(self.channels, self.height, self.width)
        return self._normalize(volume)

    def _normalize(self, volume: Tensor) -> Tensor:
        if self.normalization == "none":
            return volume
        if self.normalization == "log1p":
            return torch.log1p(volume)

        nonzero = volume != 0
        values = volume[nonzero]
        if values.numel() < 2:
            return volume
        std = values.std(unbiased=False)
        if std <= torch.finfo(volume.dtype).eps:
            return volume
        normalized = volume.clone()
        normalized[nonzero] = (values - values.mean()) / std
        return normalized
