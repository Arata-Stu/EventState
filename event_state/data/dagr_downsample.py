"""Stateful event-stream downsampling compatible with DAGR.

This module implements the algorithm used by DAGR's ``downsample_events.py``
without depending on the DAGR source tree.  Downsampling happens on the event
stream, before an event image or voxel representation is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def signed_polarity(polarity: np.ndarray) -> np.ndarray:
    """Return polarity as int8 values in ``{-1, +1}``."""

    values = np.asarray(polarity)
    if values.ndim != 1:
        raise ValueError("polarity must be one-dimensional")
    if values.size == 0:
        return values.astype(np.int8, copy=False)
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum >= 0 and maximum <= 1:
        return (2 * values.astype(np.int8) - 1).astype(np.int8, copy=False)
    if minimum >= -1 and maximum <= 1 and not np.any(values == 0):
        return values.astype(np.int8, copy=False)
    raise ValueError("polarity values must be either 0/1 or -1/+1")


def _filter_events_python(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    change_map: np.ndarray,
    factor_x: int,
    factor_y: int,
) -> np.ndarray:
    keep = np.zeros(len(x), dtype=np.bool_)
    increment = 1.0 / float(factor_x * factor_y)
    for index in range(len(x)):
        output_x = int(x[index]) // factor_x
        output_y = int(y[index]) // factor_y
        value = int(polarity[index])
        change_map[output_y, output_x] += value * increment
        if abs(float(change_map[output_y, output_x])) >= 1.0:
            keep[index] = True
            change_map[output_y, output_x] -= value
    return keep


try:  # Optional acceleration matching the official DAGR preprocessing tool.
    import numba
except ImportError:  # pragma: no cover - exercised on minimal installations.
    _filter_events = _filter_events_python
else:
    _filter_events = numba.njit(cache=True)(_filter_events_python)


@dataclass
class DAGRDownsampler:
    """Stateful signed-event downsampler.

    The residual ``change_map`` intentionally survives calls.  Resetting it at
    an HDF5 chunk or requested visualization interval would change which events
    pass the DAGR filter.
    """

    input_height: int = 720
    input_width: int = 1280
    output_height: int = 360
    output_width: int = 640
    change_map: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dimensions = (
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all dimensions must be positive")
        if self.input_height % self.output_height:
            raise ValueError("input_height must be divisible by output_height")
        if self.input_width % self.output_width:
            raise ValueError("input_width must be divisible by output_width")
        self.factor_y = self.input_height // self.output_height
        self.factor_x = self.input_width // self.output_width
        self.change_map = np.zeros(
            (self.output_height, self.output_width), dtype=np.float32
        )

    def reset(self) -> None:
        self.change_map.fill(0)

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        polarity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x)
        y = np.asarray(y)
        polarity = signed_polarity(polarity)
        if not (x.ndim == y.ndim == polarity.ndim == 1):
            raise ValueError("x, y, and polarity must be one-dimensional")
        if not (len(x) == len(y) == len(polarity)):
            raise ValueError("x, y, and polarity lengths differ")
        valid = (
            (x >= 0)
            & (x < self.input_width)
            & (y >= 0)
            & (y < self.input_height)
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            empty_u16 = np.empty(0, dtype=np.uint16)
            return empty_u16, empty_u16.copy(), np.empty(0, dtype=np.int8), valid_indices

        valid_x = x[valid_indices].astype(np.int64, copy=False)
        valid_y = y[valid_indices].astype(np.int64, copy=False)
        valid_p = polarity[valid_indices]
        keep_valid = _filter_events(
            valid_x,
            valid_y,
            valid_p,
            self.change_map,
            self.factor_x,
            self.factor_y,
        )
        source_indices = valid_indices[keep_valid]
        return (
            (valid_x[keep_valid] // self.factor_x).astype(np.uint16),
            (valid_y[keep_valid] // self.factor_y).astype(np.uint16),
            valid_p[keep_valid],
            source_indices,
        )


def bottom_padding_masks(
    *,
    content_height: int = 360,
    width: int = 640,
    padded_height: int = 448,
    patch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pixel, binary patch, and fractional patch masks for bottom padding."""

    if not 0 < content_height <= padded_height or width <= 0 or patch_size <= 0:
        raise ValueError("invalid content, padded, or patch geometry")
    if padded_height % patch_size or width % patch_size:
        raise ValueError("padded dimensions must be divisible by patch_size")
    pixel_mask = np.zeros((padded_height, width), dtype=np.bool_)
    pixel_mask[:content_height] = True
    fractions = pixel_mask.reshape(
        padded_height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).mean(axis=(1, 3), dtype=np.float32)
    return pixel_mask, fractions > 0, fractions


__all__ = ["DAGRDownsampler", "bottom_padding_masks", "signed_polarity"]
