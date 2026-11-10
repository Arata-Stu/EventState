"""Equal-weight frame metrics using the fixed F3 depth metric definitions.

The caller supplies the same GT mask used by its existing evaluation. This
module changes metric formulas and aggregation only; it does not reproduce the
F3 sample selection, input windows, coordinate system, or training protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real


METRIC_NAMES = ("mae", "abs_rel", "sq_rel", "rmse", "rmse_log",
                "d1", "d2", "d3", "log10", "silog")

F3_FRAME_PROTOCOL = {
    "format_version": 1,
    "metric_definition": "f3_eval_depth_with_additional_mae",
    "aggregation": "arithmetic mean of per-frame metrics; equal weight per scored frame",
    "arithmetic_dtype": "float64",
    "precision_note": "Same formulas; not bitwise equivalent to upstream float32 arithmetic or accumulation",
    "validity": "caller's existing GT mask; no additional F3 sample-selection filters",
    "minimum_valid_pixels": 1,
    "empty_frame_policy": "exclude from the mean denominator; report null metrics when no frames are scored",
    "prediction_policy": "reject nonfinite or nonpositive predictions on valid GT; no clipping or scale alignment",
    "sq_rel_definition": "mean((prediction - target)^2 / target^2) per frame",
    "rmse_definition": "mean over frames of sqrt(mean squared pixel error within each frame)",
    "rmse_log_definition": "natural logarithm; mean over frames of per-frame RMSElog",
    "silog_definition": "sqrt(mean(log_error^2) - 0.5 * mean(log_error)^2) per frame; no factor of 100",
    "delta_definition": "strict ratio < 1.25^k; d1/d2/d3 are fractions, not percentages",
    "additional_metrics": ["mae"],
    "metric_units": {"mae": "meters", "rmse": "meters", "abs_rel": "dimensionless",
                     "sq_rel": "dimensionless", "rmse_log": "dimensionless",
                     "d1": "fraction", "d2": "fraction", "d3": "fraction",
                     "log10": "dimensionless", "silog": "dimensionless"},
    "reference": {
        "repository": "https://github.com/grasp-lyrl/fast-feature-fields",
        "revision": "833e2c65233f9787449ce0675ae41e3f6e9c193a",
        "path": "src/f3/tasks/depth/utils/utils.py",
        "function": "eval_depth",
        "file_sha256": "7e66611dd25c61c85b56f5a00786ac9b9fd208def2c95c3e5257e190de5b54f3",
    },
}


def _validated_metrics(metrics):
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        raise ValueError("Expected the complete F3 frame metric set and MAE")
    values = {}
    for name in METRIC_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Invalid depth frame metric: {name}")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (name in {"d1", "d2", "d3"} and value > 1):
            raise ValueError(f"Nonfinite or out-of-range depth frame metric: {name}")
        values[name] = value
    return values


class FrameMeanSums:
    """Pool scalar frame metrics, with one vote for each nonempty GT frame."""

    def __init__(self):
        self.frames = 0
        self.valid_pixels = 0
        self.sums = dict.fromkeys(METRIC_NAMES, 0.0)

    def add(self, valid_pixels, metrics):
        if isinstance(valid_pixels, bool) or not isinstance(valid_pixels, int) or valid_pixels < 0:
            raise ValueError("Frame valid-pixel count must be a nonnegative integer")
        if not valid_pixels:
            if metrics:
                raise ValueError("An empty GT frame must not contribute metric values")
            return
        values = _validated_metrics(metrics)
        sums = {key: self.sums[key] + values[key] for key in METRIC_NAMES}
        if not all(math.isfinite(value) for value in sums.values()):
            raise ValueError("Nonfinite accumulated depth frame metrics")
        self.frames += 1
        self.valid_pixels += valid_pixels
        self.sums = sums

    def compute(self):
        return dict(frames=self.frames, valid_pixels=self.valid_pixels,
                    **{key: value / self.frames if self.frames else None
                       for key, value in self.sums.items()})


def compute_frame_metrics(prediction, target, valid):
    """Return (valid pixels, F3 metrics plus MAE) for one existing GT frame.

    NumPy is imported only when numerical scoring is requested. The module and
    its frame accumulator can therefore be checked with the standard library.
    """
    import numpy as np

    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Depth prediction, target and validity shapes must match")
    p, y = prediction[valid].astype(np.float64), target[valid].astype(np.float64)
    count = len(y)
    if not count:
        return 0, {}
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid target validity mask")
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Nonfinite/nonpositive predictions on valid GT; never discard failed pixels")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        difference = p - y
        log_error = np.log(p) - np.log(y)
        ratio = np.maximum(y / p, p / y)
        metrics = dict(
            mae=float(np.abs(difference).mean()),
            abs_rel=float((np.abs(difference) / y).mean()),
            sq_rel=float((difference ** 2 / y ** 2).mean()),
            rmse=float(np.sqrt((difference ** 2).mean())),
            rmse_log=float(np.sqrt((log_error ** 2).mean())),
            log10=float(np.abs(np.log10(p) - np.log10(y)).mean()),
            silog=float(np.sqrt((log_error ** 2).mean() - 0.5 * log_error.mean() ** 2)),
            **{f"d{i}": float((ratio < 1.25 ** i).mean()) for i in (1, 2, 3)},
        )
    return count, _validated_metrics(metrics)
