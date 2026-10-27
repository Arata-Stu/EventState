"""Metric computation at native target coordinates, without test-time fitting."""
import numpy as np

from .metrics import RegressionSums


def accumulate(meter, prediction, target, valid):
    valid = np.asarray(valid, dtype=bool)
    p, y = np.asarray(prediction)[valid].astype(np.float64), np.asarray(target)[valid].astype(np.float64)
    if not len(y):
        return
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid target validity mask")
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Nonfinite/nonpositive predictions on valid GT; never discard failed pixels")
    error = p - y
    sums = dict(mae=float(np.abs(error).sum()), rmse=float((error ** 2).sum()))
    if meter.task == "depth":
        ratio = np.maximum(p / y, y / p)
        sums.update(abs_rel=float((np.abs(error) / y).sum()),
                    sq_rel=float((error ** 2 / y).sum()),
                    rmse_log=float(((np.log(p) - np.log(y)) ** 2).sum()),
                    **{f"delta{i}": float((ratio < 1.25 ** i).sum()) for i in (1, 2, 3)})
    else:
        sums.update({"1pe_percent": float((np.abs(error) > 1).sum()) * 100,
                     "2pe_percent": float((np.abs(error) > 2).sum()) * 100})
    meter.add(len(y), **sums)


class DenseEvaluator:
    def __init__(self, task):
        self.task = task
        self.groups = {name: RegressionSums(task) for name in
                       ("all", "low_activity_frames", "other_frames", "low_0_100ms",
                        "low_100_500ms", "low_500ms_plus")}

    def update(self, prediction, target, valid, *, low_activity, low_duration_us):
        names = ["all", "low_activity_frames" if low_activity else "other_frames"]
        if low_activity:
            names.append("low_0_100ms" if low_duration_us < 100_000 else
                         "low_100_500ms" if low_duration_us < 500_000 else "low_500ms_plus")
        for name in names:
            accumulate(self.groups[name], prediction, target, valid)

    def compute(self):
        return {key: meter.compute() for key, meter in self.groups.items()}
