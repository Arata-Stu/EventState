"""Streaming metric sums independent of a training runtime.

Counts never depend on prediction quality. Empty strata yield null, not zero.
"""
import math


class RegressionSums:
    def __init__(self, task):
        if task not in {"depth", "disparity"}:
            raise ValueError("task must be depth or disparity")
        self.task = task
        self.count = 0
        self.frames = 0
        self.sums = {}

    def add(self, count, **sums):
        if count < 0 or any(not math.isfinite(v) or v < 0 for v in sums.values()):
            raise ValueError("Invalid regression metric sums")
        if count:
            self.frames += 1
            self.count += count
            for key, value in sums.items():
                self.sums[key] = self.sums.get(key, 0.0) + value

    def compute(self):
        keys = (["mae", "rmse", "abs_rel", "sq_rel", "rmse_log", "delta1", "delta2", "delta3"]
                if self.task == "depth" else ["mae", "rmse", "1pe_percent", "2pe_percent"])
        values = {key: None for key in keys}
        if self.count:
            values = {key: self.sums[key] / self.count for key in keys}
            for key in ("rmse", "rmse_log"):
                if key in values:
                    values[key] = math.sqrt(values[key])
        return {"valid_pixels": self.count, "frames": self.frames, **values}
