"""Pixel-pooled depth/disparity scores for shared train-fitted activity strata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dense_activity_sources(index, metadata):
    """Verify the native prepared sources recorded by the frozen feature cache."""
    from event_state.activity_reporting import validate_activity_source

    sequences = {record["sequence"] for record in metadata["records"]}
    sources = metadata.get("source_identity", {})
    for sequence in sequences:
        source = sources.get(sequence, {})
        path = source.get("path")
        expected = source.get("metadata_sha256")
        if not path or not expected:
            raise ValueError("Activity quintiles require prepared full-RGB-interval feature sources")
        if _file_sha256(path) != expected:
            raise ValueError(f"Dense activity source metadata changed for {sequence}")
        validate_activity_source(index, sequence, json.loads(Path(path).read_text()))


class DenseActivityEvaluator:
    """Route each complete frame to its group, then pool valid-pixel error sums."""

    def __init__(self, index, *, dataset, task, expected_keys, condition_label,
                 partition_factory=None, meter_factory=None, accumulate_fn=None,
                 frame_metrics=False):
        if partition_factory is None:
            from event_state.activity_reporting import ActivityPartition

            partition_factory = ActivityPartition
        if meter_factory is None:
            from .metrics import RegressionSums

            meter_factory = RegressionSums
        if accumulate_fn is None:
            from .evaluation import accumulate

            accumulate_fn = accumulate
        self.partition = partition_factory(index, dataset=dataset, expected_keys=expected_keys,
                                           condition_label=condition_label)
        self.meters = {name: meter_factory(task) for name in ("all", *self.partition.groups)}
        self.accumulate = accumulate_fn
        self.frame_meters = None
        if frame_metrics:
            if task != "depth":
                raise ValueError("F3 frame metrics require metric depth")
            from .depth_frame_metrics import FrameMeanSums

            self.frame_meters = {name: FrameMeanSums() for name in self.meters}

    def update(self, prediction, target, valid, *, sequence, timestamp, event_rate_hz,
               frame_result=None):
        group = self.partition.update(sequence, timestamp, valid_pixels=int(valid.sum()),
                                      event_rate_hz=event_rate_hz)
        for name in ("all", group):
            self.accumulate(self.meters[name], prediction, target, valid)
        if self.frame_meters is not None:
            if frame_result is None:
                from .depth_frame_metrics import compute_frame_metrics

                frame_result = compute_frame_metrics(prediction, target, valid)
            for name in ("all", group):
                self.frame_meters[name].add(*frame_result)

    def compute(self):
        report = self.partition.report({name: meter.compute() for name, meter in self.meters.items()})
        if self.frame_meters is not None:
            from .depth_frame_metrics import F3_FRAME_PROTOCOL

            report["depth_frame_metric_protocol"] = F3_FRAME_PROTOCOL
            for name, meter in self.frame_meters.items():
                report["groups"][name]["metrics_frame_mean_f3"] = meter.compute()
        return report
