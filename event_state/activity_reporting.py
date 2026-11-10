"""Shared frame partitions and portable reports for activity-stratified tasks.

Metric accumulation stays with each task: confusion matrices for Semantic and
valid-pixel error sums for Depth. This module only checks and counts samples.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any

from .activity_quantiles import ActivityIndex, canonical_sha256


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_activity_options(manifest_path, condition_label, *, dataset):
    if manifest_path is None:
        if condition_label is not None:
            raise ValueError("--condition-label requires --activity-manifest")
        return None
    if not isinstance(condition_label, str) or not condition_label.strip():
        raise ValueError("--condition-label must include the condition ID and name")
    index = ActivityIndex.load(Path(manifest_path).expanduser())
    if index.identity["source_contract"]["dataset"] != str(dataset).lower():
        raise ValueError("Activity manifest belongs to a different dataset")
    return index


def add_activity_arguments(parser):
    parser.add_argument("--activity-manifest", type=Path,
                        help="Shared train-fitted event-rate quintile manifest")
    parser.add_argument("--condition-label", help="Condition ID and name for activity reports")


def validate_activity_source(index, sequence, metadata):
    expected = index.source_metadata_sha256.get(sequence)
    if expected is None or canonical_sha256(metadata) != expected:
        raise ValueError(f"Activity manifest/event source mismatch for {sequence}")


class ActivityPartition:
    """Assign complete evaluated frames to the same train-fitted task-neutral bins."""

    def __init__(self, index, *, dataset, expected_keys, condition_label):
        if not isinstance(condition_label, str) or not condition_label.strip():
            raise ValueError("--condition-label must include the condition ID and name")
        identity = index.identity
        if identity["source_contract"]["dataset"] != str(dataset).lower():
            raise ValueError("Activity manifest belongs to a different dataset")
        keys = []
        for sequence, timestamp in expected_keys:
            if not isinstance(sequence, str) or not sequence:
                raise ValueError("Invalid activity evaluation sequence")
            if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
                raise ValueError("Activity evaluation timestamps must be integers")
            keys.append((sequence, int(timestamp)))
        self.expected = set(keys)
        if not keys or len(keys) != len(self.expected):
            raise ValueError("Activity evaluation requires unique, nonempty frame identities")
        if {seq for seq, _ in keys}.intersection(identity["reference"]["sequences"]):
            raise ValueError("Evaluation sequences overlap the activity reference train pool")
        for key in keys:
            index.lookup(*key)
        self.index = index
        self.groups = index.groups
        self.condition_label = condition_label
        self._identity = identity
        self._seen = {}
        self._frames = {name: [] for name in ("all", *self.groups)}

    def update(self, sequence, timestamp, *, valid_pixels, event_rate_hz=None):
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise ValueError("Activity evaluation timestamps must be integers")
        key = sequence, int(timestamp)
        if key in self._seen:
            raise ValueError(f"Duplicate activity evaluation frame: {sequence}/{timestamp}")
        if key not in self.expected:
            raise ValueError(f"Unexpected activity evaluation frame: {sequence}/{timestamp}")
        if isinstance(valid_pixels, bool) or not isinstance(valid_pixels, Integral) or valid_pixels < 0:
            raise ValueError("valid_pixels must be a nonnegative integer")
        record = self.index.lookup(*key)
        if event_rate_hz is not None:
            rate = float(event_rate_hz)
            if not math.isfinite(rate) or not math.isclose(
                rate, record["event_rate_hz"], rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"Feature/manifest event rate mismatch for {sequence}/{timestamp}")
        frame = dict(sequence=sequence, timestamp=int(timestamp), group=record["group"],
                     event_rate_hz=record["event_rate_hz"], valid_pixels=int(valid_pixels))
        self._seen[key] = frame
        self._frames["all"].append(frame)
        self._frames[record["group"]].append(frame)
        return record["group"]

    def validate_complete(self):
        if set(self._seen) != self.expected:
            raise ValueError(f"Missing {len(self.expected - set(self._seen))} activity evaluation frames")

    def report(self, metrics_by_group):
        self.validate_complete()
        if set(metrics_by_group) != set(self._frames):
            raise ValueError("Activity metrics do not cover exactly all requested groups")
        groups = {}
        for name, frames in self._frames.items():
            pixels = sum(frame["valid_pixels"] for frame in frames)
            rates = [frame["event_rate_hz"] for frame in frames]
            sequences = Counter(frame["sequence"] for frame in frames)
            groups[name] = dict(
                status="empty" if not frames else "no_valid_ground_truth" if not pixels else "ok",
                frames=len(frames), valid_frames=sum(f["valid_pixels"] > 0 for f in frames),
                evaluated_frame_fraction=len(frames) / len(self.expected),
                valid_pixels=pixels, sequences=len(sequences), sequence_frames=dict(sorted(sequences.items())),
                event_rate_hz_min=min(rates) if rates else None,
                event_rate_hz_max=max(rates) if rates else None,
                metrics=json_safe(metrics_by_group[name]),
            )
        assignments = [[seq, stamp, self._seen[seq, stamp]["event_rate_hz"],
                        self._seen[seq, stamp]["group"]] for seq, stamp in sorted(self._seen)]
        support = [[seq, stamp, self._seen[seq, stamp]["valid_pixels"]]
                   for seq, stamp in sorted(self._seen)]
        return dict(condition_label=self.condition_label,
                    partition="whole_frame_by_scene_event_rate", activity_manifest=self._identity,
                    samples_sha256=canonical_sha256(sorted(self._seen)),
                    activity_assignments_sha256=canonical_sha256(assignments),
                    validity_support_sha256=canonical_sha256(support), groups=groups)


def write_activity_csv(output, activity_report):
    destination = Path(output).expanduser()
    path = destination.with_suffix(".csv")
    if path.resolve() == destination.resolve():
        path = destination.with_name(destination.name + ".groups.csv")
    support_names = ["status", "frames", "valid_frames", "evaluated_frame_fraction", "valid_pixels",
                     "sequences", "event_rate_hz_min", "event_rate_hz_max"]
    metric_names = list(dict.fromkeys(key for group in activity_report["groups"].values()
                                     for key in group["metrics"] if key not in support_names))
    frame_metric_names = list(dict.fromkeys(key for group in activity_report["groups"].values()
                                           for key in group.get("metrics_frame_mean_f3", {})))
    frame_columns = ["f3_frame_mean_" + key for key in frame_metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "group", *support_names,
                                                   *metric_names, *frame_columns])
        writer.writeheader()
        for name, group in activity_report["groups"].items():
            metrics = {key: (json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value)
                       for key, value in group["metrics"].items() if key in metric_names}
            writer.writerow(dict(condition=activity_report["condition_label"], group=name,
                                 **{key: group[key] for key in support_names}, **metrics,
                                 **{"f3_frame_mean_" + key: group.get("metrics_frame_mean_f3", {}).get(key)
                                    for key in frame_metric_names}))
    return path
