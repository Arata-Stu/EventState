"""Frame activity strata for COCO detection metrics, without changing inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _values(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class ActivityDetectionEvaluator:
    """Recompute AP from all predictions/GT in each frame partition.

    The original overall evaluator is preserved. Empty/GT-free partitions have
    undefined AP, and negative COCO sentinels/nonfinite values become JSON null.
    An injected evaluator factory keeps routing tests independent of PyTorch.
    """

    def __init__(self, index, *, expected_keys, condition_label, evaluator_factory=None):
        if not condition_label or not condition_label.strip():
            raise ValueError("--condition-label must include the condition ID and name")
        if evaluator_factory is None:
            from .metrics import COCODetectionEvaluator

            evaluator_factory = COCODetectionEvaluator
        self.index = index
        self.condition_label = condition_label
        keys = [(str(seq), int(stamp)) for seq, stamp in expected_keys]
        self.expected = set(keys)
        if not keys or len(keys) != len(self.expected):
            raise ValueError("Activity evaluation requires unique, nonempty frame identities")
        reference = index.identity.get("reference", {}).get("sequences", [])
        if {sequence for sequence, _ in keys}.intersection(reference):
            raise ValueError("Evaluation sequences overlap the activity reference train pool")
        for key in keys:
            index.lookup(*key)  # Fail before inference if the manifest is incomplete.
        self.evaluators = {name: evaluator_factory() for name in ("all", *index.groups)}
        self.class_names = tuple(self.evaluators["all"].class_names)
        self.frames = {name: [] for name in self.evaluators}
        self.seen = set()
        self.target_records = []

    def update(self, predictions, targets, *, sequence_names, timestamps, image_size):
        if len({len(predictions), len(targets), len(sequence_names), len(timestamps)}) != 1:
            raise ValueError("Detection activity batch fields have different lengths")
        keys = [(str(seq), int(stamp)) for seq, stamp in zip(sequence_names, timestamps)]
        if len(keys) != len(set(keys)) or self.seen.intersection(keys):
            raise ValueError("Duplicate evaluation frame in activity partition")
        if not set(keys) <= self.expected:
            raise ValueError("Unexpected evaluation frame in activity partition")
        assignments = [self.index.lookup(*key) for key in keys]
        for target in targets:
            labels = _values(target["labels"])
            if len(labels) != len(target["boxes"]):
                raise ValueError("Detection target boxes and labels have different lengths")
            if any(isinstance(label, bool) or not 0 <= label < len(self.class_names)
                   or int(label) != label for label in labels):
                raise ValueError("Invalid detection target class in activity evaluation")
        for name, evaluator in self.evaluators.items():
            selected = [i for i, record in enumerate(assignments)
                        if name == "all" or record["group"] == name]
            if not selected:
                continue
            evaluator.update(
                [predictions[i] for i in selected], [targets[i] for i in selected],
                sequence_names=[sequence_names[i] for i in selected],
                timestamps=[timestamps[i] for i in selected], image_size=image_size,
            )
            for i in selected:
                labels = _values(targets[i]["labels"])
                boxes = _values(targets[i]["boxes"])
                self.frames[name].append(dict(
                    sequence=keys[i][0], timestamp=keys[i][1],
                    event_rate_hz=assignments[i]["event_rate_hz"],
                    group=assignments[i]["group"], labels=labels,
                    areas=[max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                           for box in boxes],
                    detections=len(predictions[i]["scores"]),
                ))
        for key, target in zip(keys, targets):
            self.target_records.append([*key, _values(target["boxes"]), _values(target["labels"])])
        self.seen.update(keys)

    def _metrics(self, name):
        evaluator, frames = self.evaluators[name], self.frames[name]
        keys = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                *(f"AP_{label}" for label in self.class_names))
        if not frames or not any(frame["labels"] for frame in frames):
            return {key: None for key in keys}
        values = evaluator.compute()
        classes = Counter(label for frame in frames for label in frame["labels"])
        no_detections = not any(frame["detections"] for frame in frames)
        areas = [area for frame in frames for area in frame["areas"]]
        area_ranges = {"AP_small": (0, 32 ** 2), "AP_medium": (32 ** 2, 96 ** 2),
                       "AP_large": (96 ** 2, 1e10)}
        result = {}
        for key in keys:
            value = values.get(key)
            if key in area_ranges:
                lower, upper = area_ranges[key]
                if not any(lower <= area <= upper for area in areas):
                    value = None
            if key.startswith("AP_") and key[3:] in self.class_names:
                class_index = self.class_names.index(key[3:])
                if not classes[class_index]:
                    value = None
                elif no_detections:
                    value = 0.0
            result[key] = (float(value) if value is not None and math.isfinite(value)
                           and value >= 0 else None)
        return result

    def compute(self):
        if self.seen != self.expected:
            raise ValueError(f"Missing {len(self.expected - self.seen)} evaluation frames")
        groups = {}
        total = len(self.seen)
        for name, frames in self.frames.items():
            labels = Counter(label for frame in frames for label in frame["labels"])
            rates = [frame["event_rate_hz"] for frame in frames]
            groups[name] = dict(
                status="empty" if not frames else "no_ground_truth" if not labels else "ok",
                frames=len(frames), evaluated_frame_fraction=len(frames) / total,
                sequences=len({frame["sequence"] for frame in frames}),
                sequence_frames=dict(sorted(Counter(f["sequence"] for f in frames).items())),
                gt_objects=sum(labels.values()),
                gt_objects_per_class={label: labels[i] for i, label in enumerate(self.class_names)},
                detections=sum(frame["detections"] for frame in frames),
                event_rate_hz_min=min(rates) if rates else None,
                event_rate_hz_max=max(rates) if rates else None,
                metrics=self._metrics(name),
            )
        population = sorted(self.target_records)
        assignments = sorted([f["sequence"], f["timestamp"], f["event_rate_hz"], f["group"]]
                             for f in self.frames["all"])
        return {
            **groups["all"]["metrics"],
            "activity_evaluation": dict(
                condition_label=self.condition_label,
                partition="whole_frame_by_scene_event_rate",
                activity_manifest=self.index.identity,
                samples_targets_sha256=_digest(population),
                activity_assignments_sha256=_digest(assignments),
                groups=groups,
            ),
        }


def validate_activity_sources(index, *, sequences, feature_cache_dir=None, event_cache_dir=None):
    """Match activity inputs to the event sources that produced the features."""
    if (feature_cache_dir is None) == (event_cache_dir is None):
        raise ValueError("Choose a feature cache or an event cache for activity provenance")
    states = {}
    for sequence in sequences:
        root = Path(feature_cache_dir or event_cache_dir)
        metadata = json.loads((root / sequence / "metadata.json").read_text())
        if feature_cache_dir is not None:
            if metadata.get("event_window_us") is not None:
                raise ValueError("Activity percentiles currently require full RGB-interval inputs")
            source = metadata.get("source_event_cache_metadata")
            if not isinstance(source, dict):
                raise ValueError("Feature cache lacks source event metadata for activity validation")
            states[sequence] = metadata.get("state_policy", "unknown")
        else:
            source = metadata
            states[sequence] = "continuous_reset_at_sequence_boundary"
        expected = index.source_metadata_sha256.get(sequence)
        if expected is None or _digest(source) != expected:
            raise ValueError(f"Activity manifest/event source mismatch for {sequence}")
        count = metadata.get("frame_count")
        paths = [path for path in (root / sequence).glob("*.pt") if path.stem.isdecimal()]
        if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
                or len(paths) != count or count != source.get("frame_count")):
            raise ValueError(f"Incomplete activity evaluation cache for {sequence}")
        if event_cache_dir is not None and not (root / sequence / "_SUCCESS").is_file():
            raise ValueError(f"Incomplete prepared event source for {sequence}")
    return states


def add_activity_arguments(parser):
    parser.add_argument("--activity-manifest", type=Path,
                        help="Train-fitted DSEC event-rate percentile manifest")
    parser.add_argument("--condition-label", help="Condition ID and name for activity reports")


def detector_checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def activity_evaluator(args, *, expected_keys, sequences, feature_cache_dir=None,
                       event_cache_dir=None):
    """Return an optional evaluator and its independently checked state policy."""
    if args.activity_manifest is None:
        if args.condition_label is not None:
            raise ValueError("--condition-label requires --activity-manifest")
        return None, None
    from event_state.activity_quantiles import ActivityIndex

    index = ActivityIndex.load(args.activity_manifest)
    states = validate_activity_sources(index, sequences=sequences,
                                      feature_cache_dir=feature_cache_dir,
                                      event_cache_dir=event_cache_dir)
    return ActivityDetectionEvaluator(index, expected_keys=expected_keys,
                                      condition_label=args.condition_label), states


def write_activity_csv(output, result):
    """Write the condition-labelled group table alongside the full JSON report."""
    activity = result["activity_evaluation"]
    metric_names = tuple(next(iter(activity["groups"].values()))["metrics"])
    fields = ("condition", "group", "status", "frames", "evaluated_frame_fraction", "gt_objects",
              "event_rate_hz_min", "event_rate_hz_max", *metric_names)
    path = Path(output).with_suffix(".csv")
    if path.resolve() == Path(output).resolve():
        path = Path(output).with_name(Path(output).name + ".groups.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, group in activity["groups"].items():
            writer.writerow(dict(condition=activity["condition_label"], group=name,
                                 **{key: group[key] for key in fields[2:8]}, **group["metrics"]))
