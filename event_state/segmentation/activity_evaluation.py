"""Train-fitted frame strata for semantic confusion matrices."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _valid_pixels(target, num_classes, ignore_index):
    if hasattr(target, "detach"):
        valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
        return int(valid.sum().item())
    stack, count = list(target), 0
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(value)
        elif value != ignore_index and 0 <= value < num_classes:
            count += 1
    return count


def _target_digest(target):
    if hasattr(target, "detach"):
        array = target.detach().cpu().contiguous().numpy()
        return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(target, separators=(",", ":")).encode()).hexdigest()


class ActivitySemanticEvaluator:
    """Pool each group's confusion matrix rather than averaging frame mIoU."""

    def __init__(self, index, *, dataset, expected_keys, condition_label, num_classes,
                 ignore_index=255, evaluator_factory=None):
        from event_state.activity_reporting import ActivityPartition

        if evaluator_factory is None:
            from .metrics import SemanticSegmentationEvaluator

            evaluator_factory = lambda: SemanticSegmentationEvaluator(
                num_classes, ignore_index=ignore_index)
        self.partition = ActivityPartition(index, dataset=dataset, expected_keys=expected_keys,
                                           condition_label=condition_label)
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.meters = {group: evaluator_factory() for group in ("all", *self.partition.groups)}
        self.target_identities = []

    def update(self, predictions, targets, *, sequence_names, timestamps):
        if len({len(predictions), len(targets), len(sequence_names), len(timestamps)}) != 1:
            raise ValueError("Semantic activity batch fields have different lengths")
        for i, (sequence, timestamp) in enumerate(zip(sequence_names, timestamps)):
            target = targets[i]
            group = self.partition.update(
                sequence, timestamp,
                valid_pixels=_valid_pixels(target, self.num_classes, self.ignore_index),
            )
            for name in ("all", group):
                self.meters[name].update(predictions[i:i + 1], targets[i:i + 1])
            self.target_identities.append([sequence, timestamp, _target_digest(target)])

    def compute(self):
        self.partition.validate_complete()
        metrics = {}
        for name, meter in self.meters.items():
            metrics[name] = dict(meter.compute())
            metrics[name]["class_support"] = [sum(row) for row in _tolist(meter.confusion)]
            if not sum(metrics[name]["class_support"]):
                for key in ("mIoU", "pixel_accuracy", "mean_class_accuracy"):
                    if key in metrics[name]:
                        metrics[name][key] = None
        report = self.partition.report(metrics)
        report["samples_targets_sha256"] = hashlib.sha256(json.dumps(
            sorted(self.target_identities), separators=(",", ":")).encode()).hexdigest()
        return {**_safe(metrics["all"]), "activity_evaluation": report}


def validate_semantic_cache(index, *, dataset, feature_cache_dir, sequences, samples,
                            load_payload, prepared_root=None):
    """Return exact frame keys after validating sparse feature/source identities.

    DSEC samples are keyed by label frame index, so timestamps are read from the
    small pickle metadata of each feature file. M3ED common caches additionally
    retain a source metadata file hash in identity.json.
    """
    from event_state.activity_reporting import validate_activity_source

    root = Path(feature_cache_dir).expanduser()
    identity_path = root / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    keys, states = [], {}
    by_sequence = {sequence: [] for sequence in sequences}
    for sample in samples:
        by_sequence[sample[0]].append(sample)
    for sequence in sequences:
        directory = root / sequence
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("event_window_us") is not None:
            raise ValueError("Semantic activity groups require full RGB-interval inputs")
        recorded = metadata.get("cached_frame_indices")
        if (not isinstance(recorded, list) or len(recorded) != len(set(recorded))
                or not recorded or any(isinstance(i, bool) or not isinstance(i, int) for i in recorded)):
            raise ValueError(f"Invalid semantic cached frame indices: {sequence}")
        actual = {int(path.stem) for path in directory.glob("*.pt") if path.stem.isdecimal()}
        selected = {sample[1] for sample in by_sequence[sequence]}
        if actual != set(recorded) or selected != set(recorded):
            raise ValueError(f"Incomplete semantic feature/target cache: {sequence}")
        if dataset == "dsec":
            source = metadata.get("source_event_cache_metadata")
            if not isinstance(source, dict):
                raise ValueError("Semantic cache lacks source event metadata")
        elif dataset == "m3ed":
            if prepared_root is None:
                raise ValueError("M3ED activity validation requires --prepared-root")
            source_path = Path(prepared_root).expanduser() / sequence / "metadata.json"
            raw = source_path.read_bytes()
            source = json.loads(raw)
            embedded = metadata.get("source_event_cache_metadata")
            if embedded is not None:
                if embedded != source:
                    raise ValueError(f"M3ED semantic prepared source changed: {sequence}")
            else:
                expected = identity.get("sources", {}).get(sequence)
                if not isinstance(expected, list) or not expected or hashlib.sha256(raw).hexdigest() != expected[0]:
                    raise ValueError("M3ED semantic cache lacks matching prepared-source provenance")
        else:
            raise ValueError("Unsupported semantic activity dataset")
        validate_activity_source(index, sequence, source)
        states[sequence] = metadata.get("state_policy", "unknown")
        for _, frame_index, feature_path, _ in by_sequence[sequence]:
            payload = load_payload(feature_path)
            if (payload.get("sequence_name") != sequence or payload.get("frame_index") != frame_index
                    or payload.get("checkpoint_sha256") != metadata.get("checkpoint_sha256")
                    or payload.get("state_policy") != metadata.get("state_policy")):
                raise ValueError(f"Semantic feature payload identity mismatch: {feature_path}")
            timestamp = payload.get("timestamp")
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise ValueError(f"Semantic feature lacks timestamp: {feature_path}")
            index.lookup(sequence, timestamp)
            if dataset == "dsec":
                stamps = index.timestamps(sequence)
                if not 1 <= frame_index <= len(stamps) or stamps[frame_index - 1] != timestamp:
                    raise ValueError(f"DSEC semantic frame-index/timestamp mismatch: {feature_path}")
            elif dataset == "m3ed":
                stamps = source.get("timestamps", [])
                if not 0 <= frame_index < len(stamps) or stamps[frame_index] != timestamp:
                    raise ValueError(f"M3ED feature/prepared timestamp mismatch: {feature_path}")
            keys.append((sequence, timestamp))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate semantic evaluation frame identities")
    return keys, states


__all__ = ["ActivitySemanticEvaluator", "validate_semantic_cache"]
