"""Portable, train-fitted event-rate groups shared by downstream evaluations."""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
M3ED_FORMAT_VERSION = 2
PERCENTILES = (20, 40, 60, 80)
QUANTILE_METHOD = "lower_order_statistic_floor_q_times_n_minus_1"
SOURCE_CONTRACT = {
    "dataset": "dsec",
    "coordinate_space": "rectified_event",
    "height": 480,
    "width": 640,
    "event_window": "rgb_interval",
    "event_window_fraction": 1.0,
    "event_window_boundary": "previous_timestamp < event_timestamp <= timestamp",
    "count_definition": "prepared_event_count_after_rectification",
}
M3ED_SOURCE_CONTRACT = {
    "dataset": "m3ed",
    "coordinate_space": "rectified_left_event",
    "height": 360,
    "width": 640,
    "event_window": "rgb_interval",
    "event_window_fraction": 1.0,
    "event_window_boundary": "previous_rgb_timestamp < event_timestamp <= rgb_timestamp",
    "downsampling": "dagr_stateful_signed_event_downsampling",
    "downsample_factor": [2, 2],
    "count_definition": "prepared_event_count_after_dagr_downsampling_and_rectification",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _sequences(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty sequence list")
    if any(not isinstance(v, str) or not v or Path(v).name != v or v in {'.', '..'}
           for v in values):
        raise ValueError(f"Invalid {name}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate sequence in {name}")
    return tuple(sorted(values))


def training_sequences(manifest: Mapping[str, Any], *, dataset: str = "dsec"
                       ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the reference train pool without confusing it with task head splits."""
    if dataset not in {"dsec", "m3ed"}:
        raise ValueError("Activity training dataset must be dsec or m3ed")
    if str(manifest.get("dataset", "")).lower() != dataset:
        raise ValueError(f"Training manifest must describe {dataset.upper()}")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Training manifest lacks splits")
    train = _sequences(splits.get("train"), "reference train sequences")
    excluded: list[str] = []
    for role in ("val", "test"):
        values = splits.get(role, [])
        if values:
            excluded.extend(_sequences(values, f"reference {role} sequences"))
    if len(set(excluded)) != len(excluded) or set(train) & set(excluded):
        raise ValueError("Training manifest splits overlap")
    return train, tuple(sorted(excluded))


def event_activity_record(payload: Mapping[str, Any], *, sequence: str,
                          metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the saved count, never estimate it from a colored event image."""
    if metadata.get("event_window_us") is not None or payload.get("event_window_us") is not None:
        raise ValueError("Activity manifests currently support RGB-interval caches, not fixed windows")
    for value in (metadata, payload):
        if float(value.get("event_window_fraction", 1.0)) != 1.0:
            raise ValueError("Activity manifests currently require the full RGB interval")
    if metadata.get("event_window") != SOURCE_CONTRACT["event_window"]:
        raise ValueError("Unsupported event accumulation window")
    if metadata.get("event_window_boundary") != SOURCE_CONTRACT["event_window_boundary"]:
        raise ValueError("Unsupported event window boundary")
    if str(metadata.get("dataset", "")).lower() != "dsec":
        raise ValueError("Expected a DSEC prepared event sequence")
    if metadata.get("sequence_name") != sequence or payload.get("sequence_name") != sequence:
        raise ValueError("Event sequence mismatch")
    if metadata.get("coordinate_space") != "rectified_event":
        raise ValueError("Activity quantiles require rectified event counts")
    for dimension in ("height", "width"):
        if metadata.get(dimension) != SOURCE_CONTRACT[dimension]:
            raise ValueError("Activity source geometry differs from full DSEC event view")
        if payload.get(dimension) != metadata[dimension]:
            raise ValueError("Event payload geometry differs from metadata")
    if metadata.get("format_version") != 2:
        raise ValueError("Unsupported DSEC event cache format")
    for key in ("format_version", "dataset", "split", "representation", "event_window_boundary"):
        if payload.get(key) != metadata.get(key):
            raise ValueError(f"Event payload {key} differs from metadata")
    if metadata.get("representation", {}).get("type") != "gep_rgb":
        raise ValueError("Activity quantiles currently require prepared GEP RGB")
    if payload.get("rectified") is not True:
        raise ValueError("Expected a rectified event payload")
    if payload.get("manifest_digest") != canonical_sha256(metadata):
        raise ValueError("Event payload manifest digest mismatch")
    fingerprint = metadata.get("input_fingerprint", {})
    if not fingerprint.get("digest") or payload.get("input_fingerprint_digest") != fingerprint["digest"]:
        raise ValueError("Event payload input fingerprint mismatch")
    timestamp = _integer(payload.get("timestamp"), "timestamp")
    previous = _integer(payload.get("previous_timestamp"), "previous_timestamp")
    window = timestamp - previous
    _integer(window, "window_us", positive=True)
    count = _integer(payload.get("event_count"), "event_count")
    return dict(sequence=sequence, timestamp=timestamp, previous_timestamp=previous,
                event_count=count, window_us=window, event_rate_hz=count * 1_000_000.0 / window)


def validate_m3ed_activity_metadata(metadata: Mapping[str, Any], *, sequence: str) -> list[int]:
    """Validate the legacy half-DAGR preparation, whose final metadata is its marker."""
    if str(metadata.get("dataset", "")).lower() != "m3ed":
        raise ValueError("Expected a M3ED prepared event sequence")
    if metadata.get("sequence_name") != sequence:
        raise ValueError("M3ED activity sequence mismatch")
    if type(metadata.get("format_version")) is not int or metadata["format_version"] != 1:
        raise ValueError("Unsupported M3ED preparation format")
    if metadata.get("coordinate_space") != M3ED_SOURCE_CONTRACT["coordinate_space"]:
        raise ValueError("M3ED activity requires rectified left-event coordinates")
    if metadata.get("event_size") != [360, 640] or metadata.get("native_event_size") != [720, 1280]:
        raise ValueError("M3ED activity source geometry differs from half-DAGR events")
    if (metadata.get("downsampling") != M3ED_SOURCE_CONTRACT["downsampling"]
            or metadata.get("downsample_factor") != [2, 2]):
        raise ValueError("M3ED activity requires the existing DAGR downsampling")
    if metadata.get("operation_order") != ["dagr_downsample_raw_event_stream",
                                             "rectify_half_scale_events",
                                             "build_event_representation"]:
        raise ValueError("Unsupported M3ED event preparation order")
    if metadata.get("representation", {}).get("type") != "gep_rgb":
        raise ValueError("M3ED activity currently requires prepared GEP RGB")
    if metadata.get("representation", {}).get("channels") != 3:
        raise ValueError("M3ED GEP activity requires three event channels")
    if metadata.get("event_window") != M3ED_SOURCE_CONTRACT["event_window_boundary"]:
        raise ValueError("Unsupported M3ED activity accumulation window")
    if metadata.get("event_window_us") is not None or float(metadata.get("event_window_fraction", 1)) != 1:
        raise ValueError("M3ED activity requires the full RGB interval")
    count = _integer(metadata.get("frame_count"), "M3ED frame_count", positive=True)
    stamps = metadata.get("timestamps")
    if not isinstance(stamps, list) or len(stamps) != count or count < 2:
        raise ValueError("M3ED timestamps must include the first RGB frame and every event interval")
    for stamp in stamps:
        _integer(stamp, "M3ED timestamp")
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("M3ED timestamps must be strictly increasing")
    return list(stamps)


def m3ed_event_activity_record(payload: Mapping[str, Any], *, sequence: str,
                              metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract half-DAGR saved counts at the exact native RGB interval."""
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid M3ED prepared event payload")
    if (type(payload.get("format_version")) is not int
            or payload.get("dataset") != metadata.get("dataset")
            or payload.get("sequence_name") != sequence
            or payload.get("format_version") != metadata.get("format_version")):
        raise ValueError("M3ED event payload provenance mismatch")
    if payload.get("representation") != "gep_rgb":
        raise ValueError("M3ED event payload representation mismatch")
    if payload.get("event_window_us") is not None or float(payload.get("event_window_fraction", 1)) != 1:
        raise ValueError("M3ED activity requires the full RGB interval")
    index = _integer(payload.get("frame_index"), "M3ED frame_index", positive=True)
    stamps = metadata["timestamps"]
    if index >= len(stamps):
        raise ValueError("M3ED event frame index is outside its RGB timeline")
    timestamp = _integer(payload.get("timestamp"), "timestamp")
    previous = _integer(payload.get("previous_timestamp"), "previous_timestamp")
    if timestamp != stamps[index] or previous != stamps[index - 1]:
        raise ValueError("M3ED event payload timeline mismatch")
    count = _integer(payload.get("event_count"), "event_count")
    if "source_event_count" in payload:
        source_count = _integer(payload["source_event_count"], "source_event_count")
        if count > source_count:
            raise ValueError("M3ED retained event count exceeds source count")
    if "dagr_event_count" in payload:
        dagr_count = _integer(payload["dagr_event_count"], "dagr_event_count")
        if count > dagr_count:
            raise ValueError("M3ED rectified event count exceeds DAGR count")
    shape = getattr(payload.get("events"), "shape", None)
    if shape is None or tuple(shape) != (3, 360, 640):
        raise ValueError("M3ED prepared event tensor geometry mismatch")
    window = _integer(timestamp - previous, "window_us", positive=True)
    return dict(sequence=sequence, timestamp=timestamp, previous_timestamp=previous,
                event_count=count, window_us=window, event_rate_hz=count * 1_000_000.0 / window)


def _records(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Invalid activity record")
        sequence = _sequences([value.get("sequence")], "record sequence")[0]
        timestamp = _integer(value.get("timestamp"), "timestamp")
        previous = _integer(value.get("previous_timestamp"), "previous_timestamp")
        window = _integer(value.get("window_us"), "window_us", positive=True)
        if timestamp - previous != window:
            raise ValueError("Activity record window does not match its timestamps")
        count = _integer(value.get("event_count"), "event_count")
        expected = count * 1_000_000.0 / window
        rate = value.get("event_rate_hz")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValueError("event_rate_hz must be finite")
        if not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Activity record rate does not match count/window")
        key = sequence, timestamp
        if key in seen:
            raise ValueError(f"Duplicate activity frame: {sequence}/{timestamp}")
        seen.add(key)
        records.append(dict(sequence=sequence, timestamp=timestamp, previous_timestamp=previous,
                            event_count=count, window_us=window, event_rate_hz=expected))
    if not records:
        raise ValueError("Activity records are empty")
    return sorted(records, key=lambda r: (r["sequence"], r["timestamp"]))


def build_activity_manifest(records: Iterable[Mapping[str, Any]], train_sequences: Sequence[str],
                            *, reference: Mapping[str, Any],
                            source_metadata_sha256: Mapping[str, str],
                            source_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit boundaries on train records only and assign every frame exactly once."""
    contract = dict(SOURCE_CONTRACT if source_contract is None else source_contract)
    if contract not in (SOURCE_CONTRACT, M3ED_SOURCE_CONTRACT):
        raise ValueError("Activity source contract mismatch")
    rows = _records(records)
    train = _sequences(list(train_sequences), "reference train sequences")
    excluded = tuple(reference.get("excluded_sequences", ()))
    if excluded:
        excluded = _sequences(list(excluded), "excluded sequences")
    if set(train) & set(excluded):
        raise ValueError("Reference train sequences overlap excluded sequences")
    if reference.get("role", "train") != "train":
        raise ValueError("Activity boundaries must be fitted on train only")
    sha = reference.get("manifest_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Reference manifest SHA256 is required")
    sequences = {r["sequence"] for r in rows}
    if set(train) - sequences:
        raise ValueError(f"Missing reference train sequences: {sorted(set(train) - sequences)}")
    if set(source_metadata_sha256) != sequences:
        raise ValueError("Source metadata hashes must cover every activity sequence exactly")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
           for v in source_metadata_sha256.values()):
        raise ValueError("Invalid source metadata SHA256")
    training = [r for r in rows if r["sequence"] in train]
    rates = sorted(r["event_rate_hz"] for r in training)
    cuts = [dict(percentile=p, event_rate_hz=rates[(p * (len(rates) - 1)) // 100])
            for p in PERCENTILES]
    # Equal-valued cutpoints merge their percentile spans, never split ties.
    effective: list[dict[str, Any]] = []
    for cut in cuts:
        if effective and effective[-1]["event_rate_hz"] == cut["event_rate_hz"]:
            effective[-1] = cut
        else:
            effective.append(cut)
    groups: list[dict[str, Any]] = []
    low_p, low_rate = 0, None
    for cut in [*effective, dict(percentile=100, event_rate_hz=None)]:
        high_p, high_rate = cut["percentile"], cut["event_rate_hz"]
        groups.append(dict(name=f"p{low_p:02d}_{high_p:02d}", percentile_start=low_p,
                           percentile_end=high_p, lower_exclusive_hz=low_rate,
                           upper_inclusive_hz=high_rate))
        low_p, low_rate = high_p, high_rate
    thresholds = [v["event_rate_hz"] for v in effective]
    for row in rows:
        row["group"] = groups[bisect.bisect_left(thresholds, row["event_rate_hz"])]["name"]
    counts = Counter(r["group"] for r in rows if r["sequence"] in train)
    for group in groups:
        group["reference_frames"] = counts[group["name"]]
    ref = dict(role="train", name=str(reference.get("name", "unnamed")), manifest_sha256=sha,
               sequences=list(train), excluded_sequences=list(excluded), frames=len(training),
               records_sha256=canonical_sha256([r for r in rows if r["sequence"] in train]))
    version = FORMAT_VERSION if contract == SOURCE_CONTRACT else M3ED_FORMAT_VERSION
    protocol = dict(format_version=version, metric="event_rate_hz", unit="events/s",
                    quantile_method=QUANTILE_METHOD, requested_percentiles=list(PERCENTILES),
                    boundary_equality="lower_group", tie_policy="merge_equal_cutpoints",
                    source_contract=copy.deepcopy(contract), reference=ref, cutpoints=cuts,
                    groups=groups, source_metadata_sha256=dict(sorted(source_metadata_sha256.items())))
    return dict(**protocol, protocol_sha256=canonical_sha256(protocol), records=rows,
                records_sha256=canonical_sha256(rows))


class ActivityIndex:
    """Validated immutable manifest view, keyed by sequence and exact timestamp."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.groups = tuple(g["name"] for g in payload["groups"])
        self._records = {(r["sequence"], r["timestamp"]): r for r in payload["records"]}

    @classmethod
    def load(cls, path: str | Path) -> ActivityIndex:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: Any) -> ActivityIndex:
        if not isinstance(payload, dict) or not isinstance(payload.get("reference"), dict):
            raise ValueError("Invalid activity manifest")
        try:
            rebuilt = build_activity_manifest(payload["records"], payload["reference"]["sequences"],
                reference=payload["reference"], source_metadata_sha256=payload["source_metadata_sha256"],
                source_contract=payload.get("source_contract"))
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete activity manifest") from error
        if rebuilt != payload:
            raise ValueError("Activity manifest protocol, assignment, or digest mismatch")
        return cls(copy.deepcopy(payload))

    @property
    def identity(self) -> dict[str, Any]:
        result = {k: copy.deepcopy(v) for k, v in self._payload.items() if k != "records"}
        result["manifest_sha256"] = canonical_sha256(self._payload)
        return result

    @property
    def source_metadata_sha256(self) -> dict[str, str]:
        return dict(self._payload["source_metadata_sha256"])

    def lookup(self, sequence: str, timestamp: int) -> dict[str, Any]:
        _integer(timestamp, "timestamp")
        try:
            return dict(self._records[sequence, timestamp])
        except KeyError as error:
            raise ValueError(f"Activity frame missing: {sequence}/{timestamp}") from error

    def timestamps(self, sequence: str) -> tuple[int, ...]:
        stamps = tuple(sorted(timestamp for name, timestamp in self._records if name == sequence))
        if not stamps:
            raise ValueError(f"Activity sequence missing: {sequence}")
        return stamps

    def validate_keys(self, keys: Iterable[tuple[str, int]], *,
                      allowed_sequences: Iterable[str] | None = None) -> None:
        allowed = set(allowed_sequences) if allowed_sequences is not None else None
        seen: set[tuple[str, int]] = set()
        for sequence, timestamp in keys:
            key = sequence, timestamp
            if key in seen:
                raise ValueError(f"Duplicate evaluated activity frame: {sequence}/{timestamp}")
            if allowed is not None and sequence not in allowed:
                raise ValueError(f"Activity evaluation sequence outside split: {sequence}")
            seen.add(key)
            self.lookup(sequence, timestamp)


__all__ = ["ActivityIndex", "SOURCE_CONTRACT", "M3ED_SOURCE_CONTRACT", "build_activity_manifest",
           "canonical_sha256", "event_activity_record", "m3ed_event_activity_record",
           "validate_m3ed_activity_metadata", "training_sequences"]
