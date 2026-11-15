"""Select fixed-duration, continuous scenes from validated event activity.

Selection uses the existing train-derived p20/p80 thresholds, never model
predictions or GT scores. Every original input timestamp within [start,end)
is retained, including frames outside the selected activity group.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from pathlib import Path

from event_state.activity_quantiles import ActivityIndex
from event_state.dense.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[2]


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    micros = round(float(value) * 1_000_000)
    if micros <= 0:
        raise ValueError(f"{name} is below microsecond resolution")
    return micros


def _semantic_test_sequences(path):
    """Read the repository's simple block-list section without a YAML runtime."""
    values, active = [], False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            active = line == "test:"
        elif active:
            item = line.strip()
            if not item.startswith("- "):
                raise ValueError("Unsupported DSEC Semantic test-list structure")
            name = item[2:].strip()
            if not name or any(not (c.isalnum() or c == "_") for c in name):
                raise ValueError("Invalid DSEC Semantic test sequence")
            values.append(name)
    if not values or len(values) != len(set(values)):
        raise ValueError("Missing or duplicate DSEC Semantic test sequences")
    return tuple(values)


def eligible_sequences(dataset, task):
    """Return current common-evaluation sequences and the split source file."""
    if dataset not in {"dsec", "m3ed"} or task not in {"semantic", "depth"}:
        raise ValueError("Expected dataset dsec/m3ed and task semantic/depth")
    if dataset == "dsec" and task == "semantic":
        source = ROOT / "tools/manifests/dsec_semantic_split.yaml"
        return _semantic_test_sequences(source), source
    protocol = load_protocol(dataset + "_f3")
    source = ROOT / "tools/manifests/dense" / (dataset + "_f3.json")
    # M3ED common Semantic uses the same two public-val sequences as Depth.
    return tuple(protocol["splits"]["val"]), source


def continuous_blocks(records, max_gap_us):
    """Never join a sequence boundary, a missing interval, or a long gap."""
    if isinstance(max_gap_us, bool) or not isinstance(max_gap_us, int) or max_gap_us <= 0:
        raise ValueError("max_gap_us must be a positive integer")
    blocks, current = [], []
    for row in records:
        if row["window_us"] > max_gap_us:
            if current:
                blocks.append(current)
                current = []
            continue
        if current and (row["sequence"] != current[-1]["sequence"]
                        or row["previous_timestamp"] != current[-1]["timestamp"]):
            blocks.append(current)
            current = []
        current.append(row)
    if current:
        blocks.append(current)
    return blocks


def _prefix_integral(rows, values):
    ends = [r["timestamp"] for r in rows]
    prefix = [0.0]
    for row, value in zip(rows, values):
        prefix.append(prefix[-1] + row["window_us"] * value)

    def integrate_until(timestamp):
        index = bisect.bisect_right(ends, timestamp)
        result = prefix[index]
        if index < len(rows):
            elapsed = max(0, timestamp - rows[index]["previous_timestamp"])
            result += elapsed * values[index]
        return result

    return integrate_until


def window_candidates(block, duration_us, low_threshold, high_threshold):
    """Full-duration windows anchored at real input timestamps.

    Activity is piecewise constant within each prepared RGB event interval;
    the last interval contributes only its overlap with [start,end).
    """
    timestamps = [r["timestamp"] for r in block]
    measures = [
        _prefix_integral(block, [float(r["event_rate_hz"] <= low_threshold) for r in block]),
        _prefix_integral(block, [float(r["event_rate_hz"] > high_threshold) for r in block]),
        _prefix_integral(block, [r["event_rate_hz"] for r in block]),
    ]
    for start in timestamps:
        end = start + duration_us
        if end > timestamps[-1]:
            break
        low, high, rate = [(fn(end) - fn(start)) / duration_us for fn in measures]
        # Integer-duration inputs should give exact [0,1] fractions; clamp only
        # floating-point accumulation noise, never relax the requested purity.
        low, high = min(1.0, max(0.0, low)), min(1.0, max(0.0, high))
        yield dict(sequence=block[0]["sequence"], start_us=start, end_us=end,
                   low_fraction=low, high_fraction=high,
                   mean_event_rate_hz_interval_weighted=rate,
                   _block_timestamps=timestamps)


def _overlaps(candidate, selected):
    return any(candidate["sequence"] == item["sequence"]
               and candidate["start_us"] < item["end_us"]
               and item["start_us"] < candidate["end_us"] for item in selected)


def select_scenes(activity_manifest, *, dataset, task, duration_seconds=10,
                  min_fraction=0.8, per_kind=1, max_gap_seconds=0.25):
    duration = _seconds(duration_seconds, "duration_seconds")
    max_gap = _seconds(max_gap_seconds, "max_gap_seconds")
    if (isinstance(min_fraction, bool) or not isinstance(min_fraction, (int, float))
            or not math.isfinite(min_fraction) or not 0 < min_fraction <= 1):
        raise ValueError("min_fraction must be in (0,1]")
    if isinstance(per_kind, bool) or not isinstance(per_kind, int) or per_kind <= 0:
        raise ValueError("per_kind must be a positive integer")
    path = Path(activity_manifest).expanduser().resolve()
    index = ActivityIndex.load(path)
    identity = index.identity
    if identity["source_contract"]["dataset"] != dataset:
        raise ValueError("Activity manifest dataset mismatch")
    allowed, split_source = eligible_sequences(dataset, task)
    if set(allowed) & set(identity["reference"]["sequences"]):
        raise ValueError("Scene split overlaps activity-reference train pool")
    thresholds = {c["percentile"]: c["event_rate_hz"] for c in identity["cutpoints"]}
    pools = {"low": [], "high": []}
    timelines = {}
    block_count = 0
    for sequence in allowed:
        timelines[sequence] = list(index.timestamps(sequence))
        records = [index.lookup(sequence, t) for t in timelines[sequence]]
        blocks = continuous_blocks(records, max_gap)
        block_count += len(blocks)
        for block in blocks:
            for candidate in window_candidates(block, duration, thresholds[20], thresholds[80]):
                for kind in pools:
                    if candidate[kind + "_fraction"] >= min_fraction:
                        pools[kind].append(dict(candidate, kind=kind))
    for kind, candidates in pools.items():
        direction = 1 if kind == "low" else -1
        candidates.sort(key=lambda c: (-c[kind + "_fraction"],
            direction * c["mean_event_rate_hz_interval_weighted"], c["sequence"], c["start_us"]))
    selected, counts = [], {"low": 0, "high": 0}
    # Round-robin selection avoids filling every slot of one kind first.
    for _ in range(per_kind):
        for kind in pools:
            for candidate in pools[kind]:
                if not _overlaps(candidate, selected):
                    selected.append(candidate)
                    counts[kind] += 1
                    break
    scenes = []
    for candidate in selected:
        source_timestamps = candidate["_block_timestamps"]
        start, end = candidate["start_us"], candidate["end_us"]
        stamps = source_timestamps[bisect.bisect_left(source_timestamps, start):
                                   bisect.bisect_left(source_timestamps, end)]
        assert stamps and stamps[0] == start
        scene = {k: v for k, v in candidate.items() if not k.startswith("_")}
        scene.update(id=f"{dataset}_{task}_{candidate['kind']}_{len(scenes)+1:02d}",
                     duration_us=duration, frames=len(stamps), timestamps_us=stamps,
                     warmup_frames_before_start=bisect.bisect_left(timelines[candidate["sequence"]], start),
                     state_policy="continuous_from_sequence_start")
        scenes.append(scene)
    shortfalls = [dict(kind=kind, requested=per_kind, selected=counts[kind],
                       reason="insufficient_nonoverlapping_windows_at_requested_duration_and_fraction")
                  for kind in pools if counts[kind] < per_kind]
    return dict(format_version=1, dataset=dataset, task=task,
        activity_manifest=str(path), activity_manifest_sha256=identity["manifest_sha256"],
        selection=dict(duration_seconds=duration / 1_000_000, min_fraction=min_fraction,
            per_kind=per_kind, max_gap_seconds=max_gap / 1_000_000,
            eligible_sequences=list(allowed), split_source=str(split_source),
            split_source_sha256=hashlib.sha256(split_source.read_bytes()).hexdigest(),
            interval_convention="[start_us,end_us)", anchor="original_input_timestamp",
            activity_weighting="RGB-interval duration overlap; piecewise-constant interval rates",
            low_threshold_hz=thresholds[20], high_threshold_hz=thresholds[80],
            threshold_policy="low <= train p20; high > train p80; ties preserved",
            selection_uses="input event activity only; no predictions or GT scores",
            all_input_frames_preserved=True, source_continuous_blocks=block_count,
            eligible_candidates={kind: len(pool) for kind, pool in pools.items()}),
        status="complete" if not shortfalls else "insufficient_candidates",
        counts=counts, shortfalls=shortfalls, scenes=scenes)


__all__ = ["continuous_blocks", "eligible_sequences", "select_scenes", "window_candidates"]
