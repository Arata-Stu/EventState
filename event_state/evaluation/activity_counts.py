"""Absolute count and persistence diagnostics for existing activity groups.

These are input-timeline statistics, not GT-supported task scores.  All counts
refer to the manifest's prepared events in each actual RGB interval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import fmean

from event_state.activity_quantiles import ActivityIndex, QUANTILE_METHOD
from event_state.evaluation.activity_scenes import continuous_blocks, eligible_sequences


MAX_GAP_US = 250_000
PERCENTILES = {"p05": 5, "p20": 20, "p40": 40, "median": 50,
               "p60": 60, "p80": 80, "p95": 95}


def distribution(values):
    """Use the existing manifest's lower-order statistic convention."""
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "min": None, "mean": None,
                **{name: None for name in PERCENTILES}, "max": None}
    return {"n": len(ordered), "min": ordered[0], "mean": fmean(ordered),
            **{name: ordered[p * (len(ordered) - 1) // 100]
               for name, p in PERCENTILES.items()}, "max": ordered[-1]}


def _interval_summary(rows, denominator_us):
    duration_us = sum(r["window_us"] for r in rows)
    count = sum(r["event_count"] for r in rows)
    return {"frames": len(rows), "observed_duration_seconds": duration_us / 1e6,
            "duration_fraction": duration_us / denominator_us if denominator_us else None,
            "total_event_count": count,
            "time_weighted_event_rate_hz": count * 1e6 / duration_us if duration_us else None,
            **{key: distribution(r[key] for r in rows)
               for key in ("event_count", "window_us", "event_rate_hz")}}


def _temporal_summary(rows, low_group, high_group):
    """Observed runs only: never infer persistence through unobserved time."""
    missing = 0
    missing_us = 0
    previous = None
    for row in rows:
        if previous is not None and row["sequence"] == previous["sequence"]:
            gap = row["previous_timestamp"] - previous["timestamp"]
            if gap < 0:
                raise ValueError("Overlapping event intervals cannot be time-weighted")
            if gap > 0:
                missing += 1
                missing_us += gap
        previous = row
    blocks = continuous_blocks(rows, MAX_GAP_US)
    runs = {"low": [], "high": []}
    for kind, group in (("low", low_group), ("high", high_group)):
        for block in blocks:
            duration = 0
            for row in block:
                if row["group"] == group:
                    duration += row["window_us"]
                elif duration:
                    runs[kind].append(duration / 1e6)
                    duration = 0
            if duration:
                runs[kind].append(duration / 1e6)
    long = [r for r in rows if r["window_us"] > MAX_GAP_US]
    return {"max_interval_us": MAX_GAP_US, "continuous_blocks": len(blocks),
            "missing_gap_count": missing, "missing_gap_seconds": missing_us / 1e6,
            "excluded_long_intervals": len(long),
            "excluded_long_interval_seconds": sum(r["window_us"] for r in long) / 1e6,
            "run_duration_seconds": {
                kind: {"group": group, **distribution(runs[kind])}
                for kind, group in (("low", low_group), ("high", high_group))}}


def _summarize_rows(rows, groups):
    denominator_us = sum(r["window_us"] for r in rows)
    return {"all": _interval_summary(rows, denominator_us),
            "groups": {g: _interval_summary([r for r in rows if r["group"] == g], denominator_us)
                       for g in groups},
            "temporal": _temporal_summary(rows, groups[0], groups[-1])}


def summarize_activity_counts(activity_manifest, *, dataset, task):
    """Validate the original assignments and summarize train/task input streams."""
    path = Path(activity_manifest)
    activity = ActivityIndex.load(path)
    identity = activity.identity
    if identity["source_contract"]["dataset"] != dataset:
        raise ValueError("Dataset differs from activity manifest")
    evaluation_sequences, split_source = eligible_sequences(dataset, task)
    reference_sequences = tuple(identity["reference"]["sequences"])
    if set(evaluation_sequences) & set(reference_sequences):
        raise ValueError("Activity reference train and task evaluation sequences overlap")

    def summarize_split(sequences):
        by_sequence = {sequence: [activity.lookup(sequence, stamp)
                                for stamp in activity.timestamps(sequence)]
                       for sequence in sequences}
        rows = [r for sequence in sorted(by_sequence) for r in by_sequence[sequence]]
        result = _summarize_rows(rows, activity.groups)
        result["sequences"] = list(sequences)
        result["per_sequence"] = {name: _summarize_rows(values, activity.groups)
                                  for name, values in by_sequence.items()}
        return result

    return {"format_version": 1, "dataset": dataset, "task": task,
            "activity_manifest": str(path.resolve()),
            "activity_manifest_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "activity_manifest_identity": identity,
            "evaluation_split_source": str(split_source.resolve()),
            "evaluation_split_source_sha256": hashlib.sha256(split_source.read_bytes()).hexdigest(),
            "sample_scope": "all_prepared_input_intervals_in_selected_sequences; not_GT_evaluation_support",
            "units": {"event_count": "prepared_events_per_actual_RGB_interval",
                      "window_us": "microseconds", "event_rate_hz": "prepared_events/second"},
            "percentile_method": QUANTILE_METHOD,
            "notes": [
                "Existing train-fitted event-rate groups are preserved; no boundaries are refitted.",
                "Count and rate quantiles describe separate marginal distributions, not converted boundaries.",
                "Time fractions use summed observed input interval durations; missing gaps are not included.",
                "Runs stop at sequence boundaries, missing intervals, and intervals longer than 250 ms.",
                "Run lengths are observed durations and can be censored by recording boundaries or missing data.",
                "DSEC and M3ED prepared counts have different source contracts; these are not matched raw sensor counts.",
            ],
            "reference_train": summarize_split(reference_sequences),
            "evaluation": summarize_split(evaluation_sequences)}


def print_summary(report):
    print(f"{report['dataset'].upper()} {report['task']}: all prepared input intervals "
          "(not GT-supported evaluation frames)")
    total = report["evaluation"]["all"]
    print(f"  all: n={total['frames']} count min..max="
          f"{total['event_count']['min']}..{total['event_count']['max']} events/RGB interval; "
          f"rate min..max={total['event_rate_hz']['min']}..{total['event_rate_hz']['max']} events/s")
    for group, summary in report["evaluation"]["groups"].items():
        count, rate = summary["event_count"], summary["event_rate_hz"]
        fraction = summary["duration_fraction"]
        print(f"  {group}: n={summary['frames']} count median={count['median']} "
              f"p05..p95={count['p05']}..{count['p95']} events/RGB interval; "
              f"window median={summary['window_us']['median']} us; "
              f"rate median={rate['median']} events/s; time={fraction:.1%}")
    for kind, run in report["evaluation"]["temporal"]["run_duration_seconds"].items():
        print(f"  {kind} ({run['group']}) runs: n={run['n']}, "
              f"median={run['median']}s p95={run['p95']}s max={run['max']}s")


__all__ = ["summarize_activity_counts", "print_summary"]
