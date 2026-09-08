#!/usr/bin/env python3
"""Summarize alignment.json files produced by multi-scene visualization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feature alignment by scene")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        help="Defaults to <output stem>_comparisons.csv",
    )
    return parser.parse_args()


def _pooled_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for source in sorted({str(row["source"]) for row in rows}):
        group = [row for row in rows if row["source"] == source]
        frames = sum(int(row["frames"]) for row in group)
        if frames <= 0:
            continue
        mean = sum(
            int(row["frames"]) * float(row["teacher_cosine_mean"])
            for row in group
        ) / frames
        variance = sum(
            int(row["frames"])
            * (
                float(row["teacher_cosine_std"]) ** 2
                + (float(row["teacher_cosine_mean"]) - mean) ** 2
            )
            for row in group
        ) / frames
        delta_weight = sum(max(int(row["frames"]) - 1, 0) for row in group)
        delta = (
            sum(
                max(int(row["frames"]) - 1, 0)
                * float(row["mean_absolute_frame_delta"])
                for row in group
            )
            / delta_weight
            if delta_weight
            else 0.0
        )
        pooled.append(
            {
                "sequence": "__overall__",
                "frames": frames,
                "source": source,
                "feature": group[0]["feature"],
                "teacher_cosine_mean": mean,
                "teacher_cosine_std": variance**0.5,
                "mean_absolute_frame_delta": delta,
            }
        )
    return pooled


def _pooled_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for comparison in sorted({str(row["comparison"]) for row in rows}):
        group = [row for row in rows if row["comparison"] == comparison]
        frames = sum(int(row["frames"]) for row in group)
        if frames <= 0:
            continue
        pooled.append(
            {
                "sequence": "__overall__",
                "frames": frames,
                "comparison": comparison,
                "mean_difference": sum(
                    int(row["frames"]) * float(row["mean_difference"])
                    for row in group
                )
                / frames,
                "positive_fraction": sum(
                    int(row["frames"]) * float(row["positive_fraction"])
                    for row in group
                )
                / frames,
            }
        )
    return pooled


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/alignment.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = int(payload.get("frame_count", 0))
        sources = payload.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"Invalid visualization summary: {path}")
        for label, values in sources.items():
            if not isinstance(values, dict):
                raise ValueError(f"Invalid source summary for {label}: {path}")
            rows.append(
                {
                    "sequence": path.parent.name,
                    "frames": frames,
                    "source": label,
                    "feature": values.get("feature"),
                    "teacher_cosine_mean": values.get("mean_teacher_cosine"),
                    "teacher_cosine_std": values.get("std_teacher_cosine"),
                    "mean_absolute_frame_delta": values.get(
                        "mean_absolute_frame_delta"
                    ),
                }
            )
        comparisons = payload.get("comparisons", {})
        if not isinstance(comparisons, dict):
            raise ValueError(f"Invalid comparison summary: {path}")
        for label, values in comparisons.items():
            if not isinstance(values, dict):
                raise ValueError(f"Invalid comparison for {label}: {path}")
            comparison_rows.append(
                {
                    "sequence": path.parent.name,
                    "frames": frames,
                    "comparison": label,
                    "mean_difference": values.get("mean"),
                    "positive_fraction": values.get("positive_fraction"),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No scene alignment.json files found below {root}")
    return rows + _pooled_source_rows(rows), comparison_rows + _pooled_comparison_rows(
        comparison_rows
    )


def _write_csv(output: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    rows, comparison_rows = collect(root)
    output = args.output.expanduser().resolve()
    comparisons_output = (
        args.comparisons_output.expanduser().resolve()
        if args.comparisons_output is not None
        else output.with_name(f"{output.stem}_comparisons.csv")
    )
    _write_csv(output, rows)
    _write_csv(comparisons_output, comparison_rows)
    print(f"Wrote {len(rows)} source rows to {output}")
    print(f"Wrote {len(comparison_rows)} comparison rows to {comparisons_output}")


if __name__ == "__main__":
    main()
