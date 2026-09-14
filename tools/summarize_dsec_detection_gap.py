#!/usr/bin/env python3
"""Combine DSEC-Detection event-gap JSON files into a compact CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("mAP", "AP50", "AP75", "AP_car", "AP_pedestrian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.expanduser().glob("*/gap*.json"))
    if not paths:
        raise FileNotFoundError(f"No gap results found under {args.input_dir}")
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = path.parent.name
        gap = int(payload["gap"]["length"])
        for subset, value in payload["subsets"].items():
            metrics = value.get("metrics") or {}
            row = {
                "source": source,
                "gap_length": gap,
                "subset": subset,
                "evaluated_frames": value["evaluated_frames"],
                "ground_truth_boxes": value["ground_truth_boxes"],
            }
            row.update({metric: metrics.get(metric) for metric in METRICS})
            records.append(row)
    baseline = {
        row["source"]: row.get("mAP")
        for row in records
        if row["gap_length"] == 0 and row["subset"] == "overall"
    }
    for row in records:
        base = baseline.get(row["source"])
        value = row.get("mAP")
        row["mAP_delta_from_gap0"] = (
            value - base if value is not None and base is not None else None
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source",
        "gap_length",
        "subset",
        "evaluated_frames",
        "ground_truth_boxes",
        *METRICS,
        "mAP_delta_from_gap0",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
