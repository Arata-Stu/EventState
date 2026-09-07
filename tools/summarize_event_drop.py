#!/usr/bin/env python3
"""Combine event-drop rendering summaries into one compact CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="MODEL=JSON",
        help="Repeat for every rendered event-drop summary",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Result must be MODEL=JSON, got {value!r}")
    model, path = value.split("=", 1)
    if not model:
        raise ValueError(f"Result model is empty: {value!r}")
    return model, Path(path).expanduser()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for specification in args.result:
        model, path = _parse_result(specification)
        payload = json.loads(path.read_text(encoding="utf-8"))
        event_drop = payload.get("event_drop") or {}
        gap_length = int(event_drop.get("length", 0))
        for policy, values in payload["sources"].items():
            rows.append(
                {
                    "model": model,
                    "gap_length": gap_length,
                    "policy": policy,
                    "dropped_frames": int(payload.get("dropped_frame_count", 0)),
                    "mean_teacher_cosine": values["mean_teacher_cosine"],
                    "dropped_teacher_cosine": values["mean_teacher_cosine_dropped"],
                    "observed_teacher_cosine": values["mean_teacher_cosine_observed"],
                    "temporal_frame_delta": values["mean_absolute_frame_delta"],
                }
            )
        for comparison, values in payload.get("comparisons", {}).items():
            rows.append(
                {
                    "model": model,
                    "gap_length": gap_length,
                    "policy": comparison,
                    "dropped_frames": int(payload.get("dropped_frame_count", 0)),
                    "mean_teacher_cosine": values["mean"],
                    "dropped_teacher_cosine": values["dropped_frames"],
                    "observed_teacher_cosine": values["observed_frames"],
                    "temporal_frame_delta": "",
                }
            )
    rows.sort(key=lambda row: (row["model"], row["gap_length"], row["policy"]))
    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "gap_length",
        "policy",
        "dropped_frames",
        "mean_teacher_cosine",
        "dropped_teacher_cosine",
        "observed_teacher_cosine",
        "temporal_frame_delta",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Event-drop summary: {args.output}")


if __name__ == "__main__":
    main()
