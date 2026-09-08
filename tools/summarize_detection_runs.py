#!/usr/bin/env python3
"""Compact best-validation summaries for frozen/fine-tune/scratch runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = ("mAP", "AP50", "AP75", "AP_car", "AP_pedestrian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the best validation epoch and aggregate detection seeds"
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output")
    return parser.parse_args()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_best(path: Path) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            score = _finite_float(record.get("val/mAP"))
            if score is None:
                continue
            if best is None or score > best["mAP"]:
                best = {
                    "epoch": int(record.get("epoch", -1)) + 1,
                    **{
                        metric: _finite_float(record.get(f"val/{metric}"))
                        for metric in METRICS
                    },
                }
    if best is None:
        raise ValueError(f"No validation mAP found: {path}")
    return best


def _run_name(root: Path, path: Path) -> tuple[str, int | None]:
    relative = path.parent.relative_to(root)
    parts: list[str] = []
    seed: int | None = None
    for part in relative.parts:
        if part.startswith("seed_") and part[5:].isdigit():
            seed = int(part[5:])
        else:
            parts.append(part)
    return "/".join(parts) or root.name, seed


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for original_root in roots:
        root = original_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Detection output root not found: {root}")
        paths = sorted(root.rglob("metrics.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No metrics.jsonl found below {root}")
        for path in paths:
            run, seed = _run_name(root, path)
            rows.append(
                {
                    "root": str(root),
                    "run": run,
                    "seed": seed,
                    "metrics_path": str(path),
                    **_read_best(path),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["root"], row["run"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (root, run), members in sorted(grouped.items()):
        result: dict[str, Any] = {"root": root, "run": run, "seeds": len(members)}
        for metric in METRICS:
            values = [row[metric] for row in members if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_std"] = statistics.pstdev(values) if values else None
        output.append(result)
    return output


def _format(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    rows = collect(args.roots)
    summary = aggregate(rows)
    print("run\tseeds\tmAP mean±std\tAP50\tAP75\tAP car\tAP pedestrian")
    for row in summary:
        map_value = f"{_format(row['mAP_mean'])}±{_format(row['mAP_std'])}"
        print(
            "\t".join(
                (
                    row["run"],
                    str(row["seeds"]),
                    map_value,
                    _format(row["AP50_mean"]),
                    _format(row["AP75_mean"]),
                    _format(row["AP_car_mean"]),
                    _format(row["AP_pedestrian_mean"]),
                )
            )
        )
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"runs": rows, "summary": summary}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
