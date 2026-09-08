#!/usr/bin/env python3
"""Summarize EventState training logs without dumping individual iterations."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LINE_RE = re.compile(r"^\[(train|validation)\]\s+step=(\d+)\s+(.*)$")
VALUE_RE = re.compile(
    r"([^\s=]+)=([-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Record:
    step: int
    values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report recent loss trends, gradient health, throughput, and optional "
            "validation summaries from EventState console logs."
        )
    )
    parser.add_argument("logs", nargs="+", type=Path, help="One or more *.log files")
    parser.add_argument(
        "--window-steps",
        type=int,
        default=10_000,
        help="Recent train interval used for averaging and trend estimation (default: 10000)",
    )
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, list[Record]]:
    records: dict[str, list[Record]] = {"train": [], "validation": []}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = LINE_RE.match(line.strip())
            if match is None:
                continue
            stage, step_text, body = match.groups()
            values = {key: float(value) for key, value in VALUE_RE.findall(body)}
            records[stage].append(Record(step=int(step_text), values=values))
    return records


def finite_values(records: Iterable[Record], key: str) -> list[float]:
    return [
        record.values[key]
        for record in records
        if key in record.values and math.isfinite(record.values[key])
    ]


def mean(records: Iterable[Record], key: str) -> float | None:
    values = finite_values(records, key)
    return statistics.fmean(values) if values else None


def percentile(records: Iterable[Record], key: str, fraction: float) -> float | None:
    values = sorted(finite_values(records, key))
    if not values:
        return None
    index = round((len(values) - 1) * fraction)
    return values[index]


def relative_half_change(records: list[Record], key: str) -> float | None:
    if len(records) < 4:
        return None
    midpoint = len(records) // 2
    first = mean(records[:midpoint], key)
    second = mean(records[midpoint:], key)
    if first is None or second is None or abs(first) < 1e-12:
        return None
    return 100.0 * (second - first) / abs(first)


def relative_change(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or abs(first) < 1e-12:
        return None
    return 100.0 * (second - first) / abs(first)


def fmt(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def fmt_change(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def run_name(path: Path) -> str:
    name = path.stem
    prefix = name.split("_", 1)[0].upper()
    return prefix if prefix in {"E0", "E1", "E2", "E3", "E4"} else name


def summarize_train(path: Path, records: list[Record], window_steps: int) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if not records:
        return [run_name(path), "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "ログなし"], warnings

    last_step = max(record.step for record in records)
    first_step = min(record.step for record in records)
    initial = [record for record in records if record.step <= first_step + window_steps]
    recent = [record for record in records if record.step > last_step - window_steps]
    skipped = sum(finite_values(recent, "optimizer_step_skipped"))
    non_finite = sum(
        1 for record in recent for value in record.values.values() if not math.isfinite(value)
    )
    grad_p95 = percentile(recent, "grad_norm", 0.95)
    initial_loss = mean(initial, "loss")
    recent_loss = mean(recent, "loss")
    total_change = relative_change(initial_loss, recent_loss)
    loss_change = relative_half_change(recent, "loss")

    if non_finite:
        warnings.append(f"{run_name(path)}: 直近区間に非有限値が{non_finite}個あります")
    if skipped:
        warnings.append(f"{run_name(path)}: 直近区間でoptimizer stepを{int(skipped)}回skipしています")
    if grad_p95 is not None and grad_p95 > 10.0:
        warnings.append(f"{run_name(path)}: grad_norm p95={grad_p95:.3g}は大きめです")
    if loss_change is not None and loss_change > 5.0:
        warnings.append(f"{run_name(path)}: 直近区間のlossが増加傾向です ({loss_change:+.1f}%)")

    return [
        run_name(path),
        str(last_step),
        fmt(initial_loss),
        fmt(recent_loss),
        fmt_change(total_change),
        fmt_change(loss_change),
        fmt(mean(recent, "h_distill_loss")),
        fmt(mean(recent, "z_distill_loss")),
        fmt(mean(recent, "h_projected_cosine")),
        fmt(mean(recent, "z_projected_cosine")),
        f"{fmt(percentile(recent, 'grad_norm', 0.5))}/{fmt(grad_p95)}",
        fmt(mean(recent, "samples_per_second")),
    ], warnings


def summarize_validation(path: Path, records: list[Record]) -> list[str] | None:
    usable = [record for record in records if "val/loss" in record.values]
    if not usable:
        return None
    latest = max(usable, key=lambda record: record.step)
    best = min(usable, key=lambda record: record.values["val/loss"])
    recent = sorted(usable, key=lambda record: record.step)[-6:]
    return [
        run_name(path),
        str(latest.step),
        fmt(latest.values.get("val/loss")),
        f"{fmt(best.values.get('val/loss'))}@{best.step}",
        fmt_change(relative_half_change(recent, "val/loss")),
        fmt(latest.values.get("val/h_projected_cosine")),
        fmt(latest.values.get("val/z_projected_cosine")),
    ]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    if args.window_steps <= 0:
        raise SystemExit("error: --window-steps must be positive")

    paths = sorted(dict.fromkeys(path.expanduser() for path in args.logs))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("error: log not found: " + ", ".join(str(path) for path in missing))

    train_rows: list[list[str]] = []
    validation_rows: list[list[str]] = []
    warnings: list[str] = []
    for path in paths:
        records = parse_log(path)
        train_row, train_warnings = summarize_train(
            path, records["train"], args.window_steps
        )
        train_rows.append(train_row)
        warnings.extend(train_warnings)
        validation_row = summarize_validation(path, records["validation"])
        if validation_row is not None:
            validation_rows.append(validation_row)

    print(f"TRAIN: 直近 {args.window_steps:,} steps の集約")
    print_table(
        [
            "run",
            "step",
            "loss初期",
            "loss直近",
            "全体Δ",
            "直近Δ",
            "h_loss",
            "z_loss",
            "h_proj_cos",
            "z_proj_cos",
            "grad med/p95",
            "sample/s",
        ],
        train_rows,
    )

    if validation_rows:
        print("\nVALIDATION: 最新値と最良値")
        print_table(
            ["run", "step", "loss", "best@step", "recentΔ", "h_proj_cos", "z_proj_cos"],
            validation_rows,
        )
    else:
        print("\nVALIDATION: なし（train-only final fit）")

    print("\nHEALTH:")
    if warnings:
        for warning in warnings:
            print(f"- WARNING: {warning}")
    else:
        print("- 非有限値、optimizer skip、顕著な勾配増大は検出されませんでした")


if __name__ == "__main__":
    main()
