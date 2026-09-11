#!/usr/bin/env python3
"""Compact live summary of DSEC-Detection fine-tune or scratch training."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = ("mAP", "AP50", "AP75", "AP_car", "AP_pedestrian")
TRAIN_PROGRESS_RE = re.compile(
    rb"DSEC-Det (finetune|scratch) (\d+)/(\d+):\s*"
    rb"(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)"
)
VALIDATION_PROGRESS_RE = re.compile(
    rb"DSEC-Det continuous validation:\s*(\d+)%\|[^\r\n]*?\|\s*(\d+)/(\d+)"
)


@dataclass(frozen=True)
class Progress:
    position: int
    stage: str
    epoch: int | None
    total_epochs: int | None
    percent: int
    current: int
    total: int


@dataclass(frozen=True)
class Run:
    label: str
    directory: Path
    metrics_path: Path
    log_path: Path
    config: dict[str, Any]
    records: list[dict[str, Any]]
    progress: Progress | None
    malformed_lines: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize current DSEC-Detection E0/E2/E4 training, validation, "
            "loss trends, and log progress without dumping JSONL records"
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Detection output root, e.g. outputs/dsec_detection_end_to_end",
    )
    parser.add_argument("--mode", choices=("finetune", "scratch"), default="finetune")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("E0", "E2", "E4"),
        help="Run directory names below MODE (default: E0 E2 E4)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of recent validation epochs in the comparison table",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Refresh continuously at this interval; omit for one snapshot",
    )
    parser.add_argument(
        "--log-tail-mib",
        type=int,
        default=4,
        help="Maximum tail of each console log inspected for live tqdm progress",
    )
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    if not path.is_file():
        return records, malformed
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                malformed += 1
    records.sort(key=lambda record: int(record.get("epoch", -1)))
    return records, malformed


def _read_tail(path: Path, maximum_bytes: int) -> bytes:
    if not path.is_file():
        return b""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read()


def _parse_progress(path: Path, maximum_bytes: int) -> Progress | None:
    content = _read_tail(path, maximum_bytes)
    candidates: list[Progress] = []
    train_progress: list[Progress] = []
    for match in TRAIN_PROGRESS_RE.finditer(content):
        progress = Progress(
            position=match.start(),
            stage="train",
            epoch=int(match.group(2)),
            total_epochs=int(match.group(3)),
            percent=int(match.group(4)),
            current=int(match.group(5)),
            total=int(match.group(6)),
        )
        train_progress.append(progress)
        candidates.append(progress)
    for match in VALIDATION_PROGRESS_RE.finditer(content):
        preceding = [value for value in train_progress if value.position < match.start()]
        training = max(preceding, key=lambda value: value.position) if preceding else None
        candidates.append(
            Progress(
                position=match.start(),
                stage="validation",
                epoch=training.epoch if training is not None else None,
                total_epochs=training.total_epochs if training is not None else None,
                percent=int(match.group(1)),
                current=int(match.group(2)),
                total=int(match.group(3)),
            )
        )
    return max(candidates, key=lambda value: value.position) if candidates else None


def _load_run(
    root: Path,
    mode: str,
    label: str,
    seed: int,
    log_tail_bytes: int,
) -> Run:
    directory = root / mode / label / f"seed_{seed}"
    metrics_path = directory / "metrics.jsonl"
    log_path = root / "logs" / f"seed_{seed}" / f"{mode}_{label}.log"
    records, malformed = _read_jsonl(metrics_path)
    return Run(
        label=label,
        directory=directory,
        metrics_path=metrics_path,
        log_path=log_path,
        config=_read_json(directory / "config.json"),
        records=records,
        progress=_parse_progress(log_path, log_tail_bytes),
        malformed_lines=malformed,
    )


def _format(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_scientific(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    def display_width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            for character in value
        )

    def pad(value: str, width: int) -> str:
        return value + " " * (width - display_width(value))

    widths = [display_width(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    print("  ".join(pad(header, widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(pad(value, widths[index]) for index, value in enumerate(row)))


def _train_change(records: list[dict[str, Any]], window: int = 3) -> float | None:
    values = [
        value
        for record in records
        if (value := _finite(record.get("train/loss"))) is not None
    ]
    if len(values) < 2:
        return None
    current = statistics.fmean(values[-min(window, len(values)) :])
    previous_end = max(0, len(values) - min(window, len(values)))
    previous_start = max(0, previous_end - window)
    previous_values = values[previous_start:previous_end]
    if not previous_values:
        previous = values[0]
    else:
        previous = statistics.fmean(previous_values)
    if abs(previous) < 1e-12:
        return None
    return 100.0 * (current - previous) / abs(previous)


def _validation_records(run: Run) -> list[dict[str, Any]]:
    return [record for record in run.records if _finite(record.get("val/mAP")) is not None]


def _best_validation(run: Run) -> dict[str, Any] | None:
    records = _validation_records(run)
    return max(records, key=lambda record: float(record["val/mAP"])) if records else None


def _stage(run: Run) -> str:
    progress = run.progress
    if progress is None:
        return "待機/未検出"
    if progress.stage == "train":
        return (
            f"train {progress.epoch}/{progress.total_epochs} "
            f"{progress.percent}% ({progress.current}/{progress.total})"
        )
    validation_epoch = progress.epoch
    if validation_epoch is None:
        validation_epoch = (
            int(run.records[-1].get("epoch", -1)) + 2 if run.records else 1
        )
    return (
        f"val after ep{validation_epoch} "
        f"{progress.percent}% ({progress.current}/{progress.total})"
    )


def _epoch_target(run: Run) -> str:
    completed = int(run.records[-1].get("epoch", -1)) + 1 if run.records else 0
    target = run.config.get("epochs", "?")
    return f"{completed}/{target}"


def _summary_rows(runs: list[Run]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        latest = run.records[-1] if run.records else {}
        latest_validation = _validation_records(run)
        latest_validation_record = latest_validation[-1] if latest_validation else None
        best = _best_validation(run)
        change = _train_change(run.records)
        best_text = "-"
        if best is not None:
            best_text = f"{_format(_finite(best.get('val/mAP')))}@{int(best['epoch']) + 1}"
        latest_val_text = "-"
        if latest_validation_record is not None:
            latest_val_text = (
                f"{_format(_finite(latest_validation_record.get('val/mAP')))}"
                f"@{int(latest_validation_record['epoch']) + 1}"
            )
        rows.append(
            [
                run.label,
                _stage(run),
                _epoch_target(run),
                _format(_finite(latest.get("train/loss"))),
                "-" if change is None else f"{change:+.1f}%",
                _format(_finite(latest.get("train/loss_box"))),
                _format(_finite(latest.get("train/loss_objectness"))),
                _format(_finite(latest.get("train/loss_classification"))),
                _format(_finite(latest.get("train/grad_norm")), 3),
                _format_scientific(_finite(latest.get("learning_rate/head"))),
                latest_val_text,
                best_text,
            ]
        )
    return rows


def _history_rows(runs: list[Run], count: int) -> list[list[str]]:
    by_run = {
        run.label: {
            int(record["epoch"]) + 1: record for record in _validation_records(run)
        }
        for run in runs
    }
    epochs = sorted({epoch for values in by_run.values() for epoch in values})[-count:]
    rows: list[list[str]] = []
    for epoch in epochs:
        row = [str(epoch)]
        for run in runs:
            record = by_run[run.label].get(epoch)
            row.extend(
                [
                    _format(_finite(record.get("val/mAP")) if record else None),
                    _format(_finite(record.get("val/AP50")) if record else None),
                    _format(_finite(record.get("val/AP75")) if record else None),
                ]
            )
        rows.append(row)
    return rows


def _health(runs: list[Run]) -> list[str]:
    messages: list[str] = []
    for run in runs:
        if not run.directory.is_dir():
            messages.append(f"{run.label}: run directoryがありません")
            continue
        if not run.records:
            messages.append(f"{run.label}: 完了epochのmetricsがまだありません")
        if run.malformed_lines:
            messages.append(
                f"{run.label}: 読み取れないJSONL行が{run.malformed_lines}行あります"
            )
        non_finite = sum(
            1
            for record in run.records
            for value in record.values()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        )
        if non_finite:
            messages.append(f"{run.label}: 非有限値が{non_finite}個あります")
        change = _train_change(run.records)
        if change is not None and change > 10.0:
            messages.append(f"{run.label}: 直近train lossが増加傾向です ({change:+.1f}%)")
        if not _validation_records(run):
            messages.append(f"{run.label}: validation結果がまだありません")
    return messages


def _render(root: Path, mode: str, seed: int, runs: list[Run], history: int) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"DSEC-Det {mode} seed={seed} | {timestamp}")
    print(f"root: {root}")
    print("\nCURRENT")
    _print_table(
        [
            "run",
            "現在位置",
            "完了ep",
            "train loss",
            "Δ3ep",
            "box",
            "obj",
            "cls",
            "grad",
            "head LR",
            "latest val",
            "best val",
        ],
        _summary_rows(runs),
    )

    history_rows = _history_rows(runs, history)
    print("\nVALIDATION HISTORY (mAP / AP50 / AP75)")
    if history_rows:
        headers = ["epoch"]
        for run in runs:
            headers.extend((f"{run.label} mAP", "AP50", "AP75"))
        _print_table(headers, history_rows)
    else:
        print("validation結果はまだありません")

    messages = _health(runs)
    print("\nHEALTH")
    if messages:
        for message in messages:
            print(f"- {message}")
    else:
        print("- JSON破損、非有限値、顕著なloss増加は検出されませんでした")
    print(
        "\n注: train値は直近の完了epoch、"
        "validationはvalidate-every時点の値です。"
    )


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.history <= 0 or args.log_tail_mib <= 0:
        raise SystemExit("error: seed must be non-negative; history/log-tail-mib positive")
    if args.watch is not None and args.watch <= 0:
        raise SystemExit("error: --watch must be positive")
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"error: detection output root not found: {root}")
    labels = list(dict.fromkeys(args.models))
    while True:
        runs = [
            _load_run(
                root,
                args.mode,
                label,
                args.seed,
                args.log_tail_mib * 1024 * 1024,
            )
            for label in labels
        ]
        if args.watch is not None and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        _render(root, args.mode, args.seed, runs, args.history)
        if args.watch is None:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nmonitor stopped")
            break


if __name__ == "__main__":
    main()
