#!/usr/bin/env python3
"""Compare frozen-head and end-to-end DSEC-Detection learning curves."""

from __future__ import annotations

import argparse
import math
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Curve:
    label: str
    regime: str
    path: Path
    records: list[dict[str, Any]]
    malformed_lines: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-backbone and end-to-end fine-tune DSEC-Detection "
            "loss/validation trajectories at matched evaluation epochs"
        )
    )
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--finetune-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=("E0", "E2", "E4"))
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of matched validation epochs to show (default: 10)",
    )
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    import json

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


def _curve(root: Path, regime: str, label: str, seed: int) -> Curve:
    if regime == "frozen":
        path = root / label / f"seed_{seed}" / "metrics.jsonl"
    else:
        path = root / "finetune" / label / f"seed_{seed}" / "metrics.jsonl"
    records, malformed = _read_jsonl(path)
    return Curve(label, regime, path, records, malformed)


def _validation(curve: Curve) -> list[dict[str, Any]]:
    return [
        record for record in curve.records if _finite(record.get("val/mAP")) is not None
    ]


def _train_losses(curve: Curve) -> list[float]:
    return [
        value
        for record in curve.records
        if (value := _finite(record.get("train/loss"))) is not None
    ]


def _best(curve: Curve) -> dict[str, Any] | None:
    values = _validation(curve)
    return max(values, key=lambda record: float(record["val/mAP"])) if values else None


def _relative_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or abs(first) < 1e-12:
        return None
    return 100.0 * (last - first) / abs(first)


def _recent_change(values: list[float], window: int = 3) -> float | None:
    if len(values) < 2:
        return None
    size = min(window, len(values) // 2)
    if size == 0:
        return None
    previous = statistics.fmean(values[-2 * size : -size])
    current = statistics.fmean(values[-size:])
    return _relative_change(previous, current)


def _epoch(record: dict[str, Any]) -> int:
    return int(record.get("epoch", -1)) + 1


def _first_epoch_at_fraction(curve: Curve, fraction: float) -> int | None:
    best = _best(curve)
    if best is None:
        return None
    threshold = float(best["val/mAP"]) * fraction
    for record in _validation(curve):
        if float(record["val/mAP"]) >= threshold:
            return _epoch(record)
    return None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_change(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}%"


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [_display_width(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], _display_width(value))

    def pad(value: str, width: int) -> str:
        return value + " " * (width - _display_width(value))

    print("  ".join(pad(header, widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(pad(value, widths[index]) for index, value in enumerate(row)))


def _validation_summary(
    frozen: dict[str, Curve], finetune: dict[str, Curve], labels: list[str]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for label in labels:
        first = frozen[label]
        second = finetune[label]
        frozen_values = _validation(first)
        finetune_values = _validation(second)
        frozen_best = _best(first)
        finetune_best = _best(second)
        frozen_score = _finite(frozen_best.get("val/mAP")) if frozen_best else None
        finetune_score = (
            _finite(finetune_best.get("val/mAP")) if finetune_best else None
        )
        delta = (
            finetune_score - frozen_score
            if frozen_score is not None and finetune_score is not None
            else None
        )
        rows.append(
            [
                label,
                (
                    f"{_fmt(frozen_score)}@{_epoch(frozen_best)}"
                    if frozen_best is not None
                    else "-"
                ),
                (
                    f"{_fmt(finetune_score)}@{_epoch(finetune_best)}"
                    if finetune_best is not None
                    else "-"
                ),
                _fmt(delta),
                _fmt(
                    _finite(frozen_values[-1].get("val/mAP"))
                    if frozen_values
                    else None
                ),
                _fmt(
                    _finite(finetune_values[-1].get("val/mAP"))
                    if finetune_values
                    else None
                ),
                str(_first_epoch_at_fraction(first, 0.95) or "-"),
                str(_first_epoch_at_fraction(second, 0.95) or "-"),
            ]
        )
    return rows


def _loss_summary(
    frozen: dict[str, Curve], finetune: dict[str, Curve], labels: list[str]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for label in labels:
        for regime, curve in (("frozen", frozen[label]), ("finetune", finetune[label])):
            values = _train_losses(curve)
            rows.append(
                [
                    label,
                    regime,
                    str(len(values)),
                    _fmt(values[0] if values else None),
                    _fmt(values[-1] if values else None),
                    _fmt_change(
                        _relative_change(values[0], values[-1]) if values else None
                    ),
                    _fmt_change(_recent_change(values)),
                ]
            )
    return rows


def _matched_history(
    frozen: dict[str, Curve],
    finetune: dict[str, Curve],
    labels: list[str],
    history: int,
) -> list[list[str]]:
    frozen_by_epoch = {
        label: {_epoch(record): record for record in _validation(frozen[label])}
        for label in labels
    }
    finetune_by_epoch = {
        label: {_epoch(record): record for record in _validation(finetune[label])}
        for label in labels
    }
    # Fine-tune determines the comparison cadence because it validates less often.
    epochs = sorted(
        {
            epoch
            for label in labels
            for epoch in finetune_by_epoch[label]
            if epoch in frozen_by_epoch[label]
        }
    )[-history:]
    rows: list[list[str]] = []
    for epoch in epochs:
        row = [str(epoch)]
        for label in labels:
            frozen_record = frozen_by_epoch[label].get(epoch)
            finetune_record = finetune_by_epoch[label].get(epoch)
            frozen_score = (
                _finite(frozen_record.get("val/mAP")) if frozen_record else None
            )
            finetune_score = (
                _finite(finetune_record.get("val/mAP")) if finetune_record else None
            )
            delta = (
                finetune_score - frozen_score
                if frozen_score is not None and finetune_score is not None
                else None
            )
            row.extend((_fmt(frozen_score), _fmt(finetune_score), _fmt(delta)))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.history <= 0:
        raise SystemExit("error: seed must be non-negative and history must be positive")
    frozen_root = args.frozen_root.expanduser().resolve()
    finetune_root = args.finetune_root.expanduser().resolve()
    if not frozen_root.is_dir():
        raise SystemExit(f"error: frozen root not found: {frozen_root}")
    if not finetune_root.is_dir():
        raise SystemExit(f"error: fine-tune root not found: {finetune_root}")
    labels = list(dict.fromkeys(args.models))
    frozen = {
        label: _curve(frozen_root, "frozen", label, args.seed) for label in labels
    }
    finetune = {
        label: _curve(finetune_root, "finetune", label, args.seed) for label in labels
    }
    missing = [
        str(curve.path)
        for curves in (frozen, finetune)
        for curve in curves.values()
        if not curve.path.is_file()
    ]
    if missing:
        raise SystemExit("error: metrics.jsonl not found:\n  " + "\n  ".join(missing))

    print(f"DSEC-Det frozen vs fine-tune | seed={args.seed}")
    print("\nVALIDATION SUMMARY")
    _print_table(
        [
            "run",
            "frozen best",
            "FT best",
            "FT−frozen",
            "frozen latest",
            "FT latest",
            "frozen 95%@ep",
            "FT 95%@ep",
        ],
        _validation_summary(frozen, finetune, labels),
    )

    print("\nMATCHED VALIDATION HISTORY (mAP)")
    history_rows = _matched_history(frozen, finetune, labels, args.history)
    if history_rows:
        headers = ["epoch"]
        for label in labels:
            headers.extend((f"{label} frozen", f"{label} FT", "Δ"))
        _print_table(headers, history_rows)
    else:
        print("共通のvalidation epochがまだありません")

    print("\nTRAIN LOSS SUMMARY")
    _print_table(
        ["run", "regime", "完了ep", "初期", "最新", "全体Δ", "直近3ep Δ"],
        _loss_summary(frozen, finetune, labels),
    )

    warnings = [
        f"{curve.regime}/{curve.label}: 読み取れないJSONL行={curve.malformed_lines}"
        for curves in (frozen, finetune)
        for curve in curves.values()
        if curve.malformed_lines
    ]
    print("\nNOTES")
    print("- mAP/APは同じofficial validation protocolなので直接比較できます。")
    print(
        "- train lossは同じYOLOX lossですが、"
        "更新対象が異なるため絶対値比較は参考値です。"
    )
    print("- 95%@epは各run自身のbest mAPの95%へ初めて到達した評価epochです。")
    for warning in warnings:
        print(f"- WARNING: {warning}")


if __name__ == "__main__":
    main()
