#!/usr/bin/env python3
"""Analyze event activity distributions without rerunning event preprocessing."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


MEASURES = {
    "count": ("event_count", "events / frame"),
    "rate": ("event_rate_kev_s", "kEvents / s"),
    "density": ("event_density_ev_mpix_s", "events / MP / s"),
}
QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
LOW_THRESHOLD_QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create bin-width-selected log histograms, exact ECDFs, KDEs, and "
            "low-activity run statistics from prepared event caches"
        )
    )
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Prepared event-cache root; repeat to compare DSEC and M3ED",
    )
    parser.add_argument(
        "--frames-csv",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help=(
            "Previously generated frames.csv; repeat to compare datasets stored "
            "on different machines without copying event caches"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--measure",
        choices=tuple(MEASURES),
        default="count",
        help="Primary quantity used for plots and threshold candidates",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional common sequence-name subset",
    )
    parser.add_argument("--kde-points", type=int, default=512)
    parser.add_argument("--kde-max-samples", type=int, default=20_000)
    parser.add_argument("--min-bins", type=int, default=20)
    parser.add_argument("--max-bins", type=int, default=200)
    return parser.parse_args()


def _parse_cache(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"cache must be LABEL:PATH, got {value!r}") from error
    root = Path(path).expanduser().resolve()
    if not label or not root.is_dir():
        raise argparse.ArgumentTypeError(f"invalid cache: {value!r}")
    return label, root


def _parse_frames_csv(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"frames-csv must be LABEL:PATH, got {value!r}"
        ) from error
    source = Path(path).expanduser().resolve()
    if not label or not source.is_file():
        raise argparse.ArgumentTypeError(f"invalid frames-csv: {value!r}")
    return label, source


def _load_frames_csv(label: str, path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "frame_index",
        "timestamp",
        "delta_us",
        "height",
        "width",
        "event_count",
    }
    float_fields = {"event_rate_kev_s", "event_density_ev_mpix_s"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            missing = (integer_fields | float_fields | {"sequence"}) - set(source)
            if missing:
                raise ValueError(
                    f"frames.csv lacks {sorted(missing)}: {path}"
                )
            row: dict[str, Any] = {"dataset": label, "sequence": source["sequence"]}
            row.update({key: int(float(source[key])) for key in integer_fields})
            row.update({key: float(source[key]) for key in float_fields})
            rows.append(row)
    if not rows:
        raise ValueError(f"frames.csv is empty: {path}")
    return rows


def _safe_load(path: Path) -> Any:
    # mmap=True reads pickle metadata immediately but pages the large event
    # tensor only if it is accessed. This makes a count-only scan practical for
    # caches containing hundreds of gigabytes of tensors.
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _sequence_directories(
    root: Path, requested: list[str] | None
) -> list[tuple[str, Path]]:
    directories: list[tuple[str, Path]] = []
    for sequence_root in sorted(path for path in root.iterdir() if path.is_dir()):
        direct = any(item.stem.isdecimal() for item in sequence_root.glob("*.pt"))
        nested_root = sequence_root / "events"
        nested = nested_root.is_dir() and any(
            item.stem.isdecimal() for item in nested_root.glob("*.pt")
        )
        if direct and nested:
            raise ValueError(
                f"Ambiguous direct and nested event caches: {sequence_root}"
            )
        if direct:
            directories.append((sequence_root.name, sequence_root))
        elif nested:
            directories.append((sequence_root.name, nested_root))
    if requested:
        mapping = dict(directories)
        missing = sorted(set(requested) - set(mapping))
        if missing:
            raise FileNotFoundError(
                f"Sequences not found under {root}: {', '.join(missing)}"
            )
        directories = [(name, mapping[name]) for name in requested]
    if not directories:
        raise ValueError(f"No event-cache sequence directories found: {root}")
    return directories


def _scan_cache(
    label: str, root: Path, requested: list[str] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directories = _sequence_directories(root, requested)
    for sequence_name, directory in directories:
        paths = sorted(
            (path for path in directory.glob("*.pt") if path.stem.isdecimal()),
            key=lambda path: int(path.stem),
        )
        for path in tqdm(paths, desc=f"activity:{label}:{sequence_name}", unit="frame"):
            payload = _safe_load(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid event-cache payload: {path}")
            count = payload.get("event_count")
            timestamp = payload.get("timestamp")
            previous = payload.get("previous_timestamp")
            frame_index = payload.get("frame_index")
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                or not isinstance(timestamp, int)
                or not isinstance(previous, int)
                or timestamp <= previous
            ):
                raise ValueError(f"Invalid event activity metadata: {path}")
            events = payload.get("events")
            height = payload.get("height")
            width = payload.get("width")
            if not isinstance(height, int) or not isinstance(width, int):
                if isinstance(events, torch.Tensor) and events.ndim == 3:
                    height, width = map(int, events.shape[-2:])
                else:
                    raise ValueError(f"Event geometry is unavailable: {path}")
            delta_us = timestamp - previous
            rate_ev_s = count * 1_000_000.0 / delta_us
            density = rate_ev_s * 1_000_000.0 / (height * width)
            rows.append(
                {
                    "dataset": label,
                    "sequence": sequence_name,
                    "frame_index": int(frame_index) if isinstance(frame_index, int) else -1,
                    "timestamp": timestamp,
                    "delta_us": delta_us,
                    "height": height,
                    "width": width,
                    "event_count": count,
                    "event_rate_kev_s": rate_ev_s / 1000.0,
                    "event_density_ev_mpix_s": density,
                }
            )
            del payload, events
        gc.collect()
    return rows


def _fd_bin_count(log_values: np.ndarray, minimum: int, maximum: int) -> int:
    if len(log_values) < 2 or float(log_values.max()) == float(log_values.min()):
        return minimum
    q25, q75 = np.quantile(log_values, (0.25, 0.75))
    width = 2.0 * float(q75 - q25) * len(log_values) ** (-1.0 / 3.0)
    if not math.isfinite(width) or width <= 0:
        count = round(math.sqrt(len(log_values)))
    else:
        count = math.ceil(float(log_values.max() - log_values.min()) / width)
    return max(minimum, min(maximum, int(count)))


def _kde(
    log_values: np.ndarray, points: int, max_samples: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(log_values) > max_samples:
        indices = np.linspace(0, len(log_values) - 1, max_samples).round().astype(int)
        values = np.sort(log_values)[indices]
    else:
        values = log_values
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    q25, q75 = np.quantile(values, (0.25, 0.75))
    robust = min(std, float(q75 - q25) / 1.34) if q75 > q25 else std
    bandwidth = 0.9 * robust * len(values) ** (-0.2) if robust > 0 else 0.05
    bandwidth = max(bandwidth, 1e-3)
    padding = max(3.0 * bandwidth, 0.02)
    grid = np.linspace(float(values.min()) - padding, float(values.max()) + padding, points)
    density = np.zeros_like(grid)
    chunk = 4096
    normalizer = bandwidth * math.sqrt(2.0 * math.pi) * len(values)
    for start in range(0, len(values), chunk):
        difference = (grid[:, None] - values[None, start : start + chunk]) / bandwidth
        density += np.exp(-0.5 * difference * difference).sum(axis=1) / normalizer
    return grid, density, bandwidth


def _kde_valleys(grid: np.ndarray, density: np.ndarray) -> list[float]:
    if len(grid) < 3:
        return []
    valleys = np.flatnonzero(
        (density[1:-1] < density[:-2]) & (density[1:-1] <= density[2:])
    ) + 1
    # Edge ripples are rarely meaningful low/high separators. Keep only minima
    # with appreciable density on both sides and convert out of log10(x+1).
    peak = float(density.max())
    selected = [
        index
        for index in valleys
        if 2 <= index < len(grid) - 2 and density[index] < 0.8 * peak
    ]
    return [float(10.0 ** grid[index] - 1.0) for index in selected]


def _runs(mask: np.ndarray) -> list[int]:
    output: list[int] = []
    current = 0
    for value in mask:
        if bool(value):
            current += 1
        elif current:
            output.append(current)
            current = 0
    if current:
        output.append(current)
    return output


def _threshold_rows(
    dataset_rows: list[dict[str, Any]], measure_key: str
) -> list[dict[str, Any]]:
    values = np.asarray([float(row[measure_key]) for row in dataset_rows])
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        by_sequence[str(row["sequence"])].append(row)
    output: list[dict[str, Any]] = []
    for quantile in LOW_THRESHOLD_QUANTILES:
        threshold = float(np.quantile(values, quantile))
        all_runs: list[int] = []
        for rows in by_sequence.values():
            ordered = sorted(rows, key=lambda row: int(row["timestamp"]))
            mask = np.asarray([float(row[measure_key]) <= threshold for row in ordered])
            all_runs.extend(_runs(mask))
        covered = int((values <= threshold).sum())
        run_array = np.asarray(all_runs, dtype=np.int64)
        output.append(
            {
                "threshold_source": f"p{quantile * 100:g}",
                "threshold": threshold,
                "frames_at_or_below": covered,
                "frame_fraction": covered / len(values),
                "run_count": len(all_runs),
                "mean_run_frames": float(run_array.mean()) if len(run_array) else 0.0,
                "p95_run_frames": (
                    float(np.quantile(run_array, 0.95)) if len(run_array) else 0.0
                ),
                "max_run_frames": int(run_array.max()) if len(run_array) else 0,
                "runs_ge_2": int((run_array >= 2).sum()),
                "runs_ge_4": int((run_array >= 4).sum()),
                "runs_ge_8": int((run_array >= 8).sum()),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_distribution(
    grouped: dict[str, list[dict[str, Any]]],
    measure_key: str,
    measure_label: str,
    minimum_bins: int,
    maximum_bins: int,
    kde_points: int,
    kde_max_samples: int,
    output: Path,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    details: dict[str, Any] = {}
    for index, (label, rows) in enumerate(grouped.items()):
        values = np.asarray([float(row[measure_key]) for row in rows], dtype=np.float64)
        log_values = np.log10(values + 1.0)
        bins = _fd_bin_count(log_values, minimum_bins, maximum_bins)
        grid, density, bandwidth = _kde(log_values, kde_points, kde_max_samples)
        color = colors(index % 10)
        axes[0].hist(
            log_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            alpha=0.8,
            color=color,
            label=f"{label} histogram (FD bins={bins})",
        )
        axes[0].plot(grid, density, color=color, linewidth=2.2, label=f"{label} KDE")
        ordered = np.sort(log_values)
        ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1].plot(ordered, ecdf, color=color, linewidth=2.0, label=label)
        for quantile in (0.01, 0.05, 0.10, 0.25):
            location = float(np.quantile(log_values, quantile))
            axes[1].scatter([location], [quantile], color=color, s=18)
        details[label] = {
            "fd_bin_count": bins,
            "kde_bandwidth_log10": bandwidth,
            "kde_valleys": _kde_valleys(grid, density),
        }
    axes[0].set_title("Fine histogram and continuous KDE")
    axes[0].set_xlabel(f"log10({measure_label} + 1)")
    axes[0].set_ylabel("density in log space")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Exact empirical CDF (no binning)")
    axes[1].set_xlabel(f"log10({measure_label} + 1)")
    axes[1].set_ylabel("fraction of frames at or below x")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return details


def _plot_sequences(
    grouped: dict[str, list[dict[str, Any]]],
    measure_key: str,
    measure_label: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    entries: list[tuple[str, float, float, float]] = []
    for dataset, rows in grouped.items():
        by_sequence: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_sequence[str(row["sequence"])].append(float(row[measure_key]))
        for sequence, values in by_sequence.items():
            array = np.log10(np.asarray(values) + 1.0)
            q10, median, q90 = np.quantile(array, (0.10, 0.50, 0.90))
            entries.append((f"{dataset}/{sequence}", median, q10, q90))
    entries.sort(key=lambda item: item[1])
    height = max(5.0, 0.25 * len(entries) + 1.5)
    figure, axis = plt.subplots(figsize=(11, height), constrained_layout=True)
    y = np.arange(len(entries))
    medians = np.asarray([item[1] for item in entries])
    lower = medians - np.asarray([item[2] for item in entries])
    upper = np.asarray([item[3] for item in entries]) - medians
    axis.errorbar(medians, y, xerr=np.vstack((lower, upper)), fmt="o", markersize=3)
    axis.set_yticks(y, [item[0] for item in entries], fontsize=7)
    axis.set_xlabel(f"log10({measure_label} + 1), median with p10-p90")
    axis.set_title("Per-sequence event activity")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.kde_points < 32 or args.kde_max_samples < 100:
        raise ValueError("KDE point/sample limits are too small")
    if not 2 <= args.min_bins <= args.max_bins:
        raise ValueError("Require 2 <= min-bins <= max-bins")
    caches = [_parse_cache(value) for value in args.cache]
    frame_sources = [_parse_frames_csv(value) for value in args.frames_csv]
    if not caches and not frame_sources:
        raise ValueError("Pass at least one --cache or --frames-csv source")
    labels = [label for label, _ in (*caches, *frame_sources)]
    if len(set(labels)) != len(labels):
        raise ValueError("Cache labels must be unique")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for label, root in caches:
        rows = _scan_cache(label, root, args.sequences)
        grouped[label] = rows
        all_rows.extend(rows)
    for label, source in frame_sources:
        rows = _load_frames_csv(label, source)
        if args.sequences:
            requested = set(args.sequences)
            available = {str(row["sequence"]) for row in rows}
            missing = sorted(requested - available)
            if missing:
                raise ValueError(
                    f"Sequences not found in {source}: {', '.join(missing)}"
                )
            rows = [row for row in rows if str(row["sequence"]) in requested]
        grouped[label] = rows
        all_rows.extend(rows)
    _write_csv(output_dir / "frames.csv", all_rows)

    measure_key, measure_label = MEASURES[args.measure]
    threshold_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "measure": args.measure,
        "measure_key": measure_key,
        "measure_label": measure_label,
        "datasets": {},
    }
    for label, rows in grouped.items():
        values = np.asarray([float(row[measure_key]) for row in rows])
        current_thresholds = _threshold_rows(rows, measure_key)
        threshold_rows.extend(
            {"dataset": label, "measure": args.measure, **row}
            for row in current_thresholds
        )
        summary["datasets"][label] = {
            "source": str(dict((*caches, *frame_sources))[label]),
            "frames": len(rows),
            "sequences": len({row["sequence"] for row in rows}),
            "zero_event_frames": int(
                sum(int(row["event_count"]) == 0 for row in rows)
            ),
            "quantiles": {
                f"p{quantile * 100:g}": float(np.quantile(values, quantile))
                for quantile in QUANTILES
            },
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "threshold_candidates": current_thresholds,
        }
    _write_csv(output_dir / "threshold_candidates.csv", threshold_rows)
    distribution_details = _plot_distribution(
        grouped,
        measure_key,
        measure_label,
        args.min_bins,
        args.max_bins,
        args.kde_points,
        args.kde_max_samples,
        output_dir / "distribution.png",
    )
    for label, details in distribution_details.items():
        summary["datasets"][label].update(details)
    _plot_sequences(
        grouped,
        measure_key,
        measure_label,
        output_dir / "sequences.png",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Event-activity analysis complete: {output_dir}", flush=True)
    print(f"Distribution: {output_dir / 'distribution.png'}", flush=True)
    print(f"Threshold table: {output_dir / 'threshold_candidates.csv'}", flush=True)


if __name__ == "__main__":
    main()
