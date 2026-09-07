#!/usr/bin/env python3
"""Render complete chronological feature exports as an MP4 and metric table."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FeatureSource:
    label: str
    directory: Path
    feature_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render RGB/events, shared normalized PCA, query cosine maps, and "
            "full-sequence teacher alignment into an MP4."
        )
    )
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL:DIR:FEATURE",
        help="Repeat for each panel, for example E0:/path/e0:Pz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=240)
    parser.add_argument("--panel-height", type=int, default=180)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional short rendering limit for smoke tests",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Artifact must contain a mapping: {path}")
    return value


def _parse_source(value: str) -> FeatureSource:
    try:
        label, directory, feature_name = value.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Source must be LABEL:DIR:FEATURE, got {value!r}"
        ) from error
    path = Path(directory).expanduser()
    if not label or not feature_name or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid feature source: {value!r}")
    return FeatureSource(label=label, directory=path, feature_name=feature_name)


def _clip_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if not paths:
        raise FileNotFoundError(f"No sequence clips found in {directory}")
    return paths


def _validate_layout(
    context_paths: list[Path], sources: list[FeatureSource]
) -> list[list[Path]]:
    expected = [path.name for path in context_paths]
    result = []
    for source in sources:
        paths = _clip_paths(source.directory)
        if [path.name for path in paths] != expected:
            raise ValueError(f"Clip manifest differs for {source.label}: {source.directory}")
        result.append(paths)
    return result


def _feature(payload: dict[str, Any], source: FeatureSource, path: Path) -> torch.Tensor:
    features = payload.get("features")
    if not isinstance(features, dict) or source.feature_name not in features:
        raise KeyError(f"{source.feature_name} is absent from {path}")
    value = features[source.feature_name]
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{source.feature_name} must have shape [T,N,D]: {path}")
    return value.float()


def _sample_tokens(value: torch.Tensor, count: int) -> torch.Tensor:
    normalized = F.normalize(value.float(), dim=-1)
    token_count = int(normalized.shape[1])
    count = min(count, token_count)
    indices = torch.linspace(0, token_count - 1, count).round().long().unique()
    return normalized[:, indices].reshape(-1, normalized.shape[-1])


def _fit_shared_pca(
    context_paths: list[Path],
    source_paths: list[list[Path]],
    sources: list[FeatureSource],
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples: list[torch.Tensor] = []
    for clip_index, context_path in enumerate(context_paths):
        context = _load(context_path)
        teacher = context.get("teacher")
        if not isinstance(teacher, torch.Tensor) or teacher.ndim != 3:
            raise ValueError(f"teacher must have shape [T,N,D]: {context_path}")
        samples.append(_sample_tokens(teacher, tokens_per_frame))
        for paths, source in zip(source_paths, sources):
            value = _feature(_load(paths[clip_index]), source, paths[clip_index])
            samples.append(_sample_tokens(value, tokens_per_frame))
    merged = torch.cat(samples)
    mean = merged.mean(0)
    centered = merged - mean
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    projected = centered @ basis[:, :3]
    lower = torch.quantile(projected, 0.01, dim=0)
    upper = torch.quantile(projected, 0.99, dim=0)
    print(f"Fitted shared PCA from {len(merged):,} normalized tokens", flush=True)
    return mean, basis[:, :3], lower, upper


def _pca_rgb(
    value: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> np.ndarray:
    normalized = F.normalize(value.float(), dim=-1)
    projected = (normalized - mean) @ basis
    rgb = ((projected - lower) / (upper - lower).clamp_min(1e-7)).clamp(0, 1)
    return (rgb.numpy() * 255).round().astype(np.uint8)


def _cosine_map(value: torch.Tensor, query_index: int, cmap: Any) -> np.ndarray:
    normalized = F.normalize(value.float(), dim=-1)
    similarity = (normalized * normalized[query_index]).sum(-1).clamp(-1, 1)
    rgba = cmap(((similarity.numpy() + 1.0) * 0.5).clip(0, 1))
    return (rgba[:, :3] * 255).round().astype(np.uint8)


def _chw_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    array = np.transpose(array[:3], (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _grid_image(
    colors: np.ndarray,
    grid_size: tuple[int, int],
    size: tuple[int, int],
) -> Image.Image:
    height, width = grid_size
    image = Image.fromarray(colors.reshape(height, width, 3), mode="RGB")
    return image.resize(size, resample=Image.Resampling.BILINEAR)


def _panel(
    image: Image.Image,
    title: str,
    *,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    title_height = 34
    result = Image.new("RGB", (width, height + title_height), "white")
    image = image.copy()
    image.thumbnail((width, height), resample=Image.Resampling.BILINEAR)
    x = (width - image.width) // 2
    y = title_height + (height - image.height) // 2
    result.paste(image, (x, y))
    draw = ImageDraw.Draw(result)
    draw.text((width // 2, 5), title, fill="black", font=font, anchor="ma")
    return result


def _timeline(
    width: int,
    height: int,
    frame_index: int,
    alignments: dict[str, list[float]],
    event_counts: list[int],
    colors: list[tuple[int, int, int]],
    font: ImageFont.ImageFont,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 55, 20, width - 18, height - 28
    draw.rectangle((left, top, right, bottom), outline=(150, 150, 150))
    total = max(2, len(event_counts))
    x_of = lambda index: left + int(index * (right - left) / (total - 1))
    y_of = lambda value: bottom - int((max(-0.05, min(1.0, value)) + 0.05) / 1.05 * (bottom - top))
    for tick in (0.0, 0.5, 1.0):
        y = y_of(tick)
        draw.line((left, y, right, y), fill=(225, 225, 225))
        draw.text((left - 6, y), f"{tick:.1f}", fill="black", font=font, anchor="rm")
    for line_index, (label, values) in enumerate(alignments.items()):
        color = colors[line_index % len(colors)]
        points = [(x_of(index), y_of(value)) for index, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text(
            (left + 8, top + 4 + 14 * line_index),
            label,
            fill=color,
            font=font,
        )
    cursor = x_of(frame_index)
    draw.line((cursor, top, cursor, bottom), fill=(20, 20, 20), width=2)
    count = event_counts[frame_index]
    draw.text(
        (right, 3),
        f"frame {frame_index + 1}/{len(event_counts)}  events={count:,}",
        fill="black",
        font=font,
        anchor="ra",
    )
    draw.text((width // 2, height - 4), "sequence time", fill="black", font=font, anchor="ms")
    return image


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.pca_tokens_per_frame <= 0:
        raise ValueError("--fps and --pca-tokens-per-frame must be positive")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("Panel dimensions must be positive")
    try:
        import imageio_ffmpeg
        from matplotlib import colormaps
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error

    context_dir = args.context_dir.expanduser()
    sources = [_parse_source(value) for value in args.source]
    context_paths = _clip_paths(context_dir)
    source_paths = _validate_layout(context_paths, sources)
    mean, basis, lower, upper = _fit_shared_pca(
        context_paths,
        source_paths,
        sources,
        args.pca_tokens_per_frame,
    )

    records: list[dict[str, Any]] = []
    global_index = 0
    for clip_index, context_path in enumerate(context_paths):
        context = _load(context_path)
        source_payloads = [_load(paths[clip_index]) for paths in source_paths]
        teacher = context["teacher"].float()
        time_steps = int(teacher.shape[0])
        source_features = [
            _feature(payload, source, source_paths[index][clip_index])
            for index, (payload, source) in enumerate(zip(source_payloads, sources))
        ]
        for local_index in range(time_steps):
            alignment = {
                source.label: float(
                    F.cosine_similarity(
                        feature[local_index], teacher[local_index], dim=-1
                    ).mean()
                )
                for source, feature in zip(sources, source_features)
            }
            record = {
                "frame": global_index,
                "timestamp": int(context["timestamps"][local_index]),
                "event_count": int(context["event_counts"][local_index]),
                **alignment,
            }
            records.append(record)
            global_index += 1
            if args.max_frames is not None and global_index >= args.max_frames:
                break
        if args.max_frames is not None and global_index >= args.max_frames:
            break
    if not records:
        raise RuntimeError("The sequence export contains no frames")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_suffix(".csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    event_order = np.argsort([record["event_count"] for record in records])
    event_groups = {
        "low": event_order[: len(event_order) // 3],
        "medium": event_order[len(event_order) // 3 : 2 * len(event_order) // 3],
        "high": event_order[2 * len(event_order) // 3 :],
    }
    source_summaries: dict[str, Any] = {}
    for source in sources:
        values = np.asarray([record[source.label] for record in records])
        source_summaries[source.label] = {
            "feature": source.feature_name,
            "mean_teacher_cosine": float(values.mean()),
            "std_teacher_cosine": float(values.std()),
            "mean_absolute_frame_delta": float(np.abs(np.diff(values)).mean())
            if len(values) > 1
            else 0.0,
            "teacher_cosine_by_event_tertile": {
                name: float(values[indices].mean())
                for name, indices in event_groups.items()
                if len(indices)
            },
        }
    summary = {
        "frame_count": len(records),
        "fps": args.fps,
        "sources": source_summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    grid_size = tuple(int(value) for value in _load(context_paths[0])["grid_size"])
    token_count = grid_size[0] * grid_size[1]
    query_index = args.query_index
    if query_index is None:
        query_index = (grid_size[0] // 2) * grid_size[1] + grid_size[1] // 2
    if not 0 <= query_index < token_count:
        raise IndexError("--query-index is outside the patch grid")

    panel_size = (args.panel_width, args.panel_height)
    columns = 2 + len(sources) + 1
    gap = 6
    panel_full_height = args.panel_height + 34
    canvas_width = columns * args.panel_width + (columns - 1) * gap
    timeline_height = 170
    canvas_height = panel_full_height * 2 + timeline_height + gap * 3
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    font = ImageFont.load_default()
    cmap = colormaps["coolwarm"]
    line_colors = [
        (214, 39, 40),
        (31, 119, 180),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
    ]
    alignment_history = {
        source.label: [record[source.label] for record in records] for source in sources
    }
    event_counts = [record["event_count"] for record in records]
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size=(canvas_width, canvas_height),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        frame_index = 0
        for clip_index, context_path in enumerate(context_paths):
            context = _load(context_path)
            source_payloads = [_load(paths[clip_index]) for paths in source_paths]
            teacher = context["teacher"].float()
            source_features = [
                _feature(payload, source, source_paths[index][clip_index])
                for index, (payload, source) in enumerate(zip(source_payloads, sources))
            ]
            for local_index in range(int(teacher.shape[0])):
                if frame_index >= len(records):
                    break
                canvas = Image.new(
                    "RGB", (canvas_width, canvas_height), (238, 238, 238)
                )
                input_images = [
                    _chw_image(context["event_rgb"][local_index]),
                    _chw_image(context["rgb"][local_index]),
                ]
                input_titles = ["Event input", "Aligned RGB"]
                for column, (image, title) in enumerate(zip(input_images, input_titles)):
                    canvas.paste(
                        _panel(
                            image,
                            title,
                            width=args.panel_width,
                            height=args.panel_height,
                            font=font,
                        ),
                        (column * (args.panel_width + gap), 0),
                    )

                all_features = [
                    *[value[local_index] for value in source_features],
                    teacher[local_index],
                ]
                titles = [source.label for source in sources] + ["DINOv3 teacher"]
                for offset, (feature, title) in enumerate(
                    zip(all_features, titles), start=2
                ):
                    pca_image = _grid_image(
                        _pca_rgb(feature, mean, basis, lower, upper),
                        grid_size,
                        panel_size,
                    )
                    cosine_image = _grid_image(
                        _cosine_map(feature, query_index, cmap),
                        grid_size,
                        panel_size,
                    )
                    x = offset * (args.panel_width + gap)
                    canvas.paste(
                        _panel(
                            pca_image,
                            f"{title} normalized PCA",
                            width=args.panel_width,
                            height=args.panel_height,
                            font=font,
                        ),
                        (x, 0),
                    )
                    alignment = (
                        "query cosine"
                        if offset == columns - 1
                        else f"query cosine / teacher={records[frame_index][title]:.3f}"
                    )
                    canvas.paste(
                        _panel(
                            cosine_image,
                            alignment,
                            width=args.panel_width,
                            height=args.panel_height,
                            font=font,
                        ),
                        (x, panel_full_height + gap),
                    )
                note = Image.new(
                    "RGB", (2 * args.panel_width + gap, panel_full_height), "white"
                )
                note_draw = ImageDraw.Draw(note)
                note_draw.multiline_text(
                    (12, 14),
                    "Continuous recurrent state\nacross the complete sequence\n\n"
                    f"query patch: {query_index}\n"
                    f"timestamp: {records[frame_index]['timestamp']}",
                    fill="black",
                    font=font,
                    spacing=6,
                )
                canvas.paste(note, (0, panel_full_height + gap))
                canvas.paste(
                    _timeline(
                        canvas_width,
                        timeline_height,
                        frame_index,
                        alignment_history,
                        event_counts,
                        line_colors,
                        font,
                    ),
                    (0, 2 * panel_full_height + gap * 2),
                )
                writer.send(np.asarray(canvas, dtype=np.uint8))
                frame_index += 1
                if frame_index % 100 == 0 or frame_index == len(records):
                    print(f"Rendered {frame_index}/{len(records)} frames", flush=True)
            if frame_index >= len(records):
                break
    finally:
        writer.close()
    print(f"Video: {output}")
    print(f"Per-frame metrics: {metrics_path}")
    print(f"Summary: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
