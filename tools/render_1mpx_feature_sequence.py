#!/usr/bin/env python3
"""Render masked 1Mpx EventState feature artifacts as an MP4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render event input, shared-PCA feature maps, and feature motion"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"invalid feature artifact: {path}")
    return value


def _fit_pca(
    paths: list[Path], feature_name: str, tokens_per_frame: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = []
    for path in paths:
        payload = _load(path)
        value = F.normalize(payload["features"][feature_name].float(), dim=-1)
        count = min(tokens_per_frame, len(value))
        indices = torch.linspace(0, len(value) - 1, count).round().long().unique()
        samples.append(value[indices])
    merged = torch.cat(samples)
    mean = merged.mean(0)
    _, _, basis = torch.pca_lowrank(merged - mean, q=3, center=False)
    projected = (merged - mean) @ basis[:, :3]
    lower = torch.quantile(projected, 0.01, dim=0)
    upper = torch.quantile(projected, 0.99, dim=0)
    return mean, basis[:, :3], lower, upper


def _feature_image(
    feature: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    size: tuple[int, int],
) -> Image.Image:
    projected = (F.normalize(feature.float(), dim=-1) - mean) @ basis
    colors = ((projected - lower) / (upper - lower).clamp_min(1e-7)).clamp(0, 1)
    canvas = torch.full((*mask.shape, 3), 0.5)
    canvas[mask] = colors
    array = (canvas.numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.NEAREST)


def _event_image(value: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    array = value.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _panel(image: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    result = Image.new("RGB", (320, 274), "white")
    result.paste(image, (0, 34))
    ImageDraw.Draw(result).text((160, 6), title, fill="black", font=font, anchor="ma")
    return result


def _timeline(
    width: int,
    frame_index: int,
    rows: list[dict[str, str]],
    feature_names: list[str],
    font: ImageFont.ImageFont,
) -> Image.Image:
    height = 180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 55, 15, width - 15, height - 30
    draw.rectangle((left, top, right, bottom), outline=(140, 140, 140))
    colors = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189)]
    for line_index, name in enumerate(feature_names):
        values = [float(row[f"{name}_previous_cosine"]) for row in rows]
        points = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                x = left + int(index * (right - left) / max(1, len(rows) - 1))
                y = bottom - int(np.clip(value, 0, 1) * (bottom - top))
                points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=colors[line_index % len(colors)], width=2)
        draw.text(
            (left + 8, top + 4 + line_index * 14),
            name,
            fill=colors[line_index % len(colors)],
            font=font,
        )
    cursor = left + int(frame_index * (right - left) / max(1, len(rows) - 1))
    draw.line((cursor, top, cursor, bottom), fill="black", width=2)
    row = rows[frame_index]
    draw.text(
        (right, 2),
        f"t={float(row['time_s']):.3f}s  events={int(row['event_count']):,}",
        fill="black",
        font=font,
        anchor="ra",
    )
    draw.text(
        (width // 2, height - 5),
        "previous-frame cosine",
        fill="black",
        font=font,
        anchor="ms",
    )
    return image


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.pca_tokens_per_frame <= 0:
        raise ValueError("--fps and --pca-tokens-per-frame must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error
    input_dir = args.input_dir.expanduser().resolve()
    paths = sorted((input_dir / "frames").glob("[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if not paths:
        raise FileNotFoundError(f"feature frames not found: {input_dir / 'frames'}")
    with (input_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(paths):
        raise ValueError("metrics and feature frame counts differ")
    first = _load(paths[0])
    feature_names = list(first["features"])
    mask = first["patch_mask"].bool()
    pca = {
        name: _fit_pca(paths, name, args.pca_tokens_per_frame)
        for name in feature_names
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / "alignment.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_width = 320
    width = panel_width * (1 + len(feature_names))
    height = 274 + 180
    width += width % 2
    height += height % 2
    font = ImageFont.load_default()
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size=(width, height),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame_index, path in enumerate(paths):
            payload = _load(path)
            canvas = Image.new("RGB", (width, height), (238, 238, 238))
            event_panel = _panel(
                _event_image(payload["event_rgb"], (320, 240)),
                "Event input",
                font,
            )
            canvas.paste(event_panel, (0, 0))
            for column, name in enumerate(feature_names, start=1):
                mean, basis, lower, upper = pca[name]
                image = _feature_image(
                    payload["features"][name],
                    mask,
                    mean,
                    basis,
                    lower,
                    upper,
                    (320, 240),
                )
                canvas.paste(
                    _panel(image, f"{name} PCA", font),
                    (column * 320, 0),
                )
            timeline = _timeline(width, frame_index, rows, feature_names, font)
            canvas.paste(timeline, (0, 274))
            writer.send(np.asarray(canvas, dtype=np.uint8))
            print(f"\r{frame_index + 1}/{len(paths)} frames", end="", flush=True)
    finally:
        writer.close()
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
