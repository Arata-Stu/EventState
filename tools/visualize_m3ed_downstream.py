#!/usr/bin/env python3
"""Render aligned M3ED RGB, DAGR events, semantics, depth, and motion for QA."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


CITYSCAPES_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
CITYSCAPES_PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)
DEPTH_ANCHORS = np.asarray(
    [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    dtype=np.float32,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create visual QA videos for prepared M3ED RGB/events and aligned "
            "downstream targets."
        )
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--downstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Defaults to every complete sequence shared by both cache roots",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Render every Nth source frame; use 1 for a contiguous full-rate check",
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument(
        "--depth-point-radius",
        type=int,
        default=1,
        help="Display-only radius for sparse depth points; 0 preserves exact pixels",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Event artifact must contain a mapping: {path}")
    return value


def _discover(prepared_root: Path, downstream_root: Path) -> list[str]:
    prepared = {
        path.parent.name
        for path in prepared_root.glob("*/metadata.json")
        if (path.parent / "aligned_rgb").is_dir()
        and (path.parent / "events").is_dir()
    }
    downstream = {
        path.parent.name
        for path in downstream_root.glob("*/_SUCCESS")
        if (path.parent / "metadata.json").is_file()
        and (path.parent / "targets.h5").is_file()
    }
    result = sorted(prepared & downstream)
    if not result:
        raise ValueError("No complete sequence is shared by the two cache roots")
    return result


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _crop_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] == (360, 640):
        image = image[4:356]
    if image.shape[:2] != (352, 640):
        raise ValueError(f"Expected aligned RGB at 640x360 or 640x352, got {image.shape}: {path}")
    return image


def _event_image(path: Path) -> tuple[np.ndarray, int]:
    payload = _safe_torch_load(path)
    tensor = payload.get("events")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError(f"Invalid event tensor: {path}")
    array = tensor.detach().cpu().float().numpy()
    if array.shape == (3, 360, 640):
        array = array[:, 4:356]
    if array.shape != (3, 352, 640):
        raise ValueError(f"Expected event tensor [3,352,640], got {array.shape}: {path}")
    image = np.transpose(array, (1, 2, 0)).clip(0.0, 1.0)
    return (image * 255).round().astype(np.uint8), int(payload.get("event_count", -1))


def _semantic_image(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(labels, dtype=np.uint8)
    valid = source != 255
    if np.any(valid & (source >= len(CITYSCAPES_PALETTE))):
        raise ValueError("Semantic labels must be Cityscapes 0..18 or ignore=255")
    result = np.zeros((*source.shape, 3), dtype=np.uint8)
    result[valid] = CITYSCAPES_PALETTE[source[valid]]
    return result, valid


def _depth_colors(depth: np.ndarray, valid: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    normalized = (np.asarray(depth, dtype=np.float32) - low) / max(high - low, 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid | ~np.isfinite(normalized)] = 0.0
    position = normalized * (len(DEPTH_ANCHORS) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(DEPTH_ANCHORS) - 1)
    fraction = (position - lower)[..., None]
    colors = DEPTH_ANCHORS[lower] * (1.0 - fraction) + DEPTH_ANCHORS[upper] * fraction
    result = colors.round().astype(np.uint8)
    result[~valid] = 0
    return result


def _dilate_for_display(colors: np.ndarray, valid: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0:
        return colors, valid
    output = colors.copy()
    output_valid = valid.copy()
    height, width = valid.shape
    y, x = np.nonzero(valid)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            target_y = y + dy
            target_x = x + dx
            inside = (
                (target_y >= 0)
                & (target_y < height)
                & (target_x >= 0)
                & (target_x < width)
            )
            output[target_y[inside], target_x[inside]] = colors[y[inside], x[inside]]
            output_valid[target_y[inside], target_x[inside]] = True
    return output, output_valid


def _panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
    *,
    width: int,
    height: int,
    title_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    header = 48
    result = Image.new("RGB", (width, height + header), "white")
    source = Image.fromarray(image, mode="RGB").resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    result.paste(source, (0, header))
    draw = ImageDraw.Draw(result)
    draw.text((width // 2, 4), title, fill="black", font=title_font, anchor="ma")
    draw.text((width // 2, 27), subtitle, fill=(60, 60, 60), font=small_font, anchor="ma")
    return result


def _depth_limits(target: h5py.File, indices: np.ndarray) -> tuple[float, float]:
    if "depth_m" not in target or "depth_valid" not in target:
        return 1.0, 80.0
    samples: list[np.ndarray] = []
    for index in indices[:: max(1, len(indices) // 100)]:
        depth = np.asarray(target["depth_m"][int(index)], dtype=np.float32)
        valid = np.asarray(target["depth_valid"][int(index)], dtype=np.bool_)
        values = depth[valid & np.isfinite(depth)]
        if len(values):
            samples.append(values[:: max(1, len(values) // 2000)])
    if not samples:
        return 1.0, 80.0
    merged = np.concatenate(samples)
    low, high = np.quantile(merged, (0.02, 0.98))
    return float(max(0.0, low)), float(max(low + 1e-3, high))


def _selected_indices(frame_count: int, args: argparse.Namespace) -> np.ndarray:
    if not 0 <= args.start_frame < frame_count:
        raise ValueError(f"--start-frame must be in [0,{frame_count - 1}]")
    indices = np.arange(args.start_frame, frame_count, args.frame_stride, dtype=np.int64)
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if not len(indices):
        raise ValueError("Frame selection is empty")
    return indices


def render_sequence(sequence: str, args: argparse.Namespace, imageio_ffmpeg: Any) -> None:
    prepared_sequence = args.prepared_root / sequence
    downstream_sequence = args.downstream_root / sequence
    prepared_metadata = _read_json(prepared_sequence / "metadata.json")
    target_metadata = _read_json(downstream_sequence / "metadata.json")
    timestamps = np.asarray(prepared_metadata.get("timestamps", []), dtype=np.int64)
    if not len(timestamps) or timestamps.tolist() != target_metadata.get("timestamps"):
        raise ValueError(f"Prepared/downstream timestamps differ: {sequence}")
    if not (downstream_sequence / "_SUCCESS").is_file():
        raise FileNotFoundError(f"Downstream completion marker missing: {sequence}")
    indices = _selected_indices(len(timestamps), args)
    output = args.output_dir / f"{sequence}_downstream_check.mp4"
    if output.exists() and not args.overwrite:
        print(f"Preserving existing video: {output}")
        return

    panel_width = args.panel_width
    panel_height = round(panel_width * 352 / 640)
    panel_full_height = panel_height + 48
    gap = 6
    top_header = 42
    canvas_width = panel_width * 3 + gap * 2
    canvas_height = top_header + panel_full_height * 2 + gap
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    title_font = _font(17)
    small_font = _font(12)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    targets_path = downstream_sequence / "targets.h5"
    with h5py.File(targets_path, "r") as target:
        cached_timestamps = np.asarray(target["timestamps"], dtype=np.int64)
        if not np.array_equal(timestamps, cached_timestamps):
            raise ValueError(f"targets.h5 timestamps differ: {sequence}")
        depth_limits = _depth_limits(target, indices)
        temporary_output = output.with_name(f".{output.stem}.partial{output.suffix}")
        temporary_output.unlink(missing_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(temporary_output),
            size=(canvas_width, canvas_height),
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            for output_index, frame_index in enumerate(
                tqdm(indices, desc=f"{sequence} QA", unit="frame")
            ):
                index = int(frame_index)
                timestamp = int(timestamps[index])
                rgb = _crop_rgb(prepared_sequence / "aligned_rgb" / f"{timestamp}.png")
                if index == 0:
                    event = np.full_like(rgb, 255)
                    event_count = 0
                else:
                    event, event_count = _event_image(
                        prepared_sequence / "events" / f"{timestamp}.pt"
                    )

                semantic_valid_frame = bool(
                    target["semantics_frame_valid"][index]
                ) if "semantics_frame_valid" in target else False
                if "semantics_19" in target and semantic_valid_frame:
                    semantics, semantic_valid = _semantic_image(target["semantics_19"][index])
                    semantic_overlay = rgb.copy()
                    semantic_overlay[semantic_valid] = (
                        0.45 * rgb[semantic_valid] + 0.55 * semantics[semantic_valid]
                    ).round().astype(np.uint8)
                    semantic_coverage = float(semantic_valid.mean())
                else:
                    semantics = np.zeros_like(rgb)
                    semantic_overlay = rgb.copy()
                    semantic_coverage = 0.0

                depth_valid_frame = bool(
                    target["depth_frame_valid"][index]
                ) if "depth_frame_valid" in target else False
                if "depth_m" in target and "depth_valid" in target and depth_valid_frame:
                    depth = np.asarray(target["depth_m"][index], dtype=np.float32)
                    depth_valid = np.asarray(target["depth_valid"][index], dtype=np.bool_)
                    depth_colors = _depth_colors(depth, depth_valid, depth_limits)
                    display_depth, display_valid = _dilate_for_display(
                        depth_colors, depth_valid, args.depth_point_radius
                    )
                    depth_overlay = rgb.copy()
                    depth_overlay[display_valid] = (
                        0.30 * rgb[display_valid] + 0.70 * display_depth[display_valid]
                    ).round().astype(np.uint8)
                    depth_points = int(depth_valid.sum())
                    median_depth = float(np.nanmedian(depth[depth_valid])) if depth_points else None
                else:
                    display_depth = np.zeros_like(rgb)
                    depth_overlay = rgb.copy()
                    depth_points = 0
                    median_depth = None

                linear_speed = None
                angular_speed = None
                if "relative_pose_valid" in target and bool(target["relative_pose_valid"][index]):
                    linear_speed = float(
                        np.linalg.norm(np.asarray(target["linear_velocity_mps"][index]))
                    )
                    angular_speed = float(
                        np.linalg.norm(np.asarray(target["angular_velocity_radps"][index]))
                    )
                depth_delta = int(target["depth_time_delta_us"][index]) if "depth_time_delta_us" in target else None
                semantics_delta = int(target["semantics_time_delta_us"][index]) if "semantics_time_delta_us" in target else None
                rows.append(
                    {
                        "output_frame": output_index,
                        "source_frame": index,
                        "timestamp_us": timestamp,
                        "event_count": event_count,
                        "semantics_valid": semantic_valid_frame,
                        "semantics_coverage": semantic_coverage,
                        "semantics_delta_us": semantics_delta,
                        "depth_valid": depth_valid_frame,
                        "depth_valid_pixels": depth_points,
                        "depth_median_m": median_depth,
                        "depth_delta_us": depth_delta,
                        "linear_speed_mps": linear_speed,
                        "angular_speed_radps": angular_speed,
                    }
                )

                canvas = Image.new("RGB", (canvas_width, canvas_height), (235, 235, 235))
                draw = ImageDraw.Draw(canvas)
                motion = (
                    "pose invalid"
                    if linear_speed is None
                    else f"speed={linear_speed:.2f}m/s omega={angular_speed:.2f}rad/s"
                )
                draw.text(
                    (canvas_width // 2, 8),
                    f"{sequence} | source frame {index}/{len(timestamps) - 1} | "
                    f"timestamp={timestamp}us | events={event_count:,} | {motion}",
                    fill="black",
                    font=small_font,
                    anchor="ma",
                )
                panels = (
                    (rgb, "Aligned RGB", "640x360 -> center crop 640x352"),
                    (
                        semantic_overlay,
                        "RGB + semantic label",
                        f"Cityscapes-19 pseudo-label | valid={semantic_valid_frame}",
                    ),
                    (
                        semantics,
                        "Semantic label",
                        f"coverage={semantic_coverage:.1%} | dt={semantics_delta}us",
                    ),
                    (event, "DAGR event input", f"causal RGB interval | count={event_count:,}"),
                    (
                        display_depth,
                        "LiDAR depth",
                        f"{depth_limits[0]:.1f}-{depth_limits[1]:.1f}m | display radius={args.depth_point_radius}px",
                    ),
                    (
                        depth_overlay,
                        "RGB + LiDAR depth",
                        f"raw points={depth_points:,} | dt={depth_delta}us",
                    ),
                )
                for panel_index, (image, title, subtitle) in enumerate(panels):
                    column = panel_index % 3
                    row = panel_index // 3
                    canvas.paste(
                        _panel(
                            image,
                            title,
                            subtitle,
                            width=panel_width,
                            height=panel_height,
                            title_font=title_font,
                            small_font=small_font,
                        ),
                        (
                            column * (panel_width + gap),
                            top_header + row * (panel_full_height + gap),
                        ),
                    )
                writer.send(np.asarray(canvas, dtype=np.uint8))
        finally:
            writer.close()
        os.replace(temporary_output, output)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {
        "sequence": sequence,
        "source_frame_count": len(timestamps),
        "rendered_frame_count": len(rows),
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "tasks": target_metadata.get("tasks", []),
        "depth_display_limits_m": list(depth_limits),
        "depth_point_radius_display_only": args.depth_point_radius,
        "semantic_source": "M3ED InternImage pseudo-labels",
        "semantic_classes": {
            str(index): {"name": name, "rgb": CITYSCAPES_PALETTE[index].tolist()}
            for index, name in enumerate(CITYSCAPES_NAMES)
        },
        "outputs": {"video": str(output), "frames": str(csv_path)},
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered M3ED downstream QA: {output}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.fps <= 0 or args.frame_stride <= 0 or args.panel_width <= 0:
        raise ValueError("fps, frame-stride, and panel-width must be positive")
    if args.start_frame < 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("start-frame must be non-negative and max-frames must be positive")
    if args.depth_point_radius < 0 or args.depth_point_radius > 4:
        raise ValueError("depth-point-radius must be in [0,4]")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.downstream_root = args.downstream_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    discovered = _discover(args.prepared_root, args.downstream_root)
    sequences = discovered if args.sequences is None else args.sequences
    unavailable = sorted(set(sequences) - set(discovered))
    if unavailable:
        raise ValueError("Incomplete or unavailable sequences: " + ", ".join(unavailable))
    for sequence in sequences:
        render_sequence(sequence, args, imageio_ffmpeg)


if __name__ == "__main__":
    main()
