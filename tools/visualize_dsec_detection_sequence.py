#!/usr/bin/env python3
"""Render DSEC-Detection predictions and frozen feature maps as one MP4."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from event_state.data.transforms import PairedSequenceTransform
from event_state.detection import (
    DSECDetectionFeatureDataset,
    EventStateYOLOX,
    detection_collate,
    load_dsec_detection_split,
)
from event_state.detection.data import (
    DAGR_CLASSES,
    build_dagr_sampling_grid,
    find_rectify_map,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_det_official_split.yaml"
PREDICTION_COLORS = ((255, 112, 67), (0, 188, 212))
GT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class DetectionSource:
    label: str
    checkpoint: Path
    feature_cache_dir: Path
    feature: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render GT/predicted DSEC-Detection boxes and shared-PCA frozen "
            "feature maps over one official sequence"
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL:CHECKPOINT:CACHE:FEATURE",
        help=(
            "Repeat for every detector, for example "
            "E2:/path/best.pt:/path/features/E2:h"
        ),
    )
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--pca-max-samples", type=int, default=50_000)
    parser.add_argument("--panel-width", type=int, default=480)
    return parser.parse_args()


def _safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_source(value: str) -> DetectionSource:
    try:
        label, checkpoint, cache, feature = value.split(":", 3)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"source must be LABEL:CHECKPOINT:CACHE:FEATURE, got {value!r}"
        ) from error
    if feature not in {"z", "h", "concat"}:
        raise argparse.ArgumentTypeError(f"unsupported feature {feature!r}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cache_path = Path(cache).expanduser().resolve()
    if not label or not checkpoint_path.is_file() or not cache_path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid detector source: {value!r}")
    return DetectionSource(label, checkpoint_path, cache_path, feature)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _dataset(
    source: DetectionSource,
    args: argparse.Namespace,
) -> DSECDetectionFeatureDataset:
    return DSECDetectionFeatureDataset(
        feature_cache_dir=source.feature_cache_dir,
        labels_root=args.labels_root,
        dataset_root=args.dataset_root,
        sequences=(args.sequence,),
        feature=source.feature,
        protocol="dsec-det",
    )


def _selected_indices(length: int, start: int, maximum: int | None) -> list[int]:
    if start < 0 or start >= length:
        raise ValueError(f"start-frame must be in [0, {length})")
    stop = length if maximum is None else min(length, start + maximum)
    if maximum is not None and maximum <= 0:
        raise ValueError("max-frames must be positive")
    return list(range(start, stop))


def _sample_keys(
    dataset: DSECDetectionFeatureDataset, indices: list[int]
) -> list[tuple[str, int]]:
    return [(dataset.samples[index][0], dataset.samples[index][1]) for index in indices]


def _load_detector(
    source: DetectionSource,
    dataset: DSECDetectionFeatureDataset,
    device: torch.device,
    score_threshold: float,
) -> EventStateYOLOX:
    checkpoint = torch.load(source.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected_feature = config.get("feature")
    if expected_feature is not None and expected_feature != source.feature:
        raise ValueError(
            f"{source.label} checkpoint expects {expected_feature}, not {source.feature}"
        )
    if str(config.get("protocol", "probe")) != "dsec-det":
        raise ValueError(f"{source.label} checkpoint was not trained with dsec-det protocol")
    sample_channels = int(dataset[0]["feature"].shape[0])
    if dataset.patch_size is None or dataset.input_size is None:
        raise RuntimeError("Detection feature geometry was not initialized")
    model = EventStateYOLOX(
        in_channels=sample_channels,
        width=int(config.get("head_width", 192)),
        input_stride=int(config.get("input_stride", dataset.patch_size)),
        image_size=dataset.input_size,
        confidence_threshold=min(0.001, score_threshold),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def _infer(
    source: DetectionSource,
    dataset: DSECDetectionFeatureDataset,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    model = _load_detector(source, dataset, device, args.score_threshold)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_collate,
    )
    use_amp = args.precision == "fp16" and device.type == "cuda"
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    observed_keys: list[tuple[str, int]] = []
    for batch in tqdm(loader, desc=f"{source.label} detection", unit="batch"):
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            batch_predictions = model(features)
        for prediction in batch_predictions:
            prediction = {key: value.detach().cpu() for key, value in prediction.items()}
            keep = prediction["scores"] >= args.score_threshold
            predictions.append({key: value[keep] for key, value in prediction.items()})
        targets.extend(
            {key: value.detach().cpu() for key, value in target.items()}
            for target in batch["targets"]
        )
        observed_keys.extend(zip(batch["sequence_names"], batch["timestamps"]))
    expected_keys = _sample_keys(dataset, indices)
    if observed_keys != expected_keys:
        raise ValueError(f"{source.label} inference order differs from feature manifest")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, targets


def _feature_map(source: DetectionSource, sequence: str, timestamp: int) -> torch.Tensor:
    path = source.feature_cache_dir / sequence / f"{timestamp}.pt"
    payload = _safe_load(path)
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        raise ValueError(f"feature payload lacks features: {path}")
    if source.feature == "concat":
        value = torch.cat((features["z"], features["h"]), dim=0)
    else:
        value = features.get(source.feature)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"invalid {source.feature} feature map: {path}")
    return value.float()


def _sample_feature_tokens(value: torch.Tensor, count: int) -> torch.Tensor:
    tokens = value.permute(1, 2, 0).reshape(-1, value.shape[0]).float()
    tokens = F.normalize(tokens, dim=-1)
    count = min(count, len(tokens))
    indices = torch.linspace(0, len(tokens) - 1, count).round().long().unique()
    return tokens[indices]


def _fit_shared_pca(
    sources: list[DetectionSource],
    keys: list[tuple[str, int]],
    tokens_per_frame: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples: list[torch.Tensor] = []
    channels: int | None = None
    for sequence, timestamp in tqdm(keys, desc="PCA sampling", unit="frame"):
        for source in sources:
            value = _feature_map(source, sequence, timestamp)
            if channels is None:
                channels = int(value.shape[0])
            elif channels != int(value.shape[0]):
                raise ValueError("Shared PCA requires equal feature dimensions")
            samples.append(_sample_feature_tokens(value, tokens_per_frame))
    merged = torch.cat(samples)
    if len(merged) > max_samples:
        indices = torch.linspace(0, len(merged) - 1, max_samples).round().long().unique()
        merged = merged[indices]
    mean = merged.mean(0)
    centered = merged - mean
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    projected = centered @ basis[:, :3]
    lower = torch.quantile(projected, 0.01, dim=0)
    upper = torch.quantile(projected, 0.99, dim=0)
    print(f"Fitted shared PCA from {len(merged):,} normalized tokens", flush=True)
    return mean, basis[:, :3], lower, upper


def _pca_image(
    value: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    size: tuple[int, int],
) -> Image.Image:
    height, width = value.shape[-2:]
    tokens = F.normalize(value.permute(1, 2, 0).reshape(-1, value.shape[0]), dim=-1)
    projected = (tokens - mean) @ basis
    scale = (upper - lower).clamp_min(1e-7)
    rgb = ((projected - lower) / scale).clamp(0, 1)
    array = (rgb.reshape(height, width, 3).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _event_sampling_grid(
    dataset_root: Path,
    sequence: str,
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, PairedSequenceTransform]:
    source_input_size = metadata.get("source_input_size")
    if not isinstance(source_input_size, list) or len(source_input_size) != 2:
        raise ValueError("Benchmark metadata lacks source_input_size")
    scale = int(metadata.get("detection_scale", 1))
    with h5py.File(find_rectify_map(dataset_root, sequence), "r") as handle:
        rectify_map = np.asarray(handle["rectify_map"], dtype=np.float32)
    source_size = (int(source_input_size[0]), int(source_input_size[1]))
    grid = build_dagr_sampling_grid(
        rectify_map,
        source_input_size=source_size,
        source_stride=1,
        scale=scale,
    )
    return grid, PairedSequenceTransform(
        height=source_size[0],
        width=source_size[1],
        training=False,
    )


def _event_image(
    event_cache_dir: Path,
    sequence: str,
    timestamp: int,
    transform: PairedSequenceTransform,
    grid: torch.Tensor,
) -> Image.Image:
    payload = _safe_load(event_cache_dir / sequence / f"{timestamp}.pt")
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, torch.Tensor) or events.ndim != 3:
        raise ValueError(f"invalid event cache at {sequence}/{timestamp}")
    transformed, _ = transform(events.float().unsqueeze(0), None)
    if transformed is None:
        raise RuntimeError("Event transform returned no tensor")
    warped = F.grid_sample(
        transformed,
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    array = warped[:3].clamp(0, 255).byte().permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array, mode="RGB")


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: int = 9,
) -> None:
    x1, y1, x2, y2 = box
    for start in np.arange(x1, x2, dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=fill, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=fill, width=width)
    for start in np.arange(y1, y2, dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=fill, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=fill, width=width)


def _label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    bounds = draw.textbbox((x, y), text, font=font, anchor="la")
    draw.rectangle(bounds, fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font, anchor="la")


def _detection_image(
    background: Image.Image,
    target: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    font: ImageFont.ImageFont,
) -> Image.Image:
    image = background.copy()
    draw = ImageDraw.Draw(image)
    for box, label in zip(target["boxes"], target["labels"]):
        coordinates = tuple(float(value) for value in box.tolist())
        _draw_dashed_rectangle(draw, coordinates, fill=GT_COLOR, width=3)
        class_name = DAGR_CLASSES[int(label)]
        _label(draw, (coordinates[0], coordinates[1]), f"GT {class_name}", GT_COLOR, font)
    for box, score, label in zip(
        prediction["boxes"], prediction["scores"], prediction["labels"]
    ):
        coordinates = tuple(float(value) for value in box.tolist())
        color = PREDICTION_COLORS[int(label) % len(PREDICTION_COLORS)]
        draw.rectangle(coordinates, outline=(0, 0, 0), width=5)
        draw.rectangle(coordinates, outline=color, width=3)
        class_name = DAGR_CLASSES[int(label)]
        _label(
            draw,
            (coordinates[0], max(0.0, coordinates[1] - 17.0)),
            f"{class_name} {float(score):.2f}",
            color,
            font,
        )
    return image


def _panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    *,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    title_height = 54
    result = Image.new("RGB", (width, height + title_height), "white")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    result.paste(image, (0, title_height))
    draw = ImageDraw.Draw(result)
    draw.text((width // 2, 5), title, fill="black", font=font, anchor="ma")
    draw.text((width // 2, 29), subtitle, fill=(70, 70, 70), font=small_font, anchor="ma")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0, 1]")
    if args.fps <= 0 or args.panel_width <= 0:
        raise ValueError("fps and panel-width must be positive")
    if args.pca_tokens_per_frame <= 0 or args.pca_max_samples < 3:
        raise ValueError("PCA sampling values must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error

    sources = [_parse_source(value) for value in args.source]
    if len(sources) < 1 or len({source.label for source in sources}) != len(sources):
        raise ValueError("source labels must be unique")
    split = load_dsec_detection_split(args.split_manifest)
    if args.sequence not in split.sequences(args.role):
        raise ValueError(f"{args.sequence!r} is not in official {args.role} split")

    device = torch.device(args.device)
    datasets = [_dataset(source, args) for source in sources]
    indices = _selected_indices(len(datasets[0]), args.start_frame, args.max_frames)
    keys = _sample_keys(datasets[0], indices)
    for source, dataset in zip(sources[1:], datasets[1:]):
        if dataset.input_size != datasets[0].input_size:
            raise ValueError(f"{source.label} input geometry differs")
        if _sample_keys(dataset, indices) != keys:
            raise ValueError(f"{source.label} evaluated-frame manifest differs")

    all_predictions: list[list[dict[str, torch.Tensor]]] = []
    targets: list[dict[str, torch.Tensor]] | None = None
    for source, dataset in zip(sources, datasets):
        predictions, current_targets = _infer(source, dataset, indices, args, device)
        all_predictions.append(predictions)
        if targets is None:
            targets = current_targets
        elif any(
            not torch.equal(first["boxes"], second["boxes"])
            or not torch.equal(first["labels"], second["labels"])
            for first, second in zip(targets, current_targets)
        ):
            raise ValueError(f"{source.label} targets differ from the first source")
    if targets is None:
        raise RuntimeError("No detection frames selected")

    mean, basis, lower, upper = _fit_shared_pca(
        sources,
        keys,
        args.pca_tokens_per_frame,
        args.pca_max_samples,
    )
    metadata_path = sources[0].feature_cache_dir / args.sequence / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    event_grid, event_transform = _event_sampling_grid(
        args.dataset_root.expanduser().resolve(), args.sequence, metadata
    )
    input_size = datasets[0].input_size
    if input_size is None:
        raise RuntimeError("Detection input size was not initialized")
    input_height, input_width = input_size
    panel_width = args.panel_width
    panel_height = round(panel_width * input_height / input_width)
    gap = 6
    header_height = 38
    panel_full_height = panel_height + 54
    canvas_width = len(sources) * panel_width + (len(sources) - 1) * gap
    canvas_height = header_height + 2 * panel_full_height + gap
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    title_font = _font(18)
    small_font = _font(13)
    box_font = _font(13)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for frame_index, ((_, timestamp), target) in enumerate(zip(keys, targets)):
        row: dict[str, Any] = {
            "frame": frame_index,
            "timestamp": timestamp,
            "gt_count": len(target["boxes"]),
        }
        for source, predictions in zip(sources, all_predictions):
            prediction = predictions[frame_index]
            row[f"{source.label}_prediction_count"] = len(prediction["boxes"])
            row[f"{source.label}_max_score"] = (
                float(prediction["scores"].max()) if len(prediction["scores"]) else 0.0
            )
        rows.append(row)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sequence": args.sequence,
                "role": args.role,
                "evaluated_frame_count": len(keys),
                "score_threshold": args.score_threshold,
                "fps": args.fps,
                "sources": [
                    {
                        "label": source.label,
                        "checkpoint": str(source.checkpoint),
                        "feature_cache_dir": str(source.feature_cache_dir),
                        "feature": source.feature,
                    }
                    for source in sources
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

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
        for frame_index, ((sequence, timestamp), target) in enumerate(zip(keys, targets)):
            background = _event_image(
                args.event_cache_dir.expanduser().resolve(),
                sequence,
                timestamp,
                event_transform,
                event_grid,
            )
            canvas = Image.new("RGB", (canvas_width, canvas_height), (235, 235, 235))
            canvas_draw = ImageDraw.Draw(canvas)
            canvas_draw.text(
                (canvas_width // 2, 7),
                f"{sequence} | evaluated frame {frame_index + 1}/{len(keys)} | "
                f"timestamp {timestamp} | GT dashed white; car orange; pedestrian cyan",
                fill="black",
                font=small_font,
                anchor="ma",
            )
            for column, (source, predictions) in enumerate(zip(sources, all_predictions)):
                prediction = predictions[frame_index]
                detection = _detection_image(background, target, prediction, box_font)
                detection_panel = _panel(
                    detection,
                    f"{source.label} detection",
                    f"{source.feature} | predictions={len(prediction['boxes'])} | "
                    f"threshold={args.score_threshold:.2f}",
                    width=panel_width,
                    height=panel_height,
                    font=title_font,
                    small_font=small_font,
                )
                feature = _feature_map(source, sequence, timestamp)
                pca = _pca_image(
                    feature,
                    mean,
                    basis,
                    lower,
                    upper,
                    (panel_width, panel_height),
                )
                feature_panel = _panel(
                    pca,
                    f"{source.label} {source.feature} feature",
                    "shared normalized PCA across all models and frames",
                    width=panel_width,
                    height=panel_height,
                    font=title_font,
                    small_font=small_font,
                )
                x = column * (panel_width + gap)
                canvas.paste(detection_panel, (x, header_height))
                canvas.paste(feature_panel, (x, header_height + panel_full_height + gap))
            writer.send(np.asarray(canvas, dtype=np.uint8))
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == len(keys):
                print(f"Rendered {frame_index + 1}/{len(keys)} frames", flush=True)
    finally:
        writer.close()
    print(f"Video: {output}")
    print(f"Per-frame summary: {output.with_suffix('.csv')}")
    print(f"Run metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
