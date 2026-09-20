#!/usr/bin/env python3
"""Render DSEC-Semantic inputs, frozen features, GT, and predictions as MP4s."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.segmentation import (
    DSEC_SEMANTIC_11_CLASSES,
    DSECSemanticFeatureDataset,
    EventStateSegmentationHead,
    load_dsec_semantic_split,
    segmentation_collate,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_semantic_split.yaml"

# Cityscapes-style colors, reordered to DSEC's official 11-class IDs.
PALETTE = np.asarray(
    [
        (0, 0, 0),          # background
        (70, 70, 70),       # building
        (190, 153, 153),    # fence
        (220, 20, 60),      # person
        (153, 153, 153),    # pole
        (128, 64, 128),     # road
        (244, 35, 232),     # sidewalk
        (107, 142, 35),     # vegetation
        (0, 0, 142),        # car
        (102, 102, 156),    # wall
        (220, 220, 0),      # traffic sign
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render event-only DSEC-Semantic predictions beside RGB reference, "
            "event input, frozen-feature PCA, GT, and an error overlay"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Subset of the selected role; omission renders every sequence",
    )
    parser.add_argument(
        "--feature",
        choices=("z", "h", "concat"),
        default=None,
        help="Defaults to the feature stored in the semantic-head checkpoint",
    )
    parser.add_argument(
        "--model-label",
        default="EventState",
        help="Short label shown above the PCA feature panel, for example E3",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--pca-max-samples", type=int, default=50_000)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional per-sequence limit for quick previews",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _selected_sequences(args: argparse.Namespace) -> tuple[str, ...]:
    split = load_dsec_semantic_split(args.split_manifest)
    available = tuple(split.sequences(args.role))
    if args.sequences is None or not args.sequences:
        return available
    unknown = sorted(set(args.sequences) - set(available))
    if unknown:
        raise ValueError(
            f"Sequences are not in semantic {args.role}: {', '.join(unknown)}"
        )
    return tuple(args.sequences)


def _feature_from_payload(payload: dict[str, Any], feature: str) -> torch.Tensor:
    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError("Semantic feature payload lacks features")
    if feature == "concat":
        value = torch.cat((features["z"], features["h"]), dim=0)
    else:
        value = features.get(feature)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"Invalid {feature} feature map")
    return value.float()


def _sample_tokens(value: torch.Tensor, count: int) -> torch.Tensor:
    tokens = value.permute(1, 2, 0).reshape(-1, value.shape[0])
    tokens = F.normalize(tokens.float(), dim=-1)
    count = min(count, len(tokens))
    indices = torch.linspace(0, len(tokens) - 1, count).round().long().unique()
    return tokens[indices]


def _fit_shared_pca(
    datasets: Iterable[DSECSemanticFeatureDataset],
    feature: str,
    tokens_per_frame: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = [sample[2] for dataset in datasets for sample in dataset.samples]
    if not paths:
        raise ValueError("No semantic feature frames are available for PCA")
    samples: list[torch.Tensor] = []
    channels: int | None = None
    for path in tqdm(paths, desc="Semantic PCA sampling", unit="frame"):
        payload = _safe_load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid semantic feature payload: {path}")
        value = _feature_from_payload(payload, feature)
        if channels is None:
            channels = int(value.shape[0])
        elif channels != int(value.shape[0]):
            raise ValueError("Semantic features use different channel dimensions")
        samples.append(_sample_tokens(value, tokens_per_frame))
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
    print(
        f"Fitted one shared PCA from {len(merged):,} tokens across all sequences",
        flush=True,
    )
    return mean, basis[:, :3], lower, upper


def _pca_image(
    value: torch.Tensor,
    pca: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Image.Image:
    mean, basis, lower, upper = pca
    height, width = value.shape[-2:]
    tokens = F.normalize(
        value.permute(1, 2, 0).reshape(-1, value.shape[0]).float(), dim=-1
    )
    projected = (tokens - mean) @ basis
    scale = (upper - lower).clamp_min(1e-7)
    rgb = ((projected - lower) / scale).clamp(0, 1)
    array = (rgb.reshape(height, width, 3).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _event_image(event_cache_dir: Path, sequence: str, timestamp: int) -> Image.Image:
    path = event_cache_dir / sequence / f"{timestamp}.pt"
    payload = _safe_load(path)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, torch.Tensor) or events.ndim != 3:
        raise ValueError(f"Invalid event cache: {path}")
    value = events[:3, :440, :640].float()
    if value.numel() and float(value.max()) <= 1.5 and float(value.min()) >= 0.0:
        value = value * 255.0
    array = value.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array, mode="RGB")


def _find_aligned_rgb(dataset_root: Path, sequence: str, timestamp: int) -> Path:
    relative = Path(sequence) / "images" / "left" / "aligned_event" / f"{timestamp}.png"
    candidates = (
        dataset_root / "train_images" / relative,
        dataset_root / "test_images" / relative,
        dataset_root / "dsec_det_extra" / "train" / relative,
        dataset_root / "dsec_det_extra" / "test" / relative,
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(path) for path in matches)
        raise FileNotFoundError(
            f"Expected one aligned RGB for {sequence}/{timestamp}; found {detail}"
        )
    return matches[0]


def _rgb_image(dataset_root: Path, sequence: str, timestamp: int) -> Image.Image:
    path = _find_aligned_rgb(dataset_root, sequence, timestamp)
    with Image.open(path) as image:
        return image.convert("RGB").crop((0, 0, 640, 440)).copy()


def _semantic_image(labels: torch.Tensor) -> Image.Image:
    values = labels.cpu().numpy().astype(np.int64, copy=False)
    output = np.full((*values.shape, 3), 40, dtype=np.uint8)
    valid = (values >= 0) & (values < len(PALETTE))
    output[valid] = PALETTE[values[valid]]
    return Image.fromarray(output, mode="RGB")


def _error_image(
    rgb: Image.Image, labels: torch.Tensor, prediction: torch.Tensor
) -> tuple[Image.Image, int, int]:
    target = labels.cpu().numpy()
    predicted = prediction.cpu().numpy()
    valid = target != 255
    incorrect = valid & (target != predicted)
    base = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = base.copy()
    overlay[valid & ~incorrect] = (40, 190, 80)
    overlay[incorrect] = (255, 45, 45)
    output = (0.45 * base + 0.55 * overlay).round().astype(np.uint8)
    output[~valid] = (35, 35, 35)
    return Image.fromarray(output, mode="RGB"), int(incorrect.sum()), int(valid.sum())


def _panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    *,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    nearest: bool = False,
) -> Image.Image:
    title_height = 54
    result = Image.new("RGB", (width, height + title_height), "white")
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    result.paste(image.resize((width, height), resampling), (0, title_height))
    draw = ImageDraw.Draw(result)
    draw.text((width // 2, 5), title, fill="black", font=font, anchor="ma")
    draw.text(
        (width // 2, 29), subtitle, fill=(70, 70, 70), font=small_font, anchor="ma"
    )
    return result


def _load_head(
    checkpoint: dict[str, Any],
    dataset: DSECSemanticFeatureDataset,
    feature: str,
    device: torch.device,
) -> EventStateSegmentationHead:
    config = checkpoint.get("config", {})
    expected = config.get("feature")
    if expected is not None and expected != feature:
        raise ValueError(f"Semantic head expects feature={expected!r}, not {feature!r}")
    channels = int(dataset[0]["feature"].shape[0])
    model = EventStateSegmentationHead(
        in_channels=channels,
        num_classes=int(config.get("num_classes", len(DSEC_SEMANTIC_11_CLASSES))),
        width=int(config.get("head_width", 192)),
        output_size=dataset.label_size,
        head_type=str(config.get("head_type", "linear")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _per_frame_miou(labels: torch.Tensor, prediction: torch.Tensor) -> float:
    valid = labels != 255
    scores: list[float] = []
    for class_index in range(len(DSEC_SEMANTIC_11_CLASSES)):
        target_class = (labels == class_index) & valid
        prediction_class = (prediction == class_index) & valid
        union = (target_class | prediction_class).sum().item()
        if union:
            intersection = (target_class & prediction_class).sum().item()
            scores.append(intersection / union)
    return float(sum(scores) / len(scores)) if scores else float("nan")


@torch.inference_mode()
def _render_sequence(
    *,
    args: argparse.Namespace,
    sequence: str,
    dataset: DSECSemanticFeatureDataset,
    model: EventStateSegmentationHead,
    feature: str,
    pca: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    imageio_ffmpeg: Any,
) -> None:
    output = args.output_dir / f"{sequence}.mp4"
    csv_output = output.with_suffix(".csv")
    json_output = output.with_suffix(".json")
    if (
        output.is_file()
        and csv_output.is_file()
        and json_output.is_file()
        and not args.overwrite
    ):
        print(f"{sequence}: existing video skipped: {output}", flush=True)
        return
    frame_count = len(dataset)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise ValueError(f"No visualization frames selected for {sequence}")

    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(frame_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=segmentation_collate,
    )
    panel_width = args.panel_width
    panel_height = round(panel_width * 440 / 640)
    gap = 6
    header_height = 38
    panel_full_height = panel_height + 54
    content_width = panel_width * 3 + gap * 2
    content_height = header_height + panel_full_height * 2 + gap
    # Avoid imageio-ffmpeg's implicit scale filter. Padding to a macroblock
    # boundary keeps every submitted RGB frame and the encoded stream identical
    # in size, while remaining compatible with conservative H.264 players.
    canvas_width = math.ceil(content_width / 16) * 16
    canvas_height = math.ceil(content_height / 16) * 16
    title_font = _font(18)
    small_font = _font(13)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    rows: list[dict[str, Any]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.partial.mp4")
    temporary_output.unlink(missing_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(temporary_output),
        size=(canvas_width, canvas_height),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    rendered = 0
    try:
        for batch in tqdm(loader, desc=f"semantic video:{sequence}", unit="batch"):
            features = batch["features"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(features)
                probabilities = logits.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            features_cpu = batch["features"].cpu()
            labels_cpu = batch["labels"].cpu()
            predictions = predictions.cpu()
            confidence = confidence.cpu()

            for local_index in range(len(predictions)):
                sequence_name = batch["sequence_names"][local_index]
                frame_index = int(batch["frame_indices"][local_index])
                timestamp = int(batch["timestamps"][local_index])
                labels = labels_cpu[local_index]
                prediction = predictions[local_index]
                rgb = _rgb_image(args.dataset_root, sequence_name, timestamp)
                events = _event_image(args.event_cache_dir, sequence_name, timestamp)
                pca_image = _pca_image(features_cpu[local_index], pca)
                gt_image = _semantic_image(labels)
                prediction_image = _semantic_image(prediction)
                error_image, errors, valid_pixels = _error_image(rgb, labels, prediction)
                pixel_accuracy = 1.0 - errors / max(1, valid_pixels)
                frame_miou = _per_frame_miou(labels, prediction)
                mean_confidence = float(confidence[local_index][labels != 255].mean())

                canvas = Image.new(
                    "RGB", (canvas_width, canvas_height), (235, 235, 235)
                )
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (canvas_width // 2, 7),
                    f"{sequence_name} | labeled frame {rendered + 1}/{frame_count} | "
                    f"index {frame_index} | timestamp {timestamp} | EVENT-ONLY prediction",
                    fill="black",
                    font=small_font,
                    anchor="ma",
                )
                panels = (
                    _panel(
                        rgb,
                        "Aligned RGB reference",
                        "display only; never passed to the semantic model",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                    _panel(
                        events,
                        "GEP 3-channel event input",
                        "causal input used to produce the frozen feature",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                    _panel(
                        pca_image,
                        f"{args.model_label} {feature} feature",
                        "shared normalized PCA across all selected sequences",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                    _panel(
                        gt_image,
                        "Ground truth",
                        "official DSEC-Semantic 11-class label",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                        nearest=True,
                    ),
                    _panel(
                        prediction_image,
                        "Event-only prediction",
                        f"frame mIoU={frame_miou:.3f} | confidence={mean_confidence:.3f}",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                        nearest=True,
                    ),
                    _panel(
                        error_image,
                        "Prediction error on RGB reference",
                        f"green=correct; red=wrong | pixel accuracy={pixel_accuracy:.3f}",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                )
                for column in range(3):
                    x = column * (panel_width + gap)
                    canvas.paste(panels[column], (x, header_height))
                    canvas.paste(
                        panels[column + 3],
                        (x, header_height + panel_full_height + gap),
                    )
                writer.send(np.asarray(canvas, dtype=np.uint8))
                rows.append(
                    {
                        "video_frame": rendered,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "valid_pixels": valid_pixels,
                        "incorrect_pixels": errors,
                        "pixel_accuracy": pixel_accuracy,
                        "frame_mIoU": frame_miou,
                        "mean_confidence": mean_confidence,
                    }
                )
                rendered += 1
    finally:
        writer.close()

    # The public path becomes visible only after FFmpeg has finalized the MP4
    # container and written its moov atom.
    temporary_output.replace(output)

    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    json_output.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "role": args.role,
                "frame_count": rendered,
                "fps": args.fps,
                "feature": feature,
                "checkpoint": str(args.checkpoint),
                "feature_cache_dir": str(args.feature_cache_dir),
                "rgb_usage": "visual_reference_only",
                "inference_input": "cached_event_feature_only",
                "pca_scope": "all_selected_sequences",
                "classes": list(DSEC_SEMANTIC_11_CLASSES),
                "palette": PALETTE.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{sequence}: video={output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.fps <= 0 or args.panel_width <= 0:
        raise ValueError("fps and panel-width must be positive")
    if args.pca_tokens_per_frame <= 0 or args.pca_max_samples < 3:
        raise ValueError("PCA sampling values must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit(
            "Install visualization dependencies with `uv sync --active --extra visualize`"
        ) from error

    for name in (
        "checkpoint",
        "feature_cache_dir",
        "event_cache_dir",
        "labels_root",
        "dataset_root",
        "output_dir",
    ):
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Semantic checkpoint not found: {args.checkpoint}")
    for name in ("feature_cache_dir", "event_cache_dir", "labels_root", "dataset_root"):
        value = getattr(args, name)
        if not value.is_dir():
            raise FileNotFoundError(f"{name} not found: {value}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    feature = args.feature or config.get("feature")
    if feature not in {"z", "h", "concat"}:
        raise ValueError("Feature is absent from the checkpoint; pass --feature")
    sequences = _selected_sequences(args)
    datasets = [
        DSECSemanticFeatureDataset(
            feature_cache_dir=args.feature_cache_dir,
            labels_root=args.labels_root,
            sequences=(sequence,),
            role=args.role,
            feature=feature,
            num_classes=int(config.get("num_classes", len(DSEC_SEMANTIC_11_CLASSES))),
        )
        for sequence in sequences
    ]
    pca = _fit_shared_pca(
        datasets, feature, args.pca_tokens_per_frame, args.pca_max_samples
    )
    device = torch.device(args.device)
    model = _load_head(checkpoint, datasets[0], feature, device)
    for sequence, dataset in zip(sequences, datasets):
        _render_sequence(
            args=args,
            sequence=sequence,
            dataset=dataset,
            model=model,
            feature=feature,
            pca=pca,
            device=device,
            imageio_ffmpeg=imageio_ffmpeg,
        )
    print(f"Semantic visualization complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
