#!/usr/bin/env python3
"""Render M3ED semantic RGB/events/features/pseudo-GT/predictions as MP4."""

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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from event_state.segmentation import (
    DSEC_SEMANTIC_11_CLASSES,
    EventStateSegmentationHead,
    M3EDSemanticFeatureDataset,
    load_m3ed_semantic_split,
    segmentation_collate,
)
from visualize_dsec_semantic_sequences import (
    PALETTE,
    _error_image,
    _feature_from_payload,
    _font,
    _panel,
    _pca_image,
    _per_frame_miou,
    _safe_load,
    _sample_tokens,
    _semantic_image,
)


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "m3ed_downstream_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "validation"), default="validation")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--feature", choices=("z", "h", "concat"), default=None)
    parser.add_argument("--model-label", default="Hybrid + dropout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--pca-max-samples", type=int, default=50_000)
    parser.add_argument("--pca-max-frames", type=int, default=2_000)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based position within the selected sequence's labeled frames",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _selected_sequences(args: argparse.Namespace) -> tuple[str, ...]:
    split = load_m3ed_semantic_split(args.split_manifest)
    available = split.sequences(args.role)
    if not args.sequences:
        return available
    unknown = sorted(set(args.sequences) - set(available))
    if unknown:
        raise ValueError(f"Sequences are not in M3ED semantic {args.role}: {', '.join(unknown)}")
    return tuple(args.sequences)


def _fit_shared_pca(
    datasets: Iterable[M3EDSemanticFeatureDataset],
    feature: str,
    *,
    tokens_per_frame: int,
    max_samples: int,
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = [sample[2] for dataset in datasets for sample in dataset.samples]
    if not paths:
        raise ValueError("No M3ED semantic feature frames are available for PCA")
    if len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames).round().astype(np.int64)
        paths = [paths[int(index)] for index in np.unique(indices)]
    samples: list[torch.Tensor] = []
    channels: int | None = None
    for path in tqdm(paths, desc="M3ED Semantic PCA sampling", unit="frame"):
        payload = _safe_load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid semantic feature payload: {path}")
        value = _feature_from_payload(payload, feature)
        if channels is None:
            channels = int(value.shape[0])
        elif channels != int(value.shape[0]):
            raise ValueError("M3ED semantic features use different channel dimensions")
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
    print(f"Fitted shared PCA from {len(merged):,} tokens", flush=True)
    return mean, basis[:, :3], lower, upper


def _rgb_image(prepared_root: Path, sequence: str, timestamp: int) -> Image.Image:
    path = prepared_root / sequence / "aligned_rgb" / f"{timestamp}.png"
    with Image.open(path) as image:
        value = image.convert("RGB")
        if value.size == (640, 360):
            value = value.crop((0, 4, 640, 356))
        if value.size != (640, 352):
            raise ValueError(f"Unexpected M3ED RGB size {value.size}: {path}")
        return value.copy()


def _event_image(prepared_root: Path, sequence: str, timestamp: int) -> Image.Image:
    path = prepared_root / sequence / "events" / f"{timestamp}.pt"
    payload = _safe_load(path)
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, torch.Tensor) or events.ndim != 3:
        raise ValueError(f"Invalid M3ED event cache: {path}")
    value = events[:3].float()
    if tuple(value.shape[-2:]) == (360, 640):
        value = value[:, 4:356]
    if tuple(value.shape[-2:]) != (352, 640):
        raise ValueError(f"Unexpected M3ED event size {tuple(value.shape)}: {path}")
    if value.numel() and float(value.max()) <= 1.5 and float(value.min()) >= 0.0:
        value = value * 255.0
    array = value.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _load_head(
    checkpoint: dict[str, Any],
    dataset: M3EDSemanticFeatureDataset,
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


@torch.inference_mode()
def _render_sequence(
    *,
    args: argparse.Namespace,
    sequence: str,
    dataset: M3EDSemanticFeatureDataset,
    model: EventStateSegmentationHead,
    feature: str,
    pca: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    imageio_ffmpeg: Any,
) -> None:
    output = args.output_dir / f"{sequence}.mp4"
    csv_output = output.with_suffix(".csv")
    json_output = output.with_suffix(".json")
    if output.is_file() and csv_output.is_file() and json_output.is_file() and not args.overwrite:
        print(f"{sequence}: existing video skipped: {output}", flush=True)
        return
    stop = len(dataset)
    if args.max_frames is not None:
        stop = min(stop, args.start_frame + args.max_frames * args.frame_stride)
    indices = list(range(args.start_frame, stop, args.frame_stride))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError(f"No visualization frames selected for {sequence}")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=segmentation_collate,
    )
    loader_iterator = iter(loader)

    panel_width = args.panel_width
    panel_height = round(panel_width * 352 / 640)
    gap = 6
    header_height = 38
    panel_full_height = panel_height + 54
    canvas_width = math.ceil((panel_width * 3 + gap * 2) / 16) * 16
    canvas_height = math.ceil((header_height + panel_full_height * 2 + gap) / 16) * 16
    title_font = _font(18)
    small_font = _font(13)
    use_amp = args.precision == "fp16" and device.type == "cuda"
    rows: list[dict[str, Any]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    temporary.unlink(missing_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
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
        for batch in tqdm(
            loader_iterator,
            total=len(loader),
            desc=f"M3ED semantic video:{sequence}",
            unit="batch",
        ):
            features = batch["features"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                probabilities = model(features).softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for local_index in range(len(predictions)):
                frame_index = int(batch["frame_indices"][local_index])
                timestamp = int(batch["timestamps"][local_index])
                labels = batch["labels"][local_index].cpu()
                prediction = predictions[local_index].cpu()
                rgb = _rgb_image(args.prepared_root, sequence, timestamp)
                events = _event_image(args.prepared_root, sequence, timestamp)
                pca_image = _pca_image(batch["features"][local_index].cpu(), pca)
                gt_image = _semantic_image(labels)
                prediction_image = _semantic_image(prediction)
                error_image, errors, valid_pixels = _error_image(rgb, labels, prediction)
                pixel_accuracy = 1.0 - errors / max(1, valid_pixels)
                frame_miou = _per_frame_miou(labels, prediction)
                mean_confidence = float(confidence[local_index].cpu()[labels != 255].mean())

                canvas = Image.new("RGB", (canvas_width, canvas_height), (235, 235, 235))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (canvas_width // 2, 7),
                    f"{sequence} | frame {rendered + 1}/{len(indices)} | index {frame_index} | "
                    f"timestamp {timestamp} | EVENT-ONLY",
                    fill="black",
                    font=small_font,
                    anchor="ma",
                )
                panels = (
                    _panel(rgb, "Aligned RGB reference", "display only; not used for inference", width=panel_width, height=panel_height, font=title_font, small_font=small_font),
                    _panel(events, "3-channel event input", "causal event interval", width=panel_width, height=panel_height, font=title_font, small_font=small_font),
                    _panel(pca_image, f"{args.model_label} {feature} feature", "shared normalized PCA", width=panel_width, height=panel_height, font=title_font, small_font=small_font),
                    _panel(gt_image, "Semantic pseudo-label", "InternImage; mapped to DSEC-11", width=panel_width, height=panel_height, font=title_font, small_font=small_font, nearest=True),
                    _panel(prediction_image, "Event-only prediction", f"frame mIoU={frame_miou:.3f} | confidence={mean_confidence:.3f}", width=panel_width, height=panel_height, font=title_font, small_font=small_font, nearest=True),
                    _panel(error_image, "Prediction error", f"green=correct; red=wrong | pixel accuracy={pixel_accuracy:.3f}", width=panel_width, height=panel_height, font=title_font, small_font=small_font),
                )
                for column in range(3):
                    x = column * (panel_width + gap)
                    canvas.paste(panels[column], (x, header_height))
                    canvas.paste(panels[column + 3], (x, header_height + panel_full_height + gap))
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
        print(f"{sequence}: finalizing MP4 after {rendered} frames...", flush=True)
        writer.close()
        print(f"{sequence}: MP4 finalized", flush=True)
    temporary.replace(output)

    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    json_output.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "role": args.role,
                "frame_count": rendered,
                "start_frame": args.start_frame,
                "frame_stride": args.frame_stride,
                "fps": args.fps,
                "feature": feature,
                "checkpoint": str(args.checkpoint),
                "label_source": "M3ED InternImage pseudo-labels mapped to DSEC-11",
                "rgb_usage": "visual_reference_only",
                "inference_input": "cached_event_feature_only",
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
    if args.batch_size <= 0 or args.num_workers < 0 or args.fps <= 0:
        raise ValueError("batch-size/fps must be positive and num-workers non-negative")
    if args.start_frame < 0 or args.frame_stride <= 0:
        raise ValueError("start-frame must be non-negative and frame-stride positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    if args.pca_max_frames <= 0 or args.pca_max_samples < 3:
        raise ValueError("PCA sampling limits must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise SystemExit("Install visualization dependencies with `uv sync --active --extra visualize`") from error
    for name in (
        "checkpoint",
        "feature_cache_dir",
        "prepared_root",
        "target_cache_dir",
        "output_dir",
    ):
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Semantic checkpoint not found: {args.checkpoint}")
    for name in ("feature_cache_dir", "prepared_root", "target_cache_dir"):
        if not getattr(args, name).is_dir():
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    feature = args.feature or config.get("feature")
    if feature not in {"z", "h", "concat"}:
        raise ValueError("Feature is absent from the checkpoint; pass --feature")
    sequences = _selected_sequences(args)
    datasets = [
        M3EDSemanticFeatureDataset(
            feature_cache_dir=args.feature_cache_dir,
            target_cache_dir=args.target_cache_dir,
            sequences=(sequence,),
            feature=feature,
        )
        for sequence in sequences
    ]
    pca = _fit_shared_pca(
        datasets,
        feature,
        tokens_per_frame=args.pca_tokens_per_frame,
        max_samples=args.pca_max_samples,
        max_frames=args.pca_max_frames,
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
    print(f"M3ED semantic visualization complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
