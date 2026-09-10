#!/usr/bin/env python3
"""Render end-to-end fine-tuned EventState+YOLOX predictions as one MP4."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_state.detection import (
    DSECDetectionEventDataset,
    EventStateYOLOX,
    detection_event_collate,
    load_dsec_detection_split,
)
from event_state.training import build_model
from visualize_dsec_detection_sequence import (
    DEFAULT_SPLIT,
    _detection_image,
    _event_image,
    _event_sampling_grid,
    _font,
    _panel,
    _pca_image,
    _rgb_image,
    _sample_feature_tokens,
)


@dataclass(frozen=True)
class FineTuneSource:
    label: str
    checkpoint: Path
    feature: str


@dataclass
class CheckpointSnapshot:
    source: FineTuneSource
    student: Mapping[str, Tensor]
    detector: Mapping[str, Tensor]
    reference_config: dict[str, Any]
    run_config: dict[str, Any]
    epoch: int
    best_map: float


@dataclass
class InferenceResult:
    source: FineTuneSource
    keys: list[tuple[str, int]]
    predictions: list[dict[str, Tensor]]
    targets: list[dict[str, Tensor]]
    evaluated: list[bool]
    features: list[Tensor]
    epoch: int
    best_map: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render current end-to-end fine-tuned EventState+YOLOX checkpoints "
            "over one complete DSEC-Detection sequence"
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL:CHECKPOINT:FEATURE",
        help="Repeat for every model, for example E2:/path/best.pt:h",
    )
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--frame-mode",
        choices=("all", "evaluated"),
        default="all",
        help="Render every event frame or only official evaluated frames",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--detection-background", choices=("rgb", "event"), default="rgb"
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--pca-max-samples", type=int, default=50_000)
    parser.add_argument("--panel-width", type=int, default=480)
    return parser.parse_args()


def _parse_source(value: str) -> FineTuneSource:
    try:
        label, checkpoint, feature = value.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"source must be LABEL:CHECKPOINT:FEATURE, got {value!r}"
        ) from error
    if feature not in {"z", "h", "concat"}:
        raise argparse.ArgumentTypeError(f"unsupported feature {feature!r}")
    path = Path(checkpoint).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid fine-tune source: {value!r}")
    return FineTuneSource(label=label, checkpoint=path, feature=feature)


def _snapshot(source: FineTuneSource) -> CheckpointSnapshot:
    # Training saves best.pt atomically, so one load observes a complete checkpoint
    # even when fine-tuning is still running.
    payload = torch.load(source.checkpoint, map_location="cpu", weights_only=False)
    run_config = payload.get("config")
    if not isinstance(run_config, dict):
        raise ValueError(f"Checkpoint lacks fine-tune config: {source.checkpoint}")
    if run_config.get("mode") != "finetune":
        raise ValueError(f"Checkpoint is not a fine-tune run: {source.checkpoint}")
    if run_config.get("feature") != source.feature:
        raise ValueError(
            f"{source.label} checkpoint expects {run_config.get('feature')!r}, "
            f"not {source.feature!r}"
        )
    reference_config = run_config.get("reference_config")
    if not isinstance(reference_config, dict):
        raise ValueError(f"Checkpoint lacks reference_config: {source.checkpoint}")
    student = payload.get("student")
    detector = payload.get("detector")
    if not isinstance(student, Mapping) or not isinstance(detector, Mapping):
        raise ValueError(f"Checkpoint lacks student/detector weights: {source.checkpoint}")
    return CheckpointSnapshot(
        source=source,
        student=student,
        detector=detector,
        reference_config=reference_config,
        run_config=run_config,
        epoch=int(payload.get("epoch", -1)),
        best_map=float(payload.get("best_mAP", float("nan"))),
    )


def _value(config: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _construction_config(reference_config: dict[str, Any]) -> Any:
    config = OmegaConf.create(reference_config)
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(result, "model.event_encoder.pretrained_init", False, merge=False)
    OmegaConf.update(result, "model.event_encoder.patch_init", "random", merge=False)
    OmegaConf.update(
        result, "model.event_encoder.checkpoint", None, merge=False, force_add=True
    )
    return result


def _freeze_projectors(model: nn.Module) -> None:
    for name in ("h_projector", "z_projector"):
        projector = getattr(model, name, None)
        if isinstance(projector, nn.Module):
            projector.eval()
            for parameter in projector.parameters():
                parameter.requires_grad_(False)


def _select_feature(outputs: Mapping[str, Any], feature: str) -> Tensor:
    z = outputs.get("z")
    h = outputs.get("h")
    if not isinstance(z, Tensor) or not isinstance(h, Tensor):
        raise ValueError("EventState must return z and h tensors")
    if feature == "z":
        return z[:, -1]
    if feature == "h":
        return h[:, -1]
    return torch.cat((z[:, -1], h[:, -1]), dim=-1)


def _detector_features(
    student: nn.Module,
    events: Tensor,
    sampling_grids: Tensor,
    *,
    feature: str,
    state: Any,
) -> tuple[Tensor, Any]:
    outputs = student(events, state=state)
    tokens = _select_feature(outputs, feature)
    patch_size = int(student.event_encoder.patch_size)
    grid_height = int(events.shape[-2]) // patch_size
    grid_width = int(events.shape[-1]) // patch_size
    if tokens.shape[1] != grid_height * grid_width:
        raise ValueError("EventState token count does not match the input patch grid")
    maps = tokens.reshape(tokens.shape[0], grid_height, grid_width, -1).permute(0, 3, 1, 2)
    warped = F.grid_sample(
        maps,
        sampling_grids.to(device=maps.device, dtype=maps.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped, outputs.get("state")


def _dataset(snapshot: CheckpointSnapshot, args: argparse.Namespace) -> DSECDetectionEventDataset:
    config = OmegaConf.create(snapshot.reference_config)
    event_mean = tuple(
        float(value)
        for value in _value(config, "dataset.representation.normalize_mean")
    )
    event_std = tuple(
        float(value)
        for value in _value(config, "dataset.representation.normalize_std")
    )
    patch_size = int(_value(config, "teacher.patch_size", 16))
    return DSECDetectionEventDataset(
        event_cache_dir=args.event_cache_dir,
        labels_root=args.labels_root,
        dataset_root=args.dataset_root,
        sequences=(args.sequence,),
        sequence_length=int(_value(config, "dataset.sequence_length", 8)),
        input_size=(
            int(_value(config, "dataset.input_height", 448)),
            int(_value(config, "dataset.input_width", 640)),
        ),
        source_stride=patch_size,
        event_mean=event_mean,
        event_std=event_std,
        continuous=True,
    )


def _selected_indices(
    dataset: DSECDetectionEventDataset, args: argparse.Namespace
) -> list[int]:
    candidates = [
        index
        for index, sample in enumerate(dataset.samples)
        if args.frame_mode == "all" or sample["target"] is not None
    ]
    if args.start_frame < 0 or args.start_frame >= len(candidates):
        raise ValueError(f"start-frame must be in [0, {len(candidates)})")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    stop = None if args.max_frames is None else args.start_frame + args.max_frames
    return candidates[args.start_frame:stop]


@torch.inference_mode()
def _infer(
    snapshot: CheckpointSnapshot,
    selected_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> InferenceResult:
    config = _construction_config(snapshot.reference_config)
    student = build_model(config, device=device)
    student.load_state_dict(snapshot.student)
    _freeze_projectors(student)
    student.eval()
    channels = int(student.event_dim)
    if snapshot.source.feature == "h":
        channels = int(student.temporal_dim)
    elif snapshot.source.feature == "concat":
        channels = int(student.event_dim) + int(student.temporal_dim)
    detector = EventStateYOLOX(
        in_channels=channels,
        width=int(snapshot.run_config.get("head_width", 192)),
        input_stride=int(student.event_encoder.patch_size),
        image_size=(430, 640),
        confidence_threshold=min(0.001, args.score_threshold),
    ).to(device)
    detector.load_state_dict(snapshot.detector)
    detector.eval()

    dataset = _dataset(snapshot, args)
    selected = set(selected_indices)
    last_required = selected_indices[-1]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=detection_event_collate,
    )
    use_amp = args.precision == "fp16" and device.type == "cuda"
    keys: list[tuple[str, int]] = []
    predictions: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    evaluated: list[bool] = []
    features: list[Tensor] = []
    state = None
    progress = tqdm(
        total=last_required + 1,
        desc=f"{snapshot.source.label} fine-tune inference",
        unit="frame",
    )
    for index, batch in enumerate(loader):
        if index > last_required:
            break
        if bool(batch["is_sequence_start"][0]):
            state = None
        events = batch["events"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            warped, state = _detector_features(
                student,
                events,
                batch["sampling_grids"],
                feature=snapshot.source.feature,
                state=state,
            )
            if index in selected:
                output = detector(warped)[0]
        if index in selected:
            output = {key: value.detach().cpu() for key, value in output.items()}
            keep = output["scores"] >= args.score_threshold
            predictions.append({key: value[keep] for key, value in output.items()})
            targets.append(
                {key: value.detach().cpu() for key, value in batch["targets"][0].items()}
            )
            keys.append((batch["sequence_names"][0], int(batch["timestamps"][0])))
            evaluated.append(bool(batch["evaluate"][0]))
            features.append(warped[0].detach().to(device="cpu", dtype=torch.float16))
        progress.update(1)
    progress.close()
    del detector, student, dataset, loader, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(keys) != len(selected_indices):
        raise RuntimeError(
            f"{snapshot.source.label} produced {len(keys)} of {len(selected_indices)} frames"
        )
    return InferenceResult(
        source=snapshot.source,
        keys=keys,
        predictions=predictions,
        targets=targets,
        evaluated=evaluated,
        features=features,
        epoch=snapshot.epoch,
        best_map=snapshot.best_map,
    )


def _fit_shared_pca(
    results: list[InferenceResult], tokens_per_frame: int, max_samples: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    channels = {int(feature.shape[0]) for result in results for feature in result.features}
    if len(channels) != 1:
        raise ValueError("Shared PCA requires equal feature dimensions")
    samples = [
        _sample_feature_tokens(feature.float(), tokens_per_frame)
        for result in results
        for feature in result.features
    ]
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


def _targets_equal(first: list[dict[str, Tensor]], second: list[dict[str, Tensor]]) -> bool:
    return len(first) == len(second) and all(
        torch.equal(left["boxes"], right["boxes"])
        and torch.equal(left["labels"], right["labels"])
        for left, right in zip(first, second)
    )


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
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
    if len({source.label for source in sources}) != len(sources):
        raise ValueError("source labels must be unique")
    split = load_dsec_detection_split(args.split_manifest)
    if args.sequence not in split.sequences(args.role):
        raise ValueError(f"{args.sequence!r} is not in official {args.role} split")

    # Snapshot all model/head weights before inference starts. Later atomic updates of
    # best.pt by an active trainer cannot mix epochs inside this visualization run.
    snapshots = [_snapshot(source) for source in sources]
    reference_dataset = _dataset(snapshots[0], args)
    selected_indices = _selected_indices(reference_dataset, args)
    del reference_dataset
    device = torch.device(args.device)
    results = [
        _infer(snapshot, selected_indices, args, device) for snapshot in snapshots
    ]
    first = results[0]
    for result in results[1:]:
        if result.keys != first.keys:
            raise ValueError(f"{result.source.label} frame order differs")
        if result.evaluated != first.evaluated:
            raise ValueError(f"{result.source.label} evaluation mask differs")
        if not _targets_equal(first.targets, result.targets):
            raise ValueError(f"{result.source.label} targets differ")

    mean, basis, lower, upper = _fit_shared_pca(
        results, args.pca_tokens_per_frame, args.pca_max_samples
    )
    first_config = OmegaConf.create(snapshots[0].reference_config)
    input_size = (
        int(_value(first_config, "dataset.input_height", 448)),
        int(_value(first_config, "dataset.input_width", 640)),
    )
    event_grid, event_transform = _event_sampling_grid(
        args.dataset_root.expanduser().resolve(),
        args.sequence,
        {"source_input_size": list(input_size), "detection_scale": 1},
    )
    panel_width = args.panel_width
    panel_height = round(panel_width * 430 / 640)
    gap = 6
    header_height = 38
    panel_full_height = panel_height + 54
    columns = len(results) + 1
    canvas_width = columns * panel_width + (columns - 1) * gap
    canvas_height = header_height + 2 * panel_full_height + gap
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    title_font = _font(18)
    small_font = _font(13)
    box_font = _font(13)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for frame_index, ((_, timestamp), target, evaluated) in enumerate(
        zip(first.keys, first.targets, first.evaluated)
    ):
        row: dict[str, Any] = {
            "frame": frame_index,
            "timestamp": timestamp,
            "evaluated": evaluated,
            "gt_count": len(target["boxes"]) if evaluated else "",
        }
        for result in results:
            prediction = result.predictions[frame_index]
            row[f"{result.source.label}_prediction_count"] = len(prediction["boxes"])
            row[f"{result.source.label}_max_score"] = (
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
                "visualization": "end_to_end_finetune",
                "sequence": args.sequence,
                "role": args.role,
                "frame_mode": args.frame_mode,
                "frame_count": len(first.keys),
                "evaluated_frame_count": sum(first.evaluated),
                "score_threshold": args.score_threshold,
                "fps": args.fps,
                "sources": [
                    {
                        "label": result.source.label,
                        "checkpoint": str(result.source.checkpoint),
                        "feature": result.source.feature,
                        "completed_epoch": result.epoch + 1,
                        "best_validation_mAP": result.best_map,
                    }
                    for result in results
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
        for frame_index, ((sequence, timestamp), target, evaluated) in enumerate(
            zip(first.keys, first.targets, first.evaluated)
        ):
            event_image = _event_image(
                args.event_cache_dir.expanduser().resolve(),
                sequence,
                timestamp,
                event_transform,
                event_grid,
            )
            rgb_image = _rgb_image(
                args.dataset_root.expanduser().resolve(),
                sequence,
                timestamp,
                event_transform,
                event_grid,
            )
            background = rgb_image if args.detection_background == "rgb" else event_image
            canvas = Image.new("RGB", (canvas_width, canvas_height), (235, 235, 235))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (canvas_width // 2, 7),
                f"{sequence} | fine-tuned end-to-end | frame {frame_index + 1}/"
                f"{len(first.keys)} | timestamp {timestamp} | "
                f"GT {'available' if evaluated else 'unavailable'} | "
                "GT dashed white; car orange; pedestrian cyan",
                fill="black",
                font=small_font,
                anchor="ma",
            )
            canvas.paste(
                _panel(
                    rgb_image,
                    "Aligned RGB input",
                    "context only; detector input is events",
                    width=panel_width,
                    height=panel_height,
                    font=title_font,
                    small_font=small_font,
                ),
                (0, header_height),
            )
            canvas.paste(
                _panel(
                    event_image,
                    "GEP 3-channel event input",
                    "same causal event window used by EventState",
                    width=panel_width,
                    height=panel_height,
                    font=title_font,
                    small_font=small_font,
                ),
                (0, header_height + panel_full_height + gap),
            )
            for column, result in enumerate(results):
                prediction = result.predictions[frame_index]
                detection = _detection_image(background, target, prediction, box_font)
                subtitle = (
                    f"{result.source.feature} | epoch={result.epoch + 1} | "
                    f"best val mAP={result.best_map:.4f} | "
                    f"predictions={len(prediction['boxes'])}"
                )
                x = (column + 1) * (panel_width + gap)
                canvas.paste(
                    _panel(
                        detection,
                        f"{result.source.label} fine-tuned detection",
                        subtitle,
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                    (x, header_height),
                )
                pca = _pca_image(
                    result.features[frame_index].float(),
                    mean,
                    basis,
                    lower,
                    upper,
                    (panel_width, panel_height),
                )
                canvas.paste(
                    _panel(
                        pca,
                        f"{result.source.label} {result.source.feature} feature",
                        "fine-tuned; shared normalized PCA across models and frames",
                        width=panel_width,
                        height=panel_height,
                        font=title_font,
                        small_font=small_font,
                    ),
                    (x, header_height + panel_full_height + gap),
                )
            writer.send(np.asarray(canvas, dtype=np.uint8))
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == len(first.keys):
                print(f"Rendered {frame_index + 1}/{len(first.keys)} frames", flush=True)
    finally:
        writer.close()
    print(f"Fine-tuned video: {output}")
    print(f"Per-frame summary: {output.with_suffix('.csv')}")
    print(f"Run metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
