#!/usr/bin/env python3
"""Export publication-ready DSEC inputs, features, and detections as PNGs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from event_state.data.transforms import PairedSequenceTransform
from event_state.detection import load_dsec_detection_split
from event_state.detection.data import DAGR_CLASSES

from visualize_dsec_detection_sequence import (
    DEFAULT_SPLIT,
    _dataset,
    _detection_image,
    _event_image,
    _event_sampling_grid,
    _feature_map,
    _fit_shared_pca,
    _font,
    _infer,
    _parse_source,
    _pca_image,
    _rgb_image,
    _sample_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one DSEC-Detection frame as separate lossless PNG files: "
            "aligned RGB, event input, PCA feature maps, and detections."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL:CHECKPOINT:CACHE:FEATURE",
        help="Repeat for each detector, for example HYBRID:/path/best.pt:/cache:h",
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
        default="evaluated",
        help="Frame numbering follows all cached frames or only evaluated frames",
    )
    frame = parser.add_mutually_exclusive_group(required=True)
    frame.add_argument(
        "--frame-number",
        type=int,
        help="One-based frame number shown by the sequence visualizer",
    )
    frame.add_argument("--timestamp", type=int, help="Exact cached frame timestamp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--detection-background",
        choices=("rgb", "event"),
        default="rgb",
    )
    parser.add_argument(
        "--keep-padding",
        action="store_true",
        help=(
            "Keep invalid rectification borders. By default every exported image "
            "is cropped to the common valid sampling-grid bounding box."
        ),
    )
    parser.add_argument("--pca-tokens-per-frame", type=int, default=8)
    parser.add_argument("--pca-max-samples", type=int, default=50_000)
    return parser.parse_args()


def _selected_index(dataset: Any, args: argparse.Namespace) -> int:
    if args.frame_number is not None:
        index = args.frame_number - 1
        if index < 0 or index >= len(dataset):
            raise ValueError(f"frame-number must be in [1, {len(dataset)}]")
        return index
    matches = [
        index
        for index, sample in enumerate(dataset.samples)
        if int(sample[1]) == args.timestamp
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one frame with timestamp {args.timestamp}; found {len(matches)}"
        )
    return matches[0]


def _safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return result or "model"


def _prediction_payload(prediction: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for box, score, label in zip(
        prediction["boxes"], prediction["scores"], prediction["labels"]
    ):
        class_index = int(label)
        rows.append(
            {
                "box_xyxy": [float(value) for value in box.tolist()],
                "score": float(score),
                "class_index": class_index,
                "class_name": DAGR_CLASSES[class_index],
            }
        )
    return rows


def _valid_crop_box(grid: torch.Tensor) -> tuple[int, int, int, int]:
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"Expected sampling grid [H,W,2], got {tuple(grid.shape)}")
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )
    coordinates = valid.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("Rectification grid contains no valid output pixels")
    top = int(coordinates[:, 0].min())
    bottom = int(coordinates[:, 0].max()) + 1
    left = int(coordinates[:, 1].min())
    right = int(coordinates[:, 1].max()) + 1
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0, 1]")
    if args.pca_tokens_per_frame <= 0 or args.pca_max_samples < 3:
        raise ValueError("PCA sampling values must be positive")

    sources = [_parse_source(value) for value in args.source]
    if len({source.label for source in sources}) != len(sources):
        raise ValueError("source labels must be unique")
    split = load_dsec_detection_split(args.split_manifest)
    if args.sequence not in split.sequences(args.role):
        raise ValueError(f"{args.sequence!r} is not in official {args.role} split")

    datasets = [_dataset(source, args) for source in sources]
    selected = _selected_index(datasets[0], args)
    selected_key = _sample_keys(datasets[0], [selected])[0]
    full_keys = _sample_keys(datasets[0], list(range(len(datasets[0]))))
    for source, dataset in zip(sources[1:], datasets[1:]):
        if len(dataset) != len(datasets[0]):
            raise ValueError(f"{source.label} frame count differs")
        if dataset.input_size != datasets[0].input_size:
            raise ValueError(f"{source.label} input geometry differs")
        if _sample_keys(dataset, [selected])[0] != selected_key:
            raise ValueError(f"{source.label} selected frame differs")

    # Fit PCA over the complete sequence, even though inference is performed for
    # only one frame. This makes the exported feature colors stable and avoids a
    # visually flattering per-frame projection.
    mean, basis, lower, upper = _fit_shared_pca(
        sources,
        full_keys,
        args.pca_tokens_per_frame,
        args.pca_max_samples,
    )

    device = torch.device(args.device)
    predictions: list[dict[str, torch.Tensor]] = []
    target: dict[str, torch.Tensor] | None = None
    for source, dataset in zip(sources, datasets):
        current_predictions, current_targets = _infer(
            source, dataset, [selected], args, device
        )
        predictions.append(current_predictions[0])
        if target is None:
            target = current_targets[0]
        elif not torch.equal(target["boxes"], current_targets[0]["boxes"]) or not torch.equal(
            target["labels"], current_targets[0]["labels"]
        ):
            raise ValueError(f"{source.label} target differs from the first source")
    if target is None:
        raise RuntimeError("No detection target loaded")

    sequence, timestamp = selected_key
    metadata_path = sources[0].feature_cache_dir / sequence / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    event_grid, event_transform = _event_sampling_grid(
        args.dataset_root.expanduser().resolve(), sequence, metadata
    )
    if not isinstance(event_transform, PairedSequenceTransform):
        raise RuntimeError("Unexpected event transform")
    rgb = _rgb_image(
        args.dataset_root.expanduser().resolve(),
        sequence,
        timestamp,
        event_transform,
        event_grid,
    )
    event = _event_image(
        args.event_cache_dir.expanduser().resolve(),
        sequence,
        timestamp,
        event_transform,
        event_grid,
    )
    background = rgb if args.detection_background == "rgb" else event
    crop_box = None if args.keep_padding else _valid_crop_box(event_grid)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_rgb = rgb if crop_box is None else rgb.crop(crop_box)
    exported_event = event if crop_box is None else event.crop(crop_box)
    exported_rgb.save(output_dir / "rgb.png")
    exported_event.save(output_dir / "event.png")

    input_size = datasets[0].input_size
    if input_size is None:
        raise RuntimeError("Detection input geometry was not initialized")
    input_height, input_width = input_size
    box_font = _font(max(13, round(input_width / 49)))
    source_metadata: list[dict[str, Any]] = []
    for source, prediction in zip(sources, predictions):
        label = _safe_label(source.label)
        feature = _feature_map(source, sequence, timestamp)
        feature_image = _pca_image(
            feature,
            mean,
            basis,
            lower,
            upper,
            (input_width, input_height),
        )
        feature_name = f"{label}_{source.feature}_feature_pca.png"
        detection_name = f"{label}_detection.png"
        detection_image = _detection_image(background, target, prediction, box_font)
        if crop_box is not None:
            feature_image = feature_image.crop(crop_box)
            detection_image = detection_image.crop(crop_box)
        feature_image.save(output_dir / feature_name)
        detection_image.save(output_dir / detection_name)
        source_metadata.append(
            {
                "label": source.label,
                "checkpoint": str(source.checkpoint),
                "feature_cache_dir": str(source.feature_cache_dir),
                "feature": source.feature,
                "feature_image": feature_name,
                "detection_image": detection_name,
                "predictions": _prediction_payload(prediction),
            }
        )

    evaluated = bool(datasets[0].samples[selected][4])
    manifest = {
        "sequence": sequence,
        "frame_mode": args.frame_mode,
        "frame_number": selected + 1,
        "frame_count": len(datasets[0]),
        "timestamp": timestamp,
        "evaluated": evaluated,
        "score_threshold": args.score_threshold,
        "detection_background": args.detection_background,
        "crop_box_xyxy": list(crop_box) if crop_box is not None else None,
        "padding_kept": args.keep_padding,
        "pca_scope": "complete sequence, shared across sources",
        "rgb_image": "rgb.png",
        "event_image": "event.png",
        "ground_truth": [
            {
                "box_xyxy": [float(value) for value in box.tolist()],
                "class_index": int(label),
                "class_name": DAGR_CLASSES[int(label)],
            }
            for box, label in zip(target["boxes"], target["labels"])
        ],
        "sources": source_metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {sequence} frame {selected + 1}/{len(datasets[0])} "
        f"timestamp={timestamp} to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
