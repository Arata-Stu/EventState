#!/usr/bin/env python3
"""Cache DINOv3 patch tokens for prepared, event-aligned M3ED RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_state.data.cache_metadata import TEACHER_CACHE_FORMAT_VERSION  # noqa: E402
from event_state.data.m3ed import discover_prepared_m3ed_sequences  # noqa: E402
from event_state.data.transforms import PairedSequenceTransform  # noqa: E402
from cache_dinov3_features import (  # noqa: E402
    atomic_torch_save,
    build_model_metadata,
    extract_patch_tokens,
    load_teacher,
    load_teacher_image,
    normalize_checkpoint,
    resolve_device,
)


@dataclass(frozen=True)
class Job:
    sequence_name: str
    frame_index: int
    timestamp: int
    image_path: Path
    output_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen DINOv3 features for prepared M3ED RGB frames"
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--model", default="dinov3_vits16")
    parser.add_argument("--backend", choices=("torch_hub", "package"), default="torch_hub")
    parser.add_argument(
        "--repository",
        default="facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa",
    )
    parser.add_argument("--source", choices=("github", "local"), default="github")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Local, manually downloaded DINOv3 weight file. M3ED cache generation "
            "never falls back to gated weight download."
        ),
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--input-height", type=int, default=352)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--image-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--image-std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_sequence(prepared_root: Path, sequence_name: str) -> tuple[list[int], list[Path]]:
    sequence_root = prepared_root / sequence_name
    metadata_path = sequence_root / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid prepared M3ED metadata: {metadata_path}") from error
    if metadata.get("dataset") != "M3ED" or metadata.get("sequence_name") != sequence_name:
        raise ValueError(f"Prepared M3ED metadata mismatch: {metadata_path}")
    timestamps = [int(value) for value in metadata.get("timestamps", [])]
    paths = [sequence_root / "aligned_rgb" / f"{timestamp}.png" for timestamp in timestamps]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared M3ED RGB frame missing: {missing[0]}")
    return timestamps, paths


def _metadata_difference(
    existing: dict[str, object], expected: dict[str, object]
) -> list[str]:
    return [
        key
        for key in sorted(set(existing) | set(expected))
        if existing.get(key) != expected.get(key)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_height % args.patch_size or args.input_width % args.patch_size:
        raise ValueError("DINOv3 input dimensions must be divisible by patch size")
    if args.batch_size <= 0 or args.num_shards <= 0:
        raise ValueError("batch-size and num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    prepared_root = args.prepared_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_sequences = discover_prepared_m3ed_sequences(prepared_root, args.sequences)
    sequences = all_sequences[args.shard_index :: args.num_shards]
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Local DINOv3 checkpoint not found: {checkpoint_path}"
        )
    checkpoint = normalize_checkpoint(str(checkpoint_path))
    print(f"Using local DINOv3 checkpoint: {checkpoint_path}")
    model_metadata = build_model_metadata(args, checkpoint)
    grid_height = args.input_height // args.patch_size
    grid_width = args.input_width // args.patch_size
    input_metadata = {
        "height": args.input_height,
        "width": args.input_width,
        "geometry": "deterministic_center_crop_to_aspect_ratio_then_bilinear_resize",
        "color_space": "RGB",
        "value_range": [0.0, 1.0],
        "normalization_mean": list(args.image_mean),
        "normalization_std": list(args.image_std),
    }
    grid_metadata = {
        "height": grid_height,
        "width": grid_width,
        "num_patches": grid_height * grid_width,
        "patch_size": args.patch_size,
    }
    transform = PairedSequenceTransform(height=args.input_height, width=args.input_width)
    jobs: list[Job] = []
    for sequence_name in sequences:
        timestamps, image_paths = _load_sequence(prepared_root, sequence_name)
        sequence_output = output_root / sequence_name
        sequence_output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": TEACHER_CACHE_FORMAT_VERSION,
            "dataset": "M3ED",
            "split": "all",
            "sequence_name": sequence_name,
            "frame_count": len(timestamps),
            "input": input_metadata,
            "grid": grid_metadata,
            "model": model_metadata,
            "cache_dtype": args.cache_dtype,
        }
        metadata_path = sequence_output / "metadata.json"
        write_metadata = args.overwrite or not metadata_path.exists()
        if metadata_path.exists() and not args.overwrite:
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing != metadata:
                cached_frames = sorted(sequence_output.glob("[0-9]*.pt"))
                if cached_frames:
                    differences = ", ".join(_metadata_difference(existing, metadata))
                    raise RuntimeError(
                        f"Incompatible teacher cache metadata: {metadata_path}; "
                        f"different fields: {differences}. Existing feature files are "
                        "preserved. Use a new --output-dir, or use --overwrite only if "
                        "replacing this cache is intentional."
                    )
                print(f"Replacing stale metadata in empty cache: {metadata_path}")
                write_metadata = True
        if write_metadata:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for index, (timestamp, image_path) in enumerate(zip(timestamps, image_paths)):
            output_path = sequence_output / f"{timestamp}.pt"
            if output_path.exists() and not args.overwrite:
                continue
            jobs.append(Job(sequence_name, index, timestamp, image_path, output_path))

    if not jobs:
        print("M3ED DINOv3 cache is already complete")
        return
    device = resolve_device(args.device)
    teacher = load_teacher(args, checkpoint).to(device).eval()
    cache_dtype = getattr(torch, args.cache_dtype)
    for start in tqdm(range(0, len(jobs), args.batch_size), desc="M3ED DINOv3", unit="batch"):
        batch_jobs = jobs[start : start + args.batch_size]
        images = torch.stack(
            [
                load_teacher_image(
                    job.image_path,
                    transform=transform,
                    mean=args.image_mean,
                    std=args.image_std,
                )
                for job in batch_jobs
            ]
        ).to(device)
        with torch.inference_mode():
            tokens: Tensor = extract_patch_tokens(teacher, images)
        if tuple(tokens.shape[1:]) != (grid_height * grid_width, args.embedding_dim):
            raise ValueError(f"Unexpected DINOv3 token shape: {tuple(tokens.shape)}")
        for job, value in zip(batch_jobs, tokens.detach().cpu().to(cache_dtype)):
            atomic_torch_save(
                job.output_path,
                {
                    "format_version": TEACHER_CACHE_FORMAT_VERSION,
                    "dataset": "M3ED",
                    "split": "all",
                    "sequence_name": job.sequence_name,
                    "frame_index": job.frame_index,
                    "timestamp": job.timestamp,
                    "source_image": str(job.image_path.relative_to(prepared_root)),
                    "grid_size": [grid_height, grid_width],
                    "input_size": [args.input_height, args.input_width],
                    "model_identifier": model_metadata["identifier"],
                    "checkpoint_identifier": model_metadata["checkpoint"],
                    "cache_dtype": args.cache_dtype,
                    "patch_tokens": value.contiguous(),
                },
                overwrite=args.overwrite,
            )
    print(f"Cached {len(jobs)} M3ED DINOv3 frames in {output_root}")


if __name__ == "__main__":
    main()
