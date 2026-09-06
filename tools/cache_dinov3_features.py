#!/usr/bin/env python3
"""Cache frozen DINOv3 patch tokens for aligned DSEC RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_state.data.cache_metadata import (  # noqa: E402
    ALIGNMENT_FORMAT_VERSION,
    SUCCESS_MARKER_NAME,
    TEACHER_CACHE_FORMAT_VERSION,
    build_input_fingerprint,
    canonical_json_sha256,
    dinov3_repository_identity,
    frame_manifest_sha256,
    reject_untracked_outputs,
    relative_file_id,
    sha256_file,
    success_marker_payload,
    validate_input_fingerprint,
    validate_success_marker,
)
from event_state.data.dsec import discover_dsec_sequences  # noqa: E402
from event_state.data.transforms import PairedSequenceTransform  # noqa: E402


CACHE_FORMAT_VERSION = TEACHER_CACHE_FORMAT_VERSION


@dataclass(frozen=True)
class CacheJob:
    split: str
    sequence_name: str
    frame_index: int
    timestamp: int
    image_path: Path
    source_image: str
    output_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache deterministic DINOv3 patch tokens for DSEC RGB frames."
    )
    parser.add_argument("--root", type=Path, required=True, help="DSEC dataset root")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Explicit sequence names; defaults to every sequence in the split",
    )
    parser.add_argument(
        "--image-directory",
        default="aligned_event",
        help="Prepared RGB directory under <split>_images/<sequence>/images/left",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/dinov3_vits16"),
        help="Cache root; outputs use <sequence>/<timestamp>.pt",
    )
    parser.add_argument("--model", default="dinov3_vits16")
    parser.add_argument("--backend", choices=("torch_hub", "package"), default="torch_hub")
    parser.add_argument(
        "--repository",
        default="facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa",
    )
    parser.add_argument("--source", choices=("github", "local"), default="github")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional local checkpoint path or URL passed as DINOv3's `weights` argument",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained weights (enabled by default)",
    )
    parser.add_argument("--input-height", type=int, default=448)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--image-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--image-std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically divide sorted sequences into this many disjoint shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to process; use one process per GPU",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing feature files; existing files are otherwise preserved",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    root = args.root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    all_sequences = discover_dsec_sequences(root, args.split, args.sequences)
    sequences = all_sequences[args.shard_index :: args.num_shards]
    print(
        f"DINOv3 cache shard {args.shard_index}/{args.num_shards}: "
        f"{len(sequences)}/{len(all_sequences)} {args.split} sequences"
    )
    if not sequences:
        print("No sequences assigned to this shard.")
        return
    checkpoint = normalize_checkpoint(args.checkpoint)
    model_metadata = build_model_metadata(args, checkpoint)
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
        "height": args.input_height // args.patch_size,
        "width": args.input_width // args.patch_size,
        "num_patches": (args.input_height // args.patch_size)
        * (args.input_width // args.patch_size),
        "patch_size": args.patch_size,
    }

    jobs: list[CacheJob] = []
    sequence_manifests: dict[str, dict[str, Any]] = {}
    sequence_outputs: dict[str, Path] = {}
    sequence_cache_files: dict[str, list[Path]] = {}
    existing_count = 0
    for sequence_name in sequences:
        frames = load_aligned_frames(root, args.split, sequence_name, args.image_directory)
        sequence_output = output_root / sequence_name
        sequence_output.mkdir(parents=True, exist_ok=True)
        timestamp_path = (
            root / f"{args.split}_images" / sequence_name / "images" / "timestamps.txt"
        )
        alignment_metadata_path = frames[0][0].parent / "metadata.json"
        input_fingerprint = build_input_fingerprint(
            root,
            files={
                "rgb_timestamps": timestamp_path,
                "alignment_metadata": alignment_metadata_path,
            },
            file_sets={"aligned_rgb_frames": [path for path, _ in frames]},
        )
        timestamp_manifest = frame_manifest_sha256(
            (
                frame_index,
                frames[frame_index - 1][1] if frame_index else timestamp,
                timestamp,
            )
            for frame_index, (_, timestamp) in enumerate(frames)
        )
        sequence_metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "dataset": "DSEC",
            "split": args.split,
            "sequence_name": sequence_name,
            "source_image_directory": relative_file_id(root, frames[0][0].parent),
            "frame_count": len(frames),
            "timestamp_manifest_sha256": timestamp_manifest,
            "input_fingerprint": input_fingerprint,
            "input": input_metadata,
            "grid": grid_metadata,
            "model": model_metadata,
            "cache_dtype": args.cache_dtype,
        }
        sequence_manifests[sequence_name] = sequence_metadata
        sequence_outputs[sequence_name] = sequence_output
        sequence_cache_files[sequence_name] = [
            sequence_output / f"{timestamp}.pt" for _, timestamp in frames
        ]
        marker_path = sequence_output / SUCCESS_MARKER_NAME
        if args.overwrite:
            marker_path.unlink(missing_ok=True)
        metadata_path = sequence_output / "metadata.json"
        reject_untracked_outputs(
            metadata_path,
            sequence_cache_files[sequence_name],
            overwrite=args.overwrite,
            description=f"Teacher cache for {sequence_name}",
        )
        ensure_json_metadata(
            metadata_path,
            sequence_metadata,
            overwrite=args.overwrite,
        )
        for frame_index, (image_path, timestamp) in enumerate(frames):
            output_path = sequence_output / f"{timestamp}.pt"
            job = CacheJob(
                split=args.split,
                sequence_name=sequence_name,
                frame_index=frame_index,
                timestamp=timestamp,
                image_path=image_path,
                source_image=relative_file_id(root, image_path),
                output_path=output_path,
            )
            if output_path.exists() and not args.overwrite:
                validate_teacher_cache_payload(
                    safe_torch_load(output_path),
                    job=job,
                    manifest=sequence_metadata,
                )
                existing_count += 1
                continue
            jobs.append(job)

    if not jobs:
        write_completion_markers(
            sequence_manifests,
            sequence_outputs,
            sequence_cache_files,
            overwrite=args.overwrite,
        )
        print(f"DINOv3 cache is already complete ({existing_count} existing files).")
        return

    device = resolve_device(args.device)
    teacher = load_teacher(args, checkpoint)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.to(device)
    transform = PairedSequenceTransform(
        height=args.input_height,
        width=args.input_width,
        training=False,
    )
    cache_dtype = getattr(torch, args.cache_dtype)
    expected_tokens = int(grid_metadata["num_patches"])

    progress = tqdm(total=len(jobs), desc="DINOv3 features", unit="frame")
    try:
        for start in range(0, len(jobs), args.batch_size):
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
                patch_tokens = extract_patch_tokens(teacher, images)
            if patch_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINOv3 returned {patch_tokens.shape[1]} patches; expected {expected_tokens} "
                    f"for grid {grid_metadata['height']}x{grid_metadata['width']}"
                )
            if patch_tokens.shape[2] != args.embedding_dim:
                raise ValueError(
                    f"DINOv3 returned embedding dim {patch_tokens.shape[2]}; "
                    f"expected {args.embedding_dim}"
                )

            cached_batch = patch_tokens.detach().to(device="cpu", dtype=cache_dtype)
            for job, tokens in zip(batch_jobs, cached_batch):
                manifest = sequence_manifests[job.sequence_name]
                payload = {
                    "format_version": CACHE_FORMAT_VERSION,
                    "dataset": "DSEC",
                    "split": job.split,
                    "sequence_name": job.sequence_name,
                    "frame_index": job.frame_index,
                    "timestamp": job.timestamp,
                    "source_image": job.source_image,
                    "input_fingerprint_digest": manifest["input_fingerprint"]["digest"],
                    "manifest_digest": canonical_json_sha256(manifest),
                    "patch_tokens": tokens.contiguous(),
                    "grid_size": [grid_metadata["height"], grid_metadata["width"]],
                    "input_size": [args.input_height, args.input_width],
                    "model_identifier": model_metadata["identifier"],
                    "checkpoint_identifier": model_metadata["checkpoint"],
                    "cache_dtype": args.cache_dtype,
                }
                atomic_torch_save(job.output_path, payload, overwrite=args.overwrite)
                progress.update(1)
    finally:
        progress.close()
    write_completion_markers(
        sequence_manifests,
        sequence_outputs,
        sequence_cache_files,
        overwrite=args.overwrite,
    )
    print(f"Cached {len(jobs)} DINOv3 features; preserved {existing_count} existing files.")


def validate_args(args: argparse.Namespace) -> None:
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("input height and width must be positive")
    if args.patch_size <= 0:
        raise ValueError("patch size must be positive")
    if args.input_height % args.patch_size or args.input_width % args.patch_size:
        raise ValueError("DINOv3 input dimensions must be divisible by patch size")
    if args.embedding_dim <= 0:
        raise ValueError("embedding dim must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_shards <= 0:
        raise ValueError("number of shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if any(value <= 0 for value in args.image_std):
        raise ValueError("image standard deviations must be positive")
    if args.checkpoint is not None and not args.pretrained:
        raise ValueError("--checkpoint requires pretrained weights; omit --no-pretrained")
    image_directory = Path(args.image_directory)
    if (
        image_directory.is_absolute()
        or len(image_directory.parts) != 1
        or image_directory.parts[0] in {".", ".."}
    ):
        raise ValueError("image directory must be one relative directory name")


def normalize_checkpoint(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    parsed = urlparse(checkpoint)
    if parsed.scheme == "file":
        candidate = Path(unquote(parsed.path)).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint file URI does not exist: {checkpoint}")
    elif parsed.scheme:
        return checkpoint
    else:
        candidate = Path(checkpoint).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
        reject_reference_repo_path(resolved, "checkpoint")
        return str(resolved)
    return checkpoint


def build_model_metadata(args: argparse.Namespace, checkpoint: str | None) -> dict[str, Any]:
    if args.backend == "torch_hub" and args.source == "local":
        repository_path = Path(args.repository).expanduser().resolve()
        reject_reference_repo_path(repository_path, "DINOv3 repository")
        if not repository_path.is_dir():
            raise FileNotFoundError(f"Local DINOv3 repository not found: {repository_path}")
        repository = repository_path.name
    elif args.backend == "package":
        repository = "dinov3"
    else:
        repository = args.repository
    repository_identity = dinov3_repository_identity(
        backend=args.backend,
        source=args.source,
        repository=args.repository,
    )
    identifier = (
        f"{args.backend}:{canonical_json_sha256(repository_identity)}:{args.model}"
    )
    return {
        "identifier": identifier,
        "name": args.model,
        "backend": args.backend,
        "repository": repository,
        "repository_identity": repository_identity,
        "source": args.source if args.backend == "torch_hub" else "installed_package",
        "pretrained": bool(args.pretrained),
        "frozen": True,
        "feature": "x_norm_patchtokens",
        "embedding_dim": args.embedding_dim,
        "checkpoint": checkpoint_identity(checkpoint, pretrained=args.pretrained),
    }


def checkpoint_identity(checkpoint: str | None, *, pretrained: bool) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "kind": "model_default" if pretrained else "random_initialization",
            "value": "official_default_pretrained_weights" if pretrained else None,
            "sha256": None,
        }
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return {
            "kind": "local_file",
            "value": checkpoint_path.name,
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        }
    return {
        "kind": "url_or_identifier",
        "value": checkpoint,
        "sha256": hashlib.sha256(checkpoint.encode("utf-8")).hexdigest(),
    }


def load_teacher(args: argparse.Namespace, checkpoint: str | None) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": args.pretrained}
    if checkpoint is not None:
        kwargs["weights"] = checkpoint

    if args.backend == "package":
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except ImportError as error:
            raise RuntimeError(
                "The installed `dinov3` package is unavailable; install it or use torch_hub"
            ) from error
        package_path = getattr(backbones, "__file__", None)
        if package_path is None:
            raise RuntimeError("The installed `dinov3` package has no source path")
        reject_reference_repo_path(
            Path(package_path).expanduser().resolve(),
            "Installed DINOv3 package",
        )
        try:
            factory = getattr(backbones, args.model)
        except AttributeError as error:
            raise ValueError(f"Unknown DINOv3 package model: {args.model}") from error
        model = factory(**kwargs)
    else:
        repository = args.repository
        if args.source == "local":
            repository = str(Path(repository).expanduser().resolve())
        if args.source == "github":
            kwargs["skip_validation"] = True
        model = torch.hub.load(
            repo_or_dir=repository,
            model=args.model,
            source=args.source,
            trust_repo=True,
            **kwargs,
        )
    if not isinstance(model, nn.Module):
        raise TypeError(f"DINOv3 loader returned {type(model).__name__}, expected torch.nn.Module")
    if not callable(getattr(model, "forward_features", None)):
        raise TypeError("DINOv3 model does not expose forward_features")
    return model


def extract_patch_tokens(model: nn.Module, images: Tensor) -> Tensor:
    features = model.forward_features(images)  # type: ignore[attr-defined]
    if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
        raise KeyError("DINOv3 forward_features did not return x_norm_patchtokens")
    tokens = features["x_norm_patchtokens"]
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("DINOv3 patch tokens must have shape [B, N, D]")
    return tokens


def load_teacher_image(
    path: Path,
    *,
    transform: PairedSequenceTransform,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    _, sequence = transform(None, tensor.unsqueeze(0))
    if sequence is None:
        raise RuntimeError("Deterministic RGB transform unexpectedly returned no image")
    tensor = sequence.squeeze(0)
    mean_tensor = tensor.new_tensor(mean).view(3, 1, 1)
    std_tensor = tensor.new_tensor(std).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def load_aligned_frames(
    root: Path,
    split: str,
    sequence_name: str,
    image_directory: str,
) -> list[tuple[Path, int]]:
    image_root = root / f"{split}_images" / sequence_name / "images"
    timestamp_path = image_root / "timestamps.txt"
    aligned_dir = image_root / "left" / image_directory
    if not timestamp_path.is_file():
        raise FileNotFoundError(f"DSEC RGB timestamp file not found: {timestamp_path}")
    if not aligned_dir.is_dir():
        raise FileNotFoundError(
            f"Aligned DSEC RGB directory not found: {aligned_dir}. Run tools/prepare_dsec.py first."
        )

    timestamps = np.atleast_1d(np.loadtxt(timestamp_path, dtype=np.int64))
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"DSEC RGB timestamps must be a non-empty 1-D array: {timestamp_path}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"DSEC RGB timestamps are not strictly increasing: {timestamp_path}")

    by_stem: dict[str, Path] = {}
    for path in aligned_dir.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if path.stem in by_stem:
            raise ValueError(f"Duplicate aligned image stem {path.stem!r} in {aligned_dir}")
        by_stem[path.stem] = path
    missing = [int(timestamp) for timestamp in timestamps if str(int(timestamp)) not in by_stem]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} aligned RGB frames in {aligned_dir}; "
            f"first timestamps: {preview}"
        )
    frames = [(by_stem[str(int(timestamp))], int(timestamp)) for timestamp in timestamps]
    alignment_metadata_path = aligned_dir / "metadata.json"
    if not alignment_metadata_path.is_file():
        raise FileNotFoundError(
            f"Aligned RGB metadata not found: {alignment_metadata_path}. "
            "Regenerate it with tools/prepare_dsec.py."
        )
    try:
        alignment_metadata = json.loads(alignment_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid aligned RGB metadata: {alignment_metadata_path}") from error
    expected_alignment = {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": split,
        "sequence_name": sequence_name,
        "frame_count": len(frames),
        "timestamp_manifest_sha256": frame_manifest_sha256(
            (
                frame_index,
                frames[frame_index - 1][1] if frame_index else timestamp,
                timestamp,
            )
            for frame_index, (_, timestamp) in enumerate(frames)
        ),
    }
    for field, expected in expected_alignment.items():
        if alignment_metadata.get(field) != expected:
            raise ValueError(
                f"Aligned RGB {field}={alignment_metadata.get(field)!r} does not match "
                f"{expected!r}: {alignment_metadata_path}"
            )
    validate_input_fingerprint(
        alignment_metadata.get("input_fingerprint"),
        required_file_roles={"rgb_timestamps", "camera_calibration"},
        required_file_set_roles={"rectified_rgb_frames"},
    )
    source_directory_id = alignment_metadata.get("source_image_directory")
    if (
        not isinstance(source_directory_id, str)
        or Path(source_directory_id).is_absolute()
        or ".." in Path(source_directory_id).parts
    ):
        raise ValueError(
            f"Aligned RGB source_image_directory must be dataset-relative: "
            f"{alignment_metadata_path}"
        )
    source_directory = root / source_directory_id
    source_images = sorted(
        path
        for path in source_directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    current_alignment_fingerprint = build_input_fingerprint(
        root,
        files={
            "rgb_timestamps": timestamp_path,
            "camera_calibration": (
                root
                / f"{split}_calibration"
                / sequence_name
                / "calibration"
                / "cam_to_cam.yaml"
            ),
        },
        file_sets={"rectified_rgb_frames": source_images},
    )
    if current_alignment_fingerprint != alignment_metadata["input_fingerprint"]:
        raise ValueError(
            f"DSEC RGB/calibration inputs changed after alignment: {alignment_metadata_path}"
        )
    validate_input_fingerprint(
        alignment_metadata.get("output_fingerprint"),
        required_file_set_roles={"aligned_rgb_frames"},
    )
    current_output_fingerprint = build_input_fingerprint(
        root,
        files={},
        file_sets={"aligned_rgb_frames": [path for path, _ in frames]},
    )
    if current_output_fingerprint != alignment_metadata["output_fingerprint"]:
        raise ValueError(
            f"Aligned RGB outputs changed after preparation: {alignment_metadata_path}"
        )
    validate_success_marker(aligned_dir, alignment_metadata)
    return frames


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reject_reference_repo_path(path: Path, description: str) -> None:
    reference_root = (PROJECT_ROOT / "reference_repo").resolve()
    try:
        path.relative_to(reference_root)
    except ValueError:
        return
    raise ValueError(
        f"{description} points inside disposable reference_repo; copy it elsewhere first: {path}"
    )


def ensure_json_metadata(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Existing metadata is unreadable; use --overwrite: {path}"
            ) from error
        if existing != payload:
            raise RuntimeError(f"Existing metadata is incompatible; use --overwrite: {path}")
        return
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes_write(path, encoded, overwrite=overwrite)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for existing environments.
        return torch.load(path, map_location="cpu")


def validate_teacher_cache_payload(
    payload: Any,
    *,
    job: CacheJob,
    manifest: dict[str, Any],
) -> Tensor:
    if not isinstance(payload, dict):
        raise TypeError(f"Teacher cache payload must be a mapping: {job.output_path}")
    expected = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": job.split,
        "sequence_name": job.sequence_name,
        "frame_index": job.frame_index,
        "timestamp": job.timestamp,
        "source_image": job.source_image,
        "input_fingerprint_digest": manifest["input_fingerprint"]["digest"],
        "manifest_digest": canonical_json_sha256(manifest),
        "grid_size": [manifest["grid"]["height"], manifest["grid"]["width"]],
        "input_size": [manifest["input"]["height"], manifest["input"]["width"]],
        "model_identifier": manifest["model"]["identifier"],
        "checkpoint_identifier": manifest["model"]["checkpoint"],
        "cache_dtype": manifest["cache_dtype"],
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(
                f"Teacher cache {field}={payload.get(field)!r} does not match "
                f"{expected_value!r}; use --overwrite: {job.output_path}"
            )
    tokens = payload.get("patch_tokens")
    if not isinstance(tokens, Tensor) or tokens.ndim != 2:
        raise ValueError(
            f"Teacher cache patch_tokens must have shape [N, D]; "
            f"use --overwrite: {job.output_path}"
        )
    expected_shape = (
        int(manifest["grid"]["num_patches"]),
        int(manifest["model"]["embedding_dim"]),
    )
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(
            f"Teacher cache shape {tuple(tokens.shape)} does not match {expected_shape}; "
            f"use --overwrite: {job.output_path}"
        )
    expected_dtype = getattr(torch, str(manifest["cache_dtype"]))
    if tokens.dtype != expected_dtype:
        raise ValueError(
            f"Teacher cache dtype {tokens.dtype} does not match {expected_dtype}; "
            f"use --overwrite: {job.output_path}"
        )
    return tokens


def write_completion_markers(
    manifests: dict[str, dict[str, Any]],
    output_directories: dict[str, Path],
    cache_files: dict[str, list[Path]],
    *,
    overwrite: bool,
) -> None:
    for sequence_name, manifest in manifests.items():
        ensure_json_metadata(
            output_directories[sequence_name] / SUCCESS_MARKER_NAME,
            success_marker_payload(
                manifest,
                output_fingerprint=build_input_fingerprint(
                    output_directories[sequence_name],
                    files={},
                    file_sets={"cache_items": cache_files[sequence_name]},
                ),
            ),
            overwrite=overwrite,
        )


def atomic_torch_save(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        commit_temporary_file(temporary_path, path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_bytes_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        commit_temporary_file(temporary_path, path, overwrite=overwrite)
    finally:
        temporary_path.unlink(missing_ok=True)


def commit_temporary_file(temporary_path: Path, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    os.replace(temporary_path, path)


if __name__ == "__main__":
    main()
