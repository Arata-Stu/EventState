#!/usr/bin/env python3
"""Cache sparse frozen EventState maps at DSEC-Semantic labeled frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from event_state.segmentation import find_semantic_label_dir, load_dsec_semantic_split
from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata


FORMAT_VERSION = 2
DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_semantic_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache EventState z/h only at DSEC-Semantic labeled frames"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--features", nargs="+", choices=("z", "h"), default=("h",))
    parser.add_argument("--state-policy", choices=("continuous", "frame"), default="continuous")
    parser.add_argument("--num-classes", type=int, choices=(11, 19), default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _semantic_transform(events: torch.Tensor, config: Any) -> torch.Tensor:
    """Crop to DSEC-Semantic geometry and pad without changing coordinates."""

    if events.ndim != 3 or events.shape[-2:] != (480, 640):
        raise ValueError(f"Expected one [C, 480, 640] DSEC event frame, got {events.shape}")
    height = int(OmegaConf.select(config, "dataset.input_height"))
    width = int(OmegaConf.select(config, "dataset.input_width"))
    means = OmegaConf.select(config, "dataset.representation.normalize_mean")
    stds = OmegaConf.select(config, "dataset.representation.normalize_std")
    if means is None or stds is None:
        raise ValueError("Checkpoint does not contain event normalization")
    value = events.float()[..., :440, :].unsqueeze(0)
    source_height, source_width = value.shape[-2:]
    if height < source_height or width < source_width:
        raise ValueError(
            "Checkpoint input is smaller than DSEC-Semantic's 440x640 geometry; "
            "resizing would invalidate pixel alignment"
        )
    if (height, width) != (source_height, source_width):
        # Preserve all native coordinates and put patch-alignment padding only
        # outside the labeled field of view, at the bottom and right.
        value = F.pad(value, (0, width - source_width, 0, height - source_height))
    mean = value.new_tensor([float(item) for item in means]).view(1, -1, 1, 1)
    std = value.new_tensor([float(item) for item in stds]).view(1, -1, 1, 1)
    if value.shape[1] != mean.shape[1]:
        raise ValueError("Event cache channels do not match checkpoint normalization")
    return ((value - mean) / std).contiguous()


def _complete(destination: Path, expected: dict[str, Any], frame_indices: set[int]) -> bool:
    path = destination / "metadata.json"
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    keys = (
        "format_version",
        "task",
        "sequence_name",
        "official_role",
        "checkpoint_sha256",
        "checkpoint_step",
        "state_policy",
        "features",
        "num_classes",
        "label_size",
        "input_size",
        "input_transform",
        "source_event_cache_metadata",
        "labeled_frame_indices",
    )
    if any(actual.get(key) != expected.get(key) for key in keys):
        return False
    cached = set(actual.get("cached_frame_indices", []))
    missing = set(actual.get("missing_labeled_frame_indices", []))
    return (
        not (cached & missing)
        and cached | missing == frame_indices
        and int(actual.get("cached_frame_count", -1)) == len(cached)
        and all(
        (destination / f"{index:06d}.pt").is_file() for index in cached
        )
    )


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    checkpoint = args.checkpoint.expanduser().resolve()
    event_cache = args.event_cache_dir.expanduser().resolve()
    labels_root = args.labels_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    split = load_dsec_semantic_split(args.split_manifest)
    sequences = split.sequences(args.role)[args.shard_index :: args.num_shards]
    if not sequences:
        raise ValueError("Selected DSEC-Semantic feature shard is empty")

    metadata = load_checkpoint_config_metadata(checkpoint)
    config = OmegaConf.create(metadata.config)
    if args.teacher_checkpoint is not None:
        teacher_checkpoint = str(args.teacher_checkpoint.expanduser().resolve())
        OmegaConf.update(config, "teacher.checkpoint", teacher_checkpoint, merge=False)
        OmegaConf.update(
            config,
            "model.event_encoder.checkpoint",
            teacher_checkpoint,
            merge=False,
            force_add=True,
        )
    device = torch.device(args.device)
    model = build_model(config, device=device)
    state = load_checkpoint(checkpoint, model=model, device=device, restore_rng=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = _sha256(checkpoint)
    requested = set(args.features)
    input_size = [
        int(OmegaConf.select(config, "dataset.input_height")),
        int(OmegaConf.select(config, "dataset.input_width")),
    ]

    with torch.inference_mode():
        for sequence in sequences:
            label_dir = find_semantic_label_dir(
                labels_root,
                role=args.role,
                sequence=sequence,
                num_classes=args.num_classes,
            )
            labeled_indices = {
                int(path.stem)
                for path in label_dir.glob("*.png")
                if path.stem.isdecimal()
            }
            source_dir = event_cache / sequence
            source_metadata_path = source_dir / "metadata.json"
            if not source_metadata_path.is_file():
                raise FileNotFoundError(f"Event cache metadata not found: {source_metadata_path}")
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
            frame_paths = sorted(
                (path for path in source_dir.glob("*.pt") if path.stem.isdecimal()),
                key=lambda path: int(path.stem),
            )
            if not frame_paths:
                raise ValueError(f"No prepared event frames found: {source_dir}")
            destination_dir = output_root / sequence
            destination_dir.mkdir(parents=True, exist_ok=True)
            sequence_metadata = {
                "format_version": FORMAT_VERSION,
                "task": "dsec_semantic",
                "sequence_name": sequence,
                "official_role": args.role,
                "num_classes": args.num_classes,
                "label_size": [440, 640],
                "label_crop": {"top": 0, "left": 0, "height": 440, "width": 640},
                "features": sorted(requested),
                "state_policy": args.state_policy,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "checkpoint_step": int(state.global_step),
                "source_event_cache": str(source_dir),
                "source_event_cache_metadata": source_metadata,
                "input_size": input_size,
                "input_transform": {
                    "crop": {"top": 0, "left": 0, "height": 440, "width": 640},
                    "pad": {
                        "bottom": input_size[0] - 440,
                        "right": input_size[1] - 640,
                    },
                    "resize": False,
                },
                "patch_size": int(model.event_encoder.patch_size),
                "labeled_frame_indices": sorted(labeled_indices),
            }
            if not args.overwrite and _complete(
                destination_dir, sequence_metadata, labeled_indices
            ):
                print(f"{sequence}: complete sequence skipped", flush=True)
                continue

            recurrent_state = None
            cached_indices: list[int] = []
            written = 0
            preserved = 0
            available_indices: set[int] = set()
            for frame_path in tqdm(frame_paths, desc=f"semantic:{args.role}:{sequence}"):
                payload = _safe_load(frame_path)
                events = payload.get("events") if isinstance(payload, dict) else None
                if not isinstance(events, torch.Tensor):
                    raise ValueError(f"Invalid prepared event frame: {frame_path}")
                frame_index = int(payload.get("frame_index", -1))
                timestamp = int(payload.get("timestamp", -1))
                available_indices.add(frame_index)
                normalized = _semantic_transform(events, config).to(device)
                if args.state_policy == "frame":
                    recurrent_state = None
                outputs = model(normalized.unsqueeze(1), state=recurrent_state)
                recurrent_state = outputs.get("state")
                if recurrent_state is not None:
                    recurrent_state = tuple(value.detach() for value in recurrent_state)
                if frame_index not in labeled_indices:
                    continue
                grid_height = normalized.shape[-2] // model.event_encoder.patch_size
                grid_width = normalized.shape[-1] // model.event_encoder.patch_size
                feature_maps: dict[str, torch.Tensor] = {}
                for feature in requested:
                    feature_maps[feature] = (
                        outputs[feature][0, 0]
                        .reshape(grid_height, grid_width, -1)
                        .permute(2, 0, 1)
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
                    )
                destination = destination_dir / f"{frame_index:06d}.pt"
                identity = {
                    "format_version": FORMAT_VERSION,
                    "task": "dsec_semantic",
                    "sequence_name": sequence,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "checkpoint_sha256": digest,
                    "checkpoint_step": int(state.global_step),
                    "state_policy": args.state_policy,
                    "features": feature_maps,
                }
                if destination.is_file() and not args.overwrite:
                    existing = _safe_load(destination)
                    valid = (
                        isinstance(existing, dict)
                        and all(
                            existing.get(key) == identity[key]
                            for key in (
                                "format_version",
                                "task",
                                "sequence_name",
                                "frame_index",
                                "timestamp",
                                "checkpoint_sha256",
                                "checkpoint_step",
                                "state_policy",
                            )
                        )
                        and requested <= set(existing.get("features", {}))
                    )
                    if valid:
                        preserved += 1
                    else:
                        _atomic_save(identity, destination)
                        written += 1
                else:
                    _atomic_save(identity, destination)
                    written += 1
                cached_indices.append(frame_index)

            missing = sorted(labeled_indices - available_indices)
            sequence_metadata.update(
                {
                    "cached_frame_indices": sorted(cached_indices),
                    "cached_frame_count": len(cached_indices),
                    "missing_labeled_frame_indices": missing,
                }
            )
            _atomic_json(sequence_metadata, destination_dir / "metadata.json")
            print(
                f"{sequence}: {written} written, {preserved} preserved, "
                f"{len(missing)} labels without an event interval",
                flush=True,
            )


if __name__ == "__main__":
    main()
