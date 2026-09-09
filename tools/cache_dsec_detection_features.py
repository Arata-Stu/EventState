#!/usr/bin/env python3
"""Cache frozen z/h maps over canonical DSEC-Detection sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from event_state.detection.split import load_dsec_detection_split
from event_state.data.transforms import PairedSequenceTransform
from event_state.training import build_model, load_checkpoint, load_checkpoint_config_metadata


FORMAT_VERSION = 1
DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_det_official_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen EventState maps for a canonical DSEC-Detection split"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--features", nargs="+", choices=("z", "h"), default=("z", "h"))
    parser.add_argument("--state-policy", choices=("continuous", "frame"), default="continuous")
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


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
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


def _event_transform(config: Any) -> PairedSequenceTransform:
    means = OmegaConf.select(config, "dataset.representation.normalize_mean")
    stds = OmegaConf.select(config, "dataset.representation.normalize_std")
    if means is None or stds is None:
        raise ValueError("Checkpoint does not contain event normalization")
    return PairedSequenceTransform(
        height=int(OmegaConf.select(config, "dataset.input_height")),
        width=int(OmegaConf.select(config, "dataset.input_width")),
        training=False,
        event_mean=tuple(float(value) for value in means),
        event_std=tuple(float(value) for value in stds),
    )


def _valid_existing(path: Path, identity: dict[str, Any], features: set[str]) -> bool:
    if not path.is_file():
        return False
    payload = _safe_load(path)
    return (
        isinstance(payload, dict)
        and all(payload.get(key) == value for key, value in identity.items())
        and isinstance(payload.get("features"), dict)
        and features <= set(payload["features"])
    )


def _completed_sequence(
    destination_dir: Path,
    frame_paths: list[Path],
    expected_metadata: dict[str, Any],
    features: set[str],
) -> bool:
    """Trust only a completion manifest whose full frame set still exists."""

    metadata_path = destination_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    identity_keys = (
        "format_version",
        "sequence_name",
        "official_role",
        "state_policy",
        "checkpoint_sha256",
        "checkpoint_step",
        "frame_count",
        "coordinate_space",
        "input_size",
        "patch_size",
    )
    if any(metadata.get(key) != expected_metadata.get(key) for key in identity_keys):
        return False
    available = metadata.get("features")
    if not isinstance(available, list) or not features <= set(available):
        return False
    if metadata.get("source_event_cache_metadata") != expected_metadata.get(
        "source_event_cache_metadata"
    ):
        return False
    return all(
        (destination_dir / frame_path.name).is_file() for frame_path in frame_paths
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    event_cache = args.event_cache_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    device = torch.device(args.device)
    split = load_dsec_detection_split(args.split_manifest)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    sequences = split.sequences(args.role)[args.shard_index :: args.num_shards]
    if not sequences:
        raise ValueError("Selected DSEC-Detection feature shard is empty")
    requested_features = set(args.features)

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
    model = build_model(config, device=device)
    state = load_checkpoint(
        checkpoint, model=model, device=device, restore_rng=False
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_digest = _sha256(checkpoint)
    event_transform = _event_transform(config)

    with torch.inference_mode():
        for sequence in sequences:
            source_dir = event_cache / sequence
            source_metadata_path = source_dir / "metadata.json"
            if not source_metadata_path.is_file():
                raise FileNotFoundError(
                    f"Event cache for official {args.role} sequence is missing: {source_dir}"
                )
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
            frame_paths = sorted(
                (path for path in source_dir.glob("*.pt") if path.stem.isdecimal()),
                key=lambda path: int(path.stem),
            )
            if not frame_paths:
                raise ValueError(f"No cached event frames found: {source_dir}")
            destination_dir = output_root / sequence
            destination_dir.mkdir(parents=True, exist_ok=True)
            sequence_metadata = {
                "format_version": FORMAT_VERSION,
                "sequence_name": sequence,
                "official_role": args.role,
                "features": sorted(requested_features),
                "state_policy": args.state_policy,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_step": int(state.global_step),
                "source_event_cache": str(source_dir),
                "source_event_cache_metadata": source_metadata,
                "frame_count": len(frame_paths),
                "coordinate_space": "rectified_event",
                "input_size": [event_transform.height, event_transform.width],
                "patch_size": int(model.event_encoder.patch_size),
            }
            if not args.overwrite and _completed_sequence(
                destination_dir,
                frame_paths,
                sequence_metadata,
                requested_features,
            ):
                print(
                    f"{sequence}: complete sequence skipped "
                    f"({len(frame_paths)} preserved)",
                    flush=True,
                )
                continue
            recurrent_state = None
            written = 0
            preserved = 0
            for frame_path in tqdm(frame_paths, desc=f"{args.role}:{sequence}", unit="frame"):
                event_payload = _safe_load(frame_path)
                events = event_payload.get("events") if isinstance(event_payload, dict) else None
                if not isinstance(events, torch.Tensor) or events.ndim != 3:
                    raise ValueError(f"Invalid event cache payload: {frame_path}")
                timestamp = int(event_payload.get("timestamp", -1))
                frame_index = int(event_payload.get("frame_index", -1))
                identity = {
                    "format_version": FORMAT_VERSION,
                    "sequence_name": sequence,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_step": int(state.global_step),
                    "state_policy": args.state_policy,
                }
                destination = destination_dir / f"{timestamp}.pt"
                # Continuous h requires replaying every frame even when its output exists.
                normalized, _ = event_transform(events.float().unsqueeze(0), None)
                if normalized is None:
                    raise RuntimeError("Event transform unexpectedly returned no tensor")
                normalized = normalized[0].to(device=device).unsqueeze(0)
                if args.state_policy == "frame":
                    recurrent_state = None
                outputs = model(normalized.unsqueeze(1), state=recurrent_state)
                recurrent_state = outputs.get("state")
                if recurrent_state is not None:
                    recurrent_state = tuple(value.detach() for value in recurrent_state)
                feature_maps: dict[str, torch.Tensor] = {}
                grid_height = int(normalized.shape[-2]) // int(model.event_encoder.patch_size)
                grid_width = int(normalized.shape[-1]) // int(model.event_encoder.patch_size)
                if "z" in requested_features:
                    feature_maps["z"] = (
                        outputs["z"][0, 0]
                        .reshape(grid_height, grid_width, -1)
                        .permute(2, 0, 1)
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
                    )
                if "h" in requested_features:
                    feature_maps["h"] = (
                        outputs["h"][0, 0]
                        .reshape(grid_height, grid_width, -1)
                        .permute(2, 0, 1)
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
                    )
                payload = {**identity, "features": feature_maps}
                if not args.overwrite and _valid_existing(
                    destination, identity, requested_features
                ):
                    preserved += 1
                else:
                    _atomic_save(payload, destination)
                    written += 1
            _atomic_write_json(sequence_metadata, destination_dir / "metadata.json")
            print(f"{sequence}: {written} written, {preserved} preserved", flush=True)


if __name__ == "__main__":
    main()
