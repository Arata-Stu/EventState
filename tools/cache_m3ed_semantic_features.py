#!/usr/bin/env python3
"""Cache continuous frozen EventState maps at labeled M3ED semantic frames."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from event_state.segmentation import load_m3ed_semantic_split
from event_state.training import load_checkpoint, load_checkpoint_config_metadata
from event_state.training.factory import build_evaluation_runtime


FORMAT_VERSION = 1
DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "m3ed_downstream_split.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--features", nargs="+", choices=("z", "h"), default=("h",))
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _set(config: Any, path: str, value: Any) -> None:
    OmegaConf.update(config, path, value, merge=False, force_add=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_bool(batch: dict[str, Any], key: str) -> bool:
    value = batch.get(key)
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    prepared_root = args.prepared_root.expanduser().resolve()
    teacher_cache = args.teacher_cache_dir.expanduser().resolve()
    target_cache = args.target_cache_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    for name, path, is_file in (
        ("checkpoint", checkpoint, True),
        ("prepared-root", prepared_root, False),
        ("teacher-cache-dir", teacher_cache, False),
        ("target-cache-dir", target_cache, False),
    ):
        if not (path.is_file() if is_file else path.is_dir()):
            raise FileNotFoundError(f"{name} not found: {path}")

    split = load_m3ed_semantic_split(args.split_manifest)
    sequences = split.sequences(args.role)
    metadata = load_checkpoint_config_metadata(checkpoint)
    config = OmegaConf.create(metadata.config)
    if str(OmegaConf.select(config, "dataset.name", default="")).lower() != "m3ed":
        raise ValueError("Checkpoint is not configured for M3ED")
    _set(config, "dataset.root", str(prepared_root))
    _set(config, "dataset.prepared_root", str(prepared_root))
    _set(config, "dataset.event_cache_dir", str(prepared_root))
    _set(config, "dataset.val_sequences", list(sequences))
    _set(config, "dataset.val_split", args.role)
    _set(config, "teacher.cache_features", True)
    _set(config, "teacher.cache_dir", str(teacher_cache))
    if args.teacher_checkpoint is not None:
        teacher_checkpoint = str(args.teacher_checkpoint.expanduser().resolve())
        _set(config, "teacher.checkpoint", teacher_checkpoint)
        _set(config, "model.event_encoder.checkpoint", teacher_checkpoint)
    _set(config, "device", args.device)
    _set(config, "training.num_workers", 0)
    _set(config, "training.pin_memory", False)
    _set(config, "training.persistent_workers", False)
    _set(config, "evaluation.checkpoint", str(checkpoint))

    runtime = build_evaluation_runtime(config)
    state = load_checkpoint(
        checkpoint,
        model=runtime.model,
        device=runtime.device,
        restore_rng=False,
        expected_config=config,
        compatibility_mode="evaluation",
    )
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    checkpoint_digest = _sha256(checkpoint)
    requested = set(args.features)
    grid_height = int(config.dataset.input_height) // int(config.teacher.patch_size)
    grid_width = int(config.dataset.input_width) // int(config.teacher.patch_size)
    label_size: list[int] | None = None
    target_validity: dict[str, np.ndarray] = {}
    target_timestamps: dict[str, np.ndarray] = {}
    for sequence in sequences:
        targets_path = target_cache / sequence / "targets.h5"
        with h5py.File(targets_path, "r") as source:
            valid = np.asarray(source["semantics_frame_valid"], dtype=np.bool_)
            timestamps = np.asarray(source["timestamps"], dtype=np.int64)
            labels = source["semantics_11"]
            current_size = [int(labels.shape[1]), int(labels.shape[2])]
        if label_size is None:
            label_size = current_size
        elif label_size != current_size:
            raise ValueError("M3ED semantic target sizes differ across sequences")
        if len(valid) != len(timestamps):
            raise ValueError(f"M3ED semantic target length mismatch: {targets_path}")
        target_validity[sequence] = valid
        target_timestamps[sequence] = timestamps
    if label_size is None:
        raise RuntimeError("No M3ED semantic target geometry was found")
    input_size = [int(config.dataset.input_height), int(config.dataset.input_width)]
    if label_size != input_size:
        raise ValueError(
            f"M3ED semantic target size {label_size} does not match model input {input_size}"
        )
    records: dict[str, dict[str, Any]] = {
        sequence: {"cached": [], "written": 0, "preserved": 0}
        for sequence in sequences
    }

    recurrent_state = None
    current_sequence: str | None = None
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if runtime.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        for batch in tqdm(runtime.validation_loader, desc=f"M3ED semantic cache {args.role}"):
            sequence = str(batch["sequence_name"][0])
            if sequence != current_sequence or _batch_bool(batch, "is_sequence_start"):
                current_sequence = sequence
                recurrent_state = None
            events = batch["events"].to(runtime.device, non_blocking=True)
            outputs = runtime.model(events, state=recurrent_state)
            recurrent_state = outputs.get("state")
            if recurrent_state is not None:
                recurrent_state = tuple(value.detach() for value in recurrent_state)
            frame_indices = batch["frame_indices"][0].detach().cpu()
            timestamps = batch["timestamps"][0].detach().cpu()
            valid = torch.from_numpy(target_validity[sequence][frame_indices.numpy()])
            expected_timestamps = torch.from_numpy(
                target_timestamps[sequence][frame_indices.numpy()]
            )
            if not torch.equal(timestamps, expected_timestamps):
                raise ValueError(f"M3ED semantic timestamps mismatch for {sequence}")
            destination_dir = output_root / sequence
            destination_dir.mkdir(parents=True, exist_ok=True)
            for local_index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                frame_index = int(frame_indices[local_index])
                timestamp = int(timestamps[local_index])
                feature_maps: dict[str, torch.Tensor] = {}
                for feature in requested:
                    feature_maps[feature] = (
                        outputs[feature][0, local_index]
                        .reshape(grid_height, grid_width, -1)
                        .permute(2, 0, 1)
                        .detach()
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
                    )
                payload = {
                    "format_version": FORMAT_VERSION,
                    "task": "m3ed_semantic",
                    "sequence_name": sequence,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_step": int(state.global_step),
                    "state_policy": "continuous",
                    "features": feature_maps,
                }
                destination = destination_dir / f"{frame_index:06d}.pt"
                preserve = False
                if destination.is_file() and not args.overwrite:
                    try:
                        existing = torch.load(destination, map_location="cpu", weights_only=True)
                        preserve = (
                            isinstance(existing, dict)
                            and existing.get("checkpoint_sha256") == checkpoint_digest
                            and int(existing.get("frame_index", -1)) == frame_index
                            and requested <= set(existing.get("features", {}))
                        )
                    except (OSError, RuntimeError, TypeError, ValueError):
                        preserve = False
                if preserve:
                    records[sequence]["preserved"] += 1
                else:
                    _atomic_torch_save(payload, destination)
                    records[sequence]["written"] += 1
                records[sequence]["cached"].append(frame_index)

    for sequence in sequences:
        destination_dir = output_root / sequence
        target_metadata_path = target_cache / sequence / "metadata.json"
        target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
        sequence_metadata = {
            "format_version": FORMAT_VERSION,
            "task": "m3ed_semantic",
            "sequence_name": sequence,
            "role": args.role,
            "features": sorted(requested),
            "state_policy": "continuous",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_step": int(state.global_step),
            "input_size": input_size,
            "label_size": label_size,
            "patch_size": int(config.teacher.patch_size),
            "target_cache": str(target_cache / sequence),
            "target_cache_metadata": target_metadata,
            "cached_frame_indices": sorted(records[sequence]["cached"]),
            "cached_frame_count": len(records[sequence]["cached"]),
        }
        _atomic_json(sequence_metadata, destination_dir / "metadata.json")
        print(
            f"{sequence}: {records[sequence]['written']} written, "
            f"{records[sequence]['preserved']} preserved, "
            f"{len(records[sequence]['cached'])} labeled event frames",
            flush=True,
        )


if __name__ == "__main__":
    main()
