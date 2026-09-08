#!/usr/bin/env python3
"""Warp cached EventState maps into distorted DSEC-Det event coordinates."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from event_state.detection.data import build_dagr_sampling_grid, find_rectify_map
from event_state.detection.split import load_dsec_detection_split


DEFAULT_SPLIT = Path(__file__).parent / "manifests" / "dsec_det_official_split.yaml"
FORMAT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reusable native or half-scale DSEC-Det feature caches"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--features", nargs="+", choices=("z", "h"), default=("z", "h"))
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 keeps native 640x430 coordinates; 2 reproduces DAGR's half scale",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _valid_existing(
    path: Path, features: set[str], source_payload: dict[str, Any], scale: int
) -> bool:
    if not path.is_file():
        return False
    payload = _safe_load(path)
    return (
        isinstance(payload, dict)
        and payload.get("benchmark_format_version") == FORMAT_VERSION
        and payload.get("coordinate_space") == "dsec_det_distorted"
        and payload.get("detection_scale") == scale
        and payload.get("checkpoint_sha256") == source_payload.get("checkpoint_sha256")
        and payload.get("timestamp") == source_payload.get("timestamp")
        and payload.get("frame_index") == source_payload.get("frame_index")
        and isinstance(payload.get("features"), dict)
        and features <= set(payload["features"])
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError("input-dir and output-dir must differ")
    features = set(args.features)
    split = load_dsec_detection_split(args.split_manifest)
    sequences = split.sequences(args.role)[args.shard_index :: args.num_shards]
    if not sequences:
        raise ValueError("Selected DSEC-Det feature shard is empty")
    device = torch.device(args.device)

    for sequence in sequences:
        source_dir = input_root / sequence
        metadata_path = source_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Source feature metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("coordinate_space", "rectified_event") != "rectified_event":
            raise ValueError(f"Source features are not rectified EventState maps: {metadata_path}")
        input_size = metadata.get("input_size")
        source_stride = metadata.get("patch_size")
        if (
            not isinstance(input_size, list)
            or len(input_size) != 2
            or not isinstance(source_stride, int)
        ):
            raise ValueError(f"Invalid source feature geometry: {metadata_path}")
        available = set(metadata.get("features", []))
        if not features <= available:
            raise ValueError(f"Source cache lacks {sorted(features - available)}: {metadata_path}")
        with h5py.File(find_rectify_map(dataset_root, sequence), "r") as handle:
            rectify_map = np.asarray(handle["rectify_map"], dtype=np.float32)
        grid = build_dagr_sampling_grid(
            rectify_map,
            source_input_size=(int(input_size[0]), int(input_size[1])),
            source_stride=source_stride,
            scale=args.scale,
        ).to(device)
        paths = sorted(
            (path for path in source_dir.glob("*.pt") if path.stem.isdecimal()),
            key=lambda path: int(path.stem),
        )
        destination_dir = output_root / sequence
        destination_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        preserved = 0
        for start in tqdm(
            range(0, len(paths), args.batch_size),
            desc=f"{args.role}:{sequence}",
            unit="batch",
        ):
            batch_paths = paths[start : start + args.batch_size]
            payloads = [_safe_load(path) for path in batch_paths]
            if any(
                not isinstance(payload, dict) or not isinstance(payload.get("features"), dict)
                for payload in payloads
            ):
                raise ValueError(f"Invalid source feature payload near {batch_paths[0]}")
            pending = [
                index
                for index, path in enumerate(batch_paths)
                if args.overwrite
                or not _valid_existing(
                    destination_dir / path.name, features, payloads[index], args.scale
                )
            ]
            preserved += len(batch_paths) - len(pending)
            if not pending:
                continue
            warped: dict[str, torch.Tensor] = {}
            with torch.inference_mode():
                for feature in features:
                    values = torch.stack(
                        [payloads[index]["features"][feature].float() for index in pending]
                    ).to(device)
                    expanded_grid = grid.unsqueeze(0).expand(len(pending), -1, -1, -1)
                    warped[feature] = F.grid_sample(
                        values,
                        expanded_grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).to(dtype=torch.float16, device="cpu")
            for output_index, payload_index in enumerate(pending):
                payload = payloads[payload_index]
                output = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"features", "coordinate_space"}
                }
                output["coordinate_space"] = "dsec_det_distorted"
                output["detection_scale"] = args.scale
                output["benchmark_format_version"] = FORMAT_VERSION
                output["features"] = {
                    feature: warped[feature][output_index].contiguous() for feature in features
                }
                _atomic_save(output, destination_dir / batch_paths[payload_index].name)
                written += 1
        output_metadata = {
            **metadata,
            "coordinate_space": "dsec_det_distorted",
            "benchmark_format_version": FORMAT_VERSION,
            "input_size": [430 // args.scale, 640 // args.scale],
            "patch_size": source_stride // args.scale,
            "features": sorted(features),
            "source_feature_cache": str(source_dir),
            "source_input_size": input_size,
            "source_patch_size": source_stride,
            "dsec_det_protocol": {
                "scale": args.scale,
                "cropped_height": 430,
                "physical_min_box_side": 20,
                "physical_min_box_diagonal": 30,
            },
        }
        (destination_dir / "metadata.json").write_text(
            json.dumps(output_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{sequence}: {written} written, {preserved} preserved", flush=True)


if __name__ == "__main__":
    main()
