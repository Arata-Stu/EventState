#!/usr/bin/env python3
"""Compute fixed per-channel moments from the official DSEC-Det train41 caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import yaml
    from event_state.data.cache_metadata import canonical_json_sha256
    from event_state.data.dsec import validate_event_cache_payload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace fixed coefficients: {args.output}")
    split_path = Path(__file__).parent / "manifests/dsec_det_official_split.yaml"
    split = yaml.safe_load(split_path.read_text())
    sequences = split["train"]
    if len(sequences) != 41 or len(set(sequences)) != 41:
        raise ValueError("Expected the official train41 split")
    if set(sequences) & (set(split["val"]) | set(split["test"])):
        raise ValueError("Overlapping official splits")

    total_pixels = 0
    frames = 0
    mean = m2 = None
    contract = None
    sources = {}
    for sequence in sequences:
        root = args.event_cache_dir / sequence
        manifest = json.loads((root / "metadata.json").read_text())
        rep = manifest["representation"]
        current = {
            "representation": rep,
            "sensor_size": [manifest["height"], manifest["width"]],
            "event_window_boundary": manifest["event_window_boundary"],
            "event_window_fraction": manifest.get("event_window_fraction", 1.0),
            "coordinate_space": manifest["coordinate_space"],
            "spatial_quantization": manifest["spatial_quantization"],
        }
        if rep["type"] != "voxel_grid" or rep["normalization"] != "none":
            raise ValueError("Use voxel caches prepared with --voxel-normalization none")
        if contract is None:
            contract = current
        if contract != current:
            raise ValueError(f"Mixed event representation contracts: {sequence}")
        digest = canonical_json_sha256(manifest)
        paths = sorted((p for p in root.glob("*.pt") if p.stem.isdecimal()), key=lambda p: int(p.stem))
        if not paths or len(paths) != manifest["frame_count"]:
            raise ValueError(f"Incomplete event cache: {sequence}")
        indices = set()
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            index = payload["frame_index"]
            if index in indices:
                raise ValueError(f"Duplicate frame index: {path}")
            indices.add(index)
            tensor, _ = validate_event_cache_payload(
                payload, cache_path=path, frame_index=index, timestamp=int(path.stem),
                previous_timestamp=payload["previous_timestamp"], sequence_name=sequence,
                split=manifest["split"], height=manifest["height"], width=manifest["width"],
                representation=rep, rectified=manifest["coordinate_space"] == "rectified_event",
                input_fingerprint_digest=manifest["input_fingerprint"]["digest"],
                manifest_digest=digest, event_window_fraction=current["event_window_fraction"],
                cache_dtype=manifest.get("cache_dtype", "float32"),
            )
            values = tensor.double().flatten(1)
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Expected finite nonnegative counts: {path}")
            count = values.shape[1]
            variance, batch_mean = torch.var_mean(values, dim=1, correction=0)
            if mean is None:
                mean, m2 = batch_mean, variance * count
            else:
                delta = batch_mean - mean
                combined = total_pixels + count
                m2 += variance * count + delta.square() * (total_pixels * count / combined)
                mean += delta * (count / combined)
            total_pixels += count
            frames += 1
        if indices != set(range(1, len(paths) + 1)):
            raise ValueError(f"Missing frame indices: {sequence}")
        sources[sequence] = {"manifest_sha256": digest, "frames": len(paths),
                             "cache_dtype": manifest.get("cache_dtype", "float32")}
        print(f"[normalization] {sequence}: {len(paths)} frames", flush=True)
    std = (m2 / total_pixels).clamp_min(0).sqrt()
    constant = std < 1e-6
    std[constant] = 1.0
    result = {
        "format": "eventstate_fixed_normalization_v1", **contract,
        "includes_zero_pixels": True, "statistics_stage": "stored_counts_before_spatial_transform",
        "split": "dsec_det_train41", "sequences": sequences, "sources": sources,
        "frames": frames, "pixels_per_channel": total_pixels,
        "mean": mean.tolist(), "std": std.tolist(),
        "std_floor": 1e-6, "constant_channels": constant.nonzero().flatten().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"[complete] {args.output}")


if __name__ == "__main__":
    main()
