#!/usr/bin/env python3
"""Fit train-only event-rate quintiles from completed prepared DSEC event caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_quantiles import (ActivityIndex, build_activity_manifest,
    canonical_sha256, event_activity_record, training_sequences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-cache-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path,
                        default=ROOT / "tools/manifests/dsec_stage1_common_v1.json")
    parser.add_argument("--sequences", nargs="+", help="Additional sequences; train pool is always included")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scan_sequence(root: Path, sequence: str, *, load: Any) -> tuple[list[dict[str, Any]], str]:
    base = root / sequence
    if not (base / "_SUCCESS").is_file():
        raise ValueError(f"Incomplete DSEC event cache: {base}")
    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    marker = json.loads((base / "_SUCCESS").read_text(encoding="utf-8"))
    expected_marker = dict(marker_format_version=1, cache_format_version=metadata.get("format_version"),
        dataset=metadata.get("dataset"), split=metadata.get("split"), sequence_name=sequence,
        frame_count=metadata.get("frame_count"),
        timestamp_manifest_sha256=metadata.get("timestamp_manifest_sha256"),
        input_fingerprint_digest=metadata.get("input_fingerprint", {}).get("digest"),
        manifest_sha256=canonical_sha256(metadata))
    if any(marker.get(key) != value for key, value in expected_marker.items()):
        raise ValueError(f"Prepared event completion marker mismatch: {base}")
    paths = sorted((p for p in base.glob("*.pt") if p.stem.isdecimal()), key=lambda p: int(p.stem))
    count = metadata.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or len(paths) != count:
        raise ValueError(f"Prepared event frame count mismatch: {base}")
    rows, timeline = [], []
    last = None
    for index, path in enumerate(paths, 1):
        payload = load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid prepared event payload: {path}")
        if payload.get("frame_index") != index or payload.get("timestamp") != int(path.stem):
            raise ValueError(f"Missing/reordered prepared event frame: {path}")
        row = event_activity_record(payload, sequence=sequence, metadata=metadata)
        if last is not None and row["previous_timestamp"] != last:
            raise ValueError(f"Gapped prepared event timeline: {path}")
        last = row["timestamp"]
        rows.append(row)
        timeline.append(dict(frame_index=index, previous_timestamp=row["previous_timestamp"],
                             timestamp=row["timestamp"]))
    # Same canonical definition as frame_manifest_sha256, without ML imports.
    expected = canonical_sha256(timeline)
    if metadata.get("timestamp_manifest_sha256") != expected:
        raise ValueError(f"Prepared event timestamp manifest mismatch: {base}")
    return rows, canonical_sha256(metadata)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Activity manifest already exists; choose a new --output: {args.output}")
    reference_bytes = args.train_manifest.read_bytes()
    manifest = json.loads(reference_bytes)
    train, excluded = training_sequences(manifest)
    root = args.event_cache_dir.expanduser().resolve()
    if args.sequences is not None:
        if len(set(args.sequences)) != len(args.sequences):
            raise ValueError("Duplicate --sequences")
        sequences = sorted(set(train) | set(args.sequences))
    else:
        sequences = sorted(p.name for p in root.iterdir() if p.is_dir())
    if set(train) - set(sequences):
        raise ValueError(f"Missing reference train sequences: {sorted(set(train) - set(sequences))}")
    import torch

    def load(path: Path) -> Any:
        # Tensor storages are not paged into RAM for this metadata-only scan.
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    rows, hashes = [], {}
    for sequence in sequences:
        records, source_hash = scan_sequence(root, sequence, load=load)
        rows.extend(records)
        hashes[sequence] = source_hash
        print(f"[activity] {sequence}: {len(records)} frames", flush=True)
    result = build_activity_manifest(rows, train, reference=dict(role="train",
        name=manifest.get("name", args.train_manifest.stem), excluded_sequences=list(excluded),
        manifest_sha256=hashlib.sha256(reference_bytes).hexdigest()), source_metadata_sha256=hashes)
    ActivityIndex.from_dict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any artifact written concurrently.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Saved {len(rows)} activity frames, {len(result['groups'])} groups: {args.output}")


if __name__ == "__main__":
    main()
