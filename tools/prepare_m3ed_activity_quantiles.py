#!/usr/bin/env python3
"""Fit train-only activity quintiles shared by M3ED Semantic and Depth probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_quantiles import (
    ActivityIndex,
    M3ED_SOURCE_CONTRACT,
    build_activity_manifest,
    canonical_sha256,
    m3ed_event_activity_record,
    training_sequences,
    validate_m3ed_activity_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", "--event-cache-dir", dest="prepared_root", type=Path,
                        required=True, help="Existing M3ED half_dagr preparation root")
    parser.add_argument("--train-manifest", type=Path,
                        default=ROOT / "tools/manifests/m3ed_stage1_common_v1.json")
    parser.add_argument("--sequences", nargs="+", help="Additional sequences; train pool is always included")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scan_sequence(root: Path, sequence: str, *, load: Any) -> tuple[list[dict[str, Any]], str]:
    base = root / sequence
    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    stamps = validate_m3ed_activity_metadata(metadata, sequence=sequence)
    folder = base / "events"
    paths = sorted((path for path in folder.glob("*.pt") if path.stem.isdecimal()),
                   key=lambda path: int(path.stem))
    if len(paths) != len(stamps) - 1 or not paths:
        raise ValueError(f"Incomplete prepared M3ED event cache: {base}")
    rows = []
    for index, path in enumerate(paths, 1):
        if int(path.stem) != stamps[index]:
            raise ValueError(f"Missing/reordered M3ED prepared timestamp: {path}")
        payload = load(path)
        row = m3ed_event_activity_record(payload, sequence=sequence, metadata=metadata)
        if payload["frame_index"] != index:
            raise ValueError(f"Missing/reordered M3ED prepared frame index: {path}")
        rows.append(row)
    declared_total = metadata.get("counts", {}).get("rectified_in_bounds_events")
    if declared_total is not None:
        if (isinstance(declared_total, bool) or not isinstance(declared_total, int)
                or declared_total < 0 or declared_total != sum(row["event_count"] for row in rows)):
            raise ValueError(f"M3ED prepared total event count mismatch: {base}")
    return rows, canonical_sha256(metadata)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Activity manifest already exists; choose a new --output: {args.output}")
    reference_bytes = args.train_manifest.read_bytes()
    manifest = json.loads(reference_bytes)
    train, excluded = training_sequences(manifest, dataset="m3ed")
    root = args.prepared_root.expanduser().resolve()
    if args.sequences is not None:
        if len(set(args.sequences)) != len(args.sequences):
            raise ValueError("Duplicate --sequences")
        sequences = sorted(set(train) | set(args.sequences))
    else:
        sequences = sorted(path.name for path in root.iterdir() if path.is_dir())
    if set(train) - set(sequences):
        raise ValueError(f"Missing reference train sequences: {sorted(set(train) - set(sequences))}")
    import torch

    def load(path: Path) -> Any:
        # Reading tensor shape and metadata does not page event storages into RAM.
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    rows, hashes = [], {}
    for sequence in sequences:
        records, source_hash = scan_sequence(root, sequence, load=load)
        rows.extend(records)
        hashes[sequence] = source_hash
        print(f"[activity] {sequence}: {len(records)} frames", flush=True)
    result = build_activity_manifest(rows, train, reference=dict(role="train",
        name=manifest.get("name", args.train_manifest.stem), excluded_sequences=list(excluded),
        manifest_sha256=hashlib.sha256(reference_bytes).hexdigest()), source_metadata_sha256=hashes,
        source_contract=M3ED_SOURCE_CONTRACT)
    ActivityIndex.from_dict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Saved {len(rows)} activity frames, {len(result['groups'])} groups: {args.output}")


if __name__ == "__main__":
    main()
