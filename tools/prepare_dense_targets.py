#!/usr/bin/env python3
"""Index official targets without moving samples between published splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_state.dense.protocol import digest, load_protocol, sequences_for, sha256_file


def unique_path(candidates):
    matches = {p.resolve() for p in candidates if p.exists()}
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source, found {sorted(map(str, matches))}: {candidates}")
    return next(iter(matches))


def parse_test_csv(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.strip().replace(",", " ").split()
        if len(fields) >= 2 and all(v.isdecimal() for v in fields[:2]):
            rows.append((int(fields[0]), int(fields[1])))
    if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Expected increasing official timestamp,index rows: {path}")
    if len({index for _, index in rows}) != len(rows):
        raise ValueError("Duplicate official prediction indices")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--role", required=True, choices=("train", "val", "test"))
    parser.add_argument("--task", required=True, choices=("depth", "disparity"))
    parser.add_argument("--target-root", type=Path, required=True,
                        help="DSEC train_disparity directory; M3ED prepare_m3ed_downstream output")
    parser.add_argument("--calibration-root", type=Path, help="DSEC directory containing SEQ/calibration/cam_to_cam.yaml")
    parser.add_argument("--test-timestamps-root", type=Path, help="Official DSEC test_disparity_timestamps CSV directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Target index already exists: {args.output}")
    protocol = load_protocol(args.protocol)
    sequences = sequences_for(protocol, args.role)
    labeled = protocol["labels_available"][args.role]
    if protocol["dataset"] == "m3ed" and args.task != "depth":
        raise ValueError("M3ED provides metric depth, not calibrated stereo-disparity targets here")
    if not labeled and protocol["dataset"] == "m3ed":
        raise ValueError("M3ED official test has no public labels or RGB timeline. "
                         "Use the published F3 train/val protocol for local depth evaluation; "
                         "do not rename labeled sequences to official test.")
    import numpy as np
    records, source_identity = [], {}
    for sequence in sequences:
        base = args.target_root / sequence
        if protocol["dataset"] == "dsec":
            if labeled:
                directory = unique_path([base / "disparity" / "event", base / "event"])
                timestamps_path = unique_path([base / "disparity" / "timestamps.txt",
                                               base / "disparity_timestamps.txt", base / "timestamps.txt"])
                timestamps = np.atleast_1d(np.loadtxt(timestamps_path, dtype=np.int64))
                files = sorted(directory.glob("*.png"))
                if timestamps.ndim != 1 or len(files) != len(timestamps) or not files:
                    raise ValueError(f"DSEC disparity timestamp/file count mismatch: {sequence}")
                q = None
                if args.task == "depth":
                    import yaml
                    if args.calibration_root is None:
                        raise ValueError("Metric DSEC depth requires --calibration-root")
                    calibration = unique_path([args.calibration_root / sequence / "calibration" / "cam_to_cam.yaml",
                                               args.calibration_root / sequence / "cam_to_cam.yaml"])
                    q = yaml.safe_load(calibration.read_text())["disparity_to_depth"]["cams_03"]
                    source_identity[str(calibration)] = sha256_file(calibration)
                for stamp, path in zip(timestamps, files):
                    if not path.stem.isdecimal():
                        raise ValueError(f"Unexpected target filename: {path}")
                    records.append(dict(sequence=sequence, timestamp=int(stamp),
                                        output_index=int(path.stem), target_path=str(path.resolve()),
                                        encoding="dsec_disparity_uint16_x256", disparity_to_depth_q=q,
                                        target_sha256=sha256_file(path)))
                source_identity[str(timestamps_path)] = sha256_file(timestamps_path)
            else:
                if args.test_timestamps_root is None:
                    raise ValueError("Hidden-label DSEC test requires official --test-timestamps-root")
                csv = unique_path([args.test_timestamps_root / f"{sequence}.csv",
                                   args.test_timestamps_root / sequence / "timestamps.csv"])
                records.extend(dict(sequence=sequence, timestamp=t, output_index=i)
                               for t, i in parse_test_csv(csv))
                source_identity[str(csv)] = sha256_file(csv)
        else:
            import h5py
            metadata_path = base / "metadata.json"
            if not (base / "_SUCCESS").is_file():
                raise ValueError(f"M3ED target preparation incomplete: {base}")
            meta = json.loads(metadata_path.read_text())
            if meta.get("sequence_name") != sequence or "depth" not in meta.get("tasks", []):
                raise ValueError("M3ED target provenance does not match sequence/task")
            path = base / "targets.h5"
            with h5py.File(path, "r") as stream:
                timestamps = np.asarray(stream["timestamps"], dtype=np.int64)
                valid = np.asarray(stream["depth_frame_valid"], dtype=bool)
                if len(valid) != len(timestamps) or len(stream["depth_m"]) != len(valid):
                    raise ValueError("M3ED target length mismatch")
                for i in np.flatnonzero(valid):
                    records.append(dict(sequence=sequence, timestamp=int(timestamps[i]),
                                        output_index=int(i), target_row=int(i), target_path=str(path.resolve()),
                                        encoding="m3ed_prepared_depth_m"))
            source_identity[str(metadata_path.resolve())] = sha256_file(metadata_path)
            # Stream once: mutable HDF5 targets must not silently change after indexing.
            source_identity[str(path.resolve())] = sha256_file(path)
        stamps = [r["timestamp"] for r in records if r["sequence"] == sequence]
        if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"Missing/nonmonotonic target timestamps: {sequence}")
    value = dict(format_version=1, protocol=protocol, protocol_sha256=digest(protocol),
                 dataset=protocol["dataset"], task=args.task, role=args.role, labeled=labeled,
                 source_identity=source_identity, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"{args.protocol} {args.role}: {len(records)} target timestamps; public labels={labeled}")


if __name__ == "__main__":
    main()
