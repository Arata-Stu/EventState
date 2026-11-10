"""Train-only activity stratification and cache provenance, without ML imports."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from event_state.activity_quantiles import (
    ActivityIndex,
    SOURCE_CONTRACT,
    build_activity_manifest,
    canonical_sha256,
    event_activity_record,
    training_sequences,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_dsec_activity_quantiles", ROOT / "tools/prepare_dsec_activity_quantiles.py"
)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def row(sequence: str, index: int, count: int, window: int = 1000) -> dict:
    return dict(sequence=sequence, timestamp=(index + 1) * window, previous_timestamp=index * window,
                event_count=count, window_us=window, event_rate_hz=count * 1e6 / window)


def build(rows: list[dict], *, train: list[str] | None = None, reference: dict | None = None) -> dict:
    return build_activity_manifest(rows, train or ["train"], reference=reference or dict(
        role="train", name="common", manifest_sha256="a" * 64, excluded_sequences=["val", "test"]),
        source_metadata_sha256={r["sequence"]: "b" * 64 for r in rows})


def cached_sequence(base: Path, *, counts: tuple[int, ...] = (1, 2)) -> tuple[dict, dict[Path, dict]]:
    base.mkdir()
    timeline = [dict(frame_index=i + 1, previous_timestamp=i * 1000, timestamp=(i + 1) * 1000)
                for i in range(len(counts))]
    metadata = dict(format_version=2, dataset="DSEC", split="train", sequence_name=base.name,
                    height=480, width=640, coordinate_space="rectified_event",
                    frame_count=len(counts), representation=dict(type="gep_rgb", channels=3),
                    event_window="rgb_interval", event_window_boundary=SOURCE_CONTRACT["event_window_boundary"],
                    timestamp_manifest_sha256=canonical_sha256(timeline),
                    input_fingerprint=dict(digest="c" * 64))
    (base / "metadata.json").write_text(json.dumps(metadata))
    marker = dict(marker_format_version=1, cache_format_version=2, dataset="DSEC", split="train",
                  sequence_name=base.name, frame_count=len(counts),
                  timestamp_manifest_sha256=metadata["timestamp_manifest_sha256"],
                  input_fingerprint_digest="c" * 64, manifest_sha256=canonical_sha256(metadata))
    (base / "_SUCCESS").write_text(json.dumps(marker))
    payloads = {}
    for entry, count in zip(timeline, counts):
        path = base / f"{entry['timestamp']}.pt"
        path.touch()
        payloads[path] = dict(entry, event_count=count, format_version=2, dataset="DSEC", split="train",
                              sequence_name=base.name, representation=metadata["representation"],
                              event_window_boundary=metadata["event_window_boundary"],
                              height=480, width=640, rectified=True,
                              manifest_digest=canonical_sha256(metadata),
                              input_fingerprint_digest="c" * 64)
    return metadata, payloads


class ActivityQuantilesTest(unittest.TestCase):
    def test_train_only_cutpoints_ignore_extreme_evaluation_frames(self):
        train = [row("train", i, i) for i in range(10)]
        original = build(train)
        expanded = build([*train, row("test", 0, 10**8)])
        self.assertEqual(original["cutpoints"], expanded["cutpoints"])
        self.assertEqual([c["event_rate_hz"] for c in original["cutpoints"]], [1000, 3000, 5000, 7000])
        self.assertEqual(expanded["reference"]["frames"], 10)
        index = ActivityIndex.from_dict(expanded)
        self.assertEqual(index.lookup("train", 2000)["group"], "p00_20")
        self.assertEqual(index.lookup("test", 1000)["group"], "p80_100")
        self.assertEqual(index.groups, ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100"))

    def test_ties_merge_and_all_zero_reference_retains_overflow_group(self):
        manifest = build([*[row("train", i, 0) for i in range(5)], row("test", 0, 1)])
        index = ActivityIndex.from_dict(manifest)
        self.assertEqual(index.groups, ("p00_80", "p80_100"))
        self.assertEqual([g["reference_frames"] for g in manifest["groups"]], [5, 0])
        self.assertEqual(index.lookup("train", 1000)["group"], "p00_80")
        self.assertEqual(index.lookup("test", 1000)["group"], "p80_100")

    def test_invalid_rates_counts_and_windows_are_rejected(self):
        for field, value in [("event_rate_hz", float("nan")), ("event_rate_hz", -1),
                             ("event_rate_hz", 9), ("event_count", True),
                             ("event_count", -1), ("window_us", 0), ("previous_timestamp", 9)]:
            with self.subTest(field=field, value=value):
                record = row("train", 0, 1)
                record[field] = value
                with self.assertRaises(ValueError):
                    build([record])

    def test_duplicates_and_missing_reference_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build([row("train", 0, 1)] * 2)
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            build([row("test", 0, 1)])

    def test_reference_role_and_split_overlap_fail_closed(self):
        for ref in (dict(role="val", manifest_sha256="a" * 64),
                    dict(role="train", manifest_sha256="a" * 64, excluded_sequences=["train"])):
            with self.assertRaises(ValueError):
                build([row("train", 0, 1)], reference=ref)
        with self.assertRaisesRegex(ValueError, "overlap"):
            training_sequences(dict(dataset="dsec", splits=dict(train=["train"], val=["train"])))
        with self.assertRaisesRegex(ValueError, "overlap"):
            training_sequences(dict(dataset="dsec", splits=dict(train=["train"], val=["val"], test=["val"])))

    def test_loaded_manifest_checks_assignments_bounds_and_digests(self):
        manifest = build([row("train", i, i) for i in range(10)])
        for mutate in (
            lambda p: p["records"][0].update(group="p80_100"),
            lambda p: p["cutpoints"][0].update(event_rate_hz=0),
            lambda p: p["groups"][0].update(reference_frames=42),
            lambda p: p.update(records_sha256="d" * 64),
            lambda p: p["source_contract"].update(height=430),
            lambda p: p.update(format_version=99),
        ):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                ActivityIndex.from_dict(changed)

    def test_lookup_key_coverage_and_result_copies(self):
        index = ActivityIndex.from_dict(build([row("train", 0, 1)]))
        index.validate_keys([("train", 1000)], allowed_sequences=["train"])
        for keys, allowed in [([("train", 2000)], ["train"]),
                              ([("train", 1000)] * 2, ["train"]),
                              ([("train", 1000)], ["test"])]:
            with self.assertRaises(ValueError):
                index.validate_keys(keys, allowed_sequences=allowed)
        index.lookup("train", 1000)["group"] = "wrong"
        index.identity["source_contract"]["height"] = 99
        self.assertEqual(index.identity["source_contract"]["height"], 480)
        self.assertNotEqual(index.lookup("train", 1000)["group"], "wrong")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(build([row("train", 0, 1)])))
            self.assertEqual(ActivityIndex.load(path).identity, index.identity)

    def test_cache_metadata_count_and_window_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, payloads = cached_sequence(Path(tmp) / "train")
            records, digest = PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)
            self.assertEqual([r["event_rate_hz"] for r in records], [1000, 2000])
            self.assertEqual(digest, canonical_sha256(metadata))

    def test_missing_and_stale_completion_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "train"
            _, payloads = cached_sequence(base)
            (base / "_SUCCESS").write_text("{}")
            with self.assertRaisesRegex(ValueError, "completion marker"):
                PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)
            (base / "_SUCCESS").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_cache_gap_missing_frame_and_fingerprint_are_rejected(self):
        for mode in ("gap", "missing", "fingerprint", "manifest", "frame_index"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "train"
                _, payloads = cached_sequence(base)
                last = payloads[base / "2000.pt"]
                if mode == "gap":
                    last["previous_timestamp"] = 1500
                elif mode == "missing":
                    (base / "2000.pt").unlink()
                elif mode == "fingerprint":
                    last["input_fingerprint_digest"] = "wrong"
                elif mode == "manifest":
                    last["manifest_digest"] = "wrong"
                else:
                    last["frame_index"] = 3
                with self.assertRaises(ValueError):
                    PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_fixed_and_fractional_windows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, payloads = cached_sequence(Path(tmp) / "train")
            original = next(iter(payloads.values()))
            for field, value in [("event_window_us", 20_000), ("event_window_fraction", .5)]:
                for location in ("metadata", "payload"):
                    meta, payload = copy.deepcopy(metadata), copy.deepcopy(original)
                    (meta if location == "metadata" else payload)[field] = value
                    with self.assertRaises(ValueError):
                        event_activity_record(payload, sequence="train", metadata=meta)

    def test_cli_help_needs_no_ml_runtime_and_output_is_preserved(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/prepare_dsec_activity_quantiles.py"),
                                 "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--train-manifest", result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("existing")
            args = type("Args", (), dict(output=path))()
            with patch.object(PREPARE, "parse_args", return_value=args):
                with self.assertRaises(FileExistsError):
                    PREPARE.main()
            self.assertEqual(path.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
