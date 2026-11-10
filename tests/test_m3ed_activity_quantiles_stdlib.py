"""M3ED native prepared timelines and dataset-specific train quantiles, no ML imports."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_state.activity_quantiles import (
    ActivityIndex,
    M3ED_SOURCE_CONTRACT,
    SOURCE_CONTRACT,
    build_activity_manifest,
    canonical_sha256,
    training_sequences,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_m3ed_activity_quantiles", ROOT / "tools/prepare_m3ed_activity_quantiles.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def prepared(base: Path, counts=(1, 2, 3)):
    (base / "events").mkdir(parents=True)
    stamps = [1000 * i for i in range(len(counts) + 1)]
    metadata = dict(format_version=1, dataset="M3ED", sequence_name=base.name,
        frame_count=len(stamps), timestamps=stamps, native_event_size=[720, 1280],
        event_size=[360, 640], recommended_model_input_size=[352, 640],
        downsampling="dagr_stateful_signed_event_downsampling", downsample_factor=[2, 2],
        operation_order=["dagr_downsample_raw_event_stream", "rectify_half_scale_events",
                         "build_event_representation"], coordinate_space="rectified_left_event",
        event_window="previous_rgb_timestamp < event_timestamp <= rgb_timestamp",
        representation=dict(type="gep_rgb", channels=3, percentile=90),
        counts=dict(rectified_in_bounds_events=sum(counts)))
    (base / "metadata.json").write_text(json.dumps(metadata))
    payloads = {}
    for index, count in enumerate(counts, 1):
        path = base / "events" / f"{stamps[index]}.pt"
        path.touch()
        payloads[path] = dict(format_version=1, dataset="M3ED", sequence_name=base.name,
            frame_index=index, timestamp=stamps[index], previous_timestamp=stamps[index - 1],
            event_count=count, source_event_count=count + 10, dagr_event_count=count + 5,
            events=SimpleNamespace(shape=(3, 360, 640)), representation="gep_rgb")
    return metadata, payloads


def make_record(sequence, index, count):
    return dict(sequence=sequence, timestamp=(index + 1) * 1000, previous_timestamp=index * 1000,
                event_count=count, window_us=1000, event_rate_hz=count * 1000.0)


def manifest(records, contract=M3ED_SOURCE_CONTRACT):
    return build_activity_manifest(records, ["train"], reference=dict(role="train",
        name="common", manifest_sha256="a" * 64, excluded_sequences=["val"]),
        source_metadata_sha256={row["sequence"]: "b" * 64 for row in records},
        source_contract=contract)


class M3EDActivityQuantilesTest(unittest.TestCase):
    def test_valid_legacy_cache_without_success_marker_and_initial_event_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "train"
            metadata, payloads = prepared(base)
            self.assertFalse((base / "_SUCCESS").exists())
            rows, digest = PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)
            self.assertEqual(len(rows), metadata["frame_count"] - 1)
            self.assertEqual([row["event_rate_hz"] for row in rows], [1000, 2000, 3000])
            self.assertEqual(rows[0]["previous_timestamp"], metadata["timestamps"][0])
            self.assertEqual(digest, canonical_sha256(metadata))

    def test_missing_extra_or_wrongly_named_event_files_are_rejected(self):
        for mode in ("missing", "extra", "wrong_stamp"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "train"
                _, payloads = prepared(base)
                if mode == "missing":
                    (base / "events/2000.pt").unlink()
                elif mode == "extra":
                    (base / "events/0.pt").touch()
                else:
                    (base / "events/2000.pt").rename(base / "events/2500.pt")
                with self.assertRaises(ValueError):
                    PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_metadata_contract_and_timestamp_order_are_checked(self):
        mutations = [dict(event_size=[352, 640]), dict(downsampling="resize"),
                     dict(downsample_factor=[1, 1]), dict(coordinate_space="raw_event"),
                     dict(operation_order=[]), dict(frame_count=True), dict(frame_count=5),
                     dict(format_version=True),
                     dict(timestamps=[0, 1000, 1000, 3000]),
                     dict(event_window_us=20000), dict(event_window_fraction=.5),
                     dict(representation=dict(type="voxel_grid", channels=20))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "train"
                metadata, payloads = prepared(base)
                metadata.update(mutation)
                (base / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_payload_provenance_timeline_count_and_shape_are_checked(self):
        mutations = [dict(sequence_name="other"), dict(dataset="DSEC"), dict(format_version=2),
                     dict(format_version=True),
                     dict(frame_index=1), dict(timestamp=2999), dict(previous_timestamp=1999),
                     dict(event_count=True), dict(event_count=-1), dict(event_count=30),
                     dict(event_window_fraction=.5), dict(event_window_us=20000),
                     dict(events=SimpleNamespace(shape=(3, 352, 640))), dict(representation="voxel_grid")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "train"
                _, payloads = prepared(base)
                payloads[base / "events/3000.pt"].update(mutation)
                with self.assertRaises(ValueError):
                    PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_total_count_is_checked_when_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "train"
            metadata, payloads = prepared(base)
            metadata["counts"]["rectified_in_bounds_events"] += 1
            (base / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "total event count"):
                PREPARE.scan_sequence(Path(tmp), "train", load=payloads.__getitem__)

    def test_m3ed_is_separate_from_legacy_dsec_protocol_and_roundtrips(self):
        rows = [make_record("train", i, i) for i in range(10)]
        m3ed = manifest(rows)
        dsec = manifest(rows, SOURCE_CONTRACT)
        self.assertEqual(m3ed["format_version"], 2)
        self.assertEqual(dsec["format_version"], 1)
        self.assertNotEqual(m3ed["protocol_sha256"], dsec["protocol_sha256"])
        index = ActivityIndex.from_dict(m3ed)
        self.assertEqual(index.identity["source_contract"], M3ED_SOURCE_CONTRACT)
        self.assertEqual(index.groups, ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100"))
        self.assertEqual(index.lookup("train", 1000)["group"], "p00_20")
        changed = copy.deepcopy(m3ed)
        changed["source_contract"]["height"] = 352
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ActivityIndex.from_dict(changed)

    def test_m3ed_train_only_cutpoints_ignore_validation_extremes(self):
        rows = [make_record("train", i, i) for i in range(10)]
        first = manifest(rows)
        more = manifest([*rows, make_record("val", 0, 100000000)])
        self.assertEqual(first["cutpoints"], more["cutpoints"])
        self.assertEqual(more["reference"]["frames"], 10)
        self.assertEqual(more["source_contract"]["count_definition"],
                         "prepared_event_count_after_dagr_downsampling_and_rectification")

    def test_timestamp_api_is_sorted_immutable_and_rejects_missing_sequences(self):
        rows = [make_record("train", 2, 3), make_record("train", 0, 1),
                make_record("val", 0, 5), make_record("train", 1, 2)]
        for contract in (SOURCE_CONTRACT, M3ED_SOURCE_CONTRACT):
            with self.subTest(dataset=contract["dataset"]):
                payload = manifest(rows, contract)
                index = ActivityIndex.from_dict(payload)
                identity = index.identity
                self.assertEqual(index.timestamps("train"), (1000, 2000, 3000))
                self.assertEqual(index.timestamps("val"), (1000,))
                self.assertIsInstance(index.timestamps("train"), tuple)
                with self.assertRaisesRegex(ValueError, "sequence missing"):
                    index.timestamps("absent")
                self.assertEqual(index.identity, identity)

    def test_training_manifest_dataset_and_holdout_are_checked(self):
        value = dict(dataset="m3ed", splits=dict(train=["train"], val=["val"], test=["test"]))
        self.assertEqual(training_sequences(value, dataset="m3ed"), (("train",), ("test", "val")))
        with self.assertRaises(ValueError):
            training_sequences(value)
        value["splits"]["train"].append("val")
        with self.assertRaisesRegex(ValueError, "overlap"):
            training_sequences(value, dataset="m3ed")

    def test_cli_help_is_stdlib_only_and_existing_outputs_are_preserved(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/prepare_m3ed_activity_quantiles.py"),
                                 "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prepared-root", result.stdout)
        self.assertIn("--event-cache-dir", result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keep.json"
            path.write_text("existing")
            with patch.object(PREPARE, "parse_args", return_value=SimpleNamespace(output=path)):
                with self.assertRaises(FileExistsError):
                    PREPARE.main()
            self.assertEqual(path.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
