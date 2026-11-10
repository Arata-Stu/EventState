"""Depth activity routing/provenance and pooled sums without scientific imports."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from event_state.activity_quantiles import (
    ActivityIndex, M3ED_SOURCE_CONTRACT, build_activity_manifest, canonical_sha256,
)
from event_state.dense.activity_evaluation import DenseActivityEvaluator, validate_dense_activity_sources
from event_state.dense.metrics import RegressionSums


ROOT = Path(__file__).resolve().parents[1]


def record(sequence, stamp, count):
    return dict(sequence=sequence, timestamp=stamp, previous_timestamp=stamp - 1000,
                window_us=1000, event_count=count, event_rate_hz=count * 1000.0)


def index_for(rows, source_hashes=None, source_contract=None):
    reference = [record("train", 1000 + i * 1000, i) for i in range(10)]
    rows = [*reference, *rows]
    return ActivityIndex.from_dict(build_activity_manifest(
        rows, ["train"], reference=dict(name="common", manifest_sha256="a" * 64,
                                        excluded_sequences=["val"]),
        source_metadata_sha256=source_hashes or {r["sequence"]: "b" * 64 for r in rows},
        source_contract=source_contract,
    ))


class Mask:
    def __init__(self, count):
        self.count = count

    def sum(self):
        return self.count


def prepared_sums(count, absolute, squared):
    # Fixed sufficient statistics isolate routing/pooling from numpy arithmetic.
    return count, dict(mae=absolute, rmse=squared, abs_rel=absolute, sq_rel=squared,
                       rmse_log=squared, delta1=0, delta2=0, delta3=0)


def accumulate_fixture(meter, prediction, target, valid):
    count, sums = prediction
    if count:
        meter.add(count, **sums)


class DenseActivityTest(unittest.TestCase):
    def test_frame_mean_is_added_without_changing_pooled_scores_or_population(self):
        from event_state.dense.depth_frame_metrics import METRIC_NAMES, F3_FRAME_PROTOCOL

        rows = [record("val", 1000, 0), record("val", 2000, 1), record("val", 3000, 99)]
        options = dict(dataset="dsec", task="depth", expected_keys=[("val", r["timestamp"]) for r in rows],
                       condition_label="E1 Encoder＋LSTM（h蒸留）", accumulate_fn=accumulate_fixture)
        original = DenseActivityEvaluator(index_for(rows), **options)
        dual = DenseActivityEvaluator(index_for(rows), frame_metrics=True, **options)
        for row, prediction, error in zip(rows, [prepared_sums(1, 3, 9), prepared_sums(3, 3, 3),
                                                prepared_sums(0, 0, 0)], [3, 1, None]):
            kwargs = dict(sequence="val", timestamp=row["timestamp"], event_rate_hz=row["event_rate_hz"])
            original.update(prediction, None, Mask(prediction[0]), **kwargs)
            values = {name: (0.0 if name.startswith("d") else float(error))
                      for name in METRIC_NAMES} if error is not None else {}
            dual.update(prediction, None, Mask(prediction[0]),
                        frame_result=(prediction[0], values), **kwargs)
        old, new = original.compute(), dual.compute()
        self.assertEqual(new["activity_assignments_sha256"], old["activity_assignments_sha256"])
        self.assertEqual(new["depth_frame_metric_protocol"], F3_FRAME_PROTOCOL)
        for name, group in old["groups"].items():
            self.assertEqual(new["groups"][name]["metrics"], group["metrics"])
        low = new["groups"]["p00_20"]
        self.assertEqual(low["metrics"]["mae"], 1.5)
        self.assertEqual(low["metrics_frame_mean_f3"]["mae"], 2)
        self.assertEqual(low["metrics_frame_mean_f3"]["rmse"], 2)
        self.assertEqual(low["metrics_frame_mean_f3"]["frames"], 2)
        self.assertEqual(low["metrics_frame_mean_f3"]["valid_pixels"], 4)
        self.assertIsNone(new["groups"]["p80_100"]["metrics_frame_mean_f3"]["mae"])
        json.dumps(new, allow_nan=False)

    def test_m3ed_uses_its_own_dataset_contract_with_the_same_pooling(self):
        rows = [record("val", 1000, 0)]
        index = index_for(rows, source_contract=M3ED_SOURCE_CONTRACT)
        meter = DenseActivityEvaluator(index, dataset="m3ed", task="depth",
            expected_keys=[("val", 1000)], condition_label="E2 Encoder＋LSTM（z+h蒸留）",
            accumulate_fn=accumulate_fixture)
        meter.update(prepared_sums(2, 4, 8), None, Mask(2), sequence="val", timestamp=1000,
                     event_rate_hz=0)
        self.assertEqual(meter.compute()["groups"]["all"]["metrics"]["mae"], 2)
        with self.assertRaisesRegex(ValueError, "different dataset"):
            DenseActivityEvaluator(index, dataset="dsec", task="depth", expected_keys=[("val", 1000)],
                condition_label="E2 Encoder＋LSTM（z+h蒸留）", accumulate_fn=accumulate_fixture)

    def test_pools_pixel_sums_instead_of_mean_frame_errors(self):
        rows = [record("val", 1000, 0), record("val", 2000, 1), record("val", 3000, 99)]
        meter = DenseActivityEvaluator(index_for(rows), dataset="dsec", task="depth",
            expected_keys=[("val", r["timestamp"]) for r in rows],
            condition_label="E1 Encoder＋LSTM（h蒸留）", accumulate_fn=accumulate_fixture)
        values = [prepared_sums(1, 3, 9), prepared_sums(3, 3, 3), prepared_sums(0, 0, 0)]
        for row, prediction in zip(rows, values):
            meter.update(prediction, None, Mask(prediction[0]), sequence="val",
                         timestamp=row["timestamp"], event_rate_hz=row["event_rate_hz"])
        result = meter.compute()
        low = result["groups"]["p00_20"]
        self.assertEqual(low["frames"], 2)
        self.assertEqual(low["valid_pixels"], 4)
        self.assertAlmostEqual(low["metrics"]["mae"], 1.5)
        self.assertAlmostEqual(low["metrics"]["rmse"], math.sqrt(3))
        self.assertNotAlmostEqual(low["metrics"]["rmse"], 2.0)
        high = result["groups"]["p80_100"]
        self.assertEqual(high["status"], "no_valid_ground_truth")
        self.assertEqual(high["frames"], 1)
        self.assertIsNone(high["metrics"]["mae"])
        self.assertEqual(result["groups"]["p40_60"]["status"], "empty")
        json.dumps(result, allow_nan=False)

    def test_disparity_and_missing_duplicate_rate_checks(self):
        rows = [record("val", 1000, 0), record("val", 2000, 99)]
        meter = DenseActivityEvaluator(index_for(rows), dataset="dsec", task="disparity",
            expected_keys=[("val", 1000), ("val", 2000)],
            condition_label="E0 Encoder（z蒸留）", accumulate_fn=lambda m, p, t, v:
                m.add(1, mae=2, rmse=4, **{"1pe_percent": 100, "2pe_percent": 0}))
        with self.assertRaisesRegex(ValueError, "rate mismatch"):
            meter.update(None, None, Mask(1), sequence="val", timestamp=1000, event_rate_hz=1)
        meter.update(None, None, Mask(1), sequence="val", timestamp=1000, event_rate_hz=0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            meter.update(None, None, Mask(1), sequence="val", timestamp=1000, event_rate_hz=0)
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            meter.compute()
        meter.update(None, None, Mask(1), sequence="val", timestamp=2000, event_rate_hz=99000)
        self.assertEqual(meter.compute()["groups"]["all"]["metrics"]["1pe_percent"], 100)

    def test_source_metadata_byte_hash_and_canonical_identity_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            source = dict(dataset="DSEC", sequence_name="val", height=480)
            path.write_text(json.dumps(source, indent=2))
            hashes = {"train": "a" * 64, "val": canonical_sha256(source)}
            index = index_for([record("val", 1000, 0)], hashes)
            metadata = dict(records=[dict(sequence="val", timestamp=1000)], source_identity=dict(
                val=dict(path=str(path), metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            validate_dense_activity_sources(index, metadata)
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                validate_dense_activity_sources(index, metadata)
            metadata["source_identity"]["val"]["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_dense_activity_sources(index, metadata)
            source["height"] = 1
            path.write_text(json.dumps(source))
            metadata["source_identity"]["val"]["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                validate_dense_activity_sources(index, metadata)

    def test_fixed_window_source_and_cli_hidden_export_are_rejected_without_torch(self):
        index = index_for([record("val", 1000, 0)])
        with self.assertRaisesRegex(ValueError, "prepared full-RGB"):
            validate_dense_activity_sources(index, dict(records=[dict(sequence="val")],
                                                        source_identity=dict(val=dict(event_window_us=20_000))))
        result = subprocess.run([sys.executable, str(ROOT / "tools/evaluate_dense_probe.py"),
            "--checkpoint", "unused", "--cache", "unused", "--output", "unused", "--export",
            "--activity-manifest", "unused", "--condition-label", "E0 Encoder（z蒸留）"],
            capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hidden-test export", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
