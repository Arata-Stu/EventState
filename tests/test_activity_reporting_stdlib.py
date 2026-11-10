"""Cross-task activity identity and support checks without ML dependencies."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from event_state.activity_quantiles import ActivityIndex, build_activity_manifest, canonical_sha256
from event_state.activity_reporting import (
    ActivityPartition, load_activity_options, validate_activity_source, write_activity_csv,
)


def fixture():
    rows = []
    for sequence, counts in (("train", range(10)), ("eval", (0, 9))):
        for i, count in enumerate(counts):
            rows.append(dict(sequence=sequence, timestamp=(i + 1) * 1000,
                             previous_timestamp=i * 1000, window_us=1000,
                             event_count=count, event_rate_hz=count * 1000.0))
    metadata = {"train": {"name": "学習"}, "eval": {"name": "評価"}}
    payload = build_activity_manifest(rows, ["train"],
        reference=dict(role="train", manifest_sha256="a" * 64, excluded_sequences=["eval"]),
        source_metadata_sha256={key: canonical_sha256(value) for key, value in metadata.items()})
    return ActivityIndex.from_dict(payload), payload, metadata


class ActivityReportingTest(unittest.TestCase):
    def partition(self):
        index, _, _ = fixture()
        return ActivityPartition(index, dataset="dsec", expected_keys=[("eval", 1000), ("eval", 2000)],
                                 condition_label="E1 Encoder＋LSTM（h蒸留）")

    def metrics(self, partition):
        return {key: dict(mae=None, frames=0, valid_pixels=0) for key in ("all", *partition.groups)}

    def test_same_assignments_across_tasks_with_different_gt_support(self):
        semantic, depth = self.partition(), self.partition()
        for meter, counts in ((semantic, (10, 20)), (depth, (1, 0))):
            for timestamp, count in zip((1000, 2000), counts):
                meter.update("eval", timestamp, valid_pixels=count)
        s, d = semantic.report(self.metrics(semantic)), depth.report(self.metrics(depth))
        self.assertEqual(s["samples_sha256"], d["samples_sha256"])
        self.assertEqual(s["activity_assignments_sha256"], d["activity_assignments_sha256"])
        self.assertNotEqual(s["validity_support_sha256"], d["validity_support_sha256"])
        self.assertEqual(s["groups"]["all"]["valid_pixels"], 30)
        self.assertEqual(d["groups"]["p80_100"]["status"], "no_valid_ground_truth")
        self.assertEqual(d["groups"]["p20_40"]["status"], "empty")
        self.assertEqual(d["groups"]["p80_100"]["frames"], 1)
        self.assertEqual(d["groups"]["p80_100"]["valid_frames"], 0)

    def test_preserves_all_frames_and_checks_cached_rate(self):
        meter = self.partition()
        with self.assertRaisesRegex(ValueError, "event rate mismatch"):
            meter.update("eval", 1000, valid_pixels=1, event_rate_hz=1)
        meter.update("eval", 1000, valid_pixels=1, event_rate_hz=0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            meter.update("eval", 1000, valid_pixels=1)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            meter.update("eval", 999, valid_pixels=1)
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            meter.report(self.metrics(meter))

    def test_no_reference_train_overlap_and_no_dataset_mix(self):
        index, _, _ = fixture()
        options = dict(index=index, expected_keys=[("eval", 1000)],
                       condition_label="E0 Encoder（z蒸留）")
        with self.assertRaisesRegex(ValueError, "different dataset"):
            ActivityPartition(dataset="m3ed", **options)
        options["expected_keys"] = [("train", 1000)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            ActivityPartition(dataset="dsec", **options)

    def test_invalid_support_and_group_coverage(self):
        for value in (True, -1, .5):
            with self.assertRaisesRegex(ValueError, "valid_pixels"):
                self.partition().update("eval", 1000, valid_pixels=value)
        meter = self.partition()
        meter.update("eval", 1000, valid_pixels=0)
        meter.update("eval", 2000, valid_pixels=0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            meter.report({"all": {}})

    def test_source_canonical_hash_and_cli_opt_in(self):
        index, payload, metadata = fixture()
        validate_activity_source(index, "eval", metadata["eval"])
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            validate_activity_source(index, "eval", {"name": "別データ"})
        self.assertIsNone(load_activity_options(None, None, dataset="dsec"))
        with self.assertRaisesRegex(ValueError, "requires --activity-manifest"):
            load_activity_options(None, "E0 Encoder（z蒸留）", dataset="dsec")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activity.json"
            path.write_text(json.dumps(payload))
            actual = load_activity_options(path, "E0 Encoder（z蒸留）", dataset="dsec")
            self.assertEqual(actual.identity, index.identity)

    def test_csv_support_fields_do_not_collide_with_dense_metrics(self):
        meter = self.partition()
        meter.update("eval", 1000, valid_pixels=1)
        meter.update("eval", 2000, valid_pixels=2)
        metrics = self.metrics(meter)
        metrics["all"].update(mae=1, class_IoU=[.5, float("nan")])
        report = meter.report(metrics)
        self.assertEqual(report["groups"]["all"]["metrics"]["class_IoU"], [.5, None])
        json.dumps(report, allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.csv"
            output.write_text("original JSON")
            path = write_activity_csv(output, report)
            self.assertEqual(output.read_text(), "original JSON")
            with path.open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
                first = next(reader)
            self.assertEqual(first["valid_pixels"], "3")
            self.assertEqual(first["frames"], "2")
            self.assertEqual(first["condition"], "E1 Encoder＋LSTM（h蒸留）")

    def test_csv_keeps_f3_frame_metrics_separate_from_pooled_metrics(self):
        meter = self.partition()
        meter.update("eval", 1000, valid_pixels=1)
        meter.update("eval", 2000, valid_pixels=3)
        metrics = self.metrics(meter)
        metrics["all"]["mae"] = 1.5
        metrics["all"]["sq_rel"] = 4
        report = meter.report(metrics)
        for group in report["groups"].values():
            group["metrics_frame_mean_f3"] = dict(frames=0, mae=None, sq_rel=None)
        report["groups"]["all"]["metrics_frame_mean_f3"] = dict(frames=2, mae=2, sq_rel=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_activity_csv(Path(tmp) / "report.json", report)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["mae"], "1.5")
        self.assertEqual(rows[0]["f3_frame_mean_mae"], "2")
        self.assertEqual(rows[0]["sq_rel"], "4")
        self.assertEqual(rows[0]["f3_frame_mean_sq_rel"], "3")
        self.assertEqual(rows[0]["frames"], "2")
        self.assertEqual(rows[0]["f3_frame_mean_frames"], "2")
        self.assertEqual(rows[2]["f3_frame_mean_mae"], "")


if __name__ == "__main__":
    unittest.main()
