"""Routing/provenance checks runnable on Mac without scientific dependencies."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "detection_activity_under_test", ROOT / "event_state/detection/activity_evaluation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Index:
    groups = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")
    identity = {"manifest_sha256": "fixed-training-thresholds"}

    def __init__(self, records, source_metadata=None):
        self.records = records
        self.source_metadata_sha256 = {
            sequence: digest(metadata) for sequence, metadata in (source_metadata or {}).items()
        }

    def lookup(self, sequence, timestamp):
        try:
            return self.records[sequence, timestamp]
        except KeyError as error:
            raise ValueError("Missing activity frame") from error


class RecordingEvaluator:
    """Records complete input frames; it deliberately does not calculate AP."""

    class_names = ("car", "pedestrian")

    def __init__(self):
        self.samples = []
        self.compute_calls = 0

    def update(self, predictions, targets, *, sequence_names, timestamps, image_size):
        self.samples.extend(zip(sequence_names, timestamps, predictions, targets))

    def compute(self):
        self.compute_calls += 1
        objects = sum(len(target["labels"]) for _, _, _, target in self.samples)
        detections = sum(len(prediction["scores"]) for _, _, prediction, _ in self.samples)
        # A non-linear aggregate makes accidentally averaging frame results observable.
        score = objects / (objects + detections) if detections else 0.0
        return dict(mAP=score, AP50=score, AP75=score, AP_small=-1.0,
                    AP_medium=float("nan"), AP_large=score, AP_car=score,
                    AP_pedestrian=float("nan"))


def target(*labels):
    return {"boxes": [[10, 10, 50, 50] for _ in labels], "labels": list(labels)}


def prediction(count=1):
    return {"boxes": [[10, 10, 50, 50]] * count, "labels": [0] * count,
            "scores": [0.9] * count}


def make_evaluator(records, keys=None):
    return MODULE.ActivityDetectionEvaluator(
        Index(records), expected_keys=list(records) if keys is None else keys,
        condition_label="E1 Encoder＋LSTM（h蒸留）", evaluator_factory=RecordingEvaluator,
    )


def update(evaluator, keys, targets, predictions=None):
    evaluator.update(
        predictions if predictions is not None else [prediction() for _ in keys], targets,
        sequence_names=[key[0] for key in keys], timestamps=[key[1] for key in keys],
        image_size=(430, 640),
    )


class ActivityDetectionTest(unittest.TestCase):
    def setUp(self):
        self.keys = [("a", 100), ("b", 200), ("a", 300), ("c", 400)]
        self.records = {
            self.keys[0]: dict(group="p00_20", event_rate_hz=1.0),
            self.keys[1]: dict(group="p80_100", event_rate_hz=100.0),
            self.keys[2]: dict(group="p00_20", event_rate_hz=2.0),
            self.keys[3]: dict(group="p20_40", event_rate_hz=4.0),
        }

    def test_routes_whole_frames_including_false_positives_on_empty_gt(self):
        evaluator = make_evaluator(self.records)
        update(evaluator, self.keys, [target(0), target(1), target(), target()],
               [prediction(1), prediction(3), prediction(4), prediction(2)])
        result = evaluator.compute()
        groups = result["activity_evaluation"]["groups"]
        low = groups["p00_20"]
        self.assertEqual([(row[0], row[1]) for row in evaluator.evaluators["p00_20"].samples],
                         [self.keys[0], self.keys[2]])
        self.assertEqual(low["frames"], 2)
        self.assertEqual(low["gt_objects"], 1)
        self.assertEqual(low["detections"], 5)
        self.assertEqual(low["sequence_frames"], {"a": 2})
        self.assertEqual(low["gt_objects_per_class"], {"car": 1, "pedestrian": 0})
        self.assertEqual((low["event_rate_hz_min"], low["event_rate_hz_max"]), (1.0, 2.0))
        self.assertAlmostEqual(low["metrics"]["mAP"], 1 / 6)
        self.assertEqual(evaluator.evaluators["all"].compute_calls, 1)
        self.assertEqual(evaluator.evaluators["p00_20"].compute_calls, 1)
        self.assertEqual(groups["all"]["frames"], sum(
            group["frames"] for name, group in groups.items() if name != "all"
        ))
        self.assertAlmostEqual(sum(group["evaluated_frame_fraction"]
                                   for name, group in groups.items() if name != "all"), 1.0)

    def test_empty_and_gt_free_groups_are_null_with_reason_and_stable_schema(self):
        evaluator = make_evaluator(self.records)
        update(evaluator, self.keys, [target(0), target(1), target(), target()])
        result = evaluator.compute()
        groups = result["activity_evaluation"]["groups"]
        self.assertEqual(groups["p40_60"]["status"], "empty")
        self.assertEqual(groups["p20_40"]["status"], "no_ground_truth")
        for name in ("p40_60", "p20_40"):
            self.assertTrue(all(value is None for value in groups[name]["metrics"].values()))
            self.assertEqual(evaluator.evaluators[name].compute_calls, 0)
        self.assertIsNone(groups["p00_20"]["metrics"]["AP_pedestrian"])
        self.assertIsNone(groups["all"]["metrics"]["AP_small"])
        self.assertIsNone(groups["all"]["metrics"]["AP_medium"])
        json.dumps(result, allow_nan=False)

    def test_zero_predictions_are_zero_only_for_supported_classes(self):
        key = self.keys[0]
        evaluator = make_evaluator({key: self.records[key]})
        update(evaluator, [key], [target(0)], [prediction(0)])
        metrics = evaluator.compute()["activity_evaluation"]["groups"]["p00_20"]["metrics"]
        self.assertEqual(metrics["mAP"], 0.0)
        self.assertEqual(metrics["AP_car"], 0.0)
        self.assertIsNone(metrics["AP_pedestrian"])
        self.assertIsNone(metrics["AP_small"])
        self.assertIsNone(metrics["AP_large"])

    def test_rejects_duplicate_missing_and_unexpected_frame_identities(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            make_evaluator(self.records, [self.keys[0], self.keys[0]])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            make_evaluator(self.records, [])
        with self.assertRaisesRegex(ValueError, "Missing activity"):
            make_evaluator(self.records, [("absent", 1)])
        evaluator = make_evaluator(self.records)
        update(evaluator, [self.keys[0]], [target(0)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            update(evaluator, [self.keys[0]], [target(0)])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            update(evaluator, [("absent", 1)], [target(0)])
        with self.assertRaisesRegex(ValueError, "Missing 3"):
            evaluator.compute()

    def test_requires_label_and_consistent_batch_fields(self):
        with self.assertRaisesRegex(ValueError, "condition-label"):
            MODULE.ActivityDetectionEvaluator(
                Index(self.records), expected_keys=self.keys, condition_label=" ",
                evaluator_factory=RecordingEvaluator,
            )
        evaluator = make_evaluator(self.records)
        with self.assertRaisesRegex(ValueError, "different lengths"):
            evaluator.update([], [target()], sequence_names=[], timestamps=[], image_size=(10, 10))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            update(evaluator, [self.keys[0]] * 2, [target(0)] * 2)

    def test_population_hash_is_order_and_prediction_independent_but_gt_sensitive(self):
        key_a, key_b = self.keys[:2]
        records = {key: self.records[key] for key in (key_a, key_b)}
        first = make_evaluator(records)
        update(first, [key_a, key_b], [target(0), target(1)])
        second = make_evaluator(records)
        update(second, [key_b, key_a], [target(1), target(0)], [prediction(4), prediction(0)])
        one, two = first.compute()["activity_evaluation"], second.compute()["activity_evaluation"]
        self.assertEqual(one["samples_targets_sha256"], two["samples_targets_sha256"])
        self.assertEqual(one["activity_assignments_sha256"], two["activity_assignments_sha256"])
        third = make_evaluator(records)
        update(third, [key_a, key_b], [target(1), target(1)])
        self.assertNotEqual(one["samples_targets_sha256"],
                            third.compute()["activity_evaluation"]["samples_targets_sha256"])

    def test_csv_preserves_condition_group_support_and_nullable_metrics(self):
        evaluator = make_evaluator(self.records)
        update(evaluator, self.keys, [target(0), target(1), target(), target()])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.json"
            MODULE.write_activity_csv(output, evaluator.compute())
            with output.with_suffix(".csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["condition"] == "E1 Encoder＋LSTM（h蒸留）" for row in rows))
        self.assertEqual(rows[1]["group"], "p00_20")
        empty = next(row for row in rows if row["group"] == "p40_60")
        self.assertEqual(empty["frames"], "0")
        self.assertEqual(empty["mAP"], "")

    def test_csv_output_does_not_overwrite_json_with_csv_suffix_or_alias(self):
        evaluator = make_evaluator(self.records)
        update(evaluator, self.keys, [target(0), target(1), target(), target()])
        result = evaluator.compute()
        encoded = json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, alias in (("metrics.csv", False), ("metrics.json", True)):
                with self.subTest(filename=filename):
                    output = root / filename
                    output.write_text(encoded)
                    if alias:
                        output.with_suffix(".csv").unlink()
                        output.with_suffix(".csv").symlink_to(output.name)
                    MODULE.write_activity_csv(output, result)
                    self.assertEqual(output.read_text(), encoded)
                    fallback = output.with_name(output.name + ".groups.csv")
                    with fallback.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 6)
                    self.assertEqual(rows[0]["group"], "all")
                    if alias:
                        self.assertTrue(output.with_suffix(".csv").is_symlink())
                        self.assertEqual(output.with_suffix(".csv").read_text(), encoded)

    def test_source_validation_matches_canonical_metadata_and_preserves_frame_policy(self):
        source = dict(sequence_name="a", dataset="DSEC", event_window="rgb_interval",
                      coordinate_space="rectified_event", height=480, width=640, frame_count=1)
        index = Index({}, {"a": source})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "a"
            directory.mkdir()
            (directory / "100.pt").touch()
            (directory / "_SUCCESS").touch()
            feature_meta = dict(source_event_cache_metadata=dict(reversed(list(source.items()))),
                                state_policy="frame", event_window_us=None, frame_count=1)
            path = directory / "metadata.json"
            path.write_text(json.dumps(feature_meta, indent=4))
            result = MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            self.assertEqual(result, {"a": "frame"})
            feature_meta["source_event_cache_metadata"]["width"] = 320
            path.write_text(json.dumps(feature_meta))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            feature_meta["event_window_us"] = 20_000
            path.write_text(json.dumps(feature_meta))
            with self.assertRaisesRegex(ValueError, "RGB-interval"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            path.write_text(json.dumps({"state_policy": "continuous"}))
            with self.assertRaisesRegex(ValueError, "lacks source"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            path.write_text(json.dumps(source))
            self.assertEqual(MODULE.validate_activity_sources(index, sequences=["a"], event_cache_dir=tmp),
                             {"a": "continuous_reset_at_sequence_boundary"})

    def test_source_validation_rejects_missing_numeric_frames_and_invalid_counts(self):
        source = dict(sequence_name="a", dataset="DSEC", event_window="rgb_interval",
                      coordinate_space="rectified_event", height=480, width=640, frame_count=1)
        index = Index({}, {"a": source})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "a"
            directory.mkdir()
            path = directory / "metadata.json"
            feature_meta = dict(source_event_cache_metadata=source, state_policy="continuous",
                                event_window_us=None, frame_count=1)
            path.write_text(json.dumps(feature_meta))
            (directory / "diagnostics.pt").touch()
            # A non-frame .pt sidecar must not satisfy numeric frame coverage.
            with self.assertRaisesRegex(ValueError, "Incomplete activity evaluation cache"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            (directory / "100.pt").touch()
            self.assertEqual(MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp),
                             {"a": "continuous"})
            for count in (None, True, 0, -1, 1.0, 2):
                with self.subTest(frame_count=count):
                    feature_meta["frame_count"] = count
                    path.write_text(json.dumps(feature_meta))
                    with self.assertRaisesRegex(ValueError, "Incomplete activity evaluation cache"):
                        MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            feature_meta["frame_count"] = 1
            path.write_text(json.dumps(feature_meta))
            (directory / "200.pt").touch()
            with self.assertRaisesRegex(ValueError, "Incomplete activity evaluation cache"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)
            (directory / "200.pt").unlink()
            # Matching manifest hash alone is insufficient if source and feature counts differ.
            source["frame_count"] = 2
            index = Index({}, {"a": source})
            path.write_text(json.dumps(feature_meta))
            with self.assertRaisesRegex(ValueError, "Incomplete activity evaluation cache"):
                MODULE.validate_activity_sources(index, sequences=["a"], feature_cache_dir=tmp)

    def test_event_source_requires_success_marker_even_with_matching_metadata(self):
        source = dict(sequence_name="a", frame_count=1)
        index = Index({}, {"a": source})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "a"
            directory.mkdir()
            (directory / "metadata.json").write_text(json.dumps(source))
            (directory / "100.pt").touch()
            with self.assertRaisesRegex(ValueError, "Incomplete prepared event source"):
                MODULE.validate_activity_sources(index, sequences=["a"], event_cache_dir=tmp)
            (directory / "_SUCCESS").touch()
            self.assertEqual(MODULE.validate_activity_sources(index, sequences=["a"], event_cache_dir=tmp),
                             {"a": "continuous_reset_at_sequence_boundary"})

    def test_activity_option_is_opt_in_without_loading_scientific_modules(self):
        args = argparse.Namespace(activity_manifest=None, condition_label=None)
        self.assertEqual(MODULE.activity_evaluator(args, expected_keys=[], sequences=[]), (None, None))
        args.condition_label = "E0 Encoder（z蒸留）"
        with self.assertRaisesRegex(ValueError, "requires --activity-manifest"):
            MODULE.activity_evaluator(args, expected_keys=[], sequences=[])


if __name__ == "__main__":
    unittest.main()
