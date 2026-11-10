"""Semantic activity routing/provenance checks without an ML runtime."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from event_state.activity_quantiles import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_activity_under_test", ROOT / "event_state/segmentation/activity_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Index:
    groups = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")

    def __init__(self, records, *, dataset="dsec", sources=None):
        self.records = records
        self.identity = dict(source_contract=dict(dataset=dataset), reference=dict(sequences=["train"]),
                             manifest_sha256="synthetic-manifest")
        self.source_metadata_sha256 = {
            seq: canonical_sha256(source) for seq, source in (sources or {}).items()}

    def lookup(self, sequence, timestamp):
        try:
            return self.records[sequence, timestamp]
        except KeyError as error:
            raise ValueError("Missing semantic activity frame") from error

    def timestamps(self, sequence):
        return tuple(sorted(timestamp for seq, timestamp in self.records if seq == sequence))


class Meter:
    def __init__(self):
        self.confusion = [[0, 0], [0, 0]]

    def update(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            for prow, trow in zip(prediction, target):
                for p, t in zip(prow, trow):
                    if 0 <= t < 2:
                        self.confusion[t][p] += 1

    def compute(self):
        matrix = self.confusion
        support = [sum(row) for row in matrix]
        total = sum(support)
        ious = []
        for i in range(2):
            union = support[i] + sum(row[i] for row in matrix) - matrix[i][i]
            ious.append(matrix[i][i] / union if union else float("nan"))
        valid = [value for value in ious if math.isfinite(value)]
        return dict(mIoU=sum(valid) / len(valid) if valid else float("nan"),
                    class_IoU=ious, evaluated_pixels=total,
                    pixel_accuracy=sum(matrix[i][i] for i in range(2)) / total if total else 0)


def make(records, keys=None):
    return MODULE.ActivitySemanticEvaluator(
        Index(records), dataset="dsec", expected_keys=list(records) if keys is None else keys,
        condition_label="E1 Encoder＋LSTM（h蒸留）", num_classes=2, evaluator_factory=Meter)


class SemanticActivityTest(unittest.TestCase):
    def test_confusion_is_pooled_not_average_frame_iou(self):
        records = {("val", 1): dict(group="p00_20", event_rate_hz=1),
                   ("val", 2): dict(group="p00_20", event_rate_hz=2),
                   ("val", 3): dict(group="p80_100", event_rate_hz=8)}
        evaluator = make(records)
        evaluator.update([[[0]], [[1] * 9], [[1]]], [[[0]], [[0] * 9], [[1]]],
                         sequence_names=["val"] * 3, timestamps=[1, 2, 3])
        report = evaluator.compute()
        low = report["activity_evaluation"]["groups"]["p00_20"]
        self.assertAlmostEqual(low["metrics"]["mIoU"], 0.05)
        self.assertNotEqual(low["metrics"]["mIoU"], 0.5)
        self.assertEqual(low["valid_pixels"], 10)
        self.assertEqual(low["metrics"]["class_support"], [10, 0])
        self.assertEqual(report["evaluated_pixels"], 11)
        self.assertEqual(report["activity_evaluation"]["groups"]["p40_60"]["status"], "empty")
        self.assertIsNone(report["activity_evaluation"]["groups"]["p40_60"]["metrics"]["mIoU"])
        self.assertIsNone(report["activity_evaluation"]["groups"]["p40_60"]["metrics"]["pixel_accuracy"])
        json.dumps(report, allow_nan=False)

    def test_ignored_pixels_and_absent_classes_are_reported(self):
        records = {("val", 1): dict(group="p00_20", event_rate_hz=1),
                   ("val", 2): dict(group="p80_100", event_rate_hz=8)}
        evaluator = make(records)
        evaluator.update([[[0, 1]], [[1]]], [[[0, 255]], [[255]]],
                         sequence_names=["val", "val"], timestamps=[1, 2])
        groups = evaluator.compute()["activity_evaluation"]["groups"]
        self.assertEqual(groups["p00_20"]["metrics"]["class_IoU"], [1.0, None])
        self.assertEqual(groups["p80_100"]["status"], "no_valid_ground_truth")
        self.assertEqual(groups["p80_100"]["frames"], 1)
        self.assertEqual(groups["p80_100"]["valid_pixels"], 0)

    def test_missing_duplicate_and_malformed_batches_fail(self):
        records = {("val", 1): dict(group="p00_20", event_rate_hz=1)}
        evaluator = make(records)
        with self.assertRaisesRegex(ValueError, "Missing"):
            evaluator.compute()
        with self.assertRaisesRegex(ValueError, "different lengths"):
            evaluator.update([], [[[0]]], sequence_names=[], timestamps=[])
        evaluator.update([[[0]]], [[[0]]], sequence_names=["val"], timestamps=[1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluator.update([[[0]]], [[[0]]], sequence_names=["val"], timestamps=[1])

    def test_target_digest_is_order_independent_and_gt_sensitive(self):
        records = {("val", 1): dict(group="p00_20", event_rate_hz=1),
                   ("val", 2): dict(group="p80_100", event_rate_hz=8)}
        digests = []
        for times, targets in [([1, 2], [[[0]], [[1]]]), ([2, 1], [[[1]], [[0]]]),
                               ([1, 2], [[[1]], [[1]]])]:
            evaluator = make(records)
            evaluator.update([[[0]], [[1]]], targets, sequence_names=["val", "val"], timestamps=times)
            digests.append(evaluator.compute()["activity_evaluation"]["samples_targets_sha256"])
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])

    def test_sparse_dsec_source_validation_preserves_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "val"
            directory.mkdir()
            feature = directory / "000001.pt"
            feature.touch()
            source = dict(sequence_name="val", frame_count=100, event_window="rgb_interval")
            metadata = dict(cached_frame_indices=[1], source_event_cache_metadata=source,
                            state_policy="frame", checkpoint_sha256="head-source")
            (directory / "metadata.json").write_text(json.dumps(metadata))
            payload = dict(sequence_name="val", frame_index=1, timestamp=1000,
                           state_policy="frame", checkpoint_sha256="head-source")
            index = Index({("val", 1000): dict(group="p00_20", event_rate_hz=1)}, sources={"val": source})
            keys, states = MODULE.validate_semantic_cache(index, dataset="dsec", feature_cache_dir=tmp,
                sequences=["val"], samples=[("val", 1, feature, "label")], load_payload=lambda _: payload)
            self.assertEqual(keys, [("val", 1000)])
            self.assertEqual(states, {"val": "frame"})
            # Sparse semantic cache does not need all 100 source event frames.
            feature.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                MODULE.validate_semantic_cache(index, dataset="dsec", feature_cache_dir=tmp,
                    sequences=["val"], samples=[("val", 1, feature, "label")], load_payload=lambda _: payload)

    def test_m3ed_legacy_common_cache_needs_matching_prepared_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "cache" / "val"
            prepared = root / "prepared" / "val"
            directory.mkdir(parents=True)
            prepared.mkdir(parents=True)
            source = dict(sequence_name="val", timestamps=[0, 1000], frame_count=2)
            raw = json.dumps(source).encode()
            (prepared / "metadata.json").write_bytes(raw)
            feature = directory / "000001.pt"
            feature.touch()
            metadata = dict(cached_frame_indices=[1], state_policy="continuous", checkpoint_sha256="source")
            (directory / "metadata.json").write_text(json.dumps(metadata))
            identity = dict(sources=dict(val=[hashlib.sha256(raw).hexdigest(), "target-hash"]))
            identity_path = root / "cache" / "identity.json"
            identity_path.write_text(json.dumps(identity))
            payload = dict(sequence_name="val", frame_index=1, timestamp=1000,
                           state_policy="continuous", checkpoint_sha256="source")
            index = Index({("val", 1000): dict(group="p00_20", event_rate_hz=1)}, dataset="m3ed",
                          sources={"val": source})
            kwargs = dict(dataset="m3ed", feature_cache_dir=root / "cache", sequences=["val"],
                          samples=[("val", 1, feature, "target")], load_payload=lambda _: payload)
            with self.assertRaisesRegex(ValueError, "prepared-root"):
                MODULE.validate_semantic_cache(index, **kwargs)
            keys, _ = MODULE.validate_semantic_cache(index, prepared_root=root / "prepared", **kwargs)
            self.assertEqual(keys, [("val", 1000)])
            identity_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "provenance"):
                MODULE.validate_semantic_cache(index, prepared_root=root / "prepared", **kwargs)

    def test_dsec_rejects_wrong_but_existing_activity_timestamp_for_frame_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "val"
            directory.mkdir()
            feature = directory / "000001.pt"
            feature.touch()
            source = dict(sequence_name="val", frame_count=2, event_window="rgb_interval")
            (directory / "metadata.json").write_text(json.dumps(dict(cached_frame_indices=[1],
                source_event_cache_metadata=source, state_policy="continuous", checkpoint_sha256="source")))
            index = Index({("val", timestamp): dict(group="p00_20", event_rate_hz=1)
                           for timestamp in (1000, 2000)}, sources={"val": source})
            payload = dict(sequence_name="val", frame_index=1, timestamp=2000,
                           state_policy="continuous", checkpoint_sha256="source")
            with self.assertRaisesRegex(ValueError, "frame-index/timestamp"):
                MODULE.validate_semantic_cache(index, dataset="dsec", feature_cache_dir=tmp,
                    sequences=["val"], samples=[("val", 1, feature, "label")], load_payload=lambda _: payload)


if __name__ == "__main__":
    unittest.main()
