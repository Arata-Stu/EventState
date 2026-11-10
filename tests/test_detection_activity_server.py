"""COCO/PyTorch integration checks; execute in the server detection environment."""

import json

import pytest
import torch

from event_state.detection.activity_evaluation import ActivityDetectionEvaluator

pytest.importorskip("pycocotools")


class Index:
    groups = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")
    identity = {"manifest_sha256": "synthetic-server-fixture"}

    def __init__(self, records):
        self.records = records

    def lookup(self, sequence, timestamp):
        return self.records[sequence, timestamp]


def target(boxes=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def prediction(boxes=(), scores=(), labels=()):
    return {**target(boxes, labels), "scores": torch.tensor(scores, dtype=torch.float32)}


def evaluator(records):
    return ActivityDetectionEvaluator(Index(records), expected_keys=list(records),
                                      condition_label="E1 Encoder＋LSTM（h蒸留）")


def update(meter, records, targets, predictions):
    keys = list(records)
    meter.update(predictions, targets, sequence_names=[key[0] for key in keys],
                 timestamps=[key[1] for key in keys], image_size=(430, 640))


def test_coco_ap_is_recomputed_for_each_activity_partition():
    records = {("sequence", 100): dict(group="p00_20", event_rate_hz=1.0),
               ("sequence", 200): dict(group="p80_100", event_rate_hz=100.0)}
    box = [10, 20, 60, 90]
    meter = evaluator(records)
    update(meter, records, [target([box], [0]), target([box], [0])],
           [prediction([box], [0.9], [0]), prediction([[300, 300, 350, 370]], [0.9], [0])])
    result = meter.compute()
    groups = result["activity_evaluation"]["groups"]
    assert groups["p00_20"]["metrics"]["mAP"] == pytest.approx(1.0)
    assert groups["p80_100"]["metrics"]["mAP"] == pytest.approx(0.0)
    assert 0.0 < result["mAP"] < 1.0
    assert groups["p40_60"]["status"] == "empty"
    assert groups["p40_60"]["metrics"]["mAP"] is None
    assert groups["p00_20"]["metrics"]["AP_pedestrian"] is None
    json.dumps(result, allow_nan=False)


def test_false_positive_on_gt_empty_frame_remains_in_group_ap():
    records = {("sequence", 100): dict(group="p00_20", event_rate_hz=1.0),
               ("sequence", 200): dict(group="p00_20", event_rate_hz=2.0),
               ("sequence", 300): dict(group="p80_100", event_rate_hz=100.0)}
    box = [10, 20, 60, 90]
    meter = evaluator(records)
    update(meter, records, [target([box], [0]), target(), target()],
           [prediction([box], [0.8], [0]), prediction([box], [0.99], [0]),
            prediction([box], [0.95], [0])])
    result = meter.compute()
    low = result["activity_evaluation"]["groups"]["p00_20"]
    assert low["frames"] == 2
    assert low["gt_objects"] == 1
    assert low["detections"] == 2
    assert low["metrics"]["mAP"] == pytest.approx(0.5)
    # A no-GT partition is undefined rather than a perfect zero-object task.
    high = result["activity_evaluation"]["groups"]["p80_100"]
    assert high["status"] == "no_ground_truth"
    assert high["detections"] == 1
    assert high["metrics"]["mAP"] is None
    json.dumps(result, allow_nan=False)


def test_no_predictions_has_zero_ap_for_supported_classes_only():
    records = {("sequence", 100): dict(group="p00_20", event_rate_hz=1.0)}
    meter = evaluator(records)
    update(meter, records, [target([[10, 20, 60, 90]], [0])], [prediction()])
    result = meter.compute()
    assert result["mAP"] == 0.0
    assert result["AP_car"] == 0.0
    assert result["AP_pedestrian"] is None
    assert result["AP_small"] is None
    assert result["AP_medium"] == 0.0
    assert result["AP_large"] is None
    json.dumps(result, allow_nan=False)
