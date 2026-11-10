"""PyTorch semantic confusion-matrix integration; run in the server env."""

import json

import pytest
import torch

from event_state.segmentation.activity_evaluation import ActivitySemanticEvaluator


class Index:
    groups = ("p00_20", "p20_40", "p40_60", "p60_80", "p80_100")
    identity = dict(source_contract=dict(dataset="dsec"), reference=dict(sequences=["train"]))

    def lookup(self, sequence, timestamp):
        return dict(group="p00_20" if timestamp < 3 else "p80_100", event_rate_hz=float(timestamp))


def test_semantic_logits_use_pooled_confusion_and_keep_empty_groups_null():
    evaluator = ActivitySemanticEvaluator(Index(), dataset="dsec", expected_keys=[("val", i) for i in (1, 2, 3)],
        condition_label="E1 Encoder＋LSTM（h蒸留）", num_classes=2)
    for timestamp, labels, prediction in [(1, [[0, 255, 255]], [[0, 1, 1]]),
                                          (2, [[0, 0, 0]], [[1, 1, 1]]),
                                          (3, [[1, 1, 1]], [[1, 1, 1]])]:
        predicted = torch.tensor([prediction])
        logits = torch.nn.functional.one_hot(predicted, 2).permute(0, 3, 1, 2).float()
        evaluator.update(logits, torch.tensor([labels]), sequence_names=["val"], timestamps=[timestamp])
    result = evaluator.compute()
    groups = result["activity_evaluation"]["groups"]
    assert groups["p00_20"]["metrics"]["mIoU"] == pytest.approx(0.125)
    assert groups["p00_20"]["metrics"]["class_support"] == [4, 0]
    assert groups["p80_100"]["metrics"]["mIoU"] == pytest.approx(1.0)
    assert groups["p40_60"]["metrics"]["mIoU"] is None
    assert groups["all"]["valid_pixels"] == 7
    json.dumps(result, allow_nan=False)


def test_semantic_all_ignore_frame_remains_in_support_but_has_no_score():
    evaluator = ActivitySemanticEvaluator(Index(), dataset="dsec", expected_keys=[("val", 1)],
        condition_label="E0 Encoder（z蒸留）", num_classes=2)
    evaluator.update(torch.zeros((1, 2, 2), dtype=torch.int64), torch.full((1, 2, 2), 255),
                     sequence_names=["val"], timestamps=[1])
    group = evaluator.compute()["activity_evaluation"]["groups"]["p00_20"]
    assert group["frames"] == 1
    assert group["valid_pixels"] == 0
    assert group["metrics"]["mIoU"] is None
