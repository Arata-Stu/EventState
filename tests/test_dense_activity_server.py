"""PyTorch/numpy depth activity integration; execute in the server environment."""

import json
import hashlib
import math

import numpy as np
import pytest

from event_state.activity_quantiles import ActivityIndex, build_activity_manifest, canonical_sha256
from event_state.dense.activity_evaluation import DenseActivityEvaluator


def index_for(rows):
    train = [dict(sequence="train", timestamp=1000 * (i + 1), previous_timestamp=1000 * i,
                  window_us=1000, event_count=i, event_rate_hz=i * 1000.0) for i in range(10)]
    return ActivityIndex.from_dict(build_activity_manifest([*train, *rows], ["train"],
        reference=dict(name="common", manifest_sha256="a" * 64, excluded_sequences=["val"]),
        source_metadata_sha256={"train": "b" * 64, "val": "c" * 64}))


def row(stamp, count):
    return dict(sequence="val", timestamp=stamp, previous_timestamp=stamp - 1000,
                window_us=1000, event_count=count, event_rate_hz=count * 1000.0)


def test_real_numpy_accumulator_pools_each_partition_and_preserves_empty_validity():
    rows = [row(1000, 0), row(2000, 1), row(3000, 99)]
    meter = DenseActivityEvaluator(index_for(rows), dataset="dsec", task="depth",
        expected_keys=[("val", r["timestamp"]) for r in rows],
        condition_label="E1 Encoder＋LSTM（h蒸留）")
    examples = [(np.array([[4.0]]), np.ones((1, 1)), np.ones((1, 1), dtype=bool)),
                (np.full((1, 3), 2.0), np.ones((1, 3)), np.ones((1, 3), dtype=bool)),
                (np.array([[100.0]]), np.ones((1, 1)), np.zeros((1, 1), dtype=bool))]
    for record, (prediction, target, valid) in zip(rows, examples):
        meter.update(prediction, target, valid, sequence="val", timestamp=record["timestamp"],
                     event_rate_hz=record["event_rate_hz"])
    report = meter.compute()
    low = report["groups"]["p00_20"]
    assert low["valid_pixels"] == 4
    assert low["metrics"]["mae"] == pytest.approx(1.5)
    assert low["metrics"]["rmse"] == pytest.approx(math.sqrt(3))
    assert low["metrics"]["abs_rel"] == pytest.approx(1.5)
    assert report["groups"]["p80_100"]["metrics"]["mae"] is None
    json.dumps(report, allow_nan=False)


def test_invalid_prediction_on_valid_gt_is_not_silently_discarded():
    record = row(1000, 0)
    meter = DenseActivityEvaluator(index_for([record]), dataset="dsec", task="depth",
        expected_keys=[("val", 1000)], condition_label="E0 Encoder（z蒸留）")
    with pytest.raises(ValueError, match="Nonfinite/nonpositive"):
        meter.update(np.array([[float("nan")]]), np.ones((1, 1)), np.ones((1, 1), dtype=bool),
                     sequence="val", timestamp=1000, event_rate_hz=0)


def test_runtime_optional_activity_preserves_existing_overall_and_low_groups(tmp_path):
    import torch
    from event_state.dense.runtime import evaluate

    records = [row(1000, 0), row(2000, 99)]
    source = dict(dataset="DSEC", sequence_name="val", height=480, width=640)
    path = tmp_path/'metadata.json'
    path.write_text(json.dumps(source))
    train = [dict(sequence="train", timestamp=1000 * (i + 1), previous_timestamp=1000 * i,
                  window_us=1000, event_count=i, event_rate_hz=i * 1000.0) for i in range(10)]
    index = ActivityIndex.from_dict(build_activity_manifest([*train, *records], ["train"],
        reference=dict(name="common", manifest_sha256="a" * 64, excluded_sequences=["val"]),
        source_metadata_sha256={"train": "b" * 64, "val": canonical_sha256(source)}))

    class Dataset:
        require_labels = True
        feature = 'z'
        def __init__(self):
            self.records = records
            self.meta = dict(dataset='dsec', task='depth', records=records, source_identity=dict(
                val=dict(path=str(path), metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        def __len__(self):
            return len(records)
        def __getitem__(self, i):
            return dict(feature=torch.full((1, 1, 2), float(i + 2)), target=torch.ones((1, 2)),
                valid=torch.ones((1, 2), dtype=torch.bool), sequence='val', timestamp=records[i]['timestamp'],
                low_activity=i == 0, low_duration_us=600000 if i == 0 else 0,
                activity_fraction=0.05, event_rate_hz=records[i]['event_rate_hz'])

    class Model(torch.nn.Module):
        def forward(self, x, size):
            return x[:, 0]

    dataset = Dataset()
    default = evaluate(Model(), dataset, torch.device('cpu'))
    grouped = evaluate(Model(), dataset, torch.device('cpu'), activity_index=index,
                       condition_label='E1 Encoder＋LSTM（h蒸留）')
    assert default['metrics'] == grouped['metrics']
    assert default['samples_targets_sha256'] == grouped['samples_targets_sha256']
    assert default['validity_sha256'] == grouped['validity_sha256']
    assert grouped['activity_evaluation']['groups']['all']['metrics'] == default['metrics']['all']
    assert 'activity_evaluation' not in default
    dual = evaluate(Model(), dataset, torch.device('cpu'), activity_index=index,
                    condition_label='E1 Encoder＋LSTM（h蒸留）', include_frame_metrics=True)
    assert dual['metrics'] == default['metrics']
    assert dual['frames'] == default['frames']
    assert dual['samples_targets_sha256'] == default['samples_targets_sha256']
    assert dual['validity_sha256'] == default['validity_sha256']
    assert dual['metrics_frame_mean_f3']['all']['mae'] == pytest.approx(1.5)
    assert dual['metrics_frame_mean_f3']['all']['rmse'] == pytest.approx(1.5)
    assert dual['metrics_frame_mean_f3']['all']['rmse'] != pytest.approx(default['metrics']['all']['rmse'])
    assert dual['activity_evaluation']['groups']['all']['metrics_frame_mean_f3'] == dual['metrics_frame_mean_f3']['all']
    assert dual['depth_frame_metric_protocol']['minimum_valid_pixels'] == 1
