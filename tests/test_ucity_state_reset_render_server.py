"""CPU fixture for the once-reset UCity Semantic video and per-frame CSV."""
import csv
import io

import numpy as np

from event_state.evaluation.comparison_semantic import SemanticBackend
from tools import render_ucity_state_reset as render
from tools.render_activity_comparison import frame_metric


class TinyBackend(SemanticBackend):
    frame_size = (1, 2)
    scored_frame_indices = {render.SEQUENCE: {2, 3, 4}}
    cfg = prepared_root = None

    def __init__(self):
        self.reset()

    def step(self, frame, predict=True):
        self.state = (self.state or 0) + 1
        if predict:
            return np.array([[self.state, 0]], dtype=np.uint8)

    def ground_truth(self, sequence, stamp):
        assert sequence == render.SEQUENCE
        if stamp < 50:
            return np.array([[stamp // 10, 255]], dtype=np.uint8)


def test_paired_canvases_matches_display_panels_and_csv_at_q1_boundaries(monkeypatch):
    scene = dict(sequence=render.SEQUENCE, start_us=20, end_us=60, timestamps_us=[20, 30, 40, 50],
                 reset_input_us=30, q1_start_us=20, q1_end_us=40)
    frames = [dict(sequence_name=render.SEQUENCE, frame_index=i, timestamp=10*i,
                   previous_timestamp=10*(i-1), events=i, event_count=1) for i in range(1, 6)]
    records = [dict(sequence=render.SEQUENCE, timestamp=f["timestamp"], previous_timestamp=f["previous_timestamp"],
                    event_count=1, window_us=10, event_rate_hz=100000, group="p00_20") for f in frames]
    monkeypatch.setattr("event_state.dense.source.prepared_frames", lambda *args: iter(frames))
    monkeypatch.setattr(render, "rgb_frame", lambda dataset, task, cfg, root, seq, stamp: stamp)
    monkeypatch.setattr(render, "event_frame", lambda events, dataset, task: events)
    monkeypatch.setattr(render, "activity_chart", lambda *args: object())
    phases = []

    def draw(task, rgb, events, target, valid, predictions, labels, scene, stamp, *args, extra_panel, **kwargs):
        assert rgb == stamp and events == stamp // 10
        assert labels == (render.CONDITION + "・状態保持", render.CONDITION + "・低活動開始時に1回リセット")
        assert "InternImage" in kwargs["target_label"]
        title, mask = extra_panel
        phases.append(title.split(" | ")[1])
        expected = [[[255 if stamp >= 30 else 0] * 3, [0] * 3]]
        np.testing.assert_array_equal(mask, expected)
        return stamp, [frame_metric(task, p, target, valid) for p in predictions]

    monkeypatch.setattr(render, "draw_frame", draw)
    stream = io.StringIO()
    assert list(render.paired_canvases(TinyBackend(), records, scene, [100000], None,
                                       csv.writer(stream))) == scene["timestamps_us"]
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert [int(r["source_timestamp_us"]) for r in rows] == scene["timestamps_us"]
    assert [r["reset_before_this_input"] for r in rows] == ["False", "True", "False", "False"]
    assert [r["valid_pixels"] for r in rows] == ["1", "1", "1", "0"]
    assert [r["continuous_frame_mIoU"] for r in rows] == ["1.0", "1.0", "1.0", ""]
    assert [r["reset_frame_mIoU"] for r in rows] == ["1.0", "0.0", "0.0", ""]
    assert [float(r["prediction_disagreement_fraction"]) for r in rows] == [0, .5, .5, .5]
    assert phases == ["介入前・共通の履歴", "この入力窓の直前にリセット",
                      "対象の低活動区間", "対象区間終了後・両方とも状態を継続"]
