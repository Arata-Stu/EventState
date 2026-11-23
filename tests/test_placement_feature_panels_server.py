"""CPU fixtures for the joint feature projection and stride-16 video crop."""
import json

import numpy as np
import pytest
import torch

from event_state.evaluation.placement_feature_panels import (
    FeatureCalibration, FeatureProjection, calibration_indices, draw_feature_frame, panel_size,
)
from event_state.evaluation.placement_features import BLOCKS, CONDITIONS


def feature_maps(frame):
    grid = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    return {key: {block: (grid + i + frame / 4 + block / 10).sin() for block in BLOCKS}
            for i, key in enumerate(CONDITIONS)}


def test_projected_colors_fixed_and_sampling_matched_across_sources():
    assert calibration_indices(3) == (0, 1, 2)
    assert calibration_indices(102, 4) == (0, 33, 67, 101)
    calibration = FeatureCalibration(tokens_per_frame=6)
    for i in calibration_indices(8, 4):
        calibration.add(feature_maps(i), i)
    projection = calibration.fit()
    saved = json.loads(json.dumps(projection.to_dict(), allow_nan=False))
    restored = FeatureProjection.from_dict(saved)
    for block in BLOCKS:
        samples = []
        for row in saved["sample_frames"]:
            pair = feature_maps(row["frame_index"])
            for key in CONDITIONS:
                samples.append(pair[key][block].reshape(-1, 4)[row["patch_indices"]])
        expected = torch.nn.functional.normalize(torch.cat(samples).double(), dim=-1).mean(0)
        torch.testing.assert_close(torch.tensor(saved["blocks"][str(block)]["mean"], dtype=torch.float64), expected)
    pair = feature_maps(3)
    keys = list(CONDITIONS)
    pair[keys[1]] = pair[keys[0]]  # Identical vectors must have identical colors in both rows.
    colors, replay = projection.render(pair), restored.render(pair)
    for block in BLOCKS:
        assert np.array_equal(colors[keys[0]][block], colors[keys[1]][block])
        assert np.array_equal(colors[keys[0]][block], replay[keys[0]][block])
        assert colors[keys[0]][block].shape == (3, 4, 3)
        assert colors[keys[0]][block].dtype == np.uint8
        assert colors[keys[0]][block].max() > colors[keys[0]][block].min()
    with pytest.raises(ValueError, match="source order"):
        calibration.add(pair, 6)


def test_nonfinite_feature_cannot_be_encoded_as_a_valid_color():
    pair = feature_maps(0)
    calibration = FeatureCalibration()
    calibration.add(pair, 0)
    projection = calibration.fit()
    pair[next(iter(CONDITIONS))][BLOCKS[0]][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        projection.render(pair)


def test_dsec_panels_show_half_of_last_patch_row_without_interpolation(monkeypatch):
    from PIL import ImageFont
    from event_state.evaluation import placement_feature_panels as panels

    monkeypatch.setattr(panels, "_fonts", lambda _: (ImageFont.load_default(), ImageFont.load_default()))
    grid = np.zeros((28, 40, 3), dtype=np.uint8)
    grid[-1, :, 0] = 139
    grid[:-1, :, 1] = 218
    pair = {key: {block: grid for block in BLOCKS} for key in CONDITIONS}
    events = np.full((440, 640, 3), 57, dtype=np.uint8)
    frame = draw_feature_frame(pair, dataset="dsec", sequence="test", timestamp_us=50000,
                               elapsed_s=.05, event_rate_hz=500000, events=events)
    assert frame.size == panel_size("dsec")
    pixels = np.asarray(frame)
    x = 320 + 37
    y = panels._TITLE_HEIGHT + panels._ROW_HEADER
    assert np.all(pixels[y:y+216, x] == [0, 218, 0])
    assert np.all(pixels[y+216:y+220, x] == [139, 0, 0])
    assert tuple(pixels[y+220+panels._ROW_HEADER+1, 37]) == (57, 57, 57)
