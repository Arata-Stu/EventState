"""Numerical parity checks; run only in the server NumPy/PyTorch environment."""

import math

import numpy as np
import pytest
import torch

from event_state.dense.depth_frame_metrics import FrameMeanSums, compute_frame_metrics


def fixed_f3_eval_depth(prediction, target):
    """Reference equations from pinned F3 eval_depth, with original key names.

    Source: grasp-lyrl/fast-feature-fields @ 833e2c65233f9787449ce0675ae41e3f6e9c193a,
    src/f3/tasks/depth/utils/utils.py, SHA256
    7e66611dd25c61c85b56f5a00786ac9b9fd208def2c95c3e5257e190de5b54f3.
    """
    ratio = torch.maximum(target / prediction, prediction / target)
    difference = prediction - target
    log_error = torch.log(prediction) - torch.log(target)
    return dict(
        **{f"d{i}": (ratio < 1.25 ** i).sum().float().item() / len(ratio) for i in (1, 2, 3)},
        abs_rel=(difference.abs() / target).mean().item(),
        sq_rel=(difference.square() / target.square()).mean().item(),
        rmse=difference.square().mean().sqrt().item(),
        rmse_log=log_error.square().mean().sqrt().item(),
        log10=(torch.log10(prediction) - torch.log10(target)).abs().mean().item(),
        silog=(log_error.square().mean() - .5 * log_error.mean().square()).sqrt().item(),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_fixed_f3_formulas_for_identical_valid_pixels(dtype):
    target = np.array([[1., 2., 4.], [8., 12., 30.]], dtype=dtype)
    prediction = np.array([[1.25, 3., 2.], [9., 9., 70.]], dtype=dtype)
    valid = np.array([[True, True, False], [True, True, True]])
    count, scores = compute_frame_metrics(prediction, target, valid)
    reference = fixed_f3_eval_depth(torch.from_numpy(prediction[valid]), torch.from_numpy(target[valid]))
    assert count == 5
    for key, value in reference.items():
        assert scores[key] == pytest.approx(value, rel=2e-6, abs=2e-7)
    assert scores["mae"] == pytest.approx(np.abs(prediction[valid].astype(np.float64) - target[valid]).mean())


def test_relative_squared_error_uses_target_squared_and_strict_delta_boundary():
    count, scores = compute_frame_metrics(np.array([4., 1.25]), np.array([2., 1.]), np.ones(2, bool))
    assert count == 2
    assert scores["sq_rel"] == pytest.approx((1 + .25 ** 2) / 2)
    assert scores["sq_rel"] != pytest.approx((2 + .25 ** 2) / 2)
    assert scores["d1"] == 0  # 1.25 exactly is excluded; 2.0 is excluded.
    assert scores["rmse_log"] == pytest.approx(math.sqrt((math.log(2) ** 2 + math.log(1.25) ** 2) / 2))


def test_empty_gt_and_masked_invalid_values_do_not_change_frame_denominator():
    meter = FrameMeanSums()
    count, scores = compute_frame_metrics(np.array([np.nan]), np.array([np.nan]), np.array([False]))
    assert (count, scores) == (0, {})
    meter.add(count, scores)
    for prediction, target in [(np.array([2., 2., 2.]), np.ones(3)), (np.array([4.]), np.ones(1))]:
        meter.add(*compute_frame_metrics(prediction, target, np.ones_like(target, bool)))
    assert meter.compute()["frames"] == 2
    assert meter.compute()["valid_pixels"] == 4
    assert meter.compute()["mae"] == 2
    assert meter.compute()["rmse"] == 2


@pytest.mark.parametrize("prediction,target", [(np.nan, 1.), (np.inf, 1.), (0., 1.), (-1., 1.), (1., 0.), (1., np.nan)])
def test_invalid_scored_values_raise_instead_of_silently_dropping_pixels(prediction, target):
    with pytest.raises(ValueError):
        compute_frame_metrics(np.array([prediction]), np.array([target]), np.array([True]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shapes"):
        compute_frame_metrics(np.ones((1, 2)), np.ones((2,)), np.ones((2,), bool))
