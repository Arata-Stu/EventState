from __future__ import annotations

import numpy as np

from event_state.data.dagr_downsample import DAGRDownsampler, bottom_padding_masks


def test_dagr_downsample_accumulates_state_across_calls() -> None:
    downsample = DAGRDownsampler(
        input_height=2, input_width=2, output_height=1, output_width=1
    )
    first = downsample(
        np.array([0, 1]), np.array([0, 1]), np.array([1, 1], dtype=np.uint8)
    )
    second = downsample(
        np.array([1, 0]), np.array([0, 1]), np.array([1, 1], dtype=np.uint8)
    )
    assert first[0].size == 0
    assert second[0].tolist() == [0]
    assert second[1].tolist() == [0]
    assert second[2].tolist() == [1]
    assert second[3].tolist() == [1]


def test_opposite_polarities_cancel_before_threshold() -> None:
    downsample = DAGRDownsampler(
        input_height=2, input_width=2, output_height=1, output_width=1
    )
    result = downsample(
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )
    assert result[0].size == 0


def test_bottom_padding_masks_include_half_valid_boundary_patch() -> None:
    pixel, patch, fraction = bottom_padding_masks()
    assert pixel.shape == (448, 640)
    assert pixel[:360].all()
    assert not pixel[360:].any()
    assert patch.shape == (28, 40)
    assert patch[:23].all()
    assert not patch[23:].any()
    assert np.all(fraction[:22] == 1)
    assert np.all(fraction[22] == 0.5)
    assert np.all(fraction[23:] == 0)
