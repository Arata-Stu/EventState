from __future__ import annotations

import torch

from event_state.data.event_representation import EventVoxelizer, GEPEventFrame


def test_gep_event_frame_matches_effective_rgb_polarity_colors() -> None:
    representation = GEPEventFrame(height=3, width=4, percentile=90)
    frame = representation(
        x=torch.tensor([1, 2]),
        y=torch.tensor([1, 1]),
        t=torch.tensor([5, 6]),
        p=torch.tensor([1, 0]),
        start_time=0,
        end_time=10,
    )

    assert frame.shape == (3, 3, 4)
    assert torch.equal(frame[:, 0, 0], torch.ones(3))
    # GEP's OpenCV save/read path makes positive blue and negative red.
    assert torch.equal(frame[:, 1, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(frame[:, 1, 2], torch.tensor([1.0, 0.0, 0.0]))


def test_event_voxelization_shape_polarity_and_timestamp_boundaries() -> None:
    voxelizer = EventVoxelizer(
        num_bins=2,
        height=2,
        width=3,
        polarity_split=True,
        normalization="none",
    )
    volume = voxelizer(
        x=torch.tensor([0, 1, 2]),
        y=torch.tensor([0, 0, 1]),
        t=torch.tensor([0, 5, 10]),
        p=torch.tensor([0, 1, 0]),
        start_time=0,
        end_time=10,
    )

    assert volume.shape == (4, 2, 3)
    assert volume.sum().item() == 2.0  # t == start is excluded; t == end is included.
    assert volume[1, 0, 1].item() == 0.5
    assert volume[3, 0, 1].item() == 0.5
    assert volume[2, 1, 2].item() == 1.0


def test_empty_gep_frame_is_white() -> None:
    representation = GEPEventFrame(height=2, width=2)
    empty = torch.empty(0)
    frame = representation(
        x=empty,
        y=empty,
        t=empty,
        p=empty,
        start_time=0,
        end_time=1,
    )
    assert torch.equal(frame, torch.ones(3, 2, 2))


def test_gep_positive_wins_ties_and_preserves_percentile_intensity() -> None:
    representation = GEPEventFrame(height=1, width=3, percentile=100)
    tie = representation(
        x=torch.tensor([0, 0]),
        y=torch.tensor([0, 0]),
        t=torch.tensor([1, 2]),
        p=torch.tensor([0, 1]),
        start_time=0,
        end_time=3,
    )
    assert torch.equal(tie[:, 0, 0], torch.tensor([0.0, 0.0, 1.0]))

    graded = representation(
        x=torch.tensor([0, 1, 1]),
        y=torch.tensor([0, 0, 0]),
        t=torch.tensor([1, 2, 3]),
        p=torch.ones(3),
        start_time=0,
        end_time=4,
    )
    assert torch.equal(graded[:, 0, 0], torch.tensor([0.5, 0.5, 1.0]))
