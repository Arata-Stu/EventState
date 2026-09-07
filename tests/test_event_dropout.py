from __future__ import annotations

import torch

from event_state.training.event_dropout import (
    apply_temporal_event_dropout,
    empty_event_input,
    sample_temporal_drop_mask,
)


def test_gep_empty_event_is_normalized_white() -> None:
    events = torch.zeros(2, 8, 3, 4, 5)
    value = empty_event_input(
        events,
        representation_type="gep_rgb",
        normalize_mean=(0.9, 0.8, 0.7),
        normalize_std=(0.2, 0.4, 0.5),
    )

    assert value.shape == (1, 1, 3, 1, 1)
    assert torch.allclose(value.flatten(), torch.tensor([0.5, 0.5, 0.6]))


def test_temporal_mask_preserves_required_context_and_recovery() -> None:
    torch.manual_seed(0)
    mask = sample_temporal_drop_mask(
        16,
        8,
        probability=1.0,
        lengths=(2,),
        min_context_frames=2,
        min_recovery_frames=1,
        device=torch.device("cpu"),
    )

    assert torch.all(mask.sum(dim=1) == 2)
    assert not bool(mask[:, :2].any())
    assert not bool(mask[:, -1].any())
    for row in mask:
        indices = row.nonzero().flatten()
        assert int(indices[1] - indices[0]) == 1


def test_dropout_replaces_only_sampled_frames() -> None:
    torch.manual_seed(0)
    events = torch.randn(4, 8, 3, 2, 2)
    dropped, mask = apply_temporal_event_dropout(
        events,
        probability=1.0,
        lengths=(1,),
        min_context_frames=2,
        min_recovery_frames=1,
        representation_type="gep_rgb",
        normalize_mean=(0.9, 0.8, 0.7),
        normalize_std=(0.2, 0.4, 0.5),
    )
    empty = empty_event_input(
        events,
        representation_type="gep_rgb",
        normalize_mean=(0.9, 0.8, 0.7),
        normalize_std=(0.2, 0.4, 0.5),
    ).expand_as(events)

    assert torch.equal(dropped[~mask], events[~mask])
    assert torch.equal(dropped[mask], empty[mask])
