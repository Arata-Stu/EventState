from __future__ import annotations

import torch

from event_state.data.transforms import PairedSequenceTransform


def test_spatial_transform_keeps_modalities_and_time_aligned() -> None:
    pattern = torch.arange(30, dtype=torch.float32).view(1, 1, 5, 6)
    events = pattern.repeat(3, 1, 1, 1)
    images = pattern.repeat(3, 3, 1, 1)
    transform = PairedSequenceTransform(height=4, width=4, training=False)

    transformed_events, transformed_images = transform(events, images)

    assert transformed_events.shape == (3, 1, 4, 4)
    assert transformed_images.shape == (3, 3, 4, 4)
    assert torch.allclose(transformed_events[:, 0], transformed_images[:, 0])
    assert torch.equal(transformed_events[0], transformed_events[1])


def test_event_normalization_is_applied_after_geometry() -> None:
    events = torch.full((2, 3, 4, 4), 0.5)
    transform = PairedSequenceTransform(
        height=4,
        width=4,
        event_mean=(0.5, 0.5, 0.5),
        event_std=(0.25, 0.25, 0.25),
    )
    normalized, _ = transform(events, None)
    assert torch.equal(normalized, torch.zeros_like(normalized))


def test_sequence_consistent_augmentation_reuses_geometry() -> None:
    pattern = torch.arange(12 * 16, dtype=torch.float32).view(1, 1, 12, 16)
    transform = PairedSequenceTransform(
        height=8,
        width=8,
        training=True,
        scale=(0.5, 0.9),
        horizontal_flip_probability=0.5,
        sequence_consistent=True,
        seed=17,
    )

    first, _ = transform(pattern, None, sequence_key="sequence_a")
    second, _ = transform(pattern, None, sequence_key="sequence_a")

    assert torch.equal(first, second)
