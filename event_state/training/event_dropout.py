"""Training-only temporal event dropout for recurrent-state supervision."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def empty_event_input(
    events: Tensor,
    *,
    representation_type: str,
    normalize_mean: Sequence[float] | None = None,
    normalize_std: Sequence[float] | None = None,
) -> Tensor:
    """Return a broadcastable normalized tensor representing no events."""

    if events.ndim != 5:
        raise ValueError("events must have shape [B,T,C,H,W]")
    channels = int(events.shape[2])
    if representation_type == "gep_rgb":
        if normalize_mean is None or normalize_std is None:
            raise ValueError("GEP dropout requires normalize_mean and normalize_std")
        if len(normalize_mean) != channels or len(normalize_std) != channels:
            raise ValueError("Event normalization length must match event channels")
        if any(float(value) <= 0 for value in normalize_std):
            raise ValueError("Event normalization std values must be positive")
        means = events.new_tensor(list(normalize_mean))
        stds = events.new_tensor(list(normalize_std))
        # GEP's no-event image is white in raw [0,1] RGB space.
        return ((1.0 - means) / stds).view(1, 1, channels, 1, 1)
    if representation_type == "voxel_grid":
        return events.new_zeros((1, 1, channels, 1, 1))
    raise ValueError(f"Unsupported event representation: {representation_type!r}")


def sample_temporal_drop_mask(
    batch_size: int,
    sequence_length: int,
    *,
    probability: float,
    lengths: Sequence[int],
    min_context_frames: int,
    min_recovery_frames: int,
    device: torch.device,
) -> Tensor:
    """Sample at most one contiguous dropout block independently per clip."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    normalized_lengths = tuple(int(value) for value in lengths)
    if not normalized_lengths or any(value <= 0 for value in normalized_lengths):
        raise ValueError("lengths must contain positive integers")
    if min_context_frames < 0 or min_recovery_frames < 0:
        raise ValueError("minimum context and recovery frames must be non-negative")
    if max(normalized_lengths) + min_context_frames + min_recovery_frames > sequence_length:
        raise ValueError("Drop block and required context/recovery do not fit the sequence")

    mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    enabled = torch.rand(batch_size, device=device) < probability
    length_choices = torch.tensor(normalized_lengths, device=device)
    selected = torch.randint(len(normalized_lengths), (batch_size,), device=device)
    selected_lengths = length_choices[selected]
    for batch_index in range(batch_size):
        if not bool(enabled[batch_index]):
            continue
        length = int(selected_lengths[batch_index])
        latest_start = sequence_length - min_recovery_frames - length
        start = int(
            torch.randint(
                min_context_frames,
                latest_start + 1,
                (1,),
                device=device,
            )
        )
        mask[batch_index, start : start + length] = True
    return mask


def apply_temporal_event_dropout(
    events: Tensor,
    *,
    probability: float,
    lengths: Sequence[int],
    min_context_frames: int,
    min_recovery_frames: int,
    representation_type: str,
    normalize_mean: Sequence[float] | None = None,
    normalize_std: Sequence[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Replace sampled frames with the representation-correct empty input."""

    if events.ndim != 5:
        raise ValueError("events must have shape [B,T,C,H,W]")
    mask = sample_temporal_drop_mask(
        int(events.shape[0]),
        int(events.shape[1]),
        probability=probability,
        lengths=lengths,
        min_context_frames=min_context_frames,
        min_recovery_frames=min_recovery_frames,
        device=events.device,
    )
    if not bool(mask.any()):
        return events, mask
    empty = empty_event_input(
        events,
        representation_type=representation_type,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    return torch.where(mask[:, :, None, None, None], empty, events), mask


__all__ = [
    "apply_temporal_event_dropout",
    "empty_event_input",
    "sample_temporal_drop_mask",
]
