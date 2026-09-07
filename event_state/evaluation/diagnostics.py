"""Small, dependency-light diagnostics for dense token representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def temporal_feature_stability(features: Tensor, max_lag: int | None = None) -> dict[int, Tensor]:
    """Mean token-wise cosine similarity for each temporal lag.

    ``features`` may have shape ``[T, N, D]`` or ``[B, T, N, D]``.
    """

    if features.ndim == 3:
        features = features.unsqueeze(0)
    if features.ndim != 4:
        raise ValueError("features must have shape [T, N, D] or [B, T, N, D]")
    time_steps = features.shape[1]
    if time_steps < 2:
        return {}
    max_lag = min(max_lag or time_steps - 1, time_steps - 1)
    normalized = F.normalize(features.float(), dim=-1)
    return {
        lag: (normalized[:, :-lag] * normalized[:, lag:]).sum(-1).mean()
        for lag in range(1, max_lag + 1)
    }


def token_similarity_map(features: Tensor, query_index: int) -> Tensor:
    """Cosine similarity from one query patch to every patch.

    Accepts ``[N, D]`` or ``[T, N, D]`` and preserves optional time.
    """

    if features.ndim not in {2, 3}:
        raise ValueError("features must have shape [N, D] or [T, N, D]")
    token_count = features.shape[-2]
    if not 0 <= query_index < token_count:
        raise IndexError(f"query_index {query_index} is outside [0, {token_count})")
    normalized = F.normalize(features.float(), dim=-1)
    query = normalized[..., query_index : query_index + 1, :]
    return (normalized * query).sum(-1)


def joint_pca_rgb(*feature_sets: Tensor) -> list[Tensor]:
    """Project multiple ``[..., N, D]`` tensors into one shared PCA basis."""

    if not feature_sets:
        raise ValueError("At least one feature tensor is required")
    embedding_dim = feature_sets[0].shape[-1]
    if any(tensor.ndim < 2 or tensor.shape[-1] != embedding_dim for tensor in feature_sets):
        raise ValueError("All feature tensors must share their final embedding dimension")
    flattened = [tensor.float().reshape(-1, embedding_dim) for tensor in feature_sets]
    lengths = [tensor.shape[0] for tensor in flattened]
    merged = torch.cat(flattened)
    centered = merged - merged.mean(0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    projected = centered @ basis[:, :3]

    raw_outputs: list[Tensor] = []
    offset = 0
    for source, length in zip(feature_sets, lengths):
        value = projected[offset : offset + length].reshape(*source.shape[:-1], 3)
        raw_outputs.append(value)
        offset += length
    # A shared PCA basis is only visually comparable when every source also
    # uses the same RGB range. Per-source percentile scaling can make unrelated
    # representations appear deceptively similar.
    merged_rgb = torch.cat([value.reshape(-1, 3) for value in raw_outputs])
    lower = torch.quantile(merged_rgb, 0.01, dim=0)
    upper = torch.quantile(merged_rgb, 0.99, dim=0)
    scale = (upper - lower).clamp_min(torch.finfo(merged_rgb.dtype).eps)
    return [((value - lower) / scale).clamp(0, 1) for value in raw_outputs]


def cosine_similarity_by_event_level(
    prediction: Tensor,
    target: Tensor,
    event_counts: Tensor,
) -> dict[str, Tensor]:
    """Split frame-level alignment into low/medium/high event-count thirds."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share shape [B, T, N, D]")
    if event_counts.shape != prediction.shape[:2]:
        raise ValueError("event_counts must have shape [B, T]")
    similarity = F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean(-1)
    flat_counts = event_counts.reshape(-1).float()
    flat_similarity = similarity.reshape(-1)
    thresholds = torch.quantile(flat_counts, flat_counts.new_tensor([1 / 3, 2 / 3]))
    masks = {
        "low": flat_counts <= thresholds[0],
        "medium": (flat_counts > thresholds[0]) & (flat_counts <= thresholds[1]),
        "high": flat_counts > thresholds[1],
    }
    nan = flat_similarity.new_tensor(float("nan"))
    return {
        name: flat_similarity[mask].mean() if torch.any(mask) else nan
        for name, mask in masks.items()
    }
