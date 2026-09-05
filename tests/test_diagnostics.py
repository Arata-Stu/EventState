from __future__ import annotations

import torch

from event_state.evaluation.diagnostics import (
    cosine_similarity_by_event_level,
    temporal_feature_stability,
    token_similarity_map,
)


def test_temporal_stability_is_one_for_persistent_features() -> None:
    frame = torch.randn(2, 5)
    features = frame.unsqueeze(0).repeat(4, 1, 1)
    stability = temporal_feature_stability(features)
    assert set(stability) == {1, 2, 3}
    assert all(torch.allclose(value, torch.tensor(1.0)) for value in stability.values())


def test_token_similarity_map_uses_selected_query() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    similarity = token_similarity_map(features, query_index=0)
    assert torch.allclose(similarity, torch.tensor([1.0, 0.0, -1.0]))


def test_event_level_alignment_returns_three_groups() -> None:
    target = torch.randn(1, 3, 2, 4)
    prediction = target.clone()
    metrics = cosine_similarity_by_event_level(
        prediction,
        target,
        event_counts=torch.tensor([[1, 10, 100]]),
    )
    assert set(metrics) == {"low", "medium", "high"}
    assert all(torch.allclose(value, torch.tensor(1.0)) for value in metrics.values())

