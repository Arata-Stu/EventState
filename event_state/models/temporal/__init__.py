"""Temporal model registry."""

from __future__ import annotations

from .base import TemporalBackbone, TemporalState, detach_temporal_state
from .identity import IdentityTemporal, IdentityTemporalBackbone
from .lstm import LSTMState, PatchwiseLSTM, PatchwiseLSTMTemporalBackbone
from .mamba import MambaTemporal, MambaTemporalBackbone


def build_temporal_backbone(
    kind: str,
    *,
    input_dim: int = 384,
    hidden_dim: int = 384,
    output_dim: int = 384,
    num_layers: int = 1,
    dropout: float = 0.0,
    detach_state_every: int | None = None,
) -> TemporalBackbone:
    normalized_kind = str(kind).lower()
    if normalized_kind in {"none", "identity"}:
        if output_dim != input_dim:
            raise ValueError("Identity temporal backbone requires output_dim == input_dim")
        return IdentityTemporalBackbone(input_dim=input_dim)
    if normalized_kind in {"lstm", "patchwise_lstm"}:
        return PatchwiseLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            detach_state_every=detach_state_every,
        )
    if normalized_kind == "mamba":
        return MambaTemporalBackbone(input_dim=input_dim, output_dim=output_dim)
    raise ValueError("temporal kind must be one of: none, identity, lstm, mamba")


__all__ = [
    "IdentityTemporal",
    "IdentityTemporalBackbone",
    "LSTMState",
    "MambaTemporal",
    "MambaTemporalBackbone",
    "PatchwiseLSTM",
    "PatchwiseLSTMTemporalBackbone",
    "TemporalBackbone",
    "TemporalState",
    "build_temporal_backbone",
    "detach_temporal_state",
]
