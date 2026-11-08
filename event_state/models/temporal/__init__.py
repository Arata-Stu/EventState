"""Temporal model registry."""

from __future__ import annotations

from .base import TemporalBackbone, TemporalState, detach_temporal_state
from .convlstm import ConvLSTM
from .identity import IdentityTemporal, IdentityTemporalBackbone
from .lstm import LSTMState, PatchwiseLSTM, PatchwiseLSTMTemporalBackbone
from .stages import FourStageLSTM
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
    first_stage: int = 1,
    grid_size: tuple[int, int] | None = None,
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
    if normalized_kind == "four_stage_lstm":
        if num_layers != 1 or dropout or detach_state_every is not None:
            raise ValueError("Expected one LSTM layer per ViT stage")
        return FourStageLSTM(input_dim, hidden_dim, output_dim, first_stage)
    if normalized_kind == "convlstm":
        if grid_size is None or num_layers != 1 or dropout or detach_state_every is not None:
            raise ValueError("ConvLSTM requires a patch grid and one layer without internal dropout or truncation")
        return ConvLSTM(input_dim, hidden_dim, output_dim, grid_size)
    if normalized_kind == "mamba":
        return MambaTemporalBackbone(input_dim=input_dim, output_dim=output_dim)
    raise ValueError("temporal kind must be one of: none, identity, lstm, convlstm, mamba")


__all__ = [
    "ConvLSTM",
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
