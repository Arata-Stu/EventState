"""Patch-wise LSTM temporal baseline."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn

from .base import (
    TemporalBackbone,
    TemporalState,
    detach_temporal_state,
    validate_token_sequence,
)


LSTMState: TypeAlias = tuple[Tensor, Tensor]


class PatchwiseLSTM(TemporalBackbone):
    """Apply one shared LSTM independently at every spatial patch location."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,
        output_dim: int = 384,
        num_layers: int = 1,
        dropout: float = 0.0,
        detach_state_every: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if detach_state_every is not None and detach_state_every <= 0:
            raise ValueError("detach_state_every must be positive or None")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.detach_state_every = detach_state_every
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)

    def _validate_state(
        self,
        state: TemporalState,
        *,
        flattened_batch: int,
    ) -> LSTMState | None:
        if state is None:
            return None
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("PatchwiseLSTM state must be an (h, c) tensor tuple")
        hidden, cell = state
        if not isinstance(hidden, Tensor) or not isinstance(cell, Tensor):
            raise TypeError("PatchwiseLSTM hidden and cell state must be tensors")
        expected = (self.num_layers, flattened_batch, self.hidden_dim)
        if tuple(hidden.shape) != expected or tuple(cell.shape) != expected:
            raise ValueError(
                f"PatchwiseLSTM state tensors must both have shape {expected}; "
                f"got {tuple(hidden.shape)} and {tuple(cell.shape)}"
            )
        return hidden, cell

    def _run_lstm(
        self,
        sequence: Tensor,
        state: LSTMState | None,
    ) -> tuple[Tensor, LSTMState]:
        truncate = self.detach_state_every
        if truncate is None or sequence.shape[1] <= truncate:
            return self.lstm(sequence, state)

        outputs: list[Tensor] = []
        current_state = state
        for start in range(0, sequence.shape[1], truncate):
            if start > 0:
                current_state = detach_temporal_state(current_state)
            output, current_state = self.lstm(sequence[:, start : start + truncate], current_state)
            outputs.append(output)
        if current_state is None:  # pragma: no cover - nn.LSTM always returns state
            raise RuntimeError("nn.LSTM unexpectedly returned no state")
        return torch.cat(outputs, dim=1), current_state

    def forward(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, LSTMState]:
        validate_token_sequence(z, feature_dim=self.input_dim)
        batch_size, sequence_length, num_tokens, _ = z.shape
        flattened_batch = batch_size * num_tokens
        lstm_input = z.permute(0, 2, 1, 3).reshape(
            flattened_batch, sequence_length, self.input_dim
        )
        checked_state = self._validate_state(state, flattened_batch=flattened_batch)
        output, new_state = self._run_lstm(lstm_input, checked_state)
        output = self.output_projection(output)
        output = output.reshape(
            batch_size, num_tokens, sequence_length, self.output_dim
        ).permute(0, 2, 1, 3).contiguous()
        return output, new_state


PatchwiseLSTMTemporalBackbone = PatchwiseLSTM


__all__ = ["LSTMState", "PatchwiseLSTM", "PatchwiseLSTMTemporalBackbone"]
