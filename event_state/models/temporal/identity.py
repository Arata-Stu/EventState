"""No-memory temporal baseline."""

from __future__ import annotations

from torch import Tensor

from .base import TemporalBackbone, TemporalState, validate_token_sequence


class IdentityTemporalBackbone(TemporalBackbone):
    def __init__(self, input_dim: int = 384) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)

    def forward(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, None]:
        validate_token_sequence(z, feature_dim=self.input_dim)
        if state is not None:
            raise ValueError("IdentityTemporalBackbone does not accept recurrent state")
        return z, None


IdentityTemporal = IdentityTemporalBackbone


__all__ = ["IdentityTemporal", "IdentityTemporalBackbone"]
