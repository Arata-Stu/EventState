"""Reserved Mamba temporal-backbone interface for Phase 4."""

from __future__ import annotations

from torch import Tensor

from .base import TemporalBackbone, TemporalState, validate_token_sequence


class MambaTemporalBackbone(TemporalBackbone):
    """Shape-validating placeholder; the Mamba implementation is Phase 4."""

    def __init__(self, input_dim: int = 384, output_dim: int = 384, **_: object) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(
        self,
        z: Tensor,
        state: TemporalState = None,
    ) -> tuple[Tensor, TemporalState]:
        del state
        validate_token_sequence(z, feature_dim=self.input_dim)
        raise NotImplementedError(
            "MambaTemporalBackbone is a Phase 4 placeholder; use temporal.type=none or lstm"
        )


MambaTemporal = MambaTemporalBackbone


__all__ = ["MambaTemporal", "MambaTemporalBackbone"]
