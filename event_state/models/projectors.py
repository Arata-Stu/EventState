"""Projection heads from event state to the teacher feature space."""

from __future__ import annotations

from torch import Tensor, nn


class TeacherProjection(nn.Module):
    """Token-wise Linear-GELU-Linear projection."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 384,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim < 2:
            raise ValueError("tokens must have shape [..., N, D]")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Projection expects feature dim {self.input_dim}, got {tokens.shape[-1]}"
            )
        return self.projection(tokens)


ProjectionHead = TeacherProjection


__all__ = ["ProjectionHead", "TeacherProjection"]
