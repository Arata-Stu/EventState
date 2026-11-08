"""Single-layer ConvLSTM over the encoder's rectangular patch grid."""

import torch
from torch import Tensor, nn

from .base import TemporalBackbone, TemporalState, validate_token_sequence


class ConvLSTM(TemporalBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 grid_size: tuple[int, int]) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, *grid_size) <= 0:
            raise ValueError("ConvLSTM dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, state: TemporalState = None):
        validate_token_sequence(z, feature_dim=self.input_dim)
        batch, _, tokens, _ = z.shape
        height, width = self.grid_size
        if tokens != height * width:
            raise ValueError(f"Expected tokens for grid {self.grid_size}, got {tokens}")
        shape = (batch, self.hidden_dim, height, width)
        if state is None:
            h = z.new_zeros(shape)
            c = z.new_zeros(shape)
        else:
            # Stream batching splits and joins recurrent tensors on dimension 1.
            if not isinstance(state, tuple) or len(state) != 2 or any(
                not isinstance(t, Tensor) or t.shape != (1, *shape) for t in state
            ):
                raise ValueError(f"Expected ConvLSTM (h, c) with shape {(1, *shape)}")
            h, c = (t.squeeze(0) for t in state)

        outputs = []
        for frame in z.unbind(dim=1):
            x = frame.transpose(1, 2).reshape(batch, self.input_dim, height, width)
            i, f, g, o = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
            outputs.append(h.flatten(2).transpose(1, 2))
        return self.output_projection(torch.stack(outputs, dim=1)), (h.unsqueeze(0), c.unsqueeze(0))
