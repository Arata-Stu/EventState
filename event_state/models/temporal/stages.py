"""Direct recurrent connections after DINOv3 blocks 3, 6, 9 and 12."""
import torch
from torch import nn

from .lstm import PatchwiseLSTM


class FourStageLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("Direct stage connections require the ViT token dimension")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.stages = nn.ModuleList(
            PatchwiseLSTM(input_dim, hidden_dim, output_dim) for _ in range(4)
        )

    def forward(self, backbone, frames, batch, time, state=None):
        if len(backbone.blocks) != 12:
            raise ValueError("Four-stage LSTM requires a 12-block DINOv3")
        if state is None:
            state = (None,) * 4
        if len(state) != 4:
            raise ValueError("Expected four stage states")
        tokens, (height, width) = backbone.prepare_tokens_with_masks(frames)
        prefix = 1 + backbone.n_storage_tokens
        outputs, states = [], []
        for index, block in enumerate(backbone.blocks, 1):
            rope = backbone.rope_embed(H=height, W=width) if backbone.rope_embed is not None else None
            tokens = block(tokens, rope)
            if index % 3 == 0:
                patches = backbone.norm(tokens[:, prefix:])
                z = patches.reshape(batch, time, height * width, self.input_dim)
                h, carried = self.stages[len(outputs)](z, state[len(outputs)])
                outputs.append(h)
                states.append(carried)
                if index < 12:
                    tokens = torch.cat((tokens[:, :prefix], h.flatten(0, 1)), dim=1)
        return z, tuple(outputs), tuple(states)
