"""Direct recurrent connections after DINOv3 blocks 3, 6, 9 and 12."""
import torch
from torch import nn

from .lstm import PatchwiseLSTM


class FourStageLSTM(nn.Module):
    """Insert independent patchwise LSTMs between selected ViT blocks.

    The four readouts always correspond to blocks 3, 6, 9 and 12. Inactive
    readouts retain the ViT features produced downstream of earlier insertions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, first_stage=1, *,
                 after_blocks=None, residual_skip=False):
        super().__init__()
        if input_dim != output_dim:
            raise ValueError("Direct stage connections require the ViT token dimension")
        self.input_dim, self.output_dim = input_dim, output_dim
        if first_stage not in (1, 2, 3, 4):
            raise ValueError("first_stage must be 1, 2, 3 or 4")
        if after_blocks is None:
            after_blocks = tuple(range(3 * first_stage, 13, 3))
        else:
            if first_stage != 1:
                raise ValueError("Use after_blocks or a non-default first_stage, not both")
            after_blocks = tuple(after_blocks)
            if (not after_blocks or any(type(block) is not int or block not in (3, 6, 9, 12)
                                        for block in after_blocks)
                    or after_blocks != tuple(sorted(set(after_blocks)))):
                raise ValueError("after_blocks must be a nonempty, increasing subset of [3, 6, 9, 12]")
        self.after_blocks = after_blocks
        if not isinstance(residual_skip, bool):
            raise ValueError("residual_skip must be a boolean")
        # Keep the scalar state_dict format of single-insertion checkpoints.
        scale_shape = () if len(after_blocks) == 1 else (len(after_blocks),)
        self.skip_scale = nn.Parameter(torch.zeros(scale_shape)) if residual_skip else None
        self.first_stage = after_blocks[0] // 3
        self.stages = nn.ModuleList(
            PatchwiseLSTM(input_dim, hidden_dim, output_dim) for _ in self.after_blocks
        )

    def forward(self, backbone, frames, batch, time, state=None):
        if len(backbone.blocks) != 12:
            raise ValueError("Four-stage LSTM requires a 12-block DINOv3")
        if state is None:
            state = (None,) * len(self.stages)
        if len(state) != len(self.stages):
            raise ValueError("Expected one state per active LSTM stage")
        tokens, (height, width) = backbone.prepare_tokens_with_masks(frames)
        prefix = 1 + backbone.n_storage_tokens
        outputs, states = [], []
        for index, block in enumerate(backbone.blocks, 1):
            rope = backbone.rope_embed(H=height, W=width) if backbone.rope_embed is not None else None
            tokens = block(tokens, rope)
            if index % 3 == 0:
                patches = backbone.norm(tokens[:, prefix:])
                z = patches.reshape(batch, time, height * width, self.input_dim)
                if index not in self.after_blocks:
                    outputs.append(z)
                    continue
                active = len(states)
                h, carried = self.stages[active](z, state[active])
                next_patches = h.flatten(0, 1)
                if self.skip_scale is not None:
                    # Keep the unnormalized ViT residual stream as the identity path.
                    gain = self.skip_scale.reshape(-1)[active]
                    if index == 12:
                        # The head receives the normalized final ViT feature plus memory.
                        h = z + gain * h
                    else:
                        next_patches = tokens[:, prefix:] + gain * next_patches
                        h = backbone.norm(next_patches).reshape_as(z)
                outputs.append(h)
                states.append(carried)
                if index < 12:
                    tokens = torch.cat((tokens[:, :prefix], next_patches), dim=1)
        return z, tuple(outputs), tuple(states)
