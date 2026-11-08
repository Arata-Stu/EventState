import torch
from torch import nn

from event_state.dense.head import build_dense_head
from event_state.models.temporal.stages import FourStageLSTM


class TinyBlock(nn.Linear):
    def forward(self, tokens, rope):
        return super().forward(tokens)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(4, 4) for _ in range(12))
        self.norm = nn.LayerNorm(4)
        self.rope_embed = None
        self.n_storage_tokens = 1

    def prepare_tokens_with_masks(self, frames):
        return frames, (2, 3)


def test_four_stage_carry_fusion_and_feedback():
    torch.manual_seed(3)
    vit = TinyViT()
    memory = FourStageLSTM(4, 4, 4)
    frames = torch.randn(3, 8, 4)  # two prefix tokens and six patches
    z, stages, _ = memory(vit, frames, 1, 3)
    state = None
    for i in range(3):
        _, step, state = memory(vit, frames[i:i+1], 1, 1, state)
        for trace, value in zip(stages, step):
            torch.testing.assert_close(trace[:, i:i+1], value)
    # The final representation must backpropagate through every recurrent insertion.
    grads = torch.autograd.grad(stages[-1].sum(), [m.lstm.weight_ih_l0 for m in memory.stages])
    assert all(g.abs().sum() > 0 for g in grads)
    maps = [x.detach().reshape(3, 2, 3, 4).permute(0, 3, 1, 2).requires_grad_() for x in stages]
    prediction = build_dense_head("stages", 16, 4)(torch.cat(maps, dim=1), (32, 48))
    assert prediction.shape == (3, 32, 48)
    assert all(g.abs().sum() > 0 for g in torch.autograd.grad(prediction.mean(), maps))
