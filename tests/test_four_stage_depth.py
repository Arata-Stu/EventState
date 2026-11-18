import pytest
import torch
from torch import nn

from event_state.dense.head import build_dense_head
from event_state.models import EventStateModel, build_temporal_backbone, detach_temporal_state
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
        return frames.reshape(-1, 8, 4), (2, 3)


@pytest.mark.parametrize("first_stage", [1, 2, 3, 4])
def test_four_stage_carry_fusion_and_feedback(first_stage):
    torch.manual_seed(3)
    vit = TinyViT()
    memory = FourStageLSTM(4, 4, 4, first_stage)
    assert len(memory.stages) == 5 - first_stage
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


@pytest.mark.parametrize("blocks", [(3,), (6,), (9,), (9, 12), (6, 9, 12), (3, 6, 9, 12)])
def test_placement_exposes_final_representation_and_carries_only_active_states(blocks):
    torch.manual_seed(7)
    encoder = nn.Module()
    encoder.embed_dim = 4
    encoder.backbone = TinyViT()
    memory = build_temporal_backbone("four_stage_lstm", input_dim=4, hidden_dim=4, output_dim=4,
                                     after_blocks=list(blocks))
    model = EventStateModel(encoder, memory, nn.Identity(), nn.Identity(), teacher_dim=4)
    order = []
    def trace(name):
        def hook(_module, _inputs, _output):
            order.append(name)
        return hook

    hooks = [block.register_forward_hook(trace(i)) for i, block in enumerate(encoder.backbone.blocks, 1)]
    hooks.extend(stage.register_forward_hook(trace("lstm")) for stage in memory.stages)
    events = torch.randn(2, 3, 1, 8, 4)
    outputs = model(events)
    for handle in hooks:
        handle.remove()
    expected = []
    for i in range(1, 13):
        expected.append(i)
        if i in blocks:
            expected.append("lstm")
    assert order == expected
    assert len(outputs["state"]) == len(blocks)

    # Both distillation and the Semantic head read the final representation.
    levels, _ = model.forward_depth_stages(events)
    torch.testing.assert_close(outputs["h"], levels[-1])
    if 12 not in blocks:
        torch.testing.assert_close(outputs["h"], outputs["z"])
        assert not torch.allclose(outputs["h"], levels[blocks[-1] // 3 - 1])
    loss = model.project_h(outputs["h"])[..., 0].square().mean()
    gradients = torch.autograd.grad(loss, [stage.lstm.weight_ih_l0 for stage in memory.stages])
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gradients)

    # Match continuous, frame-by-frame Semantic cache extraction to the Stage 1 clip.
    state = None
    for i in range(events.shape[1]):
        step = model(events[:, i:i+1], state)
        state = detach_temporal_state(step["state"])
        torch.testing.assert_close(step["h"], outputs["h"][:, i:i+1])


@pytest.mark.parametrize("first_stage", [1, 2, 3, 4])
def test_explicit_suffix_loads_legacy_weights(first_stage):
    legacy = FourStageLSTM(4, 4, 4, first_stage)
    explicit = FourStageLSTM(4, 4, 4, after_blocks=list(range(3 * first_stage, 13, 3)))
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    vit = TinyViT()
    frames = torch.randn(2, 8, 4)
    torch.testing.assert_close(explicit(vit, frames, 1, 2), legacy(vit, frames, 1, 2))


def test_final_block_insertion_matches_conventional_encoder_then_patchwise_lstm():
    torch.manual_seed(11)
    vit = TinyViT()
    memory = FourStageLSTM(4, 4, 4, after_blocks=[12])
    baseline = build_temporal_backbone("lstm", input_dim=4, hidden_dim=4, output_dim=4)
    baseline.load_state_dict(memory.stages[0].state_dict())
    frames = torch.randn(6, 8, 4)
    tokens, _ = vit.prepare_tokens_with_masks(frames)
    for block in vit.blocks:
        tokens = block(tokens, None)
    baseline_z = vit.norm(tokens)[:, 2:].reshape(2, 3, 6, 4)
    baseline_h, baseline_state = baseline(baseline_z)
    z, levels, states = memory(vit, frames, 2, 3)
    torch.testing.assert_close((z, levels[-1], states[0]), (baseline_z, baseline_h, baseline_state))


@pytest.mark.parametrize("blocks", [[], [0], [4], [12, 3], [9, 9], [3.0], ["3"]])
def test_invalid_placements_fail_before_forward(blocks):
    with pytest.raises(ValueError, match="after_blocks"):
        FourStageLSTM(4, 4, 4, after_blocks=blocks)
