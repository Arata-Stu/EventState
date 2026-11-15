import pytest
import torch
from torch import nn

from event_state.models.event_state import EventStateModel
from event_state.models.temporal import build_temporal_backbone


def test_convlstm_rectangular_grid_step_carry_and_neighbor_gradient():
    torch.manual_seed(1)
    model = build_temporal_backbone(
        "convlstm", input_dim=2, hidden_dim=4, output_dim=3, grid_size=(3, 5)
    )
    x = torch.randn(2, 3, 15, 2, requires_grad=True)
    sequence, state = model(x)
    assert sequence.shape == (2, 3, 15, 3)
    assert state[0].shape == state[1].shape == (1, 2, 4, 3, 5)
    carried = None
    outputs = []
    for frame in x.unbind(1):
        out, carried = model.forward_step(frame, carried)
        outputs.append(out)
    torch.testing.assert_close(sequence, torch.stack(outputs, dim=1))
    for actual, expected in zip(carried, state):
        torch.testing.assert_close(actual, expected)

    # The center output must depend on a neighboring previous hidden patch.
    past_h = torch.zeros_like(state[0], requires_grad=True)
    out, _ = model.forward_step(x[:, 0], (past_h, torch.zeros_like(past_h)))
    grad, = torch.autograd.grad(out[0, 7].sum(), past_h)
    assert grad[0, 0, :, 1, 1].abs().sum() > 0
    assert grad[0, 0, :, 1, 0].abs().sum() == 0


class TinyDenseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def get_intermediate_layers(self, frames, n):
        assert list(n) == [2, 5, 8, 11]
        pooled = torch.nn.functional.avg_pool2d(frames, 16)
        tokens = self.projection(pooled.flatten(2).transpose(1, 2))
        return tuple(tokens * scale for scale in (1, 2, 3, 4))


class TinyDenseEncoder(nn.Module):
    embed_dim = 4
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.backbone = TinyDenseBackbone()

    def forward(self, frames):
        return self.backbone.get_intermediate_layers(frames, [2, 5, 8, 11])[-1]


def test_convlstm_depth_last_level_rectangular_runtime_grid_and_continuous_state():
    """Depth retains the first three ViT levels and uses the actual input grid."""
    torch.manual_seed(2)
    memory = build_temporal_backbone(
        'convlstm', input_dim=4, hidden_dim=4, output_dim=4, grid_size=(2, 2)
    )
    model = EventStateModel(TinyDenseEncoder(), memory, teacher_dim=4, projector_hidden_dim=8)
    events = torch.randn(2, 3, 3, 32, 48)  # runtime grid 2x3, training grid 2x2
    backbone_levels = tuple(x.unflatten(0, (2, 3)) for x in
                            model.event_encoder.backbone.get_intermediate_layers(
                                events.flatten(0, 1), [2, 5, 8, 11]))
    levels, state = model.forward_depth_stages(events)
    for actual, expected in zip(levels[:3], backbone_levels[:3]):
        torch.testing.assert_close(actual, expected)
    expected_last, expected_state = memory(backbone_levels[-1], grid_size=(2, 3))
    torch.testing.assert_close(levels[-1], expected_last)
    for actual, expected in zip(state, expected_state):
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (1, 2, 4, 2, 3)
    assert memory.grid_size == (2, 2)

    # The ordinary Semantic path and the final Depth level share the same h.
    semantic = model(events)
    torch.testing.assert_close(semantic['h'], levels[-1])
    carried = None
    stepped = []
    for frame in events.unbind(1):
        step, carried = model.forward_depth_stages(frame[:, None], carried)
        stepped.append(step[-1])
    torch.testing.assert_close(torch.cat(stepped, dim=1), levels[-1])
    for actual, expected in zip(carried, state):
        torch.testing.assert_close(actual, expected)
    reset, _ = model.forward_depth_stages(events[:, 1:2])
    assert not torch.allclose(reset[-1], levels[-1][:, 1:2])
    assert memory.grid_size == (2, 2)

    # A carried state from a different geometry must not be silently reset.
    with pytest.raises(ValueError, match='Expected ConvLSTM'):
        model.forward_depth_stages(events[..., :32], state)
