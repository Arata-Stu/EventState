import torch

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
