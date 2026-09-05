from __future__ import annotations

import torch

from event_state.losses import DistillationLoss, NoZObjective


def test_distillation_loss_backpropagates_only_to_student() -> None:
    prediction = torch.randn(2, 3, 5, 8, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    criterion = DistillationLoss(cosine_weight=1.0, mse_weight=1.0)

    components = criterion.components(prediction, target)
    components["loss"].backward()

    assert components["loss"].ndim == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None


def test_none_z_objective_is_differentiable_zero() -> None:
    z = torch.randn(1, 2, 3, 4, requires_grad=True)
    loss = NoZObjective()(z)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(z.grad, torch.zeros_like(z))

