"""ML-host checks for soft h weights and matched z-only ablation."""

from pathlib import Path

import pytest
import torch

from event_state.losses.activity import ActivityDistillationLoss


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("drop", [False, True])
def test_branch_routing_and_dropout(alpha, drop):
    from test_training_runtime import TinySequenceModel, _make_trainer, _trainer_config

    config = _trainer_config(h_enabled=True, z_type="direct_dino")
    config["dataset"] = {"activity_mask": True}
    config["loss"]["h_distill"]["activity_active_weight"] = alpha
    trainer = _make_trainer(TinySequenceModel(), config)
    trainer.h_distillation_loss = ActivityDistillationLoss()
    trainer.z_distillation_loss = ActivityDistillationLoss()
    batch = {"events": torch.randn(1, 3, 3, 16, 16),
             "teacher_features": torch.randn(1, 3, 1, 4),
             "event_activity": torch.tensor([[[True], [False], [True]]])}
    dropout = torch.tensor([[False, False, True]]) if drop else None
    result = trainer._forward_objectives(batch, event_dropout_mask=dropout)
    weights = torch.tensor([alpha, 1., 1. if drop else alpha]).view(1, 3, 1)
    z_weights = torch.tensor([1., 0., 0. if drop else 1.]).view(1, 3, 1)
    torch.testing.assert_close(result["h_components"]["loss"], ActivityDistillationLoss()(
        result["h_prediction"], batch["teacher_features"], weights))
    torch.testing.assert_close(result["z_components"]["loss"], ActivityDistillationLoss()(
        result["z_prediction"], batch["teacher_features"], z_weights))
    torch.testing.assert_close(result["loss"], result["h_components"]["loss"] +
                               result["z_components"]["loss"])


def test_soft_loss_uses_weight_sum_and_teacher_is_detached():
    # Per-token cosine+squared-L2 distances are 3 and 6, respectively.
    p = torch.tensor([[[1., 0.], [1., 0.]]], requires_grad=True)
    q = torch.tensor([[[0., 1.], [-1., 0.]]], requires_grad=True)
    loss = ActivityDistillationLoss()(p, q, torch.tensor([[0.5, 1.]]))
    torch.testing.assert_close(loss, torch.tensor(5.))  # (0.5*3 + 6) / 1.5
    loss.backward()
    assert p.grad is not None
    assert q.grad is None


def test_z_only_excludes_temporal_and_h_projector_gradients():
    from test_training_runtime import TinySequenceModel, _make_trainer, _trainer_config

    config = _trainer_config(h_enabled=False, z_type="direct_dino")
    config["dataset"] = {"activity_mask": True}
    model = TinySequenceModel()
    trainer = _make_trainer(model, config)
    trainer.h_distillation_loss = ActivityDistillationLoss()
    trainer.z_distillation_loss = ActivityDistillationLoss()
    batch = {"events": torch.randn(1, 2, 3, 16, 16),
             "teacher_features": torch.randn(1, 2, 1, 4),
             "event_activity": torch.tensor([[[True], [False]]])}
    result = trainer._forward_objectives(batch)
    result["z_prediction"].retain_grad()
    torch.testing.assert_close(result["loss"], result["z_components"]["loss"])
    result["loss"].backward()
    assert all(p.grad is None for p in model.temporal_model.parameters())
    assert all(p.grad is None for p in model.h_projector.parameters())
    assert model.event_encoder.weight.grad is not None
    assert torch.count_nonzero(result["z_prediction"].grad[:, 1]) == 0


@pytest.mark.parametrize("experiment,enabled,alpha", [
    ("activity_z_only", False, 0.),
    ("activity_z_h_all", True, 1.),
    ("activity_z_h_soft", True, 0.5),
])
def test_configs_preserve_matched_training(experiment, enabled, alpha):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from event_state.training.factory import validate_config, _build_losses

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        base = compose(config_name="config", overrides=[
            "dataset=dsec_det_train41", "experiment=activity_dual"])
        config = compose(config_name="config", overrides=[
            "dataset=dsec_det_train41", f"experiment={experiment}"])
    validate_config(config)
    for section in ("model", "dataset", "teacher", "training", "optimizer", "scheduler", "seed"):
        if section == "seed":
            assert config.seed == base.seed
        else:
            assert OmegaConf.to_container(config[section]) == OmegaConf.to_container(base[section])
    assert config.loss.h_distill.enabled == enabled
    assert config.loss.h_distill.activity_active_weight == alpha
    assert config.loss.z_objective == base.loss.z_objective
    assert all(isinstance(loss, ActivityDistillationLoss) for loss in _build_losses(config))
    for invalid in (-1., 1.1, float("nan"), float("inf")):
        config.loss.h_distill.activity_active_weight = invalid
        with pytest.raises(ValueError, match="activity_active_weight"):
            validate_config(config)
    config.loss.h_distill.activity_active_weight = 0.5
    config.loss.kind = "scale_event"
    with pytest.raises(ValueError, match="dense loss"):
        validate_config(config)
    config.loss.kind = "dense"
    config.dataset.activity_mask = False
    with pytest.raises(ValueError, match="activity_mask"):
        validate_config(config)
