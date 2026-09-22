"""Numerical reference checks to run on the training host (torch + OpenCV)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from event_state.data.activity import scale_event_activation
from event_state.data.transforms import PairedSequenceTransform
from event_state.losses.scale_event import ScaleEventLoss


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_composition_enables_data_and_both_loss_branches():
    from hydra import compose, initialize_config_dir

    from event_state.training.factory import _build_losses, validate_config

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="config", overrides=[
            "dataset=dsec_joint_clean", "experiment=scale_event_dual",
        ])
    validate_config(config)
    assert config.dataset.activity_mask
    assert config.loss.h_distill.enabled
    assert config.loss.z_objective.type == "direct_dino"
    assert all(isinstance(loss, ScaleEventLoss) for loss in _build_losses(config))
    config.dataset.train_sequences.append("zurich_city_07_a")
    with pytest.raises(ValueError, match="leakage"):
        validate_config(config)


def test_activity_only_experiment_preserves_existing_final_fit_settings():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from event_state.training.factory import _build_losses, validate_config
    from event_state.losses.activity import ActivityDistillationLoss

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        baseline = compose(config_name="config", overrides=[
            "dataset=dsec_det_train41", "experiment=h_distill_lstm_zloss",
            "training.validation_enabled=false",
        ])
        activity = compose(config_name="config", overrides=[
            "dataset=dsec_det_train41", "experiment=activity_dual",
        ])
    validate_config(activity)
    for section in ("training", "optimizer", "scheduler", "model", "teacher"):
        assert OmegaConf.to_container(baseline[section]) == OmegaConf.to_container(activity[section])
    assert baseline.dataset.train_sequences == activity.dataset.train_sequences
    assert baseline.dataset.val_sequences == activity.dataset.val_sequences
    assert all(isinstance(loss, ActivityDistillationLoss) for loss in _build_losses(activity))


def reference_module(relative):
    path = ROOT / "reference_repo/ScaleEvent/scale_event" / relative
    if not path.is_file():
        pytest.skip("ScaleEvent reference checkout is required for numerical comparison")
    spec = importlib.util.spec_from_file_location("scale_event_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["empty", "full", "sparse", "near_white"])
def test_activation_matches_released_png_path(tmp_path, kind):
    cv2 = pytest.importorskip("cv2")
    image = np.full((48, 80, 3), 255, dtype=np.uint8)
    if kind == "full":
        image[:] = [0, 0, 255]
    elif kind == "sparse":
        rng = np.random.default_rng(9)
        image[rng.random((48, 80)) < 0.1] = [255, 0, 0]
    elif kind == "near_white":
        image[::2, :, 0] = 254
    path = tmp_path / "events.png"
    cv2.imwrite(str(path), image)
    expected = reference_module("dataset/event_utils.py").prepare_mask(str(path), W=64, H=32)
    actual = scale_event_activation(image, height=32, width=64)
    np.testing.assert_array_equal(actual, expected.numpy()[..., 0].astype(bool))


@pytest.mark.parametrize("chunk_size", [1, 3, 99])
@pytest.mark.parametrize("mask_kind", ["mixed", "empty", "full"])
def test_loss_and_gradients_match_released_crossgram(chunk_size, mask_kind):
    torch.manual_seed(42)
    prediction = torch.randn(2, 3, 7, 5, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    mask = torch.rand(2, 3, 7) > 0.5
    if mask_kind != "mixed":
        mask.fill_(mask_kind == "full")
    actual = ScaleEventLoss(chunk_size).components(prediction, target, mask)
    reference_prediction = prediction.detach().clone().requires_grad_()
    criterion = reference_module("pretrain/criterion.py").DistillLoss_With_CrossGram()
    expected = criterion(
        {"x_norm_patchtokens": target.detach().flatten(0, 1)},
        {"x_norm_patchtokens": reference_prediction.flatten(0, 1)},
        mask.flatten(0, 1).unsqueeze(-1).float(),
    )
    for key, value in zip(("loss", "l1", "intra", "cross"), expected):
        torch.testing.assert_close(actual[key], value, atol=1e-6, rtol=1e-5)
    actual["loss"].backward()
    expected[0].backward()
    torch.testing.assert_close(prediction.grad, reference_prediction.grad, atol=1e-6, rtol=1e-5)
    assert target.grad is None
    assert torch.count_nonzero(prediction.grad[~mask]) == 0


def test_activity_uses_same_crop_and_flip_before_normalization(monkeypatch):
    pytest.importorskip("cv2")
    transform = PairedSequenceTransform(
        height=32, width=64, training=True, horizontal_flip_probability=1.0,
        event_mean=(0.9, 0.8, 0.9), event_std=(0.2, 0.3, 0.2),
    )
    monkeypatch.setattr(transform, "_sample_crop", lambda *args: (8, 16, 32, 64))
    events = torch.ones(2, 3, 48, 96)
    events[:, :2, 8:40, 16:48] = 0
    _, _, activity = transform(events, None, return_activity=True)
    source = (events[0, :, 8:40, 16:80] * 255).byte().permute(1, 2, 0).numpy()
    expected = torch.from_numpy(scale_event_activation(source, height=32, width=64)).flip(-1)
    assert torch.equal(activity[0], expected.flatten())
    assert torch.equal(activity[0], activity[1])


def test_h_receives_complement_and_dropout_goes_to_h():
    # Import the existing tiny trainer fixture without building a real DINO model.
    from test_training_runtime import TinySequenceModel, _make_trainer, _trainer_config

    config = _trainer_config(h_enabled=True, z_type="direct_dino")
    config["loss"]["kind"] = "scale_event"
    config["dataset"] = {"activity_mask": True}
    trainer = _make_trainer(TinySequenceModel(), config)
    trainer.h_distillation_loss = ScaleEventLoss()
    trainer.z_distillation_loss = ScaleEventLoss()
    batch = {"events": torch.randn(1, 3, 3, 16, 16),
             "teacher_features": torch.randn(1, 3, 1, 4),
             "event_activity": torch.tensor([[[True], [False], [True]]])}
    dropout = torch.tensor([[False, False, True]])
    batch["events"][:, 2] = 0
    result = trainer._forward_objectives(batch, event_dropout_mask=dropout)
    active = torch.tensor([[[True], [False], [False]]])
    torch.testing.assert_close(result["z_components"]["loss"], ScaleEventLoss()(
        result["z_prediction"], batch["teacher_features"], active))
    torch.testing.assert_close(result["h_components"]["loss"], ScaleEventLoss()(
        result["h_prediction"], batch["teacher_features"], ~active))
    result["loss"].backward()
    assert trainer.model.z_projector.weight.grad is not None
    assert trainer.model.h_projector.weight.grad is not None


def test_activity_moves_with_the_batch_without_changing_bool_dtype():
    from test_training_runtime import TinySequenceModel, _make_trainer, _trainer_config

    trainer = _make_trainer(TinySequenceModel(), _trainer_config())
    # Meta exercises transfer without needing an accelerator on the test host.
    trainer.device = torch.device("meta")
    activity = torch.ones(1, 2, 3, dtype=torch.bool)
    moved = trainer._move_batch({"event_activity": activity})
    assert moved["event_activity"].device.type == "meta"
    assert moved["event_activity"].dtype == torch.bool
    assert activity.device.type == "cpu"
