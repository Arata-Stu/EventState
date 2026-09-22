"""Self-contained specification checks; no reference repository is imported."""

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


@pytest.mark.parametrize("color,active", [
    ((255, 255, 255), False), ((255, 254, 255), True), ((255, 0, 0), True),
])
def test_activation_constant_colors_have_known_classification(color, active):
    pytest.importorskip("cv2")
    image = np.empty((48, 80, 3), dtype=np.uint8)
    image[:] = color
    np.testing.assert_array_equal(
        scale_event_activation(image, height=32, width=64), np.full((2, 4), active),
    )


def test_activation_retains_two_stage_sampling_not_pooling():
    pytest.importorskip("cv2")
    # 32 -> 8 -> 2 samples original coordinates 5,6,9,10 for output index 0.
    # An event at (0,0) is present in that patch but absent from those samples.
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    assert not scale_event_activation(image, height=32, width=32).any()
    image[5, 5] = (255, 0, 0)
    expected = np.array([[True, False], [False, False]])
    np.testing.assert_array_equal(scale_event_activation(image, height=32, width=32), expected)


def _scalar_crossgram_oracle(prediction, target, mask):
    """Float64, scalar pair enumeration from the documented loss definition.

    Kept independent of torch autograd, tensor matmul, row chunking, and the
    implementation under test. Used also for finite-difference gradients.
    """
    p = np.asarray(prediction, dtype=np.float64) * mask[..., None]
    q = np.asarray(target, dtype=np.float64) * mask[..., None]
    l1 = np.abs(p - q).mean()
    p = p.reshape(-1, p.shape[-2], p.shape[-1])
    q = q.reshape(p.shape)
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    intra = cross = 0.0
    for frame in range(len(p)):
        for i in range(p.shape[1]):
            for j in range(p.shape[1]):
                reference = float(np.dot(q[frame, i], q[frame, j]))
                if reference <= 0.1:
                    continue
                event = max(0.0, float(np.dot(p[frame, i], p[frame, j])))
                mixed = max(0.0, float(np.dot(q[frame, i], p[frame, j])))
                intra += (event - reference) ** 2
                cross += (mixed - reference) ** 2
    denominator = len(p) * p.shape[1] ** 2
    intra /= denominator
    cross /= denominator
    return l1 + 10 * intra + 4 * cross, l1, intra, cross


@pytest.mark.parametrize("chunk_size", [1, 3, 99])
@pytest.mark.parametrize("mask_kind", ["mixed", "empty", "full"])
def test_loss_matches_scalar_definition_and_finite_difference_gradients(chunk_size, mask_kind):
    torch.manual_seed(42)
    prediction = torch.randn(2, 3, 7, 5, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    mask = torch.rand(2, 3, 7) > 0.5
    if mask_kind != "mixed":
        mask.fill_(mask_kind == "full")
    actual = ScaleEventLoss(chunk_size).components(prediction, target, mask)
    p = prediction.detach().numpy().astype(np.float64)
    q = target.detach().numpy().astype(np.float64)
    m = mask.numpy()
    expected = _scalar_crossgram_oracle(p, q, m)
    for key, value in zip(("loss", "l1", "intra", "cross"), expected):
        assert actual[key].item() == pytest.approx(value, abs=1e-6, rel=1e-5)
    actual["loss"].backward()
    for index in (0, 6, 23, 78, 130, 209):
        delta = np.zeros_like(p)
        delta.flat[index] = 1e-5
        derivative = (_scalar_crossgram_oracle(p + delta, q, m)[0]
                      - _scalar_crossgram_oracle(p - delta, q, m)[0]) / 2e-5
        assert prediction.grad.flatten()[index].item() == pytest.approx(
            derivative, abs=2e-5, rel=2e-3,
        )
    assert target.grad is None
    assert torch.count_nonzero(prediction.grad[~mask]) == 0


def test_crossgram_hand_calculated_example():
    prediction = torch.tensor([[[1., 0.], [0., 1.]]])
    teacher = torch.tensor([[[1., 0.], [1., 0.]]])
    values = ScaleEventLoss().components(prediction, teacher)
    assert values["l1"].item() == pytest.approx(0.5)
    assert values["intra"].item() == pytest.approx(0.5)
    assert values["cross"].item() == pytest.approx(0.5)
    assert values["loss"].item() == pytest.approx(7.5)


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
