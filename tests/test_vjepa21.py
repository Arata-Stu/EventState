"""Run on the training server: tiny synthetic teacher, no downloads/data needed."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from event_state.causal_rgb import history_positions
from event_state.models.teacher.vjepa21 import FrozenVJEPA21Teacher


class FakeVideoEncoder(nn.Module):
    embed_dim = 768

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, video):
        self.calls.append(tuple(video.shape))
        # All-time mean mimics bidirectional attention; catches future leakage.
        pooled = torch.nn.functional.avg_pool3d(video, (1 if video.shape[2] == 1 else 2, 16, 16)).mean(1)
        pooled = pooled + pooled.mean((1, 2, 3), keepdim=True)
        return pooled.flatten(1).unsqueeze(-1).expand(-1, -1, 768) * self.scale


def teacher(micro=1):
    return FrozenVJEPA21Teacher(backbone=FakeVideoEncoder(), context_frames=4,
                               micro_batch_size=micro, image_mean=(0, 0, 0), image_std=(1, 1, 1))


def inputs(start, length):
    p = history_positions(start, length, 4)
    images = torch.tensor(p).float().reshape(1, -1, 1, 1, 1).expand(1, -1, 3, 32, 48).clone()
    times = torch.tensor([p]) * 50000
    targets = times[:, 3:]
    return images, times, targets


def test_future_perturbation_and_chunk_invariance():
    model = teacher()
    images, times, targets = inputs(1, 6)
    full = model.forward_history(images, times, targets)
    changed = images.clone()
    changed[:, 6:] = 10000  # Targets 1,2,3 (positions <=3) must not change.
    modified = model.forward_history(changed, times, targets)
    torch.testing.assert_close(full[:, :3], modified[:, :3])
    assert not torch.equal(full[:, 3:], modified[:, 3:])
    parts = [model.forward_history(*inputs(1, 2)), model.forward_history(*inputs(3, 4))]
    torch.testing.assert_close(full, torch.cat(parts, dim=1))
    torch.testing.assert_close(full, teacher(3).forward_history(images, times, targets))
    assert full.shape == (1, 6, 6, 768)
    assert not full.requires_grad
    assert max(shape[0] for shape in model.backbone.calls) == 1


def test_batch_sequence_isolation_and_freeze():
    model = teacher(3)
    model.train()
    assert not model.training and not model.backbone.training
    assert all(not p.requires_grad for p in model.parameters())
    images, times, targets = inputs(5, 2)
    batch = model.forward_history(torch.cat((images, images + 500)), times.repeat(2, 1), targets.repeat(2, 1))
    single = model.forward_history(images, times, targets)
    torch.testing.assert_close(batch[:1], single)


def test_endpoint_and_shape_fail_closed():
    model = teacher()
    images, times, targets = inputs(1, 2)
    with pytest.raises(ValueError, match="misaligned"):
        model.forward_history(images, times, targets + 1)
    with pytest.raises(ValueError, match="length"):
        model.forward_history(images[:, 1:], times[:, 1:], targets)


@pytest.mark.parametrize("dataset_name", ["dsec", "m3ed"])
@pytest.mark.parametrize("mode,k,fps", [("video", 4, None), ("video", 4, 4), ("image", 1, None)])
def test_dataset_history_and_shared_geometry(dataset_name, mode, k, fps):
    from event_state.data.dsec import DSECSequenceDataset
    from event_state.data.m3ed import M3EDSequenceDataset
    from event_state.data.transforms import PairedSequenceTransform
    from event_state.data.video_history import configure_video_history
    cls = DSECSequenceDataset if dataset_name == "dsec" else M3EDSequenceDataset
    data = cls.__new__(cls)
    data._clips = [("seq", 1, 4), ("seq", 3, 2)]
    data._frames_by_sequence = {"seq": [SimpleNamespace(timestamp=i * 50000, frame_index=i, image_path=i) for i in range(6)]}
    data.load_events = data.load_images = True
    data.activity_mask = False
    data.feature_cache_dir = None
    data.target_tasks = ()
    data.transform = PairedSequenceTransform(height=32, width=48)
    configure_video_history(data, k, mode, fps)
    image = lambda index: torch.full((3, 48, 64), float(index))
    data._load_image = image
    data._load_event_frame = data._load_event = lambda record: (image(record.frame_index), 1)
    whole, part = data[0], data[(1, 17)]
    expanded = whole["teacher_images"][whole["teacher_indices"]]
    torch.testing.assert_close(expanded[2:], part["teacher_images"][part["teacher_indices"]])
    torch.testing.assert_close(whole["events"], expanded[:, -1])
    assert whole["teacher_timestamps"][:, -1].tolist() == [50000, 100000, 150000, 200000]
    assert "teacher_features" not in whole and "images" not in whole


def test_image_teacher_uses_one_frame_and_no_temporal_leakage():
    model = FrozenVJEPA21Teacher(backbone=FakeVideoEncoder(), input_mode="image", context_frames=1,
                                 image_mean=(0, 0, 0), image_std=(1, 1, 1), micro_batch_size=2)
    images = torch.randn(1, 3, 3, 32, 48)
    times = torch.tensor([[0, 50000, 100000]])
    before = model.forward_history(images, times, times)
    images[:, 1:] += 100
    after = model.forward_history(images, times, times)
    torch.testing.assert_close(before[:, :1], after[:, :1])
    assert before.shape == (1, 3, 6, 768)
    assert all(shape[2] == 1 for shape in model.backbone.calls)


def test_packed_history_collation_padding_and_timestamp_checks():
    from event_state.data.video_history import collate_teacher_history
    model = FrozenVJEPA21Teacher(backbone=FakeVideoEncoder(), context_frames=4, sample_fps=4,
                                 image_mean=(0, 0, 0), image_std=(1, 1, 1), micro_batch_size=2)
    short = dict(teacher_images=torch.ones(1, 3, 32, 48), teacher_image_count=1,
                 teacher_indices=torch.tensor([[0, 0, 0, 0]]),
                 teacher_timestamps=torch.tensor([[0, 0, 0, 0]]), timestamps=torch.tensor([0]),
                 teacher_requested_timestamps=torch.tensor([[-750000, -500000, -250000, 0]]),
                 teacher_prefix_padding=torch.tensor([[True, True, True, False]]),
                 teacher_sequence_start_timestamp=0)
    long = dict(short, teacher_images=torch.arange(4.).view(4, 1, 1, 1).expand(4, 3, 32, 48),
                teacher_image_count=4, teacher_indices=torch.tensor([[0, 1, 2, 3]]),
                teacher_timestamps=torch.tensor([[0, 250000, 500000, 750000]]), timestamps=torch.tensor([750000]),
                teacher_requested_timestamps=torch.tensor([[0, 250000, 500000, 750000]]),
                teacher_prefix_padding=torch.zeros(1, 4, dtype=torch.bool))
    batch = collate_teacher_history([short, long])
    kwargs = dict(indices=batch["teacher_indices"], image_counts=batch["teacher_image_count"],
                  requested_timestamps=batch["teacher_requested_timestamps"],
                  prefix_padding=batch["teacher_prefix_padding"],
                  sequence_start_timestamps=batch["teacher_sequence_start_timestamp"])
    before = model.forward_history(batch["teacher_images"], batch["teacher_timestamps"], batch["timestamps"], **kwargs)
    batch["teacher_images"][0, 1:] = 1e9  # Collation padding must never be read.
    after = model.forward_history(batch["teacher_images"], batch["teacher_timestamps"], batch["timestamps"], **kwargs)
    torch.testing.assert_close(before, after)
    bad_times = batch["teacher_timestamps"].clone()
    bad_times[1, 0, 1] += 1
    with pytest.raises(ValueError, match="sampling time"):
        model.forward_history(batch["teacher_images"], bad_times, batch["timestamps"], **kwargs)
    kwargs["indices"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="batch padding"):
        model.forward_history(batch["teacher_images"], batch["teacher_timestamps"], batch["timestamps"], **kwargs)


def test_engine_dispatch_keeps_video_history_on_cpu():
    from event_state.training.engine import EventStateTrainer
    trainer = EventStateTrainer.__new__(EventStateTrainer)
    trainer.teacher = teacher()
    trainer.device = torch.device("cpu")
    images, times, targets = inputs(1, 2)
    batch = dict(teacher_images=images, teacher_timestamps=times, timestamps=targets)
    actual = trainer._teacher_features(trainer._move_batch(batch))
    torch.testing.assert_close(actual, trainer.teacher.forward_history(images, times, targets))
    with pytest.raises(ValueError, match="caches"):
        trainer._teacher_features(dict(batch, teacher_features=torch.zeros_like(actual)))


@pytest.mark.parametrize("mode", ["image", "video"])
def test_h_only_backward_does_not_train_teacher_or_z_projector(mode):
    from event_state.models import EventStateModel, TeacherProjection
    from event_state.models.temporal import PatchwiseLSTM
    from event_state.training.engine import EventStateTrainer

    class Encoder(nn.Module):
        embed_dim = 384

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 384, 16, stride=16)

        def forward(self, x):
            return self.conv(x).flatten(2).transpose(1, 2)

    student = EventStateModel(Encoder(), PatchwiseLSTM(input_dim=384, hidden_dim=384, output_dim=384),
                              h_projector=TeacherProjection(384, 768, 768),
                              z_projector=TeacherProjection(384, 768, 768), teacher_dim=768)
    frozen = teacher() if mode == "video" else FrozenVJEPA21Teacher(
        backbone=FakeVideoEncoder(), input_mode="image", context_frames=1,
        image_mean=(0, 0, 0), image_std=(1, 1, 1))
    config = dict(training=dict(precision="fp32"), teacher=dict(cache_features=False),
                  model=dict(temporal=dict(detach_state_every=None)),
                  loss=dict(h_distill=dict(enabled=True), z_objective=dict(type="none", weight=0)),
                  output=dict(directory="."), evaluation=dict(event_count_thresholds=None))
    trainer = EventStateTrainer(model=student, teacher=frozen, train_loader=None,
                                validation_loader=[], optimizer=None, scheduler=None,
                                h_distillation_loss=nn.MSELoss(), z_distillation_loss=nn.MSELoss(),
                                config=config, device=torch.device("cpu"), logger=object(),
                                checkpoint_config=config)
    images, times, targets = inputs(1, 2)
    events = images[:, 3:]
    if mode == "image":
        images, times = images[:, 3:], times[:, 3:]
    result = trainer._forward_objectives(dict(events=events, teacher_images=images,
                                              teacher_timestamps=times, timestamps=targets))
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert student.event_encoder.conv.weight.grad is not None
    assert any(p.grad is not None for p in student.temporal_model.parameters())
    assert any(p.grad is not None for p in student.h_projector.parameters())
    assert all(p.grad is None for p in student.z_projector.parameters())
    assert all(p.grad is None for p in frozen.parameters())
    assert torch.isnan(result["z_frame_cosine"]).all()
