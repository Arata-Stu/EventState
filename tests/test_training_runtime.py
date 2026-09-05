from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from evaluate import compose_evaluation_config
from event_state.training import data as training_data
from event_state.training.checkpoint import critical_config
from event_state.training.engine import EventStateTrainer
from event_state.training.factory import validate_config
from event_state.training.metrics import EventCountAnalysis
from event_state.training.optim import WarmupCosineScheduler, build_optimizer


class TinyEventState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.event_encoder = nn.Linear(3, 4)
        self.temporal_model = nn.LSTM(4, 4, batch_first=True)
        self.h_projector = nn.Linear(4, 4)
        self.z_projector = nn.Linear(4, 4)


class TinySequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.event_encoder = nn.Linear(3, 4)
        self.temporal_model = nn.Linear(4, 4)
        self.h_projector = nn.Linear(4, 4)
        self.z_projector = nn.Linear(4, 4)

    def forward_sequence(self, events: torch.Tensor, state=None) -> dict:
        del state
        pooled = events.mean(dim=(-1, -2))
        z = self.event_encoder(pooled).unsqueeze(2)
        h = self.temporal_model(z)
        return {"z": z, "h": h, "state": None}

    def project_z(self, value: torch.Tensor) -> torch.Tensor:
        return self.z_projector(value)

    def project_h(self, value: torch.Tensor) -> torch.Tensor:
        return self.h_projector(value)


class RecordingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)
        self.output_dtype: torch.dtype | None = None
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projection(images).mean(dim=(-1, -2)).unsqueeze(1)
        self.output_dtype = features.dtype
        return features


def _trainer_config(*, h_enabled: bool = True, z_type: str = "none") -> dict:
    return {
        "training": {"precision": "fp32"},
        "teacher": {"cache_features": True},
        "model": {"temporal": {"detach_state_every": None}},
        "loss": {
            "h_distill": {"enabled": h_enabled},
            "z_objective": {"type": z_type, "weight": 1.0},
        },
        "output": {"directory": "."},
        "evaluation": {"event_count_thresholds": None},
    }


def _make_trainer(
    model: nn.Module,
    config: dict,
    *,
    teacher: nn.Module | None = None,
    device: torch.device | None = None,
) -> EventStateTrainer:
    return EventStateTrainer(
        model=model,
        teacher=teacher,
        train_loader=None,
        validation_loader=[],
        optimizer=None,
        scheduler=None,
        h_distillation_loss=nn.MSELoss(),
        z_distillation_loss=nn.MSELoss(),
        config=config,
        device=device or torch.device("cpu"),
        logger=object(),
        checkpoint_config=config,
    )


def test_optimizer_groups_cover_parameters_once_and_apply_lr_multipliers() -> None:
    model = TinyEventState()
    optimizer = build_optimizer(
        model,
        {
            "type": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "event_encoder_lr_mult": 0.1,
            "temporal_lr_mult": 0.5,
            "projector_lr_mult": 1.0,
        },
    )
    grouped_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)
    group_lrs = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert all(
        math.isclose(value, 1e-4)
        for name, value in group_lrs.items()
        if name.startswith("event_encoder/")
    )


def test_warmup_cosine_scheduler_round_trip() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=0.1
    )
    first_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    scheduler.step()
    state = scheduler.state_dict()
    restored = WarmupCosineScheduler(
        optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=0.1
    )
    restored.load_state_dict(state)
    assert first_lr < 1e-3
    assert restored.num_updates == scheduler.num_updates
    assert restored.get_last_lr() == scheduler.get_last_lr()


def test_event_count_analysis_uses_three_non_overlapping_ranges() -> None:
    analysis = EventCountAnalysis()
    counts = torch.tensor([[1, 5, 10, 50, 100, 500]])
    z_cosine = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    h_cosine = z_cosine + 0.1
    analysis.update(counts, z_cosine, h_cosine)
    metrics = analysis.compute((10, 100))
    assert metrics["event_count/low_frames"] == 3
    assert metrics["event_count/medium_frames"] == 2
    assert metrics["event_count/high_frames"] == 1
    assert metrics["alignment/low_h_cosine"] > metrics["alignment/low_z_cosine"]


def _valid_config() -> dict:
    return {
        "dataset": {
            "event_window": "rgb_interval",
            "sequence_length": 8,
            "input_height": 448,
            "input_width": 640,
            "representation": {
                "type": "gep_rgb",
                "event_bins": 10,
                "polarity_split": True,
            },
            "augmentation": {"enabled": False},
        },
        "model": {
            "event_encoder": {"in_channels": 3, "patch_init": "pretrained"},
            "temporal": {"output_dim": 384},
            "projector": {"input_dim": 384, "output_dim": 384},
        },
        "teacher": {
            "patch_size": 16,
            "embedding_dim": 384,
            "frozen": True,
            "pretrained": True,
            "feature": "x_norm_patchtokens",
            "cache_features": True,
            "cache_dir": "/cache/dinov3_vits16",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "loss": {
            "h_distill": {"enabled": True},
            "z_objective": {"type": "none"},
        },
        "training": {
            "max_steps": 10,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "gradient_accumulation": 1,
            "log_every": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "validation_batches": 1,
            "gradient_clip": 1.0,
        },
        "scheduler": {"warmup_steps": 1},
        "evaluation": {"event_count_thresholds": None, "max_batches": None},
    }


def test_cached_teacher_rejects_stochastic_geometry() -> None:
    config = _valid_config()
    config["dataset"]["augmentation"]["enabled"] = True
    with pytest.raises(ValueError, match="Cached DINO features"):
        validate_config(config)


def test_cached_teacher_requires_cache_directory() -> None:
    config = _valid_config()
    config["teacher"]["cache_dir"] = None
    with pytest.raises(ValueError, match="teacher.cache_dir"):
        validate_config(config)


def test_voxel_grid_requires_matching_event_encoder_channels() -> None:
    config = _valid_config()
    config["dataset"]["representation"]["type"] = "voxel_grid"
    with pytest.raises(ValueError, match="produces 20 channels"):
        validate_config(config)


@pytest.mark.parametrize(
    ("h_enabled", "z_type", "active_projector", "inactive_projector"),
    [
        (True, "none", "h_projector", "z_projector"),
        (False, "direct_dino", "z_projector", "h_projector"),
    ],
)
def test_disabled_distillation_projector_stays_out_of_training_graph(
    h_enabled: bool,
    z_type: str,
    active_projector: str,
    inactive_projector: str,
) -> None:
    model = TinySequenceModel()
    trainer = _make_trainer(
        model,
        _trainer_config(h_enabled=h_enabled, z_type=z_type),
    )
    batch = {
        "events": torch.randn(2, 3, 3, 2, 2),
        "teacher_features": torch.randn(2, 3, 1, 4),
    }

    result = trainer._forward_objectives(batch)
    result["loss"].backward()

    assert getattr(model, active_projector).weight.grad is not None
    assert getattr(model, inactive_projector).weight.grad is None
    assert result[f"{inactive_projector[0]}_prediction"].requires_grad is False
    assert all(
        not result[name].requires_grad
        for name in (
            "z_frame_cosine",
            "h_frame_cosine",
            "z_projected_frame_cosine",
            "h_projected_frame_cosine",
        )
    )


def test_online_teacher_uses_fp32_outside_student_autocast() -> None:
    model = TinySequenceModel()
    teacher = RecordingTeacher()
    config = _trainer_config()
    config["training"]["precision"] = "bf16"
    config["teacher"]["cache_features"] = False
    trainer = _make_trainer(model, config, teacher=teacher)

    features = trainer._teacher_features({"images": torch.randn(1, 2, 3, 2, 2)})

    assert features.dtype == torch.float32
    assert teacher.output_dtype == torch.float32


@pytest.mark.parametrize(
    ("device_type", "precision"),
    [("mps", "bf16"), ("cpu", "fp16")],
)
def test_unavailable_amp_reports_fp32_fallback(
    device_type: str,
    precision: str,
) -> None:
    config = _trainer_config()
    config["training"]["precision"] = precision
    with pytest.warns(RuntimeWarning, match="falling back to fp32"):
        trainer = _make_trainer(
            TinySequenceModel(),
            config,
            device=torch.device(device_type),
        )
    assert trainer.actual_precision == "fp32"
    assert trainer.effective_precision == trainer.actual_precision


def test_evaluation_config_restores_checkpoint_semantics_and_runtime_locations() -> None:
    saved = _valid_config()
    saved.update(
        {
            "seed": 7,
            "optimizer": {"type": "adamw", "lr": 1e-4},
            "output": {"directory": "/old/output"},
            "device": "cuda",
        }
    )
    saved["dataset"].update(
        {
            "root": "/old/data",
            "event_cache_dir": "/old/event-cache",
            "train_split": "train",
            "val_split": "test",
            "train_sequences": ["train_a"],
            "val_sequences": ["test_a"],
        }
    )
    saved["teacher"]["cache_dir"] = "/old/teacher-cache"
    current = copy.deepcopy(saved)
    current["model"]["temporal"]["output_dim"] = 999
    current["loss"]["z_objective"]["type"] = "direct_dino"
    current["dataset"].update(
        {
            "root": "/new/data",
            "event_cache_dir": None,
            "val_split": "validation",
            "val_sequences": ["validation_a"],
        }
    )
    current["teacher"]["cache_dir"] = "/new/teacher-cache"
    current["training"]["num_workers"] = 0
    current["evaluation"] = {
        "checkpoint": "/checkpoints/model.pt",
        "max_batches": None,
        "event_count_thresholds": [10, 100],
    }
    current["output"] = {"directory": "/new/output"}
    current["device"] = "cpu"

    composed = compose_evaluation_config(
        saved,
        OmegaConf.create(current),
        explicit_runtime_paths={
            "dataset.root",
            "dataset.event_cache_dir",
            "dataset.val_split",
            "dataset.val_sequences",
            "teacher.cache_dir",
        },
    )

    assert composed.model.temporal.output_dim == 384
    assert composed.loss.z_objective.type == "none"
    assert composed.dataset.root == "/new/data"
    assert composed.dataset.event_cache_dir is None
    assert composed.dataset.val_split == "validation"
    assert list(composed.dataset.val_sequences) == ["validation_a"]
    assert composed.teacher.cache_dir == "/new/teacher-cache"
    assert composed.training.num_workers == 0
    assert composed.evaluation.event_count_thresholds == [10, 100]
    assert composed.output.directory == "/new/output"
    assert composed.device == "cpu"


def test_evaluation_config_does_not_replace_saved_cache_paths_with_defaults() -> None:
    saved = _valid_config()
    saved["dataset"]["root"] = "/saved/data"
    saved["dataset"]["event_cache_dir"] = "/saved/event-cache"
    saved["teacher"]["cache_dir"] = "/saved/teacher-cache"
    saved["output"] = {"directory": "/saved/output"}
    saved["device"] = "cuda"
    current = copy.deepcopy(saved)
    current["dataset"]["root"] = None
    current["dataset"]["event_cache_dir"] = None
    current["teacher"]["cache_dir"] = "cache/dinov3_vits16"
    current["output"] = {"directory": "/evaluation/output"}
    current["device"] = "auto"

    composed = compose_evaluation_config(saved, OmegaConf.create(current))

    assert composed.dataset.root == "/saved/data"
    assert composed.dataset.event_cache_dir == "/saved/event-cache"
    assert composed.teacher.cache_dir == "/saved/teacher-cache"
    assert composed.output.directory == "/evaluation/output"
    assert composed.device == "auto"


def test_validation_only_loader_does_not_construct_the_train_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict] = []

    class DummyRepresentation:
        channels = 3

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            del index
            return {"events": torch.zeros(1, 3, 2, 2)}

    monkeypatch.setattr(
        training_data,
        "build_event_representation",
        lambda dataset_config: DummyRepresentation(),
    )
    monkeypatch.setattr(training_data, "DSECSequenceDataset", DummyDataset)
    config = {
        "seed": 0,
        "dataset": {
            "name": "dsec",
            "root": "/validation-only",
            "sequence_length": 8,
            "val_split": "test",
            "val_sequences": ["sequence_v"],
            "image_directory": "aligned_event",
            "rectify_events": True,
            "event_cache_dir": None,
            "representation": {
                "type": "gep_rgb",
                "normalize_mean": [0.0, 0.0, 0.0],
                "normalize_std": [1.0, 1.0, 1.0],
            },
            "augmentation": {"enabled": False},
            "input_height": 448,
            "input_width": 640,
        },
        "teacher": {"cache_features": False},
        "training": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }

    loader = training_data.build_validation_dataloader(config)

    assert len(constructed) == 1
    assert constructed[0]["split"] == "test"
    assert constructed[0]["sequences"] == ["sequence_v"]
    assert loader.batch_size == 1


def test_evaluation_local_dino_relocation_uses_content_identity(tmp_path: Path) -> None:
    saved_repository = tmp_path / "old_mount" / "dinov3"
    current_repository = tmp_path / "new_mount" / "dinov3"
    saved_repository.mkdir(parents=True)
    current_repository.mkdir(parents=True)
    (saved_repository / "hubconf.py").write_text("MODEL = 'vits16'\n", encoding="utf-8")
    (current_repository / "hubconf.py").write_text("MODEL = 'vits16'\n", encoding="utf-8")
    saved_checkpoint = tmp_path / "old_mount" / "weights.pt"
    current_checkpoint = tmp_path / "new_mount" / "weights.pt"
    saved_checkpoint.write_bytes(b"same weights")
    current_checkpoint.write_bytes(b"same weights")

    saved = _valid_config()
    saved.update({"device": "cpu", "output": {"directory": "/saved/output"}})
    saved["teacher"].update(
        {
            "backend": "torch_hub",
            "source": "local",
            "repository": str(saved_repository),
            "checkpoint": str(saved_checkpoint),
        }
    )
    # Repository is individually specified while backend/source/checkpoint use
    # the same effective fallback as the factory.
    saved["model"]["event_encoder"]["repository"] = str(saved_repository)
    current = copy.deepcopy(saved)
    current["teacher"]["repository"] = str(current_repository)
    current["teacher"]["checkpoint"] = current_checkpoint.as_uri()
    current["model"]["event_encoder"]["repository"] = str(current_repository)
    signature = critical_config(saved, mode="evaluation")

    composed = compose_evaluation_config(
        saved,
        OmegaConf.create(current),
        explicit_runtime_paths={
            "teacher.repository",
            "teacher.checkpoint",
            "model.event_encoder.repository",
        },
        checkpoint_signature=signature,
    )

    assert composed.teacher.repository == str(current_repository)
    assert composed.teacher.checkpoint == current_checkpoint.as_uri()
    assert composed.model.event_encoder.repository == str(current_repository)

    changed_repository = tmp_path / "different_mount" / "dinov3"
    changed_repository.mkdir(parents=True)
    (changed_repository / "hubconf.py").write_text("MODEL = 'different'\n", encoding="utf-8")
    current["teacher"]["repository"] = str(changed_repository)
    current["model"]["event_encoder"]["repository"] = str(changed_repository)
    with pytest.raises(ValueError, match="repository_identity"):
        compose_evaluation_config(
            saved,
            OmegaConf.create(current),
            explicit_runtime_paths={
                "teacher.repository",
                "teacher.checkpoint",
                "model.event_encoder.repository",
            },
            checkpoint_signature=signature,
        )


def test_teacher_cache_identity_treats_file_uri_as_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher weights.pt"
    checkpoint.write_bytes(b"same checkpoint")
    teacher = {
        "type": "dinov3_vits16",
        "backend": "torch_hub",
        "source": "github",
        "repository": (
            "facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa"
        ),
        "checkpoint": str(checkpoint),
        "pretrained": True,
    }

    plain_identity = training_data._expected_teacher_identity(teacher)
    teacher["checkpoint"] = checkpoint.as_uri()
    uri_identity = training_data._expected_teacher_identity(teacher)

    assert uri_identity["checkpoint"] == plain_identity["checkpoint"]
    assert uri_identity["checkpoint"]["kind"] == "local_file"
