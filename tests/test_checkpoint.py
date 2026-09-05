from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from event_state.training.checkpoint import (
    assert_checkpoint_config_compatible,
    load_checkpoint,
    save_checkpoint,
)
from event_state.training.optim import WarmupCosineScheduler


def test_checkpoint_restores_model_optimizer_scheduler_and_step(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=1,
        total_steps=10,
        min_lr_ratio=0.1,
    )
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        global_step=7,
        best_validation_loss=0.25,
        data_state={"epoch": 2, "batch_in_epoch": 3},
        config={"experiment": "test"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    restored = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=False,
    )

    assert restored.global_step == 7
    assert restored.best_validation_loss == 0.25
    assert restored.data_state == {"epoch": 2, "batch_in_epoch": 3}
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def _experiment_config() -> dict:
    return {
        "seed": 0,
        "model": {"temporal": {"type": "lstm", "hidden_dim": 4}},
        "teacher": {"type": "dinov3_vits16", "cache_dir": "/cache/a"},
        "dataset": {
            "root": "/data/a",
            "event_cache_dir": "/cache/events-a",
            "train_split": "train",
            "val_split": "test",
            "train_sequences": ["sequence_a"],
            "val_sequences": ["sequence_b"],
            "representation": {"type": "gep_rgb", "channels": 3},
        },
        "loss": {
            "h_distill": {"enabled": True},
            "z_objective": {"type": "none"},
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "scheduler": {"warmup_steps": 10},
        "training": {
            "max_steps": 100,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "num_workers": 4,
            "log_every": 20,
            "resume": None,
        },
        "device": "cuda",
        "output": {"directory": "/run/a"},
    }


def test_resume_compatibility_allows_runtime_paths_but_rejects_objective_change() -> None:
    saved = _experiment_config()
    relocated = copy.deepcopy(saved)
    relocated["dataset"]["root"] = "/mounted/data"
    relocated["dataset"]["event_cache_dir"] = "/mounted/event-cache"
    relocated["teacher"]["cache_dir"] = "/mounted/teacher-cache"
    relocated["training"]["num_workers"] = 8
    relocated["training"]["log_every"] = 5
    relocated["device"] = "cpu"
    relocated["output"]["directory"] = "/run/b"

    assert_checkpoint_config_compatible(saved, relocated, mode="resume")

    relocated["loss"]["z_objective"]["type"] = "direct_dino"
    with pytest.raises(ValueError, match=r"loss\.z_objective\.type"):
        assert_checkpoint_config_compatible(saved, relocated, mode="resume")


def test_evaluation_compatibility_allows_split_and_sequence_selection() -> None:
    saved = _experiment_config()
    evaluation = copy.deepcopy(saved)
    evaluation["dataset"]["val_split"] = "validation"
    evaluation["dataset"]["val_sequences"] = ["sequence_c"]

    assert_checkpoint_config_compatible(saved, evaluation, mode="evaluation")


def test_checkpoint_load_rejects_critical_config_before_state_restore(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=1, total_steps=10, min_lr_ratio=0.1
    )
    saved_config = _experiment_config()
    checkpoint_path = tmp_path / "incompatible.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        global_step=1,
        best_validation_loss=1.0,
        data_state={},
        config=saved_config,
    )
    current_config = copy.deepcopy(saved_config)
    current_config["loss"]["z_objective"]["type"] = "direct_dino"

    with pytest.raises(ValueError, match=r"loss\.z_objective\.type"):
        load_checkpoint(
            checkpoint_path,
            model=model,
            restore_rng=False,
            expected_config=current_config,
        )


def test_local_dino_paths_compare_by_source_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    saved_repository = tmp_path / "machine_a" / "dinov3"
    current_repository = tmp_path / "machine_b" / "dinov3"
    saved_repository.mkdir(parents=True)
    current_repository.mkdir(parents=True)
    (saved_repository / "hubconf.py").write_text("MODEL = 'vits16'\n", encoding="utf-8")
    (current_repository / "hubconf.py").write_text("MODEL = 'vits16'\n", encoding="utf-8")
    saved_weights = tmp_path / "machine_a" / "weights.pt"
    current_weights = tmp_path / "machine_b" / "weights.pt"
    saved_weights.write_bytes(b"same frozen teacher weights")
    current_weights.write_bytes(b"same frozen teacher weights")

    saved = _experiment_config()
    saved["teacher"].update(
        {
            "backend": "torch_hub",
            "source": "local",
            "repository": str(saved_repository),
            "checkpoint": str(saved_weights),
            "pretrained": True,
        }
    )
    current = copy.deepcopy(saved)
    current["teacher"]["repository"] = str(current_repository)
    current["teacher"]["checkpoint"] = current_weights.as_uri()

    assert_checkpoint_config_compatible(saved, current, mode="resume")

    # The identity cache represents immutable source during one process, so use
    # a third path to exercise a genuinely different source identity.
    changed_repository = tmp_path / "machine_c" / "dinov3"
    changed_repository.mkdir(parents=True)
    (changed_repository / "hubconf.py").write_text("MODEL = 'changed'\n", encoding="utf-8")
    current["teacher"]["repository"] = str(changed_repository)
    with pytest.raises(ValueError, match="repository_identity"):
        assert_checkpoint_config_compatible(saved, current, mode="resume")
