"""Hydra entry point for EventState Phase 0/1 pretraining."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from event_state.training.checkpoint import load_checkpoint_config
from event_state.training.engine import EventStateTrainer
from event_state.training.logging import TrainingLogger
from event_state.training.factory import build_runtime


def _resume_path(value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    expanded = Path(str(value)).expanduser()
    path = Path(to_absolute_path(str(expanded)))
    if path.is_dir():
        candidates = sorted(path.glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No step checkpoints found in {path}")
        return candidates[-1]
    return path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True)
    checkpoint_config = OmegaConf.to_container(config, resolve=True)
    resume_path = _resume_path(config.training.resume)
    if resume_path is not None:
        # Reject objective/model/data-semantic drift before building datasets or
        # loading DINO weights. EventStateTrainer.resume validates it again at
        # the state-restoration boundary.
        load_checkpoint_config(
            resume_path,
            expected_config=checkpoint_config,
            compatibility_mode="resume",
        )
    runtime = build_runtime(config)
    logger = TrainingLogger(config.output.directory, resolved_yaml)
    try:
        trainer = EventStateTrainer(
            model=runtime.model,
            teacher=runtime.teacher,
            train_loader=runtime.dataloaders.train,
            validation_loader=runtime.dataloaders.validation,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            h_distillation_loss=runtime.h_distillation_loss,
            z_distillation_loss=runtime.z_distillation_loss,
            config=config,
            device=runtime.device,
            logger=logger,
            checkpoint_config=checkpoint_config,
        )
        print(
            f"Starting {config.experiment.name} on {runtime.device}; "
            f"teacher={'cache' if runtime.teacher is None else 'online'}",
            flush=True,
        )
        if resume_path is not None:
            trainer.resume(resume_path)
            print(f"Resumed {resume_path} at step {trainer.global_step}", flush=True)
        trainer.fit()
    finally:
        logger.close()


if __name__ == "__main__":
    main()
