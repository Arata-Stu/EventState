"""Evaluate z/h alignment, including low/medium/high event-count strata."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from event_state.training.checkpoint import (
    assert_checkpoint_config_compatible,
    assert_checkpoint_signature_compatible,
    load_checkpoint,
    load_checkpoint_config_metadata,
)
from event_state.training.engine import EventStateTrainer
from event_state.training.factory import build_evaluation_runtime
from event_state.training.logging import TrainingLogger


_ALWAYS_RUNTIME_OVERRIDE_PATHS = (
    "device",
    "training.num_workers",
    "training.pin_memory",
    "training.persistent_workers",
    "evaluation",
    "output",
)
_EXPLICIT_RUNTIME_OVERRIDE_PATHS = {
    "dataset.root",
    "dataset.event_cache_dir",
    "dataset.train_split",
    "dataset.val_split",
    "dataset.train_sequences",
    "dataset.val_sequences",
    "teacher.cache_dir",
    "teacher.backend",
    "teacher.source",
    "teacher.repository",
    "teacher.checkpoint",
    "model.event_encoder.backend",
    "model.event_encoder.source",
    "model.event_encoder.repository",
    "model.event_encoder.checkpoint",
}


def _explicit_override_paths(overrides: list[str]) -> set[str]:
    paths: set[str] = set()
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in _EXPLICIT_RUNTIME_OVERRIDE_PATHS:
            paths.add(key)
    return paths


def compose_evaluation_config(
    checkpoint_config: object,
    runtime_config: DictConfig,
    *,
    explicit_runtime_paths: set[str] | None = None,
    checkpoint_signature: object | None = None,
) -> DictConfig:
    """Restore experiment semantics while accepting machine/evaluation overrides."""

    saved = OmegaConf.create(checkpoint_config)
    current = OmegaConf.create(OmegaConf.to_container(runtime_config, resolve=True))
    override_paths = (*_ALWAYS_RUNTIME_OVERRIDE_PATHS, *(explicit_runtime_paths or set()))
    for path in override_paths:
        OmegaConf.update(
            saved,
            path,
            OmegaConf.select(current, path),
            merge=False,
            force_add=True,
        )
    composed = OmegaConf.to_container(saved, resolve=True)
    if checkpoint_signature is None:
        # Compatibility path for schema-v1 checkpoints written before
        # content-aware signatures were added. Local source relocation is not
        # possible for those checkpoints when the old path no longer exists.
        assert_checkpoint_config_compatible(
            checkpoint_config,
            composed,
            mode="evaluation",
        )
    else:
        assert_checkpoint_signature_compatible(
            checkpoint_signature,
            composed,
            mode="evaluation",
        )
    return saved


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    checkpoint_value = config.evaluation.checkpoint or config.training.resume
    if not checkpoint_value:
        raise ValueError(
            "Set evaluation.checkpoint=/path/to/checkpoint.pt (or training.resume)"
        )
    expanded_checkpoint = Path(str(checkpoint_value)).expanduser()
    checkpoint_path = Path(to_absolute_path(str(expanded_checkpoint)))
    checkpoint_metadata = load_checkpoint_config_metadata(checkpoint_path)
    saved_config = checkpoint_metadata.config
    explicit_runtime_paths = _explicit_override_paths(
        list(HydraConfig.get().overrides.task)
    )
    evaluation_config = compose_evaluation_config(
        saved_config,
        config,
        explicit_runtime_paths=explicit_runtime_paths,
        checkpoint_signature=checkpoint_metadata.signatures.get("evaluation"),
    )
    # Preserve the actual selected artifact in the resolved evaluation config,
    # including when the backwards-compatible training.resume alias was used.
    OmegaConf.update(
        evaluation_config,
        "evaluation.checkpoint",
        str(checkpoint_path),
        merge=False,
    )
    runtime = build_evaluation_runtime(evaluation_config)
    resolved_yaml = OmegaConf.to_yaml(evaluation_config, resolve=True)
    logger = TrainingLogger(evaluation_config.output.directory, resolved_yaml)
    try:
        trainer = EventStateTrainer(
            model=runtime.model,
            teacher=runtime.teacher,
            train_loader=None,
            validation_loader=runtime.validation_loader,
            optimizer=None,
            scheduler=None,
            h_distillation_loss=runtime.h_distillation_loss,
            z_distillation_loss=runtime.z_distillation_loss,
            config=evaluation_config,
            device=runtime.device,
            logger=logger,
            checkpoint_config=OmegaConf.to_container(evaluation_config, resolve=True),
        )
        validation_batches_value = evaluation_config.evaluation.max_batches
        validation_batches = (
            None if validation_batches_value is None else int(validation_batches_value)
        )
        state = load_checkpoint(
            checkpoint_path,
            model=runtime.model,
            device=runtime.device,
            restore_rng=False,
            expected_config=evaluation_config,
            compatibility_mode="evaluation",
        )
        metrics = trainer.validate(max_batches=validation_batches)
        logger.scalars(metrics, state.global_step)
        logger.console("evaluation", state.global_step, metrics)
        metrics_path = logger.write_metrics_json("alignment_metrics.json", metrics)
        logger.flush()
        print(f"Alignment metrics written to {metrics_path}", flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
