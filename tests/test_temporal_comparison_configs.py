"""Server-only Hydra composition checks; no pretrained weights or GPU needed."""

import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from event_state.dense.common_protocol import audit_common, load_common
from event_state.dense.protocol import digest, pretraining_audit_config
from event_state.training.factory import validate_config
from tools.run_temporal_comparison import plan


def test_both_temporal_models_compose_as_matched_final_stage_h_distillation(tmp_path):
    for directory in ("DSEC", "DSEC_cache/events/gep_rgb", "DSEC_cache/dinov3_vits16",
                      "m3ed_cache/half_dagr", "m3ed_cache/dinov3_vits16_640x352"):
        (tmp_path / directory).mkdir(parents=True)
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"planning-only checkpoint fixture")
    statistics_dir = tmp_path / "statistics"
    statistics_dir.mkdir()
    for dataset in ("dsec", "m3ed"):
        protocol = load_common(dataset)
        (statistics_dir / f"{dataset}_common.json").write_text(json.dumps(dict(
            protocol_sha256=digest(protocol), sequences=protocol["splits"]["train"],
            normalize_mean=[0.1, 0.2, 0.3], normalize_std=[0.4, 0.5, 0.6], event_window_us=None)))
    args = SimpleNamespace(architectures="patchwise,convlstm", stage="smoke", data_root=tmp_path,
                           teacher_checkpoint=teacher, statistics_dir=statistics_dir,
                           output_root=tmp_path / "smoke", gpus="0,1", seed=0, num_workers=0,
                           execute=False, skip_preflight=False)
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    for stage, steps in (("smoke", 99), ("full", 99999)):
        args.stage = stage
        args.output_root = tmp_path / stage
        configs = {}
        for job in plan(args):
            command = job["command"]
            overrides = command[command.index("train.py") + 1:]
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name="config", overrides=overrides)
            validate_config(cfg)
            audit_common(pretraining_audit_config(cfg, to_plain=lambda value:
                OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value))
            dataset = job["name"].split("_", 1)[0]
            architecture = "patchwise" if "_patchwise_" in job["name"] else "convlstm"
            expected_kind = "lstm" if architecture == "patchwise" else "convlstm"
            assert cfg.model.temporal.type == expected_kind
            assert cfg.model.temporal.num_layers == 1
            assert cfg.model.temporal.input_dim == cfg.model.temporal.hidden_dim == cfg.model.temporal.output_dim == 384
            assert cfg.model.temporal.dropout == 0
            assert cfg.model.temporal.detach_state_every is None
            assert cfg.loss.h_distill.enabled
            assert cfg.loss.h_distill.cosine_weight == cfg.loss.h_distill.mse_weight == 1
            assert cfg.loss.z_objective.type == "none" and cfg.loss.z_objective.weight == 0
            assert not cfg.dataset.activity_mask
            assert not cfg.dataset.augmentation.enabled
            assert not cfg.training.event_dropout.enabled
            assert not cfg.training.validation_enabled
            assert cfg.training.max_steps == steps
            assert cfg.training.distributed_world_size == 1
            assert cfg.training.sampling.mode == "mixed" and cfg.training.sampling.batching == "joined"
            assert cfg.training.sampling.random_weight == cfg.training.sampling.stream_weight == 1
            assert cfg.training.batch_size == (8 if dataset == "dsec" else 4)
            assert cfg.dataset.sequence_length == (8 if dataset == "dsec" else 16)
            assert cfg.training.gradient_accumulation == 1
            assert cfg.training.resume is None
            assert cfg.optimizer.lr == 0.0001
            assert cfg.training.precision == "fp16"
            assert cfg.seed == 0
            assert architecture in cfg.experiment.name.lower()
            configs[dataset, architecture] = OmegaConf.to_container(cfg.model, resolve=True)
        for dataset in ("dsec", "m3ed"):
            # The temporal kind is the sole model-config difference, including head/projector settings.
            patchwise = configs[dataset, "patchwise"]
            convlstm = configs[dataset, "convlstm"]
            patchwise["temporal"]["type"] = "convlstm"
            assert patchwise == convlstm
        assert not args.output_root.exists()
