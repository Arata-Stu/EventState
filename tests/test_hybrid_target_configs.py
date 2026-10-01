"""GPU-host preflight: four target ablations compose and train intended branches."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.run_hybrid_target_ablation import TRAIN4, build_plan


def test_four_target_configs(tmp_path):
    from hydra import compose, initialize_config_dir
    from event_state.training.factory import validate_config

    for name in ('DSEC', 'DSEC_cache/events/gep_rgb', 'DSEC_cache/dinov3_vits16',
                 'm3ed_cache/half_dagr', 'm3ed_cache/dinov3_vits16_640x352'):
        (tmp_path / name).mkdir(parents=True)
    teacher = tmp_path / 'teacher.pt'
    teacher.touch()
    (tmp_path / 'm3ed_cache/half_dagr/event_statistics_train4.json').write_text(json.dumps(
        dict(sequences=sorted(TRAIN4), representation='gep_rgb',
             normalize_mean=[0.1, 0.2, 0.3], normalize_std=[0.4, 0.5, 0.6])))
    args = SimpleNamespace(gpus='0,1,2', seed=0, num_workers=4, data_root=tmp_path,
        teacher_checkpoint=teacher, event_statistics=None, output_root=tmp_path / 'out', stage='full')
    _, _, jobs = build_plan(args)
    root = Path(__file__).resolve().parents[1]
    for job in jobs:
        with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
            cfg = compose(config_name='config', overrides=job['command'][2:])
        validate_config(cfg)
        assert cfg.model.temporal.type == 'lstm'
        assert not cfg.dataset.activity_mask
        assert not cfg.dataset.augmentation.enabled
        assert not cfg.training.event_dropout.enabled
        assert cfg.training.sampling.mode == 'mixed'
        assert cfg.training.resume is None
        assert cfg.training.max_steps == 100000
        assert cfg.training.checkpoint_every == 10000
        is_dsec = job['name'].startswith('dsec')
        assert cfg.training.batch_size == (8 if is_dsec else 4)
        assert cfg.dataset.sequence_length == (8 if is_dsec else 16)
        assert cfg.training.validation_enabled == (not is_dsec)
        z_only = job['name'].endswith('z_only')
        assert cfg.loss.h_distill.enabled == (not z_only)
        assert cfg.loss.z_objective.type == ('none' if job['name'].endswith('h_only') else 'direct_dino')


@pytest.mark.parametrize('h_enabled,z_type', [(False, 'direct_dino'), (True, 'none')])
def test_full_region_single_branch_gradients(h_enabled, z_type):
    from test_training_runtime import TinySequenceModel, _make_trainer, _trainer_config
    model = TinySequenceModel()
    trainer = _make_trainer(model, _trainer_config(h_enabled=h_enabled, z_type=z_type))
    result = trainer._forward_objectives({
        'events': torch.randn(1, 2, 3, 16, 16),
        'teacher_features': torch.randn(1, 2, 1, 4)})
    result['loss'].backward()
    assert model.event_encoder.weight.grad is not None
    if not h_enabled:
        assert all(p.grad is None for p in model.temporal_model.parameters())
        assert all(p.grad is None for p in model.h_projector.parameters())
    else:
        assert any(p.grad is not None for p in model.temporal_model.parameters())
        assert all(p.grad is None for p in model.z_projector.parameters())
