"""Server-side Hydra composition and runtime validation; no pretrained weights."""
from pathlib import Path
import json
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import digest
from event_state.training.factory import validate_config
from tools.run_stage1_common import plan


def test_all_launch_commands_compose_and_validate(tmp_path):
    for d in ('DSEC','DSEC_cache/events/gep_rgb','DSEC_cache/dinov3_vits16',
              'm3ed_cache/half_dagr','m3ed_cache/dinov3_vits16_640x352'):
        (tmp_path/d).mkdir(parents=True)
    teacher = tmp_path/'teacher.pt'
    teacher.write_bytes(b'fixture')
    for ds in ('dsec','m3ed'):
        p = load_common(ds)
        (tmp_path/f'{ds}_common.json').write_text(json.dumps(dict(protocol_sha256=digest(p),
            sequences=p['splits']['train'], normalize_mean=[0.,0.,0.], normalize_std=[1.,1.,1.])))
    a = SimpleNamespace(gpus='0,1,2', output_root=tmp_path/'out', data_root=tmp_path,
        statistics_dir=tmp_path, teacher_checkpoint=teacher, seed=0, num_workers=0, phase='all', stage='smoke')
    for job in plan(a):
        overrides = job['command'][job['command'].index('train.py')+1:]
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
            cfg = compose(config_name='config', overrides=overrides)
        validate_config(cfg)
        assert cfg.training.sampling.mode == 'mixed'
        assert cfg.model.temporal.type == ('identity' if '_E0_' in job['name'] else 'lstm')
