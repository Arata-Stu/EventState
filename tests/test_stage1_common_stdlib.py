"""Portable planning/leakage tests; no torch or Hydra imports."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from event_state.dense.common_protocol import load_common, audit_common
from event_state.dense.protocol import digest, load_protocol
from tools.run_stage1_common import plan


class CommonTests(unittest.TestCase):
    def test_common_pool_excludes_both_task_holdouts(self):
        for ds, count in [('dsec', 35), ('m3ed', 3)]:
            p = load_common(ds)
            self.assertEqual(len(p['splits']['train']), count)
            depth = load_protocol(ds + '_f3')
            self.assertFalse(set(p['splits']['train']) & set(depth['splits']['val']))
        self.assertFalse(set(load_common('dsec')['splits']['train']) &
                         {'zurich_city_07_a','zurich_city_08_a'})

    def test_plan_and_statistics_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ('DSEC','DSEC_cache/events/gep_rgb','DSEC_cache/dinov3_vits16',
                      'm3ed_cache/half_dagr','m3ed_cache/dinov3_vits16_640x352'):
                (root / d).mkdir(parents=True)
            teacher = root / 'teacher.pt'
            teacher.write_bytes(b'fixture')
            for ds in ('dsec','m3ed'):
                p = load_common(ds)
                stats = dict(protocol_sha256=digest(p), sequences=p['splits']['train'],
                             normalize_mean=[0.,0.,0.], normalize_std=[1.,1.,1.], event_window_us=None)
                (root / f'{ds}_common.json').write_text(json.dumps(stats))
                cfg = dict(dataset=dict(name=ds, train_sequences=p['splits']['train'],
                           val_sequences=[], normalization_provenance=stats,
                           representation={k:stats[k] for k in ('normalize_mean','normalize_std')}),
                           training=dict(validation_enabled=False))
                audit_common(cfg)
                leaked = copy.deepcopy(cfg)
                leaked['dataset']['train_sequences'].append(p['splits']['val'][0])
                with self.assertRaises(ValueError): audit_common(leaked)
            a = SimpleNamespace(data_root=root, statistics_dir=root, output_root=root/'out',
                                teacher_checkpoint=teacher, gpus='0,1,2', seed=0, num_workers=2,
                                phase='all', stage='full')
            jobs = plan(a)
            self.assertEqual(len(jobs), 8)
            for ds in ('dsec','m3ed'):
                subset = [j for j in jobs if j['name'].startswith(ds)]
                self.assertEqual(len({j['planned_frames'] for j in subset}), 1)
                self.assertEqual([j['steps'] for j in subset], [99999]*3+[33333])
                self.assertIn('model=no_memory', subset[0]['command'])
                self.assertIn('model=lstm', subset[1]['command'])
                self.assertIn('torch.distributed.run', subset[-1]['command'])
            self.assertFalse(a.output_root.exists())
            stats['sequences'] = ['wrong']
            (root/'m3ed_common.json').write_text(json.dumps(stats))
            with self.assertRaises(ValueError): plan(a)


if __name__ == '__main__':
    unittest.main()
