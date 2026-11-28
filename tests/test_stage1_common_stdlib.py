"""Portable planning/leakage tests; no torch or Hydra imports."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from event_state.dense.common_protocol import load_common, audit_common
from event_state.dense.protocol import digest, load_protocol
from tools.placement_conditions import PLACEMENTS, SKIP_CONDITION, placement_label
from tools.placement_gpu import PROJECT, semantic_worker_gpus
from tools.run_stage1_common import plan
from tools.stage1_progress import Stage1JobProgress, read_stage1_progress
from tools.progress_eta import format_eta
from event_state.dense.stream_lanes import segments, FullStreamBatchSampler


class CommonTests(unittest.TestCase):
    def test_running_semantic_worker_keeps_its_gpu_between_cuda_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            launcher = str(PROJECT / 'tools/run_placement_semantic.py').encode()
            for pid, gpu, args in [('10', '2', [launcher, b'--worker-result', b'result.json']),
                                   ('11', '1', [launcher, b'--execute'])]:
                process = proc / pid
                process.mkdir()
                (process / 'cmdline').write_bytes(b'\0'.join([b'python', *args, b'']))
                (process / 'environ').write_bytes(f'CUDA_VISIBLE_DEVICES={gpu}\0'.encode())
            self.assertEqual(semantic_worker_gpus(proc), {'2'})

    def test_progress_uses_recent_complete_training_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            self.assertIsNone(read_stage1_progress(log))
            log.write_bytes(b"initializing\n[train] step=5 ")
            self.assertIsNone(read_stage1_progress(log))
            log.write_text("startup " + "x" * (160 * 1024) + "\n"
                           "[train] step=1 loss=0.3 clips_per_update=4 samples_per_second=2\n"
                           "[validation] step=1 loss=1.9\n"
                           "[train] step=2 loss=0.2 clips_per_update=4 samples_per_second=4\n"
                           "[train] step=3 loss=0.1 clips_per_update=4 samples_per_second=nan\n"
                           "[train] step=4 loss=99")
            self.assertEqual(read_stage1_progress(log), (3, 0.1, 1.5))

    def test_live_eta_calibrates_to_wall_time_and_flags_stalled_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text("[train] step=1 loss=0.25 clips_per_update=8 samples_per_second=16\n")
            progress = Stage1JobProgress()
            first = progress.status(log, 10081, 2)
            self.assertIn("step 1/10,081", first)
            self.assertIn("約1.4時間", first)
            self.assertIn("JST", first)
            self.assertIsNone(progress.status(log, 10081, 60))
            with log.open("a") as stream:
                stream.write("[train] step=61 loss=0.123\n")
            update = progress.status(log, 10081, 62)
            self.assertIn("loss=0.123", update)
            self.assertIn("約2.8時間", update)
            self.assertIn("実測更新速度", update)
            self.assertIn("再計測中", progress.status(log, 10081, 122))
            with log.open("a") as stream:
                stream.write("[train] step=10081 loss=0.1\n")
            self.assertIn("最終checkpoint", progress.status(log, 10081, 182))
        self.assertIn("約1分", format_eta(2))
        self.assertIn("待ち", format_eta(float("nan")))

    def test_full_stream_lanes_are_disjoint_and_chronological(self):
        records = [(name, start, 8) for name, count in [('a',19),('b',25),('c',32)]
                   for start in range(count)]
        groups = segments(records, 2)
        flattened = [i for values in groups.values() for i in values]
        self.assertEqual(sorted(flattened), list(range(len(records))))
        owners = []
        for rank in range(3):
            sampler = FullStreamBatchSampler(groups, 2, 7, rank, 3)
            batches = list(sampler)
            self.assertTrue(all(len(b) == 2 for b in batches))
            owned = {name for slot in sampler.slots for name in slot}
            owners.append(owned)
            self.assertTrue({i for name in owned for i in groups[name]} <=
                            {i for b in batches for i, epoch in b})
            for slot in range(2):
                seq = [b[slot][0] for b in batches]
                firsts = {values[0] for values in groups.values()}
                for prev, cur in zip(seq, seq[1:]):
                    self.assertTrue(cur == prev+1 or cur in firsts)
        self.assertEqual(set.union(*owners), set(groups))
        self.assertFalse(owners[0] & owners[1] or owners[1] & owners[2] or owners[0] & owners[2])
        with self.assertRaises(ValueError): FullStreamBatchSampler(groups, 3, 7, 0, 3)

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
                for job in subset:
                    self.assertIn('training.gradient_accumulation=1', job['command'])
                    self.assertIn('++training.sampling.batching=joined', job['command'])
            a.four_stage_lstm = True
            a.phase = 'baseline'
            staged = plan(a)
            self.assertEqual([j['name'] for j in staged],
                             ['dsec_four_stage_lstm_w1', 'm3ed_four_stage_lstm_w1'])
            for job in staged:
                self.assertIn('model=four_stage_lstm', job['command'])
                self.assertIn('experiment=h_distill_lstm', job['command'])
            a.four_stage_lstm = False
            for first in range(1, 5):
                a.lstm_from_stage = first
                for job in plan(a):
                    self.assertIn(f"model.temporal.first_stage={first}", job["command"])
            a.placement_suite = True
            with self.assertRaisesRegex(ValueError, 'placement-suite'):
                plan(a)
            a.lstm_from_stage = None
            a.phase = 'ddp'
            with self.assertRaisesRegex(ValueError, 'placement-suite'):
                plan(a)
            a.phase = 'all'
            a.gpus = '2'
            expected = {'E1_b3': [3], 'E1_b6': [6], 'E1_b9': [9], 'E1_b9_12': [9, 12],
                        'E1_b6_9_12': [6, 9, 12], 'E1_b3_6_9_12': [3, 6, 9, 12]}
            self.assertEqual({key: list(value) for key, value in PLACEMENTS.items()}, expected)
            for stage, steps in [('smoke', 99), ('full', 99999)]:
                a.stage = stage
                suite = plan(a)
                self.assertEqual(len(suite), 12)
                self.assertEqual({job['name'] for job in suite}, {
                    f'{ds}_{key}_w1' for ds in ('dsec', 'm3ed') for key in expected})
                for job in suite:
                    key = job['condition']
                    self.assertEqual((job['world'], job['steps'], job['after_blocks']),
                                     (1, steps, expected[key]))
                    self.assertEqual(job['label'], placement_label(key))
                    self.assertEqual(job['planned_frames'], 64 * steps)
                    for override in ('model=four_stage_lstm', 'experiment=h_distill_lstm',
                                     'training.resume=null', 'seed=0',
                                     '++model.temporal.after_blocks=' + json.dumps(expected[key], separators=(',', ':')),
                                     f"hydra.run.dir={(a.output_root / job['name']).resolve()}"):
                        self.assertIn(override, job['command'])
                a.placement_suite, a.b3_skip = False, True
                skip_jobs = plan(a)
                self.assertEqual([job['name'] for job in skip_jobs],
                                 [f'{ds}_{SKIP_CONDITION}_w1' for ds in ('dsec', 'm3ed')])
                for job, direct in zip(skip_jobs, (j for j in suite if j['condition'] == 'E1_b3')):
                    self.assertEqual(job['condition'], SKIP_CONDITION)
                    self.assertTrue(job['residual_skip'])
                    self.assertEqual(job['steps'], steps)
                    self.assertEqual(job['after_blocks'], [3])
                    self.assertIn('残差接続', job['label'])
                    self.assertIn('++model.temporal.residual_skip=true', job['command'])
                    skip_recipe = [arg.replace(SKIP_CONDITION, 'E1_b3') for arg in job['command']
                                   if arg != '++model.temporal.residual_skip=true']
                    self.assertEqual(skip_recipe, direct['command'])
                a.placement_suite = True
                with self.assertRaisesRegex(ValueError, 'b3-skip'):
                    plan(a)
                a.b3_skip = False
                a.skip_placement_suite = True
                with self.assertRaisesRegex(ValueError, 'skip-placement-suite'):
                    plan(a)
                a.placement_suite = False
                extra = plan(a)
                blocks = {'E1_b3_6_9': [3, 6, 9], 'E1_b6_skip': [6], 'E1_b9_skip': [9],
                          'E1_b3_6_9_skip': [3, 6, 9]}
                self.assertEqual([job['name'] for job in extra],
                                 [f'{ds}_{name}_w1' for ds in ('dsec', 'm3ed') for name in blocks])
                for job in extra:
                    name = job['condition']
                    self.assertEqual((job['steps'], job['after_blocks'], job['world']), (steps, blocks[name], 1))
                    self.assertEqual('++model.temporal.residual_skip=true' in job['command'], name.endswith('_skip'))
                    self.assertEqual(job.get('residual_skip', False), name.endswith('_skip'))
                a.final_skip_suite = True
                final_jobs = plan(a)
                self.assertEqual([j['condition'] for j in final_jobs],
                    ['E1_b12_skip', 'E1_b3_6_9_12_skip'] * 2)
                self.assertEqual([j['after_blocks'] for j in final_jobs], [[12], [3, 6, 9, 12]] * 2)
                self.assertTrue(all(j['residual_skip'] and j['steps'] == steps for j in final_jobs))
                a.final_skip_suite = False
                a.placement_suite, a.skip_placement_suite = True, False
            self.assertFalse(a.output_root.exists())
            stats['sequences'] = ['wrong']
            (root/'m3ed_common.json').write_text(json.dumps(stats))
            with self.assertRaises(ValueError): plan(a)


if __name__ == '__main__':
    unittest.main()
