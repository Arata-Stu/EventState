"""Safe local launcher checks; no torch, GPU, or real training data."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.run_hybrid_target_ablation import TRAIN4, build_plan, execute


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('DSEC', 'DSEC_cache/events/gep_rgb', 'DSEC_cache/dinov3_vits16',
                     'm3ed_cache/half_dagr', 'm3ed_cache/dinov3_vits16_640x352'):
            (self.root / name).mkdir(parents=True)
        self.teacher = self.root / 'teacher.pt'
        self.teacher.touch()
        self.stats = self.root / 'm3ed_cache/half_dagr/event_statistics_train4.json'
        self.stats.write_text(json.dumps(dict(sequences=sorted(TRAIN4), representation='gep_rgb',
                                             normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])))
        self.args = SimpleNamespace(gpus='0,1,2', seed=0, num_workers=4, data_root=self.root,
            teacher_checkpoint=self.teacher, event_statistics=None, output_root=self.root / 'out',
            stage='full', skip_tests=True)

    def test_plans_and_rejections(self):
        for stage, steps in [('smoke', '100'), ('full', '100000')]:
            self.args.stage = stage
            output, _, jobs = build_plan(self.args)
            self.assertFalse(output.exists())
            self.assertEqual([j['gpu'] for j in jobs], ['0', '1', '2', '2'])
            for job in jobs:
                self.assertIn('training.max_steps=' + steps, job['command'])
                self.assertIn('training.sampling.mode=mixed', job['command'])
                self.assertIn('training.resume=null', job['command'])
                self.assertIn('dataset.activity_mask=false', job['command'])
        self.args.gpus = '0,0,2'
        with self.assertRaises(ValueError):
            build_plan(self.args)
        self.args.gpus = '0,1,2'
        data = json.loads(self.stats.read_text())
        data['sequences'][0] = 'car_urban_day_ucity_small_loop'
        self.stats.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            build_plan(self.args)

    def test_gpu_queue_with_fake_jobs(self):
        output, stats, jobs = build_plan(self.args)
        for job in jobs:
            # Exclusive file creation detects simultaneous jobs on the same GPU.
            code = ("import pathlib,time; "
                    f"lock=pathlib.Path({str(self.root / ('gpu' + job['gpu']))!r}); "
                    "lock.open('x').close(); time.sleep(0.05); lock.unlink()")
            job['command'] = [sys.executable, '-c', code]
        with patch('tools.run_hybrid_target_ablation.shutil.disk_usage',
                   return_value=SimpleNamespace(free=100 * 2**30)):
            execute(self.args, output, stats, jobs)
        self.assertEqual(len(list((output / 'logs').glob('*.log'))), 4)
        self.assertTrue((output / 'launch.json').exists())
        with self.assertRaises(ValueError):
            build_plan(self.args)


if __name__ == '__main__':
    unittest.main()
