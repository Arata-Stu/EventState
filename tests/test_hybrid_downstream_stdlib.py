import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import re
import tempfile
import unittest
import zipfile
import subprocess
from unittest.mock import patch

from tools.run_hybrid_target_downstream import model_plan, safe_cleanup, verify_results, PROJECT
from tools.run_hybrid_target_downstream import main


class DownstreamTests(unittest.TestCase):
    def test_activity_alpha_suite(self):
        argv = ['runner', '--pretrain-root', '/pretrain', '--output-root', '/new-output',
                '--cache-root', '/new-cache', '--suite', 'activity-alpha', '--dry-run']
        output = io.StringIO()
        with patch('sys.argv', argv), redirect_stdout(output):
            main()
        plan = output.getvalue()
        self.assertEqual(plan.count('tools/train_dsec_'), 18)
        self.assertEqual(plan.count('tools/train_m3ed_'), 3)
        self.assertNotIn('m3ed_alpha_0', plan)
        self.assertNotIn('dsec_z_only', plan)

    def test_selected_h_only_dry_run(self):
        argv = ['runner', '--pretrain-root', '/unused', '--output-root', '/new-output',
                '--cache-root', '/new-cache', '--only', 'dsec_h_only', '--dry-run']
        with patch('sys.argv', argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        output = io.StringIO()
        with patch('sys.argv', argv + ['--dsec-h-checkpoint', '/old-h.pt']), redirect_stdout(output):
            main()
        plan = output.getvalue()
        self.assertEqual(plan.count('tools/train_dsec_'), 6)
        self.assertEqual(plan.count('tools/evaluate_dsec_'), 6)
        self.assertNotIn('m3ed', plan)
        self.assertNotIn('dsec_z_only', plan)
        self.assertNotIn('dsec_z_h', plan)

    def test_plans_and_cli_arguments(self):
        counts = []
        for name in ('dsec_z_only', 'dsec_z_h', 'm3ed_z_only', 'm3ed_h_only', 'dsec_h_only', 'm3ed_z_h'):
            stages, results = model_plan(name, Path('/checkpoint'), Path('/data'), Path('/teacher'),
                                        Path('/scratch') / name, Path('/results') / name)
            counts.append(len(results))
            for _, cmd, _ in stages:
                source = (PROJECT / cmd[1]).read_text()
                supported = set(re.findall(r'["\'](--[\w-]+)["\']', source))
                self.assertTrue({s for s in cmd[2:] if s.startswith('--')} <= supported, cmd)
                self.assertNotIn('test', cmd)
            if name.endswith('z_only'):
                self.assertEqual({r[2] for r in results}, {'z'})
        self.assertEqual(counts, [2, 6, 1, 3, 6, 3])

    def test_cleanup_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            cache = root / 'model'; cache.mkdir()
            protected = root / 'keep'; protected.write_text('keep')
            identity = {'run': 'x'}
            with self.assertRaises(ValueError):
                safe_cleanup(cache, root, identity)
            (cache / '.owner.json').write_text(json.dumps(identity))
            (cache / 'link').symlink_to(protected)
            with self.assertRaises(ValueError):
                safe_cleanup(cache, root, identity)
            (cache / 'link').unlink()
            (cache / 'feature.pt').write_text('owned')
            safe_cleanup(cache, root, identity)
            self.assertFalse(cache.exists())
            self.assertTrue(protected.exists())
            with self.assertRaises(ValueError):
                safe_cleanup(root, root, identity)

    def test_result_gate(self):
        with tempfile.TemporaryDirectory() as d:
            head = Path(d)
            metric = head / 'validation_metrics.json'
            results = [(head, metric, 'z', 'mIoU', 'validation')]
            for name in ('best.pt', 'last.pt'):
                with zipfile.ZipFile(head / name, 'w') as z:
                    z.writestr('archive/data.pkl', 'fixture')
            metric.write_text(json.dumps(dict(feature='z', role='validation', mIoU=0.3, evaluated_pixels=100)))
            verify_results(results)
            metric.write_text(json.dumps(dict(feature='z', role='validation', mIoU=float('nan'), evaluated_pixels=100)))
            with self.assertRaises(ValueError):
                verify_results(results)
            (head / 'best.pt').unlink()
            with self.assertRaises(ValueError):
                verify_results(results)

    def test_failure_preserves_cache_and_resume_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = root / 'data'; pretrain = root / 'pretrain'
            for rel in ('DSEC_cache/events/gep_rgb', 'DSEC/task_labels/semantic', 'DSEC/dsec_det_labels',
                        'm3ed_cache/half_dagr', 'm3ed_cache/dinov3_vits16_640x352', 'm3ed_cache/m3ed_downstream'):
                (data / rel).mkdir(parents=True)
            for name in ('dsec_z_only', 'dsec_z_h', 'm3ed_z_only', 'm3ed_h_only'):
                ck = pretrain / name / 'checkpoints/step_00100000.pt'
                ck.parent.mkdir(parents=True); ck.write_text('fixture')
            teacher = root / 'teacher.pt'; teacher.write_text('fixture')
            output, cache = root / 'output', root / 'cache'
            argv = ['runner', '--pretrain-root', str(pretrain), '--output-root', str(output),
                    '--cache-root', str(cache), '--data-root', str(data), '--teacher-checkpoint', str(teacher)]
            fail_once = [True]
            resumed = []
            def fake_run(command, **kwargs):
                def value(flag): return command[command.index(flag) + 1]
                script = Path(command[1]).name
                if script.startswith('train_'):
                    head = Path(value('--output-dir')); head.mkdir(parents=True, exist_ok=True)
                    for name in ('best.pt', 'last.pt'):
                        with zipfile.ZipFile(head / name, 'w') as z:
                            z.writestr('archive/data.pkl', 'fixture')
                    if 'm3ed_z_only' in str(head) and fail_once[0]:
                        fail_once[0] = False
                        raise subprocess.CalledProcessError(1, command)
                    if '--resume' in command:
                        resumed.append(str(head))
                    if script == 'train_m3ed_semantic.py':
                        (head / 'validation_metrics.json').write_text(json.dumps(dict(
                            role='validation', feature=value('--feature'), mIoU=.3, evaluated_pixels=100)))
                elif script.startswith('evaluate_'):
                    metric = Path(value('--output'))
                    result = dict(role='val', feature=value('--feature'), evaluated_pixels=100)
                    if 'detection' in script:
                        result.update(mAP=.3, protocol='dsec-det')
                    else:
                        result['mIoU'] = .3
                    metric.write_text(json.dumps(result))
            with patch('sys.argv', argv), patch('tools.run_hybrid_target_downstream.subprocess.run', side_effect=fake_run):
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
            self.assertFalse((cache / 'dsec_z_only').exists())
            self.assertTrue((cache / 'm3ed_z_only/.owner.json').exists())
            with patch('sys.argv', argv + ['--resume']), patch('tools.run_hybrid_target_downstream.subprocess.run', side_effect=fake_run):
                main()
            self.assertEqual(len(resumed), 1)
            self.assertTrue((output / 'm3ed_h_only/complete.json').exists())
            self.assertEqual(list(cache.iterdir()), [cache / '.owner.json'])


if __name__ == '__main__':
    unittest.main()
