"""Portable safeguards for the remaining three ConvLSTM downstream jobs."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from event_state.dense.protocol import digest, load_protocol
from tools import run_temporal_downstream as runner


ROOT = Path(__file__).resolve().parents[1]


class RemainingDownstreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            project_root=ROOT, data=self.root / 'data', stage1_root=self.root / 'stage1',
            output=self.root / 'new_output', scratch=self.root / 'scratch',
            depth_run=self.root / 'old_depth', statistics=self.root / 'statistics',
            weights=self.root / 'weights.pth', activity_root=self.root / 'activity',
            baseline_semantic_config=self.root / 'old_semantic.json', seed=0)
        self.write(self.args.weights, b'fixture weights')
        for dataset in ('dsec', 'm3ed'):
            self.dataset(dataset)

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(json.dumps(content))
        return path

    def dataset(self, dataset):
        protocol = load_protocol(dataset + '_f3')
        common = runner.read(ROOT / f'tools/manifests/{dataset}_stage1_common_v1.json')
        self.write(self.args.statistics / f'{dataset}_common.json',
                   dict(sequences=common['splits']['train'], event_window_us=None,
                        event_rate_p10_hz=10))
        self.write(self.args.stage1_root / f'{dataset}_E1_convlstm_w1/checkpoints/step_00099999.pt',
                   b'fixture checkpoint')
        self.write(self.args.activity_root / f'{dataset}.json', {'dataset': dataset})
        target = self.write(self.root / f'{dataset}_target.bin', b'fixture original GT')
        events = self.args.data / ('DSEC_cache/events/gep_rgb' if dataset == 'dsec'
                                   else 'm3ed_cache/half_dagr')
        for role in ('train', 'val'):
            records = []
            for sequence in protocol['splits'][role]:
                records.append(dict(sequence=sequence, timestamp=100_000,
                                    target_path=str(target), target_sha256=runner.sha256(target)))
                self.write(events / sequence / 'metadata.json', dict(dataset=dataset))
                if dataset == 'dsec' and role == 'val':
                    for name in ('events.h5', 'rectify_map.h5'):
                        self.write(self.args.data / 'DSEC/train_events' / sequence /
                                   'events/left' / name, b'fixture raw events')
                    self.write(self.args.data / 'DSEC/train_images' / sequence /
                               'images/timestamps.txt', b'0\n100000\n')
            self.write(self.args.depth_run / 'targets' / f'{dataset}_{role}.json',
                       dict(protocol=protocol, protocol_sha256=digest(protocol),
                            dataset=dataset, task='depth', role=role, labeled=True,
                            records=records, source_identity={str(target): runner.sha256(target)}))

    def activity(self, path):
        dataset = Path(path).stem

        def lookup(sequence, timestamp):
            if timestamp != 100_000:
                raise ValueError(f'missing activity interval: {sequence}/{timestamp}')
            return {'group': 'p00_20', 'timestamp': timestamp, 'previous_timestamp': 0}

        return SimpleNamespace(identity={'source_contract': {'dataset': dataset}},
                               timestamps=lambda sequence: (100_000,), lookup=Mock(side_effect=lookup))

    def semantic(self, _args):
        backbone = self.args.stage1_root / 'dsec_E1_convlstm_w1/checkpoints/step_00099999.pt'
        return dict(name='dsec_semantic', kind='semantic', dataset='dsec', label=runner.LABEL,
                    backbone=str(backbone), inputs=[str(self.args.weights), str(backbone)],
                    remaining_cache_bytes=100)

    def plan(self):
        with patch.object(runner, 'build_semantic_job', self.semantic), \
                patch.object(runner.ActivityIndex, 'load', self.activity), \
                contextlib.redirect_stdout(io.StringIO()):
            return runner.build_plan(self.args)

    def test_idle_gpu_filter_preserves_physical_order_and_excludes_compute_jobs(self):
        responses = iter(['0, GPU-a\n1, GPU-b\n2, GPU-c\n', 'GPU-b, 123\n'])
        run = Mock(side_effect=lambda *a, **k: SimpleNamespace(stdout=next(responses)))
        self.assertEqual(runner.idle_gpus(('2', '1', '0'), run=run), ('2', '0'))
        self.assertEqual(run.call_count, 2)
        self.assertIn('--query-compute-apps=gpu_uuid,pid', run.call_args.args[0])

    def test_unknown_gpu_unparseable_process_or_smi_failure_is_not_assumed_idle(self):
        for requested, processes in ((('7',), ''), (('0',), 'GPU-a, N/A\n')):
            results = iter(['0, GPU-a\n', processes])
            with self.subTest(requested=requested, processes=processes), self.assertRaises(ValueError):
                runner.idle_gpus(requested, run=lambda *a, **k: SimpleNamespace(stdout=next(results)))
        with self.assertRaises(subprocess.CalledProcessError):
            runner.idle_gpus(('0',), run=Mock(side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')))

    def test_depth_plan_preserves_protocol_readout_paths_and_capacity_for_each_dataset(self):
        before = set(self.root.rglob('*'))
        with patch.object(runner.ActivityIndex, 'load', self.activity):
            for dataset, patches in (('dsec', 30 * 40), ('m3ed', 22 * 40)):
                with self.subTest(dataset=dataset):
                    job = runner.build_depth_job(self.args, dataset)
                    self.assertEqual(job['name'], dataset + '_E1_convlstm')
                    self.assertEqual(job['roles'], ['train', 'val'])
                    self.assertTrue(job['need_main'] and job['need_activity'])
                    self.assertEqual(job['need_disparity'], dataset == 'dsec')
                    self.assertEqual(job['head'], str(self.args.output / 'heads' / job['name'] / 'stages'))
                    for role in ('train', 'val'):
                        count = len(load_protocol(dataset + '_f3')['splits'][role])
                        self.assertEqual(job['role_bytes'][role], count * patches * 1536 * 2)
                        self.assertEqual(job['indexes'][role],
                                         str(self.args.output / 'targets' / f'{dataset}_{role}.json'))
                        self.assertEqual(job['index_sources'][role],
                                         str(self.args.depth_run / 'targets' / f'{dataset}_{role}.json'))
                    self.assertEqual(job['remaining_cache_bytes'], sum(job['role_bytes'].values()))
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_stale_protocol_or_statistics_are_rejected_before_any_output(self):
        path = self.args.depth_run / 'targets/m3ed_val.json'
        payload = runner.read(path)
        payload['protocol_sha256'] = 'incorrect'
        path.write_text(json.dumps(payload))
        with patch.object(runner.ActivityIndex, 'load', self.activity), self.assertRaisesRegex(ValueError, 'protocol'):
            runner.build_depth_job(self.args, 'm3ed')
        stats = self.args.statistics / 'dsec_common.json'
        payload = runner.read(stats)
        payload['sequences'].append('evaluation_sequence')
        stats.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'statistics'):
            runner.build_depth_job(self.args, 'dsec')
        self.assertFalse(self.args.output.exists())

    def test_only_initial_rgb_target_may_lack_an_activity_interval(self):
        path = self.args.depth_run / 'targets/m3ed_val.json'
        payload = runner.read(path)
        initial = dict(payload['records'][0], timestamp=0)
        payload['records'].insert(0, initial)
        path.write_text(json.dumps(payload))
        with patch.object(runner.ActivityIndex, 'load', self.activity):
            job = runner.build_depth_job(self.args, 'm3ed')
        self.assertEqual(job['role_bytes']['val'], len(payload['records']) * 22 * 40 * 1536 * 2)
        payload['records'][0]['timestamp'] = 50_000
        path.write_text(json.dumps(payload))
        with patch.object(runner.ActivityIndex, 'load', self.activity), \
                self.assertRaisesRegex(ValueError, 'missing activity interval'):
            runner.build_depth_job(self.args, 'm3ed')

    def backbone_config(self, dataset='m3ed'):
        return dict(dataset=dict(name=dataset), seed=0, training=dict(max_steps=99999),
                    model=dict(temporal=dict(type='convlstm', num_layers=1,
                                            input_dim=384, hidden_dim=384, output_dim=384)),
                    loss=dict(h_distill=dict(enabled=True, cosine_weight=1, mse_weight=1),
                              z_objective=dict(type='none', weight=0)))

    def test_checkpoint_recipe_requires_matched_full_convlstm_h_only_configuration(self):
        config = self.backbone_config()
        runner.validate_backbone_config(config, 'm3ed')
        runner.validate_backbone_config(self.backbone_config('dsec'), 'dsec')
        invalid = [
            (('dataset', 'name'), 'dsec'), (('seed',), 1),
            (('training', 'max_steps'), 99),
            (('model', 'temporal', 'type'), 'lstm'),
            (('model', 'temporal', 'num_layers'), 2),
            (('model', 'temporal', 'input_dim'), 192),
            (('model', 'temporal', 'hidden_dim'), 192),
            (('model', 'temporal', 'output_dim'), 192),
            (('loss', 'h_distill', 'enabled'), False),
            (('loss', 'h_distill', 'cosine_weight'), .5),
            (('loss', 'h_distill', 'mse_weight'), 0),
            (('loss', 'z_objective', 'type'), 'cosine_mse'),
            (('loss', 'z_objective', 'weight'), 1),
        ]
        for keys, value in invalid:
            altered = copy.deepcopy(config)
            branch = altered
            for key in keys[:-1]:
                branch = branch[key]
            branch[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, 'full99999 recipe'):
                runner.validate_backbone_config(altered, 'm3ed')

    def test_checkpoint_metadata_is_loaded_only_by_the_worker_check(self):
        load = Mock(return_value=SimpleNamespace(config=self.backbone_config()))
        module = SimpleNamespace(load_checkpoint_config_metadata=load)
        with patch.dict(sys.modules, {'event_state.training': module}):
            runner.check_backbone(self.args.weights, 'm3ed')
        load.assert_called_once_with(self.args.weights)

    def test_original_gt_hash_mismatch_and_existing_output_are_preserved(self):
        target = self.root / 'm3ed_target.bin'
        target.write_bytes(b'modified ground truth')
        with self.assertRaisesRegex(ValueError, 'source hash differs'):
            self.plan()
        self.assertEqual(target.read_bytes(), b'modified ground truth')
        self.args.output.mkdir()
        marker = self.args.output / 'existing.txt'
        marker.write_text('retain')
        with self.assertRaises(FileExistsError):
            self.plan()
        self.assertEqual(marker.read_text(), 'retain')

    def test_preflight_is_read_only_and_reports_two_gpu_limit(self):
        plan = self.plan()
        before = set(self.root.rglob('*'))
        output = io.StringIO()
        with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)), \
                contextlib.redirect_stdout(output):
            runner.preflight(self.args, plan, ('0', '2'))
        self.assertIn('Initially feasible parallel jobs: 2/3', output.getvalue())
        self.assertEqual(before, set(self.root.rglob('*')))
        with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
            runner.preflight(self.args, plan, ('0',))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'No requested GPU'):
            runner.preflight(self.args, plan, ())

    def worker(self, plan, fd):
        path = self.root / 'plan.json'
        runner.save(path, plan)
        return SimpleNamespace(worker_plan=path, worker_name='dsec_semantic', parent_lock_fd=fd,
                               worker_status=self.root / 'status.json', worker_result=self.root / 'result.json')

    def test_worker_rejects_wrong_lock_mutated_inputs_or_new_gpu_occupancy(self):
        plan = self.plan()
        self.args.output.mkdir()
        lock = self.args.output / 'run.lock'
        lock.write_text('lock')
        other = self.root / 'other.lock'
        other.write_text('another lock')
        with other.open() as stream, self.assertRaisesRegex(ValueError, 'run lock'):
            runner._worker(self.worker(plan, stream.fileno()))
        with lock.open() as stream:
            a = self.worker(plan, stream.fileno())
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}), \
                    patch.object(runner, 'idle_gpus', return_value=()), \
                    patch.object(runner, 'run_semantic_job') as run, \
                    self.assertRaisesRegex(RuntimeError, 'become busy'):
                runner._worker(a)
            run.assert_not_called()
            self.args.weights.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                runner._worker(a)

    def test_execute_copies_canonical_target_indexes_and_preserves_assigned_physical_gpus(self):
        plan = self.plan()
        observed = {}

        def schedule(jobs, gpus, **kwargs):
            observed['gpus'] = gpus
            for job in jobs:
                for role, source in job.get('index_sources', {}).items():
                    self.assertEqual(Path(job['indexes'][role]).read_bytes(), Path(source).read_bytes())
                command = kwargs['worker_command'](job, self.root / 's', self.root / 'r')
                self.assertIn('--parent-lock-fd', command)
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '1')
            return {j['name']: {'completed': True} for j in jobs}

        with patch.object(runner, 'run_parallel', side_effect=schedule), \
                patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.args, plan, ('1', '2'))
        self.assertEqual(observed['gpus'], ('1', '2'))
        self.assertTrue((self.args.output / '_SUCCESS').is_file())
        self.assertEqual(len(runner.read(self.args.output / 'summary.json')), 3)
        with self.assertRaises(FileExistsError):
            runner.execute(self.args, plan, ('1',))

    def test_worker_dispatches_semantic_and_depth_with_the_inherited_lock(self):
        from tools import run_common_depth

        plan = self.plan()
        self.args.output.mkdir()
        lock = self.args.output / 'run.lock'
        lock.write_text('lock')

        def done(args, job, status_callback):
            self.assertIsInstance(args.weights, Path)
            self.assertEqual(args.seed, 0)
            self.assertFalse(args.activity_only)
            self.assertEqual(args.parent_lock_fd, descriptor)
            status_callback(dict(phase='completed', remaining_cache_bytes=0))
            return dict(job=job['name'], completed=True)

        with lock.open() as stream, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), \
                patch.object(runner, 'idle_gpus', return_value=('1',)), \
                patch.object(runner, 'check_backbone') as check_backbone:
            descriptor = stream.fileno()
            worker = self.worker(plan, descriptor)
            with patch.object(runner, 'run_semantic_job', side_effect=done) as semantic:
                runner._worker(worker)
            semantic.assert_called_once()
            check_backbone.assert_called_once_with(plan['jobs'][0]['backbone'], 'dsec')
            self.assertEqual(runner.read(worker.worker_result)['job'], 'dsec_semantic')
            worker.worker_name = 'm3ed_E1_convlstm'
            with patch.object(run_common_depth, 'run_condition', side_effect=done) as depth, \
                    patch.object(run_common_depth, '_LOCK_FD', None):
                runner._worker(worker)
                self.assertEqual(run_common_depth._LOCK_FD, descriptor)
            depth.assert_called_once()
            self.assertEqual(check_backbone.call_count, 2)
            self.assertEqual(runner.read(worker.worker_result)['job'], 'm3ed_E1_convlstm')
            self.assertEqual(runner.read(worker.worker_status)['remaining_cache_bytes'], 0)

    def test_worker_rejects_changed_raw_disparity_input_before_dispatch(self):
        plan = self.plan()
        self.args.output.mkdir()
        lock = self.args.output / 'run.lock'
        lock.write_text('lock')
        depth = next(j for j in plan['jobs'] if j['name'] == 'dsec_E1_convlstm')
        raw = Path(depth['stat_inputs'][0])
        raw.write_bytes(raw.read_bytes() + b'new event bytes')
        with lock.open() as stream, self.assertRaisesRegex(ValueError, 'Raw disparity-evaluation input changed'):
            worker = self.worker(plan, stream.fileno())
            worker.worker_name = depth['name']
            runner._worker(worker)

    def test_cli_help_and_import_require_no_training_dependencies(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'tools/run_temporal_downstream.py'), '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--execute', result.stdout)
        result = subprocess.run([sys.executable, '-S', '-c',
                                 'import tools.run_temporal_downstream; import sys; '
                                 'assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
