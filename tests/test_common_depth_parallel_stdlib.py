"""Exercise GPU orchestration with tiny stdlib workers, never Torch or real data."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tools import common_depth_parallel as parallel
from tools import run_common_depth as runner


WORKER = r'''
import json, os, sys, time
from pathlib import Path
job, status_path, result_path, events_path = json.loads(sys.argv[1]), *map(Path, sys.argv[2:])

def event(kind, **extra):
    data = dict(kind=kind, name=job['name'], pid=os.getpid(),
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), **extra)
    fd = os.open(events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(data) + '\n').encode())
    finally:
        os.close(fd)

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)

event('start')
write(status_path, dict(phase='cache', remaining_cache_bytes=job['remaining_cache_bytes']))
mode = job.get('mode', 'normal')
if mode == 'release':
    time.sleep(.12)
    event('release')
    write(status_path, dict(phase='train', remaining_cache_bytes=0))
time.sleep(.3)
write(result_path, dict(condition=job['label'], depth=dict(mae=1)))
event('end')
'''


def events_at(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class ParallelDepthSchedulerTest(unittest.TestCase):
    def fixture(self, root):
        worker = root/'worker.py'
        worker.write_text(WORKER)
        events = root/'events.jsonl'

        def command(job, status_path, result_path):
            return [sys.executable, str(worker), json.dumps(job), str(status_path),
                    str(result_path), str(events)]

        return events, command

    @staticmethod
    def job(name, *, size=0, mode='normal'):
        return dict(name=name, label=name + ' fixture', remaining_cache_bytes=size, mode=mode)

    def test_gpu_list_rejects_ambiguous_or_oversubscribed_devices(self):
        self.assertEqual(parallel.parse_gpus('0,1,2'), ('0', '1', '2'))
        self.assertEqual(parallel.parse_gpus('2'), ('2',))
        for value in ('', '0,0', '0,00', '0,,1', '-1', 'cuda:0', '0,1,2,3'):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                parallel.parse_gpus(value)

    def test_reservations_use_integer_margin_and_add_active_unwritten_cache(self):
        for size in (0, 1, 10, 100, 2**54):
            with self.subTest(size=size):
                self.assertEqual(parallel.reservation_bytes(size), (size*11 + 9)//10)
        self.assertEqual(parallel.admission_bytes(self.job('next', size=100),
            [dict(remaining_cache_bytes=100), dict(remaining_cache_bytes=0)]), 220)

    def test_three_workers_use_distinct_gpu_environments_and_never_exceed_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, command = self.fixture(root)
            jobs = [self.job('condition' + str(i)) for i in range(6)]
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'inherited-parent-value'}):
                results = parallel.run_parallel(jobs, ('0', '1', '2'), scratch=root,
                    state_dir=root/'state', worker_command=command, disk_free=lambda: 10**12,
                    poll_interval=.01)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'inherited-parent-value')
            active, maximum, used = {}, 0, set()
            for event in events_at(events):
                if event['kind'] == 'start':
                    self.assertNotIn(event['gpu'], active.values())
                    active[event['name']] = event['gpu']
                    maximum = max(maximum, len(active))
                    used.add(event['gpu'])
                    self.assertLessEqual(len(active), 3)
                elif event['kind'] == 'end':
                    del active[event['name']]
            self.assertFalse(active)
            self.assertEqual(maximum, 3)
            self.assertEqual(used, {'0', '1', '2'})
            self.assertEqual(set(results), {job['name'] for job in jobs})
            for job in jobs:
                self.assertEqual(results[job['name']]['condition'], job['label'])
                self.assertTrue((root/'state'/(job['name'] + '.log')).is_file())

    def test_pending_job_waits_for_reserved_cache_and_starts_before_training_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, command = self.fixture(root)
            jobs = [self.job('condition' + str(i), size=100, mode='release') for i in range(3)]
            parallel.run_parallel(jobs, ('0', '1', '2'), scratch=root, state_dir=root/'state',
                worker_command=command, disk_free=lambda: 165, poll_interval=.01)
            unreleased, active, maximum = set(), set(), 0
            for event in events_at(events):
                name, kind = event['name'], event['kind']
                if kind == 'start':
                    # 2 * ceil(100 * 1.1) exceeds the available 165 bytes.
                    self.assertFalse(unreleased, 'two uncached reservations were admitted together')
                    unreleased.add(name)
                    active.add(name)
                    maximum = max(maximum, len(active))
                elif kind == 'release':
                    unreleased.remove(name)
                elif kind == 'end':
                    active.remove(name)
            self.assertGreater(maximum, 1, 'a completed cache reservation must release before the job exits')
            self.assertFalse(unreleased | active)

    def test_single_job_larger_than_available_scratch_fails_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(parallel.subprocess, 'Popen') as launch:
                with self.assertRaises((OSError, ValueError)):
                    parallel.run_parallel([self.job('too_large', size=100)], ('0',),
                        scratch=root, state_dir=root/'state', worker_command=lambda *args: [],
                        disk_free=lambda: 109, poll_interval=.01)
                launch.assert_not_called()

    def test_smaller_pending_job_is_not_blocked_by_an_oversized_first_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, command = self.fixture(root)
            with self.assertRaises(OSError):
                parallel.run_parallel([self.job('large', size=1000), self.job('small', size=100)],
                    ('0', '1'), scratch=root, state_dir=root/'state', worker_command=command,
                    disk_free=lambda: 165, poll_interval=.01)
            self.assertEqual([(event['name'], event['kind']) for event in events_at(events)],
                             [('small', 'start'), ('small', 'end')])
            self.assertTrue((root/'state/small.result.json').is_file())

    def test_worker_failure_stops_other_process_groups_and_preserves_existing_artifacts(self):
        self.exercise_aborted_scheduler(interrupt=False)

    def test_keyboard_interrupt_stops_workers_and_their_children(self):
        self.exercise_aborted_scheduler(interrupt=True)

    def exercise_aborted_scheduler(self, *, interrupt):
        # Fake the signal boundary here: the separate real subprocess test below
        # verifies descendants. macOS app sandboxes can reject killpg for an
        # already-disappeared group with EPERM instead of ESRCH.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root/'completed_best.pt'
            artifact.write_bytes(b'completed model')
            processes, signals = {}, []

            class Process:
                def __init__(self, name, log):
                    self.pid = 900000 + len(processes)
                    self.name, self.log, self.returncode = name, log, None

                def poll(self):
                    if not interrupt and self.name == 'failure':
                        self.returncode = 7
                    return self.returncode

                def wait(self, timeout=None):
                    if self.returncode is None:
                        raise subprocess.TimeoutExpired(self.name, timeout)
                    return self.returncode

            def launch(command, **options):
                self.assertTrue(options['start_new_session'])
                self.assertEqual(options['pass_fds'], (42,))
                proc = Process(command[0], options['stdout'])
                processes[proc.pid] = proc
                return proc

            def kill_group(pid, signum):
                signals.append((pid, signum))
                if processes[pid].returncode is None:
                    processes[pid].returncode = -signum

            def sleep(_interval):
                if interrupt:
                    raise KeyboardInterrupt

            expected = KeyboardInterrupt if interrupt else RuntimeError
            with patch.object(parallel.os, 'killpg', side_effect=kill_group):
                with self.assertRaises(expected):
                    parallel.run_parallel([self.job('survivor'), self.job('failure')],
                        ('0', '1'), scratch=root, state_dir=root/'state',
                        worker_command=lambda job, *_: [job['name']], lock_fd=42,
                        popen_factory=launch, disk_free=lambda: 10**12, sleep=sleep)
            self.assertEqual(len(processes), 2)
            for pid, proc in processes.items():
                self.assertIn((pid, signal.SIGTERM), signals)
                self.assertIn((pid, signal.SIGKILL), signals)
                self.assertTrue(proc.log.closed)
                self.assertIsNotNone(proc.returncode)
            self.assertEqual(artifact.read_bytes(), b'completed model')

    def test_group_stop_escalates_when_worker_and_descendant_ignore_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat = Path(tmp)/'child.heartbeat'
            child = "import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);p=Path(sys.argv[1]);n=0\nwhile True:\n p.write_text(str(n));n+=1;time.sleep(.01)"
            source = "import signal,subprocess,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(120)"
            proc = subprocess.Popen([sys.executable, '-c', source, child, str(heartbeat)],
                                    start_new_session=True)
            try:
                deadline = time.monotonic() + 2
                while not heartbeat.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(heartbeat.is_file())
                first = heartbeat.read_text()
                time.sleep(.05)
                self.assertNotEqual(heartbeat.read_text(), first, 'descendant must be alive before stop')
                parallel.stop_process_group(proc, timeout=.1)
                self.assertIsNotNone(proc.poll())
                time.sleep(.03)
                stopped = heartbeat.read_bytes()
                time.sleep(.08)
                self.assertEqual(heartbeat.read_bytes(), stopped,
                                 'SIGKILL must stop the descendant heartbeat as well')
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


class ParallelDepthRunnerTest(unittest.TestCase):
    def fixture(self, root):
        args = SimpleNamespace(output=root/'run', scratch=root/'scratch', data=root/'data',
            weights=root/'weights.pt', m3ed_targets=root/'m3ed_train',
            m3ed_val_targets=root/'m3ed_val', baselines_only=True, local_only=True,
            activity_only=False, dsec_activity_manifest=None, m3ed_activity_manifest=None,
            gpus='0,1,2', preflight=False)
        args.output.mkdir()
        args.scratch.mkdir()
        args.weights.write_bytes(b'fixture weights')
        indexes = {}
        for ds in ('dsec', 'm3ed'):
            stats = root/'outputs/stage1_common_20261105/statistics'/f'{ds}_common.json'
            stats.parent.mkdir(parents=True, exist_ok=True)
            stats.write_text('{}')
            events = args.data/('DSEC_cache/events/gep_rgb' if ds == 'dsec' else 'm3ed_cache/half_dagr')
            events.mkdir(parents=True)
            for condition in runner.CONDITIONS[:3]:
                _, _, folder, model, step = condition
                checkpoint = root/'outputs'/folder/f'{ds}_{model}'/'checkpoints'/f'step_{step:08d}.pt'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes((ds + condition[0]).encode())
            for role in ('train', 'val'):
                path = args.output/'targets'/f'{ds}_{role}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(dict(records=[{}] * (2 if role == 'train' else 1))))
                indexes[ds, role] = path
        return args, indexes

    def completed_heads(self, args):
        for ds in ('dsec', 'm3ed'):
            for condition in runner.CONDITIONS[:3]:
                head = args.output/'heads'/(ds + '_' + condition[0])/'stages'
                head.mkdir(parents=True)
                (head/'best.pt').write_bytes(b'completed trained head')
                (head/'_SUCCESS').write_text('complete')
                (head/'continuous.json').write_text('{"metrics":{"all":{"mae":9}}}')
                if ds == 'dsec':
                    (head/'disparity_val.json').write_text('{"mean_sequence_mae_px":1}')

    @staticmethod
    def tree(root):
        return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
                for path in root.rglob('*')}

    def test_completed_e0_head_and_depth_report_reuse_only_missing_disparity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, indexes = self.fixture(root)
            self.completed_heads(args)
            head = args.output/'heads/dsec_E0/stages'
            (head/'disparity_val.json').unlink()
            summary = args.output/'summary.json'
            summary.write_bytes(b'parent-owned summary')
            originals = {path: path.read_bytes() for path in head.iterdir()}
            calls, statuses = [], []

            def fake(tool, *cmd):
                calls.append(tool)
                self.assertEqual(tool, 'evaluate_dsec_disparity_window.py')
                self.assertEqual(cmd[cmd.index('--name') + 1], 'dsec_E0')
                Path(cmd[cmd.index('--output') + 1]).write_text('{"mean_sequence_mae_px":0.7}')

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'run', side_effect=fake):
                job = runner.build_jobs(args, indexes, {})[0]
                self.assertFalse(job['need_main'])
                self.assertTrue(job['need_disparity'])
                self.assertEqual(job['roles'], [])
                self.assertEqual(job['remaining_cache_bytes'], 0)
                result = runner.run_condition(args, job, status_callback=statuses.append)
                runner.run_condition(args, runner.build_jobs(args, indexes, {})[0])
            self.assertEqual(calls, ['evaluate_dsec_disparity_window.py'])
            self.assertEqual(result['depth'], dict(mae=9))
            self.assertEqual(result['mae_px'], .7)
            self.assertEqual(summary.read_bytes(), b'parent-owned summary')
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(args.scratch.iterdir()), [])
            self.assertTrue(all(status['remaining_cache_bytes'] == 0 for status in statuses))

    def test_interrupted_json_is_preserved_and_only_validation_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, indexes = self.fixture(root)
            self.completed_heads(args)
            head = args.output/'heads/dsec_E0/stages'
            report = head/'continuous.json'
            report.write_bytes(b'{"metrics":')
            calls = []

            def fake(tool, *cmd):
                calls.append(tool)
                if tool == 'cache_dense_features.py':
                    self.assertEqual(Path(cmd[cmd.index('--target-index') + 1]), indexes['dsec', 'val'])
                    Path(cmd[cmd.index('--output-dir') + 1]).mkdir()
                elif tool == 'evaluate_dense_probe.py':
                    Path(cmd[cmd.index('--output') + 1]).write_text('{"metrics":{"all":{"mae":2}}}')
                else:
                    self.fail('A completed head must not train or export: ' + tool)

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'run', side_effect=fake):
                job = runner.build_jobs(args, indexes, {})[0]
                self.assertEqual(job['roles'], ['val'])
                self.assertEqual(job['invalid_reports'], [str(report)])
                result = runner.run_condition(args, job)
            self.assertEqual(calls, ['cache_dense_features.py', 'evaluate_dense_probe.py'])
            self.assertEqual(result['depth'], dict(mae=2))
            preserved = list(head.glob('continuous.json.partial_*'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(preserved[0].read_bytes(), b'{"metrics":')
            self.assertEqual(list(args.scratch.iterdir()), [])

    def test_incomplete_head_never_reuses_old_depth_or_disparity_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, indexes = self.fixture(root)
            self.completed_heads(args)
            head = args.output/'heads/dsec_E0/stages'
            for marker in ('best.pt', '_SUCCESS'):
                path = head/marker
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.subTest(missing=marker), patch.object(runner, 'ROOT', root):
                        job = runner.build_jobs(args, indexes, {})[0]
                        self.assertEqual(job['roles'], ['train', 'val'])
                        self.assertTrue(job['need_main'])
                        self.assertTrue(job['need_disparity'])
                        self.assertEqual(set(job['invalid_reports']),
                            {str(head/'continuous.json'), str(head/'disparity_val.json')})
                finally:
                    path.write_bytes(original)

    def test_failed_extraction_preserves_partial_cache_and_outstanding_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, indexes = self.fixture(root)
            statuses = []

            def fail(tool, *cmd):
                self.assertEqual(tool, 'cache_dense_features.py')
                cache = Path(cmd[cmd.index('--output-dir') + 1])
                cache.mkdir()
                (cache/'partial.bin').write_bytes(b'partial feature cache')
                raise RuntimeError('fixture extraction failed')

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'run', side_effect=fail):
                job = runner.build_jobs(args, indexes, {})[0]
                with self.assertRaisesRegex(RuntimeError, 'fixture extraction failed'):
                    runner.run_condition(args, job, status_callback=statuses.append)
            self.assertIn('preserved', statuses[-1]['phase'])
            self.assertEqual(statuses[-1]['remaining_cache_bytes'], job['remaining_cache_bytes'])
            self.assertEqual((Path(statuses[-1]['cache_dir'])/'train/partial.bin').read_bytes(),
                             b'partial feature cache')
            self.assertFalse((args.output/'summary.json').exists())

    def test_parent_writes_one_summary_and_exports_once_after_all_worker_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _indexes = self.fixture(root)
            args.local_only = False
            self.completed_heads(args)
            timeline, summary_writes = [], []
            original_save = runner.save

            def save(path, value):
                if path == args.output/'summary.json':
                    summary_writes.append(value.copy())
                original_save(path, value)

            def dispatch(jobs, gpus, **options):
                self.assertEqual(gpus, ('0', '1', '2'))
                self.assertEqual(options['lock_fd'], 42)
                results = {}
                for job in reversed(jobs):
                    command = options['worker_command'](job, root/'status.json', root/'result.json')
                    self.assertEqual(command[command.index('--worker-name') + 1], job['name'])
                    self.assertNotIn('--gpus', command)
                    results[job['name']] = dict(condition=job['label'], depth=dict(mae=9))
                    if job['dataset'] == 'dsec':
                        results[job['name']]['mae_px'] = .2 if job['name'] == 'dsec_E2' else 1
                    timeline.append('done ' + job['name'])
                return results

            def fake(tool, *cmd):
                self.assertEqual(tool, 'export_dsec_depth_submission.py')
                if '--preflight' in cmd:
                    timeline.append('export preflight')
                    return
                self.assertEqual(len([entry for entry in timeline if entry.startswith('done ')]), 6)
                self.assertEqual(len(json.loads((args.output/'summary.json').read_text())), 6)
                self.assertEqual(cmd[cmd.index('--name') + 1], 'dsec_E2')
                timeline.append('export')
                submission = Path(cmd[cmd.index('--output') + 1])
                submission.mkdir()
                (submission/'submission.zip').write_bytes(b'fixture ZIP')

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'verify_decoder'), \
                 patch.object(runner, 'timestamps', return_value=root), \
                 patch.object(runner, 'run', side_effect=fake), patch.object(runner, 'save', side_effect=save), \
                 patch.object(runner, 'run_condition') as direct_worker, \
                 patch.object(parallel, 'run_parallel', side_effect=dispatch):
                runner.pipeline(args, lock_fd=42)
                runner._export(args, json.loads((args.output/'summary.json').read_text()), [])
            direct_worker.assert_not_called()
            self.assertEqual(len(summary_writes), 1)
            self.assertEqual(set(summary_writes[0]), {ds + '_' + c[0] for ds in ('dsec', 'm3ed')
                                                    for c in runner.CONDITIONS[:3]})
            self.assertEqual(timeline[-1], 'export')
            self.assertEqual(timeline.count('export'), 1)

    def test_internal_worker_writes_only_its_status_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, indexes = self.fixture(root)
            self.completed_heads(args)
            (args.output/'summary.json').write_text('parent summary')
            (args.output/'run.json').write_text('parent identity')
            before = self.tree(root)
            plan = root/'plan.json'
            with patch.object(runner, 'ROOT', root):
                jobs = runner.build_jobs(args, indexes, {})
            values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            plan.write_text(json.dumps(dict(arguments=values, jobs=jobs)))
            status, result = root/'status.json', root/'result.json'
            with (args.output/'run.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                worker = SimpleNamespace(parent_lock_fd=lock.fileno(), worker_plan=plan,
                    worker_name='dsec_E0', worker_status=status, worker_result=result)
                with patch.object(runner, 'ROOT', root), patch.object(runner, '_LOCK_FD', None), \
                     patch.object(runner, 'run') as launch, patch.object(runner, 'verify_decoder') as decoder:
                    runner._worker(worker)
            launch.assert_not_called()
            decoder.assert_not_called()
            for name, content in before.items():
                path = root/name
                self.assertEqual(path.read_bytes() if path.is_file() else None, content)
            self.assertEqual(json.loads(result.read_text())['condition'], runner.CONDITIONS[0][1])
            self.assertEqual(json.loads(status.read_text())['phase'], 'completed')
            self.assertFalse((args.output/'submission').exists())

    def test_export_uses_first_selected_gpu_and_restores_parent_environment_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _indexes = self.fixture(root)
            args.gpus = '2,1'
            summary = {'dsec_' + condition[0]: dict(mae_px=i)
                       for i, condition in enumerate(runner.CONDITIONS[:3])}

            def fail(tool, *cmd):
                self.assertEqual(tool, 'export_dsec_depth_submission.py')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
                raise RuntimeError('fixture export failed')

            with patch.object(runner, 'ROOT', root), patch.object(runner, 'run', side_effect=fail), \
                 patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'parent-value'}):
                with self.assertRaisesRegex(RuntimeError, 'fixture export failed'):
                    runner._export(args, summary, ['--output', args.output/'submission'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'parent-value')

    def command_line(self, args, *, preflight=False):
        command = ['run_common_depth.py', '--output', str(args.output), '--scratch', str(args.scratch),
            '--data', str(args.data), '--weights', str(args.weights), '--baselines-only',
            '--local-only', '--gpus', '0,1,2']
        return command + (['--preflight'] if preflight else [])

    def test_preflight_checks_all_conditions_without_writes_importing_models_or_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _indexes = self.fixture(root)
            args.scratch = root/'not_created/scratch'
            before = self.tree(root)
            output = StringIO()
            with patch.object(runner, 'ROOT', root), patch.object(runner.os, 'chdir'), \
                 patch.object(sys, 'path', sys.path.copy()), \
                 patch.object(sys, 'argv', self.command_line(args, preflight=True)), \
                 patch.dict(sys.modules, {'torch': None, 'numpy': None}), \
                 patch.object(runner, 'run') as launch, patch.object(runner, 'verify_decoder') as decoder, \
                 patch.object(runner, 'timestamps') as download, \
                 patch.object(parallel.subprocess, 'Popen') as workers, redirect_stdout(output):
                runner.main()
            for operation in (launch, decoder, download, workers):
                operation.assert_not_called()
            self.assertEqual(self.tree(root), before)
            self.assertIn('GPU(s) 0, 1, 2', output.getvalue())
            for ds in ('dsec', 'm3ed'):
                for condition in runner.CONDITIONS[:3]:
                    self.assertIn('[' + ds + '_' + condition[0] + '] ' + condition[1], output.getvalue())
            self.assertIn('no model loading or inference', output.getvalue())

    def test_old_runner_lock_refuses_parallel_start_and_readonly_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, _indexes = self.fixture(root)
            with (args.output/'run.lock').open('a') as old_runner:
                fcntl.flock(old_runner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for preflight in (False, True):
                    with self.subTest(preflight=preflight), patch.object(runner, 'ROOT', root), \
                         patch.object(runner.os, 'chdir'), patch.object(sys, 'path', sys.path.copy()), \
                         patch.object(sys, 'argv', self.command_line(args, preflight=preflight)), \
                         patch.object(runner, 'pipeline') as launch, \
                         patch('sys.stderr', new_callable=StringIO) as error:
                        with self.assertRaises(SystemExit) as exit_result:
                            runner.main()
                        self.assertEqual(exit_result.exception.code, 2)
                        launch.assert_not_called()
                        self.assertIn('another Depth runner is still active', error.getvalue())


if __name__ == '__main__':
    unittest.main()
