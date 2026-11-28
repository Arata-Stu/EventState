import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import b3_skip_progress as progress


class PipelineProgressTests(unittest.TestCase):
    def test_barrier_between_datasets_and_partial_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # DSEC takes longest in full, M3ED takes longest in Semantic.
            estimates = [(1, "smoke"), (2, "smoke"), (7200, "full"), (3600, "full")]
            semantic = [(100, "", "DSEC完了見込み"), (10800, "履歴", "M3ED head")]
            with patch.object(progress, "stage1_remaining", side_effect=estimates), \
                    patch.object(progress, "semantic_remaining", side_effect=semantic):
                lines, done = progress.snapshot(root, 2, time.time())
            self.assertFalse(done)
            self.assertIn("約5.0時間", next(line for line in lines if "見積可能な工程の残り" in line))
            self.assertIn("6.0時間", next(line for line in lines if "1 GPU" in line))
            self.assertIn("M3ED 履歴", lines[-1])
            self.assertEqual(list(root.iterdir()), [])

    def test_new_worker_rate_supersedes_history_and_completion_marker_ends_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "dsec_E1_b3_skip_w1.log").write_text(
                "[train] step=20 loss=0.2 clips_per_update=8 samples_per_second=16\n")
            seconds, detail = progress.stage1_remaining(root, "dsec", 100, 12, time.time())
            self.assertEqual(seconds, 40)
            self.assertIn("今回の速度", detail)
            self.assertIn("loss=0.2", detail)
            stalled, detail = progress.stage1_remaining(root, "dsec", 100, 12, time.time() + 900)
            self.assertIsNone(stalled)
            self.assertIn("再計測待ち", detail)
            finished = root / "stage2_b3_skip_semantic_20261126"
            finished.mkdir()
            (finished / "_SUCCESS").touch()
            self.assertTrue(progress.snapshot(root, 1, time.time())[1])

    def test_extended_suite_counts_all_eight_training_jobs_and_has_its_own_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "stage2_b3_skip_semantic_20261126"
            original.mkdir()
            (original / "_SUCCESS").touch()
            lines, done = progress.snapshot(root, 3, time.time(), suite=True)
            self.assertFalse(done)
            training = [line for line in lines if "Stage 1 full " in line]
            self.assertEqual(len(training), 8)
            self.assertEqual(sum("E1_b3_6_9_skip" in line for line in training), 2)
            self.assertTrue(any("3 GPU使用" in line for line in lines))
            self.assertEqual(progress.queued_seconds([12, 8, 10, 11], 3), 19)
            current = root / "stage2_skip_placement_semantic_20261128"
            current.mkdir()
            (current / "_SUCCESS").touch()
            self.assertTrue(progress.snapshot(root, 3, time.time(), suite=True)[1])

    def test_background_follow_preserves_log_exit_status_and_survives_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            tools = project / "tools"
            tools.mkdir()
            for name in ("run_b3_skip.sh", "run_skip_placements.sh", "run_final_skip.sh", "b3_skip_progress.py", "b3_semantic_timing.py",
                         "common_depth_parallel.py", "depth_slots.py", "progress_eta.py",
                         "placement_conditions.py", "stage1_progress.py", "summarize_training_logs.py"):
                shutil.copy(progress.PROJECT / "tools" / name, tools)
            record = project / progress.RATE_RECORD.relative_to(progress.PROJECT)
            record.parent.mkdir(parents=True)
            shutil.copy(progress.RATE_RECORD, record)
            log = project / "outputs/b3_skip_20261126.log"
            log.parent.mkdir()
            log.write_text("previous invocation\n")
            worker = project / "env/bin/python"
            worker.parent.mkdir(parents=True)
            worker.write_text('#!/bin/sh\necho worker-ready\nexit 23\n')
            worker.chmod(0o755)
            command = ["bash", "tools/run_b3_skip.sh", "--follow", "0"]
            failed = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=12)
            self.assertEqual(failed.returncode, 23, failed.stderr)
            self.assertIn("worker-ready", failed.stdout)
            self.assertNotIn("previous invocation", failed.stdout)
            self.assertIn("previous invocation", log.read_text())

            worker.write_text('#!/bin/sh\necho waiting-for-release\n'
                              'for i in 1 2 3 4 5 6 7 8; do\n'
                              '  if [ -e release ]; then echo detached-job-finished; exit 23; fi\n'
                              '  sleep 1\ndone\nexit 2\n')
            follower = subprocess.Popen(command, cwd=project, start_new_session=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                self.wait_for_text(log, "waiting-for-release")
                os.killpg(follower.pid, signal.SIGINT)
                follower.communicate(timeout=5)

                (project / "release").touch()
                self.wait_for_text(log, "detached-job-finished")
            finally:
                (project / "release").touch()
                if follower.poll() is None:
                    follower.kill()
                follower.communicate(timeout=5)

            worker.write_text('#!/bin/sh\necho "worker $*"\n')
            new_run = subprocess.run(["bash", "tools/run_skip_placements.sh", "--follow", "0,2"],
                                     cwd=project, capture_output=True, text=True, timeout=12)
            self.assertEqual(new_run.returncode, 0, new_run.stderr)
            commands = [line for line in new_run.stdout.splitlines() if line.startswith("worker ")]
            self.assertEqual(len(commands), 3, new_run.stdout)
            self.assertTrue(all("--skip-placement-suite" in line and "--gpus 0,2" in line for line in commands))
            self.assertIn("--stage smoke", commands[0])
            self.assertIn("stage1_skip_placement_full_20261128", commands[1])
            self.assertIn("tools/run_placement_semantic.py", commands[2])
            self.assertIn(commands[2], (log.parent / "skip_placement_20261128.log").read_text())

            final_run = subprocess.run(["bash", "tools/run_final_skip.sh", "--follow", "0,2"],
                                       cwd=project, capture_output=True, text=True, timeout=12)
            self.assertEqual(final_run.returncode, 0, final_run.stderr)
            commands = [line for line in final_run.stdout.splitlines() if line.startswith("worker ")]
            self.assertEqual(len(commands), 3, final_run.stdout)
            self.assertTrue(all("--final-skip-suite" in line and "--gpus 0,2" in line for line in commands))
            self.assertIn("stage1_final_skip_full_20261128", commands[1])
            self.assertIn(commands[2], (log.parent / "final_skip_20261128.log").read_text())
            lines, done = progress.snapshot(project / "outputs", 3, time.time(), final=True)
            self.assertFalse(done)
            self.assertEqual(sum("Stage 1 full " in line for line in lines), 4)

    def wait_for_text(self, path, expected):
        until = time.monotonic() + 6
        while time.monotonic() < until:
            if expected in path.read_text():
                return
            time.sleep(.05)
        self.fail(f"Missing {expected} in {path.read_text()}")
