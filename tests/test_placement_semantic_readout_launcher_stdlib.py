"""Exercise the four-head Semantic launcher without checkpoints, CUDA or datasets."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_placement_semantic_readout as suite
from tools.placement_semantic_readout_results import GROUPS


class ReadoutLauncherTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.output = self.root / "output"
        self.scratch = self.root / "scratch"
        self.flags = ["--output", str(self.output), "--scratch", str(self.scratch),
                      "--data", str(self.root / "data"), "--gpus", "2,0"]
        self.controls = {ds: {} for ds in suite.DATASETS}
        for ds, cond in suite.JOBS:
            self.controls[ds][cond] = dict(
                dataset=ds, condition=cond, feature="h", seed=0, global_step=99999,
                role="test" if ds == "dsec" else "validation", head_recipe={"seed": 0},
                backbone_sha256=f"{ds}-{cond}", head_sha256=f"{ds}-{cond}-old",
                mIoU=dict.fromkeys(GROUPS, 0.5))

    def test_dry_plan_has_four_cells_and_requires_no_server_inputs(self):
        with patch.object(suite, "controls_for_jobs", side_effect=AssertionError), \
                redirect_stdout(io.StringIO()) as output:
            suite.main(self.flags)
        for ds, cond in suite.JOBS:
            self.assertIn(f"[{ds}_{cond}] {suite.label(cond)}", output.getvalue())
        self.assertIn("2,0", output.getvalue())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.scratch.exists())

    def test_preflight_routes_all_cells_to_their_exact_frozen_checkpoint(self):
        with patch.dict(os.environ), redirect_stdout(io.StringIO()), \
                patch.object(suite, "controls_for_jobs", return_value=self.controls), \
                patch.object(suite, "run_condition", return_value=None) as route, patch.object(suite, "run_parallel") as queue:
            suite.main(["--preflight", *self.flags])
        self.assertEqual(route.call_count, 4)
        for call, (ds, cond) in zip(route.call_args_list, suite.JOBS):
            args, dataset, condition, *_ = call.args
            self.assertFalse(args.execute)
            self.assertEqual((dataset, condition), (ds, cond))
            self.assertEqual(call.kwargs["feature"], "stages")
            self.assertEqual(call.kwargs["checkpoint"], suite.checkpoint_paths(ds, suite.ROOT)[cond])
            self.assertEqual(call.kwargs["label"], suite.label(cond))
        queue.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_partial_run_resume_and_read_only_reverification(self):
        flags = ["--execute", *self.flags]
        submitted, verified = [], []
        def resume(a, ds, cond, *args, **kwargs):
            self.assertEqual(kwargs["feature"], "stages")
            verified.append((ds, cond))
            return suite.read(a.output / f"{ds}_{cond}" / "verified.json")

        def schedule(jobs, gpus, *, scratch, idle_devices, worker_command, on_complete, **kwargs):
            available = idle_devices(gpus)
            self.assertEqual(available, ("2",))
            self.assertEqual(scratch, self.scratch)
            self.assertTrue(scratch.is_dir())
            for job in jobs:
                ds, cond, name = job["dataset"], job["condition"], job["name"]
                self.assertEqual(job["kind"], "semantic")
                self.assertEqual(job["remaining_cache_bytes"], 10 if ds == "dsec" else 20)
                argv = worker_command(job, self.root / "status", self.root / "worker-result")
                self.assertEqual(argv[argv.index("--worker")+1:argv.index("--worker")+3], [ds, cond])
                self.assertEqual(int(argv[argv.index("--parent-lock-fd")+1]), kwargs["lock_fd"])
                self.assertEqual(argv[argv.index("--scratch")+1], str(self.scratch))
                self.assertFalse((self.output / "_SUCCESS").exists())
                submitted.append((ds, cond))
                score = 0.52 if cond == "E1" else 0.55
                row = dict(self.controls[ds][cond], feature="stages", head=name+"/best.pt", head_sha256=name+"-new",
                           mIoU=dict.fromkeys(GROUPS, score))
                suite.save(self.output / name / "verified.json", row)
                on_complete(name, row)
                partial = suite.read(self.output / "summary.json")
                self.assertEqual(len(partial["four_layer_heads"]), len(submitted))
                if len(submitted) == 2:
                    raise RuntimeError("interrupted fixture")

        with patch.dict(os.environ), redirect_stdout(io.StringIO()), \
                patch.object(suite, "controls_for_jobs", return_value=self.controls), \
                patch.object(suite, "idle_gpus", return_value=("2",)) as idle, \
                patch.object(suite, "cache_bytes", side_effect=lambda a, ds, *args, **kw:
                             (10 if ds == "dsec" else 20) if kw["feature"] == "stages" else 0), \
                patch.object(suite, "run_condition", side_effect=resume), \
                patch.object(suite, "verify_head_recipe", return_value={"seed": 0}), \
                patch.object(suite, "run_parallel", side_effect=schedule) as queue:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                suite.main(flags)
            self.assertFalse((self.output / "_SUCCESS").exists())
            suite.main(flags)
            self.assertEqual(set(submitted), set(suite.JOBS))
            self.assertEqual(len(submitted), 4)
            self.assertEqual(verified, submitted[:2])
            self.assertTrue((self.output / "_SUCCESS").is_file())
            summary = suite.read(self.output / "summary.json")
            self.assertEqual(summary["final_only_controls"], self.controls)
            for ds in suite.DATASETS:
                self.assertAlmostEqual(summary["comparisons"][ds]["interaction_pp"]["all"], 3)
            verified.clear()
            suite.main(flags)
            self.assertEqual(set(verified), set(suite.JOBS))
            self.assertEqual(queue.call_count, 2)
            self.assertEqual(idle.call_count, 2)
            self.assertEqual(idle.call_args.args, (("2", "0"),))
            self.controls["m3ed"]["E1"]["head_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "Different data or control"):
                suite.main(flags)
            self.assertEqual(queue.call_count, 2)

    def test_incomplete_summary_invalidates_previous_completion_marker(self):
        self.output.mkdir()
        (self.output / "_SUCCESS").touch()
        suite.write_summary(self.output, self.controls, {})
        self.assertFalse((self.output / "_SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
