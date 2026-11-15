"""Portable temporal-comparison planning checks; no Torch, Hydra or GPU."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import digest
from tools import run_stage1_common as common
from tools import run_temporal_comparison as runner


class TemporalComparisonPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("DSEC", "DSEC_cache/events/gep_rgb", "DSEC_cache/dinov3_vits16",
                          "m3ed_cache/half_dagr", "m3ed_cache/dinov3_vits16_640x352"):
            (self.root / directory).mkdir(parents=True)
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"planning-only checkpoint fixture")
        self.statistics = self.root / "statistics"
        self.statistics.mkdir()
        for dataset in ("dsec", "m3ed"):
            protocol = load_common(dataset)
            stats = dict(protocol_sha256=digest(protocol), sequences=protocol["splits"]["train"],
                         normalize_mean=[0.1, 0.2, 0.3], normalize_std=[0.4, 0.5, 0.6],
                         event_window_us=None)
            (self.statistics / f"{dataset}_common.json").write_text(json.dumps(stats))
        self.args = SimpleNamespace(architectures="convlstm", stage="full", data_root=self.root,
                                    teacher_checkpoint=self.teacher, statistics_dir=self.statistics,
                                    output_root=self.root / "new_comparison", gpus="0,1", seed=0,
                                    num_workers=2, execute=False, skip_preflight=False)

    @staticmethod
    def overrides(job):
        command = job["command"]
        return command[command.index("train.py") + 1:]

    def snapshot(self):
        return {str(path.relative_to(self.root)):
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                for path in self.root.rglob("*")}

    def test_default_is_two_fresh_single_gpu_convlstm_jobs(self):
        jobs = runner.plan(self.args)
        self.assertEqual([j["name"] for j in jobs],
                         ["dsec_E1_convlstm_w1", "m3ed_E1_convlstm_w1"])
        for job in jobs:
            overrides = self.overrides(job)
            self.assertEqual(job["world"], 1)
            self.assertIn("model=convlstm", overrides)
            self.assertIn("experiment=h_distill_lstm", overrides)
            self.assertIn("E1 Encoder＋LSTM（h蒸留）", job["label"])
            self.assertIn("ConvLSTM", job["label"])
            self.assertNotIn("torch.distributed.run", job["command"])
            self.assertIn(f"hydra.run.dir={(self.args.output_root / job['name']).resolve()}", overrides)
            experiment_names = [v.split("=", 1)[1] for v in overrides if v.startswith("experiment.name=")]
            self.assertEqual(len(experiment_names), 1)
            self.assertIn("convlstm", experiment_names[0].lower())

    def test_optional_paired_plan_has_four_separate_outputs_and_equal_budgets(self):
        self.args.architectures = "patchwise,convlstm"
        jobs = runner.plan(self.args)
        self.assertEqual({j["name"] for j in jobs},
                         {f"{dataset}_E1_{architecture}_w1"
                          for dataset in ("dsec", "m3ed")
                          for architecture in ("patchwise", "convlstm")})
        directories = []
        for dataset in ("dsec", "m3ed"):
            subset = [j for j in jobs if j["name"].startswith(dataset + "_")]
            self.assertEqual(len(subset), 2)
            self.assertEqual(subset[0]["steps"], subset[1]["steps"])
            self.assertEqual(subset[0]["planned_frames"], subset[1]["planned_frames"])
            self.assertEqual(subset[0]["effective_clips"], subset[1]["effective_clips"])
            for job in subset:
                architecture = "patchwise" if "_patchwise_" in job["name"] else "convlstm"
                kind = "lstm" if architecture == "patchwise" else "convlstm"
                overrides = self.overrides(job)
                self.assertIn(f"model={kind}", overrides)
                self.assertIn("Patchwise LSTM" if architecture == "patchwise" else "ConvLSTM", job["label"])
                directories.extend(v for v in overrides if v.startswith("hydra.run.dir="))
        self.assertEqual(len(set(directories)), 4)

    def test_only_temporal_kind_and_run_identity_differ_from_existing_e1_recipe(self):
        self.args.architectures = "patchwise,convlstm"
        baseline_args = copy.copy(self.args)
        baseline_args.phase = "baseline"
        baseline_args.lstm_from_stage = None
        baseline_args.four_stage_lstm = False
        baseline = {j["name"].split("_", 1)[0]: j for j in common.plan(baseline_args)
                    if "_E1_" in j["name"]}

        def recipe(job):
            return [v for v in self.overrides(job)
                    if not v.startswith(("model=", "hydra.run.dir=", "experiment.name="))]

        for job in runner.plan(self.args):
            dataset = job["name"].split("_", 1)[0]
            self.assertEqual(recipe(job), recipe(baseline[dataset]))
            self.assertEqual(job["statistics_sha256"], baseline[dataset]["statistics_sha256"])
            self.assertEqual(job["protocol"], baseline[dataset]["protocol"])

    def test_smoke_and_full_are_fresh_runs_with_fixed_final_step_and_frame_budget(self):
        for stage, steps, warmup in (("smoke", 99, 9), ("full", 99999, 999)):
            with self.subTest(stage=stage):
                self.args.stage = stage
                self.args.output_root = self.root / stage
                for job in runner.plan(self.args):
                    dataset = job["name"].split("_", 1)[0]
                    clips, length = (8, 8) if dataset == "dsec" else (4, 16)
                    overrides = self.overrides(job)
                    self.assertEqual(job["steps"], steps)
                    self.assertEqual(job["effective_clips"], clips)
                    self.assertEqual(job["planned_frames"], steps * clips * length)
                    self.assertIn(f"training.max_steps={steps}", overrides)
                    self.assertIn(f"scheduler.warmup_steps={warmup}", overrides)
                    self.assertIn(f"dataset.sequence_length={length}", overrides)
                    self.assertIn("training.resume=null", overrides)
                    self.assertIn("training.gradient_accumulation=1", overrides)
                    interval = steps if stage == "smoke" else 9999
                    self.assertIn(f"training.checkpoint_every={interval}", overrides)
                    # The common executor selects exactly the final step, not a smoke or val-best weight.
                    expected_name = "step_00000099.pt" if stage == "smoke" else "step_00099999.pt"
                    self.assertEqual(f"step_{job['steps']:08d}.pt", expected_name)

    def test_planning_is_read_only_and_does_not_mutate_supplied_arguments(self):
        before_files = self.snapshot()
        before_args = copy.deepcopy(vars(self.args))
        runner.plan(self.args)
        self.assertEqual(self.snapshot(), before_files)
        self.assertEqual(vars(self.args), before_args)
        self.assertFalse(self.args.output_root.exists())

    def test_one_to_three_distinct_gpu_ids_keep_each_job_single_gpu(self):
        for gpu_ids in ("2", "0,2", "0,1,2"):
            with self.subTest(gpus=gpu_ids):
                self.args.gpus = gpu_ids
                self.assertTrue(all(j["world"] == 1 for j in runner.plan(self.args)))

    def test_invalid_gpu_lists_are_rejected_without_outputs(self):
        for gpu_ids in ("", "0,0", "0,00", "0,1,2,3", "-1", "gpu0", "0,"):
            with self.subTest(gpus=gpu_ids):
                self.args.gpus = gpu_ids
                with self.assertRaises(ValueError):
                    runner.plan(self.args)
                self.assertFalse(self.args.output_root.exists())

    def test_repeated_unknown_and_empty_architectures_are_rejected(self):
        for architectures in ("", "convlstm,convlstm", "patchwise,patchwise", "unknown", "patchwise,unknown"):
            with self.subTest(architectures=architectures):
                self.args.architectures = architectures
                with self.assertRaises(ValueError):
                    runner.plan(self.args)
                self.assertFalse(self.args.output_root.exists())

    def test_existing_root_is_preserved_including_empty_or_partial_runs(self):
        self.args.output_root.mkdir()
        for partial in (False, True):
            with self.subTest(partial=partial):
                if partial:
                    (self.args.output_root / "checkpoint.pt").write_bytes(b"preserve me")
                before = self.snapshot()
                with self.assertRaises(FileExistsError):
                    runner.plan(self.args)
                self.assertEqual(self.snapshot(), before)

    def test_protected_train_pool_and_original_statistics_remain_required(self):
        path = self.statistics / "m3ed_common.json"
        statistics = json.loads(path.read_text())
        statistics["sequences"].append(load_common("m3ed")["splits"]["val"][0])
        path.write_text(json.dumps(statistics))
        with self.assertRaises(ValueError):
            runner.plan(self.args)
        self.assertFalse(self.args.output_root.exists())

    def test_fixed_window_statistics_cannot_silently_replace_rgb_interval_recipe(self):
        path = self.statistics / "dsec_common.json"
        statistics = json.loads(path.read_text())
        statistics["event_window_us"] = 20000
        path.write_text(json.dumps(statistics))
        with self.assertRaises(ValueError):
            runner.plan(self.args)
        self.assertFalse(self.args.output_root.exists())

    def test_missing_teacher_or_input_cache_stops_before_creating_run(self):
        self.teacher.unlink()
        with self.assertRaises(FileNotFoundError):
            runner.plan(self.args)
        self.teacher.write_bytes(b"fixture")
        (self.root / "m3ed_cache/dinov3_vits16_640x352").rmdir()
        with self.assertRaises(FileNotFoundError):
            runner.plan(self.args)
        self.assertFalse(self.args.output_root.exists())


if __name__ == "__main__":
    unittest.main()
