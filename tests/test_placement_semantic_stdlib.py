"""Placement Semantic routing, identity, cleanup and resumption without Torch or data."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_placement_semantic as suite


class PlacementSemanticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.baselines = suite.baseline_results()

    def test_defaults_plan_all_six_for_both_datasets_without_writes(self):
        output = self.root / "output"
        with redirect_stdout(io.StringIO()) as stream:
            suite.main(["--output", str(output)])
        self.assertFalse(output.exists())
        text = stream.getvalue()
        self.assertEqual(text.count("| Semantic |"), 12)
        for key in suite.PLACEMENTS:
            self.assertIn(f"[m3ed_{key}]", text)
            self.assertIn(f"[dsec_{key}]", text)
        self.assertNotIn("depth", text.lower())

    def test_gpu_list_and_legacy_gpu_select_the_expected_pool(self):
        for flags, pool in (([], "0"), (["--gpu", "2"], "2"), (["--gpus", "2,0,1"], "2,0,1")):
            with redirect_stdout(io.StringIO()) as stream:
                suite.main(flags)
            self.assertEqual(stream.getvalue().count(f"| GPU pool {pool}\n"), 12)
        for flags in (["--gpu", "0", "--gpus", "1"], ["--gpus", "0,0"], ["--gpu", "-1"]):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                suite.main(flags)

    def test_skip_plans_two_isolated_jobs_and_requires_a_distinct_output(self):
        flags = ["--conditions", suite.SKIP_CONDITION, "--gpus", "0,1,2"]
        with redirect_stdout(io.StringIO()) as stream:
            suite.main(flags)
        self.assertEqual(stream.getvalue().count("| Semantic |"), 2)
        for ds in suite.DATASETS:
            self.assertIn(ds + "_" + suite.SKIP_CONDITION, stream.getvalue())
        for args in (["--output", str(suite.PLACEMENT_OUTPUT)],
                     ["--conditions", suite.SKIP_CONDITION, "E1_b3"]):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                suite.main([*flags, *args])

    def test_skip_placement_suite_selects_all_eight_and_keeps_workers_in_the_same_run(self):
        flags = ["--skip-placement-suite"]
        with redirect_stdout(io.StringIO()) as stream:
            suite.main(flags)
        self.assertEqual(stream.getvalue().count("| Semantic | GPU pool 0,1,2"), 8)
        for cond in ("E1_b3_6_9", "E1_b6_skip", "E1_b9_skip", "E1_b3_6_9_skip"):
            for ds in suite.DATASETS:
                self.assertIn(f"[{ds}_{cond}]", stream.getvalue())
        with patch.object(suite, "run_worker") as dispatch:
            suite.main(flags + ["--conditions", "E1_b6_skip", "--datasets", "m3ed",
                                "--worker-result", str(self.root / "result"), "--execute"])
            args, ds, condition = dispatch.call_args.args
            self.assertEqual((ds, condition), ("m3ed", "E1_b6_skip"))
            self.assertEqual(args.stage1_root.name, "stage1_skip_placement_full_20261128")
            self.assertEqual(args.output.name, "stage2_skip_placement_semantic_20261128")
            command = suite.worker_command(args, dict(condition=condition, dataset=ds), "status", "result", 25)
            self.assertIn("--skip-placement-suite", command)
            self.assertIn(str(args.placement_semantic_root), command)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            suite.main(flags + ["--output", str(suite.PLACEMENT_OUTPUT)])

    def test_final_skip_suite_dispatch_summary_and_baseline_control(self):
        args = SimpleNamespace(conditions=suite.FINAL_SKIP_SUITE, output=self.root / "final",
                               placement_semantic_root=self.root / "archive", skip_placement_suite=True,
                               final_skip_suite=True)
        for ds in suite.DATASETS:
            for cond in suite.FINAL_SKIP_SUITE:
                suite.save(args.output / f"{ds}_{cond}/verified.json",
                           dict(condition=cond, dataset=ds,
                                mIoU={g: .61 for g in self.baselines[ds]["groups"]}))
        with patch.object(suite, "replacement_control",
                          return_value=dict(condition="E1_b3_6_9_12",
                                            mIoU={g: .5 for g in self.baselines["dsec"]["groups"]})), \
                patch.object(suite, "skip_history", return_value={"state_policy": "reset_each_window"}):
            suite.write_summary(args, self.baselines, workers_finished=False)
            self.assertFalse((args.output / "_SUCCESS").exists())
            suite.write_summary(args, self.baselines)
        rows = suite.read(args.output / "summary.json")["semantic"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["dsec_E1_b12_skip"]["replacement_control"]["condition"], "E1")
        self.assertEqual(rows["dsec_E1_b3_6_9_12_skip"]["replacement_control"]["condition"], "E1_b3_6_9_12")
        self.assertTrue((args.output / "_SUCCESS").exists())
        with patch.object(suite, "run_worker") as worker:
            suite.main(["--final-skip-suite", "--conditions", "E1_b12_skip", "--datasets", "dsec",
                        "--worker-result", str(self.root / "worker"), "--execute"])
        parsed, ds, cond = worker.call_args.args
        self.assertEqual(parsed.output.name, "stage2_final_skip_semantic_20261128")
        self.assertEqual(parsed.stage1_root.name, "stage1_final_skip_full_20261128")
        self.assertIn("--final-skip-suite",
                      suite.worker_command(parsed, dict(condition=cond, dataset=ds), "s", "r", 25))

    def test_new_skip_summary_requires_own_history_and_exact_replacement(self):
        output, archive = self.root / "new", self.root / "old"
        args = SimpleNamespace(conditions=suite.SKIP_PLACEMENT_SUITE, output=output,
                               placement_semantic_root=archive, skip_placement_suite=True)
        def scores(ds, cond):
            return {g: 0.50 if cond in suite.SKIP_CONTROLS else 0.38 for g in self.baselines[ds]["groups"]}
        def create(ds, cond):
            suite.save(output / f"{ds}_{cond}/verified.json",
                       dict(condition=cond, dataset=ds, mIoU=scores(ds, cond)))
        def control(root, ds, baseline, cond):
            self.assertEqual(root, archive if cond in suite.PLACEMENTS else output)
            return dict(condition=cond, mIoU=scores(ds, cond))
        delayed = ("m3ed", "E1_b3_6_9_skip")
        def history(a, ds, result):
            return None if (ds, result["condition"]) == delayed else {"state_policy": "reset_each_window"}
        for ds in suite.DATASETS:
            for cond in suite.SKIP_PLACEMENT_SUITE:
                if cond in suite.SKIP_CONTROLS:
                    create(ds, cond)
        with patch.object(suite, "replacement_control", side_effect=control), \
                patch.object(suite, "skip_history", side_effect=history):
            suite.write_summary(args, self.baselines)
            rows = suite.read(output / "summary.json")["semantic"]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows["dsec_E1_b6_skip"]["replacement_control"]["condition"], "E1_b6")
            self.assertAlmostEqual(rows["dsec_E1_b9_skip"]["delta_from_replacement_pp"]["all"], 12.)
            for ds in suite.DATASETS:
                create(ds, "E1_b3_6_9")
            suite.write_summary(args, self.baselines)
            rows = suite.read(output / "summary.json")["semantic"]
            self.assertEqual(len(rows), 7)
            self.assertNotIn("history", rows["dsec_E1_b3_6_9"])
            self.assertEqual(rows["dsec_E1_b3_6_9_skip"]["replacement_control"]["condition"], "E1_b3_6_9")
            self.assertFalse((output / "_SUCCESS").exists())
            delayed = None
            suite.write_summary(args, self.baselines, workers_finished=False)
            self.assertEqual(len(suite.read(output / "summary.json")["semantic"]), 8)
            self.assertFalse((output / "_SUCCESS").exists())
            suite.write_summary(args, self.baselines)
            self.assertTrue((output / "_SUCCESS").is_file())
            # Published comparisons do not mutate the head receipt that signed the history evaluation.
            self.assertNotIn("replacement_control", suite.read(output / "dsec_E1_b6_skip/verified.json"))

    def test_only_this_workers_new_verified_cache_is_temporary(self):
        ds, condition, attempt = "dsec", "E1_b6_skip", "run-abcdef123456"
        key = f"{ds}_{condition}"
        receipt = self.root / key / "verified.json"
        cache = self.root.resolve() / "scratch" / key / attempt
        cache.mkdir(parents=True)
        (cache / "_SUCCESS").touch()
        args = SimpleNamespace(output=self.root, scratch=self.root.resolve() / "scratch", execute=True,
                               skip_placement_suite=True, worker_status=self.root / "status",
                               worker_result=self.root / "result")
        def fit(*_args, **_kwargs):
            row = dict(attempt=attempt, cache=str(cache))
            suite.save(receipt, row)
            return row
        with (self.root / ".lock").open("w") as lock, patch.dict(os.environ, CUDA_VISIBLE_DEVICES="1"), \
                patch.object(suite, "idle_gpus", return_value=("1",)), patch.object(suite, "run_condition", side_effect=fit), \
                patch.object(suite, "skip_history", side_effect=[{"ok": True}, {"ok": True}, RuntimeError("reset interrupted")]):
            args.parent_lock_fd = lock.fileno()
            suite.run_worker(args, ds, condition)
            self.assertFalse(cache.exists())
            self.assertFalse(suite.read(args.worker_result)["cache_retained"])
            cache.mkdir(parents=True)
            (cache / "_SUCCESS").touch()
            suite.run_worker(args, ds, condition)
            self.assertTrue(cache.is_dir())
            receipt.unlink()
            with self.assertRaisesRegex(RuntimeError, "reset interrupted"):
                suite.run_worker(args, ds, condition)
            self.assertTrue(cache.is_dir())

    def test_multiple_workers_share_one_summary_and_use_each_available_gpu_exclusively(self):
        output, scratch = self.root/"output", self.root/"scratch"
        flags = ["--execute", "--gpus", "0,1,2", "--output", str(output), "--scratch", str(scratch)]
        # Stand-ins execute as actual child processes, with the real scheduler's GPU masks,
        # process groups and inherited run lock. GPU 2 starts slow and the other GPUs refill.
        program = """
import json, os, sys, time
from pathlib import Path
root, scratch, key, result, fd = sys.argv[1:]
root, scratch = Path(root), Path(scratch)
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
assert os.fstat(int(fd)).st_ino == (root / ".lock").stat().st_ino
assert os.getsid(0) == os.getpid()
slot = scratch / ("gpu-" + gpu)
with slot.open("x"): pass
if key == "dsec_E1_b3":
    deadline = time.monotonic() + 8
    while not (root / "m3ed_E1_b6" / "verified.json").exists():
        if time.monotonic() > deadline: raise TimeoutError("No GPU was refilled")
        time.sleep(0.01)
receipt = root / key / "verified.json"
receipt.parent.mkdir()
temporary = receipt.with_suffix(".tmp")
temporary.write_text(json.dumps(dict(gpu=gpu)))
temporary.replace(receipt)
Path(result).write_text(receipt.read_text())
slot.unlink()
"""
        original_command = suite.worker_command
        def command(a, job, status, result, fd):
            args = original_command(a, job, status, result, fd)
            self.assertEqual(args[args.index("--datasets")+1], job["dataset"])
            self.assertEqual(args[args.index("--conditions")+1], job["condition"])
            self.assertEqual(Path(args[args.index("--output")+1]), output.resolve())
            return [sys.executable, "-c", program, str(output), str(scratch), job["name"], str(result), str(fd)]
        actual_scheduler = suite.run_parallel
        def dispatch(jobs, *args, **kwargs):
            self.assertTrue(all(j["kind"] == "semantic" and j["remaining_cache_bytes"] == 1024 for j in jobs))
            return actual_scheduler(jobs, *args, poll_interval=0.01, **kwargs)
        with patch.object(suite, "idle_gpus", return_value=("2","0")) as idle, \
                patch.object(suite, "cache_bytes", return_value=1024) as estimate, \
                patch.object(suite, "worker_command", side_effect=command), \
                patch.object(suite, "run_parallel", side_effect=dispatch), redirect_stdout(io.StringIO()), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "parent"}):
            suite.main(flags)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "parent")
            idle.assert_called_once_with(("0", "1", "2"))
            self.assertEqual(estimate.call_count, 2)
            with patch.object(suite, "run_condition") as verify:
                suite.main(flags)
                self.assertEqual(verify.call_count, 12)
            self.assertEqual(estimate.call_count, 2)
            self.assertEqual(idle.call_count, 1)
        results = suite.read(output/"summary.json")["semantic"]
        self.assertEqual(len(results), 12)
        self.assertTrue((output/"_SUCCESS").exists())
        self.assertEqual({r["gpu"] for r in results.values()}, {"2", "0"})
        self.assertEqual(results["dsec_E1_b3"]["gpu"], "2")
        self.assertEqual(results["m3ed_E1_b6"]["gpu"], "0")

    def test_worker_rechecks_gpu_and_validates_inherited_lock_before_training(self):
        a = SimpleNamespace(output=self.root, execute=True, worker_status=self.root/"status.json",
                            worker_result=self.root/"result.json")
        with (self.root/".lock").open("w") as lock, (self.root/"other").open("w") as wrong, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                patch.object(suite, "idle_gpus", return_value=()) as idle, \
                patch.object(suite, "run_condition", return_value=dict(gpu="2")) as fit:
            a.parent_lock_fd = wrong.fileno()
            with self.assertRaisesRegex(ValueError, "different run lock"):
                suite.run_worker(a, "dsec", "E1_b3")
            a.parent_lock_fd = lock.fileno()
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                suite.run_worker(a, "dsec", "E1_b3")
            fit.assert_not_called()
            idle.return_value = ("2",)
            suite.run_worker(a, "dsec", "E1_b3")
            fit.assert_called_once()
        self.assertEqual(suite.read(a.worker_result), dict(gpu="2"))

    def metadata(self, ds="dsec", cond="E1_b9"):
        return SimpleNamespace(global_step=99999, config=dict(dataset=dict(name=ds), seed=0,
            training=dict(max_steps=99999), model=dict(temporal=dict(type="four_stage_lstm",
                after_blocks=list(suite.PLACEMENTS[cond]), input_dim=384, output_dim=384, hidden_dim=384)),
            loss=dict(h_distill=dict(enabled=True, cosine_weight=1, mse_weight=1),
                      z_objective=dict(type="none", weight=0))))

    def test_wrong_dataset_seed_position_or_encoder_budget_rejected(self):
        config = self.metadata().config
        with patch.object(suite, "audit_common") as audit:
            suite.validate_config(config, "dsec", "E1_b9")
            audit.assert_called_once()
            for dataset, condition, seed in (("m3ed", "E1_b9", 0), ("dsec", "E1_b3", 0),
                                              ("dsec", "E1_b9_12", 0), ("dsec", "E1_b9", 1)):
                with self.subTest(dataset=dataset, condition=condition, seed=seed), self.assertRaises(ValueError):
                    suite.validate_config(dict(config, seed=seed), dataset, condition)
            with self.assertRaises(ValueError):
                suite.validate_config(dict(config, training=dict(max_steps=99)), "dsec", "E1_b9")

    def test_identical_placement_needs_matching_residual_recipe(self):
        config = self.metadata(cond="E1_b3").config
        with patch.object(suite, "audit_common"):
            with self.assertRaisesRegex(ValueError, "recipe/placement"):
                suite.validate_config(config, "dsec", suite.SKIP_CONDITION)
            config["model"]["temporal"]["residual_skip"] = True
            suite.validate_config(config, "dsec", suite.SKIP_CONDITION)
            with self.assertRaisesRegex(ValueError, "recipe/placement"):
                suite.validate_config(config, "dsec", "E1_b3")

    def test_skip_history_reuses_verified_head_and_requires_both_histories_for_success(self):
        args = SimpleNamespace(output=self.root, activity_root=self.root/"activity", data=self.root/"data",
                               conditions=[suite.SKIP_CONDITION])
        args.activity_root.mkdir()
        for ds in suite.DATASETS:
            baseline = self.baselines[ds]
            scores = {k: g["metrics"]["mIoU"] for k, g in baseline["groups"].items()}
            directory = self.root / f"{ds}_{suite.SKIP_CONDITION}"
            result = dict(head_sha256="head", backbone_sha256="backbone", report="report", mIoU=scores)
            suite.save(directory/"verified.json", result)
            activity = args.activity_root / (ds + ".json")
            suite.save(activity, dict(dataset=ds))
            reset = dict.fromkeys(scores, 0.25)
            evaluation = dict(continuous_mIoU=scores, reset_each_window_mIoU=reset,
                continuous_minus_reset_pp={g: 100 * (v - 0.25) for g, v in scores.items()},
                continuous_report="report", state_policy="reset_each_window", learned_skip_scale=0.01,
                activity_manifest_sha256=baseline["manifest_sha256"], head_sha256="head", backbone_sha256="backbone",
                comparison_hashes={key: baseline[key] for key in suite.COMPARISON_HASHES})
            suite.write_summary(args, self.baselines)
            self.assertFalse((self.root/"_SUCCESS").is_file())
            with patch.dict("sys.modules", {"tools.evaluate_b3_skip_history": SimpleNamespace(
                    evaluate_history=lambda dataset, receipt, manifest, *, data_root: evaluation)}):
                suite.skip_history(args, ds, result, execute=True)
            self.assertEqual(suite.skip_history(args, ds, result), evaluation)
        suite.write_summary(args, self.baselines)
        summary = suite.read(self.root/"summary.json")
        self.assertEqual(len(summary["semantic"]), 2)
        self.assertTrue((self.root/"_SUCCESS").is_file())
        stored = suite.read(directory/"history.json")
        stored["evaluation"]["head_sha256"] = "other"
        suite.save(directory/"history.json", stored)
        with self.assertRaisesRegex(ValueError, "verified continuous head"):
            suite.skip_history(args, ds, result)

    def test_skip_dispatch_resumes_history_without_reserving_another_full_cache(self):
        directory = self.root / f"dsec_{suite.SKIP_CONDITION}"
        suite.save(directory / "verified.json", {"attempt": "run-a"})
        args = SimpleNamespace(output=self.root, scratch=self.root/"scratch", conditions=[suite.SKIP_CONDITION])
        def check(jobs, gpus, **kwargs):
            self.assertEqual(gpus, ("0", "1", "2"))
            self.assertIs(kwargs["idle_devices"], suite.idle_gpus)
            self.assertEqual([(j["dataset"], j["history_only"], j["remaining_cache_bytes"]) for j in jobs],
                             [("dsec", True, 0), ("m3ed", False, 4096)])
        with patch.object(suite, "cache_bytes", return_value=4096) as estimate, \
                patch.object(suite, "run_parallel", side_effect=check), patch.object(suite, "write_summary"), \
                redirect_stdout(io.StringIO()):
            suite.execute_jobs(args, [(ds, suite.SKIP_CONDITION) for ds in suite.DATASETS],
                               ("0", "1", "2"), self.baselines, None)
        estimate.assert_called_once_with(args, "m3ed", suite.SKIP_CONDITION)

    def test_skip_per_activity_comparison_uses_verified_archived_replacement_head(self):
        cond, ds = "E1_b3", "dsec"
        directory = self.root / f"{ds}_{cond}"
        path = directory / "run-mock"
        self.make_report(ds, cond, path)
        checkpoint = self.root / "replacement.pt"
        checkpoint.write_bytes(b"replacement")
        row = suite.verified_metrics(ds, cond, path, self.baselines[ds])
        row.update(attempt=path.name, global_step=99999, seed=0, backbone=str(checkpoint),
                   backbone_sha256=suite.sha256_file(checkpoint))
        suite.save(directory/"verified.json", row)
        training = SimpleNamespace(load_checkpoint_config_metadata=lambda _: self.metadata(ds, cond))
        with patch.dict("sys.modules", {"event_state.training": training}), patch.object(suite, "audit_common"):
            control = suite.replacement_control(self.root, ds, self.baselines[ds])
            scores = dict(row["mIoU"], all=row["mIoU"]["all"] + 0.03)
            delta = suite.skip_comparison(scores, control)["delta_from_b3_replacement_pp"]
            self.assertAlmostEqual(delta["all"], 3.0)
            self.assertEqual(delta["p00_20"], 0)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Stage 1 replacement control"):
                suite.replacement_control(self.root, ds, self.baselines[ds])

    def make_report(self, ds, cond, output):
        ref = self.baselines[ds]
        output.mkdir(parents=True)
        head = output / ("h/best.pt" if ds == "m3ed" else "final/last.pt")
        head.parent.mkdir()
        head.write_bytes(b"fixture")
        if ds == "dsec":
            suite.save(output/"final/config.json", dict(selected_development_epoch=1, epochs=2))
            development = output/"development"
            development.mkdir()
            records = [{"epoch": i, "learning_rate": 1e-3, "train/loss": 1.1, "val/mIoU": miou}
                       for i, miou in enumerate((0.32, 0.45, 0.40))]
            (development/"metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (output/"_SUCCESS").touch()
        activity = dict(groups=deepcopy(ref["groups"]), activity_manifest=dict(manifest_sha256=ref["manifest_sha256"]),
                        condition_label=suite.placement_label(cond), head_sha256=suite.sha256_file(head),
                        state_policy_by_sequence=ref["state_policy_by_sequence"],
                        **{k: ref[k] for k in suite.COMPARISON_HASHES})
        path = output / ("h/activity_groups.json" if ds == "m3ed" else "activity_test.json")
        data = dict(role="validation" if ds == "m3ed" else "test", feature="h", activity_evaluation=activity)
        suite.save(path, data)
        return path, data

    def test_head_gt_support_and_all_five_quintiles_match_reference(self):
        for ds in suite.DATASETS:
            output, cond = self.root/ds, "E1_b9_12"
            path, data = self.make_report(ds, cond, output)
            values = suite.verified_metrics(ds, cond, output, self.baselines[ds])
            self.assertEqual(set(values["mIoU"]), set(self.baselines[ds]["groups"]))
            self.assertEqual(set(values["delta_from_final_block_only_pp"].values()), {0})
            if ds == "dsec":
                selection = values["development_selection"]
                self.assertEqual((selection["epoch_zero_based"], selection["val_mIoU"]), (1, 0.45))
                suite.save(output/"final/config.json", dict(selected_development_epoch=2, epochs=3))
                with self.assertRaisesRegex(ValueError, "development selection"):
                    suite.verified_metrics(ds, cond, output, self.baselines[ds])
            else:
                self.assertNotIn("development_selection", values)
            data["activity_evaluation"]["samples_targets_sha256"] = "changed"
            suite.save(path, data)
            with self.assertRaisesRegex(ValueError, "matched Semantic protocol"):
                suite.verified_metrics(ds, cond, output, self.baselines[ds])

    def test_m3ed_fits_once_and_keeps_cached_features_on_resume(self):
        ds, condition = "m3ed", "E1_b6_9_12"
        a = SimpleNamespace(stage1_root=self.root/"stage1", output=self.root/"outputs",
            scratch=self.root/"scratch", data=self.root/"data", statistics=self.root/"stats.json",
            m3ed_target_root=self.root/"targets", activity_root=self.root/"activity",
            execute=True, worker_status=self.root/"m3ed.status.json")
        key = f"{ds}_{condition}"
        checkpoint = a.stage1_root/f"{key}_w1/checkpoints/step_00099999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"backbone")
        training = SimpleNamespace(load_checkpoint_config_metadata=lambda p: self.metadata(ds, condition))
        def run(command, **kwargs):
            self.assertEqual(command[command.index("--features")+1], "h")
            self.assertEqual(command[command.index("--checkpoint")+1], str(checkpoint))
            self.assertEqual(command[command.index("--condition-label")+1], suite.placement_label(condition))
            self.assertEqual(command[command.index("--parent-lock-fd")+1], "18")
            self.assertEqual(command[command.index("--worker-status")+1], str(a.worker_status))
            self.assertEqual(kwargs["pass_fds"], (18,))
            cache = Path(command[command.index("--cache")+1])
            (cache/"seq").mkdir(parents=True)
            (cache/"seq/metadata.json").write_text("{}")
            (cache/"seq/000099.pt").write_bytes(b"large sparse tensor")
            (cache/"_SUCCESS").touch()
            self.make_report(ds, condition, Path(command[command.index("--output")+1]))
        with patch.dict("sys.modules", {"event_state.training": training}), patch.object(suite, "audit_common"), \
                patch.object(suite.subprocess, "run", side_effect=run) as child:
            for _ in range(2):
                suite.run_condition(a, ds, condition, self.baselines[ds], 18)
            self.assertEqual(child.call_count, 1)
            result = suite.read(a.output/key/"verified.json")
            cache = Path(result["cache"])
            self.assertTrue((cache/"seq/metadata.json").is_file())
            self.assertTrue((cache/"seq/000099.pt").is_file())
            self.assertTrue((cache/"_SUCCESS").is_file())
            training.load_checkpoint_config_metadata = lambda p: SimpleNamespace(global_step=99)
            with self.assertRaisesRegex(ValueError, "99,999-update"):
                suite.run_condition(a, ds, condition, self.baselines[ds], 18)
        suite.write_summary(a, self.baselines)
        self.assertEqual(list(suite.read(a.output/"summary.json")["semantic"]), [key])
        self.assertFalse((a.output/"_SUCCESS").exists())

    def test_four_layer_report_must_match_the_trained_head_and_requested_backbone(self):
        for ds in suite.DATASETS:
            directory = self.root / ds
            old_path, report = self.make_report(ds, "E1_b3_6_9_12", directory)
            if ds == "m3ed":
                (directory/"h").rename(directory/"stages")
                path = directory/"stages/activity_groups.json"
            else:
                path = old_path
            config_path = path.parent/"config.json" if ds == "m3ed" else directory/"final/config.json"
            config = suite.read(config_path) if config_path.exists() else {}
            provenance = dict(backbone_sha256="frozen", stage_blocks=[3, 6, 9, 12])
            suite.save(config_path, dict(config, **provenance))
            report.update(feature="stages", **provenance)
            report["activity_evaluation"]["condition_label"] = "four-block E1"
            suite.save(path, report)

            def measure():
                return suite.verified_metrics(ds, "E1", directory, self.baselines[ds], feature="stages",
                                              label="four-block E1", backbone_sha256="frozen")

            self.assertEqual(measure()["after_blocks"], [12])
            for field, wrong_value in (("backbone_sha256", "stale"), ("stage_blocks", [12, 9, 6, 3])):
                suite.save(path, dict(report, **{field: wrong_value}))
                with self.subTest(ds=ds, field=field), self.assertRaisesRegex(ValueError, "block order"):
                    measure()


if __name__ == "__main__":
    unittest.main()
