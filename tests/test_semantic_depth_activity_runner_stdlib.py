"""Portable checks of the existing-head batch runner; no Torch or GPU imports."""

from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_state.activity_quantiles import (
    ActivityIndex, M3ED_SOURCE_CONTRACT, build_activity_manifest, canonical_sha256,
)
from tools import run_semantic_depth_activity as runner


class BatchActivityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(runner.ROOT / "tools/manifests", self.root / "tools/manifests")
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.args = SimpleNamespace(data=self.root / "data", depth_run=self.root / "depth",
            output=self.root / "reports", scratch=None, baselines_only=False, preflight=False)
        self.args.depth_run.mkdir()
        (self.args.depth_run / "run.lock").touch()
        self.commands = []
        self.create_inputs()

    def file(self, path, text="fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def create_inputs(self):
        for dataset in ("dsec", "m3ed"):
            _, sequences, evaluation = runner.source_sequences(dataset)
            event_root = self.args.data / ("DSEC_cache/events/gep_rgb" if dataset == "dsec" else "m3ed_cache/half_dagr")
            for sequence in sequences:
                base = event_root / sequence
                self.file(base / "metadata.json", "{}")
                self.file(base / "_SUCCESS", "")
                self.file(base / ("1.pt" if dataset == "dsec" else "events/1.pt"))
            target = self.file(self.root / "targets" / dataset / "target.h5")
            self.file(self.args.depth_run / "targets" / f"{dataset}_val.json", json.dumps(dict(
                dataset=dataset, task="depth", role="val", labeled=True,
                records=[dict(sequence=evaluation[0], timestamp=1000, target_path=str(target))])))
            self.file(self.root / f"outputs/stage1_common_20261105/statistics/{dataset}_common.json", "{}")
            for condition, _, feature in runner.CONDITIONS:
                run = runner.semantic_run(dataset, condition)
                head_dir = run / (feature if dataset == "m3ed" else "final")
                cache = self.root / "features" / dataset / condition
                config = dict(feature=feature, feature_cache_dir=str(cache),
                    split_manifest=str(self.root / "tools/manifests" / ("m3ed_semantic_common_v1.yaml" if dataset == "m3ed" else "dsec_semantic_split.yaml")),
                    batch_size=3, precision="fp32", num_workers=0)
                semantic_sequences = runner.M3ED_VAL if dataset == "m3ed" else runner.DSEC_TEST
                for sequence in semantic_sequences:
                    self.file(cache / sequence / "metadata.json", json.dumps(dict(
                        task=dataset + "_semantic", features=[feature], state_policy="continuous")))
                    self.file(cache / sequence / "000001.pt")
                if dataset == "dsec":
                    labels = self.root / "labels"
                    config["labels_root"] = str(labels)
                    for sequence in semantic_sequences:
                        self.file(labels / "test" / sequence / "11classes/000001.png")
                else:
                    self.file(cache / "identity.json", "{}")
                    self.file(cache / "_SUCCESS", "")
                    targets = run / "targets"
                    config["target_cache_dir"] = str(targets)
                    for sequence in semantic_sequences:
                        self.file(targets / sequence / "targets.h5")
                self.file(head_dir / "config.json", json.dumps(config))
                self.file(head_dir / ("best.pt" if dataset == "m3ed" else "last.pt"))
                depth_head = self.args.depth_run / "heads" / f"{dataset}_{condition}" / "stages"
                self.file(depth_head / "best.pt")
                self.file(depth_head / "_SUCCESS", "")
                folder = "stage1_joined_baseline_full_20261107"
                model, step = condition + "_w1", 99999
                if condition.startswith("E1_"):
                    folder = "stage1_rvt_ddp_full_20261105" if condition == "E1_old_ddp" else "stage1_joined_ddp_full_new_20261106"
                    model, step = "E1_w3", 33333
                self.file(self.root / "outputs" / folder / f"{dataset}_{model}" / "checkpoints" / f"step_{step:08d}.pt")

    def generate_manifest(self, dataset, output):
        reference, sequences, _ = runner.source_sequences(dataset)
        event_root = self.args.data / ("DSEC_cache/events/gep_rgb" if dataset == "dsec" else "m3ed_cache/half_dagr")
        records = [dict(sequence=sequence, timestamp=1000, previous_timestamp=0,
                        event_count=i + 1, window_us=1000, event_rate_hz=(i + 1) * 1000.)
                   for i, sequence in enumerate(sequences)]
        value = build_activity_manifest(records, runner.read(reference)["splits"]["train"],
            reference=dict(role="train", name=dataset, manifest_sha256=runner.sha256(reference)),
            source_metadata_sha256={s: canonical_sha256(runner.read(event_root / s / "metadata.json")) for s in sequences},
            source_contract=M3ED_SOURCE_CONTRACT if dataset == "m3ed" else None)
        self.file(output, json.dumps(value))

    def fake_launch(self, cmd):
        self.commands.append(cmd)
        tool = Path(cmd[2]).name
        def option(name):
            return cmd[cmd.index(name) + 1]
        if tool.startswith("prepare_"):
            dataset = "dsec" if "dsec" in tool else "m3ed"
            self.generate_manifest(dataset, Path(option("--output")))
        elif tool in ("evaluate_dsec_semantic.py", "evaluate_m3ed_semantic.py"):
            index = ActivityIndex.load(option("--activity-manifest"))
            output = Path(option("--output"))
            self.file(output, json.dumps(dict(activity_evaluation=dict(
                activity_manifest=index.identity, condition_label=option("--condition-label")))))
            self.file(output.with_suffix(".csv"), "condition,group\n")
        elif tool == "run_common_depth.py":
            self.assertIn("--activity-only", cmd)
            conditions = runner.CONDITIONS[:3] if "--baselines-only" in cmd else runner.CONDITIONS
            depth_summary = {}
            for dataset in ("dsec", "m3ed"):
                for condition, _, _ in conditions:
                    name = f"{dataset}_{condition}"
                    output = self.args.depth_run / "heads" / name / "stages/activity_continuous_frame_mean.json"
                    self.file(output, "{}")
                    self.file(output.with_suffix(".csv"), "condition,group\n")
                    depth_summary[name] = dict(activity_report=str(output), overall_frame_mean_f3=dict(mae=1))
            self.file(self.args.depth_run / "activity_summary.json", json.dumps(depth_summary))
        else:
            self.fail(f"Unexpected operation; the batch must never train: {tool}")

    def execute(self):
        with patch.object(runner, "launch", self.fake_launch), redirect_stdout(io.StringIO()):
            runner.execute(self.args)

    def test_preflight_checks_all_inputs_without_writes_or_inference(self):
        self.args.preflight = True
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.execute()
        self.assertFalse(self.commands)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_all_conditions_use_existing_configs_and_resume_verified_reports(self):
        self.execute()
        self.assertEqual(len(self.commands), 13)  # 2 manifests + 10 Semantic + 1 Depth batch
        semantic = [c for c in self.commands if Path(c[2]).name.startswith("evaluate_")]
        self.assertEqual(len(semantic), 10)
        for cmd in semantic:
            self.assertEqual(cmd[cmd.index("--batch-size") + 1], "3")
            self.assertEqual(cmd[cmd.index("--precision") + 1], "fp32")
            self.assertEqual(cmd[cmd.index("--num-workers") + 1], "0")
            if "evaluate_dsec_semantic.py" in cmd[2]:
                self.assertTrue(cmd[cmd.index("--checkpoint") + 1].endswith("final/last.pt"))
        self.commands.clear()
        self.execute()
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(Path(self.commands[0][2]).name, "run_common_depth.py")
        summary = runner.read(self.args.output / "summary.json")
        self.assertEqual(len(summary["semantic"]), 10)
        self.assertEqual(len(summary["depth"]), 10)
        self.assertTrue(all(Path(report).name == "activity_continuous_frame_mean.json"
                            for report in summary["depth"].values()))

    def test_baselines_only_excludes_ddp_and_forwards_scratch(self):
        self.args.baselines_only = True
        self.args.scratch = self.root / "scratch"
        self.execute()
        self.assertEqual(len(self.commands), 9)
        depth_cmd = self.commands[-1]
        self.assertIn("--baselines-only", depth_cmd)
        self.assertEqual(depth_cmd[depth_cmd.index("--scratch") + 1], str(self.args.scratch))

    def test_late_missing_head_stops_before_any_expensive_operation(self):
        (self.args.depth_run / "heads/m3ed_E1_new_ddp/stages/_SUCCESS").unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute()
        self.assertFalse(self.commands)
        self.assertFalse(self.args.output.exists())

    def test_live_depth_lock_stops_without_writes(self):
        with (self.args.depth_run / "run.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "still active"):
                self.execute()
        self.assertFalse(self.commands)
        self.assertFalse(self.args.output.exists())

    def test_changed_head_config_or_report_never_overwrites_saved_scores(self):
        self.execute()
        self.commands.clear()
        output = self.args.output / "semantic/dsec_E0.json"
        saved = output.read_bytes()
        config = runner.semantic_run("dsec", "E0") / "final/config.json"
        value = runner.read(config)
        value["batch_size"] = 7
        config.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Different/stale"):
            self.execute()
        self.assertEqual(saved, output.read_bytes())
        self.assertFalse(self.commands)

    def test_missing_csv_refuses_to_treat_existing_json_as_complete(self):
        self.execute()
        self.commands.clear()
        (self.args.output / "semantic/m3ed_E2.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete existing"):
            self.execute()
        self.assertFalse(self.commands)


if __name__ == "__main__":
    unittest.main()
