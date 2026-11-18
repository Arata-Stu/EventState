"""Matched DSEC head commands and input/failure guards without Torch or datasets."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_state.activity_quantiles import build_activity_manifest, canonical_sha256
from tools import temporal_downstream_semantic as semantic


ROOT = Path(__file__).resolve().parents[1]


class SemanticJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="semantic plan ")
        self.addCleanup(self.temp.cleanup)
        project = Path(self.temp.name)
        self.args = SimpleNamespace(project_root=project, data=project / "data",
            stage1_root=project / "stage1", output=project / "output",
            scratch=project / "scratch", activity_root=project / "activity",
            baseline_semantic_config=project / "old/final/config.json", seed=0)
        split = project / "tools/manifests/dsec_semantic_split.yaml"
        split.parent.mkdir(parents=True)
        split.write_bytes((ROOT / "tools/manifests/dsec_semantic_split.yaml").read_bytes())
        for name in ("run_dsec_semantic_frozen.sh", "cache_dsec_semantic_features.py",
                     "train_dsec_semantic.py", "evaluate_dsec_semantic.py"):
            (project / "tools" / name).write_bytes((ROOT / "tools" / name).read_bytes())
        self.split = semantic._split(split)
        labels = project / "labels"
        sources = {"reference": "a" * 64}
        records = [dict(sequence="reference", previous_timestamp=i * 100000,
                        timestamp=(i + 1) * 100000, window_us=100000,
                        event_count=i, event_rate_hz=i * 10.0) for i in range(10)]
        self.metadata = {}
        for role, names in self.split.items():
            for name in names:
                directory = labels / ("test" if role == "test" else "train") / name / "11classes"
                directory.mkdir(parents=True)
                (directory / "000001.png").write_bytes(b"PNG input checked only by portable preflight")
                prepared = self.args.data / "DSEC_cache/events/gep_rgb" / name
                prepared.mkdir(parents=True)
                meta = dict(dataset="DSEC", sequence_name=name,
                            coordinate_space="rectified_event", height=480, width=640,
                            event_window="rgb_interval", representation=dict(type="gep_rgb"))
                (prepared / "metadata.json").write_text(json.dumps(meta))
                (prepared / "000001.pt").write_bytes(b"prepared fixture")
                self.metadata[name] = meta
                if role == "test":
                    sources[name] = canonical_sha256(meta)
                    records.append(dict(sequence=name, previous_timestamp=0, timestamp=100000,
                                        window_us=100000, event_count=1, event_rate_hz=10.0))
        activity = build_activity_manifest(records, ["reference"], reference=dict(
            role="train", name="fixture", manifest_sha256="b" * 64,
            excluded_sequences=self.split["test"]), source_metadata_sha256=sources)
        self.args.activity_root.mkdir()
        (self.args.activity_root / "dsec.json").write_text(json.dumps(activity))
        checkpoint = self.args.stage1_root / "dsec_E1_convlstm_w1/checkpoints/step_00099999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"final checkpoint fixture")
        config = dict(feature="h", head_type="linear", loss="ce", precision="fp16",
                      protocol="official-final", batch_size=8, seed=0,
                      learning_rate=2e-4, weight_decay=.05, horizontal_flip_probability=.5,
                      activity_active_weight=1.0, activity_inactive_weight=1.0,
                      num_classes=11, num_workers=4, head_width=192,
                      training_sequences=self.split["train"] + self.split["val"],
                      labels_root=str(labels), split_manifest=str(split))
        self.args.baseline_semantic_config.parent.mkdir(parents=True)
        self.args.baseline_semantic_config.write_text(json.dumps(config))

    def test_plan_is_read_only_and_reserves_all_three_semantic_roles(self):
        before = set(self.args.project_root.rglob("*"))
        job = semantic.build_semantic_job(self.args)
        self.assertEqual(before, set(self.args.project_root.rglob("*")))
        self.assertEqual(job["name"], "dsec_semantic")
        self.assertEqual(job["kind"], "semantic")
        self.assertEqual(len(job["identity"]["sources"]), 11)
        self.assertEqual(job["cache_bytes"], 11 * (384 * 28 * 40 * 2 + 16384))
        self.assertEqual(job["remaining_cache_bytes"], job["cache_bytes"])
        self.assertTrue(all(Path(p).is_absolute() for p in job["inputs"]))
        self.assertFalse(Path(job["output"]).exists())

    def test_explicit_placement_uses_requested_checkpoint_and_attempt(self):
        checkpoint = self.args.stage1_root / "dsec_E1_b9_w1/checkpoints/step_00099999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"single block 9 fixture")
        label = "E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（9層後）"
        job = semantic.build_semantic_job(self.args, checkpoint=checkpoint, name="attempt-01", label=label)
        self.assertEqual(job["backbone"], str(checkpoint.resolve()))
        self.assertEqual(job["label"], label)
        self.assertEqual(job["output"], str(self.args.output.resolve() / "attempt-01"))
        self.assertEqual(job["cache"], str(self.args.scratch.resolve() / "attempt-01"))

    def test_wrong_recipe_or_split_is_rejected(self):
        original = self.args.baseline_semantic_config.read_text()
        for key, value in (("feature", "z"), ("head_type", "nonlinear"), ("loss", "ce-dice"),
                           ("seed", 1), ("learning_rate", .01), ("batch_size", 4)):
            config = json.loads(original)
            config[key] = value
            self.args.baseline_semantic_config.write_text(json.dumps(config))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recipe differs"):
                semantic.build_semantic_job(self.args)
        config = json.loads(original)
        other = self.args.project_root / "other_split.yaml"
        other.write_text("train: []")
        config["split_manifest"] = str(other)
        self.args.baseline_semantic_config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "different split"):
            semantic.build_semantic_job(self.args)

    def test_missing_labels_or_modified_activity_source_fail_before_writing(self):
        test = self.split["test"][0]
        meta = self.args.data / "DSEC_cache/events/gep_rgb" / test / "metadata.json"
        value = json.loads(meta.read_text())
        value["new_field"] = "changed source"
        meta.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            semantic.build_semantic_job(self.args)
        meta.write_text(json.dumps(self.metadata[test]))
        labels = self.args.project_root / "labels/test" / test / "11classes/000001.png"
        labels.unlink()
        with self.assertRaises(FileNotFoundError):
            semantic.build_semantic_job(self.args)
        self.assertFalse(self.args.output.exists())

    def test_existing_outputs_or_cache_are_preserved(self):
        for target in (self.args.output / "dsec_semantic", self.args.scratch / "dsec_semantic"):
            target.mkdir(parents=True)
            with self.subTest(target=target), self.assertRaises(FileExistsError):
                semantic.build_semantic_job(self.args)
            target.rmdir()

    def fake_wrapper(self, job):
        output, cache = Path(job["output"]), Path(job["cache"])
        (output / "final").mkdir()
        (output / "final/last.pt").write_bytes(b"new final fitted head")
        (output / "test_metrics.json").write_text(json.dumps(dict(role="test", feature="h", mIoU=.6)))
        for sequence in job["identity"]["sources"]:
            directory = cache / sequence
            directory.mkdir(parents=True)
            (directory / "000001.pt").write_bytes(b"features")
            (directory / "metadata.json").write_text(json.dumps(dict(
                state_policy="continuous", features=["h"], cached_frame_indices=[1],
                checkpoint_sha256=job["identity"]["backbone_sha256"])))

    def fake_activity_report(self, job):
        groups = {name: dict(frames=1, valid_pixels=100, metrics=dict(mIoU=.6))
                  for name in job["identity"]["activity_group_names"]}
        groups["all"] = dict(frames=5, valid_pixels=500, metrics=dict(mIoU=.6))
        activity = dict(condition_label=job["label"], groups=groups,
            head_sha256=semantic._sha(Path(job["output"]) / "final/last.pt"),
            activity_manifest=dict(manifest_sha256=job["identity"]["activity_manifest_sha256"]),
            state_policy_by_sequence={s: "continuous" for s in self.split["test"]})
        return dict(role="test", feature="h", mIoU=.6, activity_evaluation=activity)

    def test_worker_keeps_refit_wrapper_and_activity_recipe_and_lock(self):
        job = semantic.build_semantic_job(self.args)
        self.args.parent_lock_fd = 123
        calls, updates = [], []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            if command[0] == "bash":
                self.fake_wrapper(job)
            else:
                Path(job["activity_report"]).write_text(json.dumps(self.fake_activity_report(job)))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"), patch.object(semantic.subprocess, "run", side_effect=run):
            result = semantic.run_semantic_job(self.args, job, updates.append)
        wrapper = calls[0][0]
        self.assertEqual(wrapper[wrapper.index("--gpu") + 1], "2")
        self.assertEqual(wrapper[wrapper.index("--epochs") + 1], "50")
        self.assertEqual(wrapper[wrapper.index("--feature") + 1], "h")
        self.assertTrue(all(kwargs["pass_fds"] == (123,) for _, kwargs in calls))
        evaluator = calls[1][0]
        self.assertIn("final/last.pt", evaluator[evaluator.index("--checkpoint") + 1])
        self.assertEqual(evaluator[evaluator.index("--role") + 1], "test")
        self.assertIn("--activity-manifest", evaluator)
        self.assertTrue((Path(job["output"]) / "_SUCCESS").is_file())
        self.assertTrue((Path(job["cache"]) / "_SUCCESS").is_file())
        self.assertEqual(updates[0]["remaining_cache_bytes"], job["cache_bytes"])
        self.assertEqual(updates[-1]["remaining_cache_bytes"], 0)
        self.assertEqual(result["semantic_mIoU"], .6)

    def test_worker_failure_preserves_outputs_without_success(self):
        job = semantic.build_semantic_job(self.args)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="1"), patch.object(semantic.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, "fixture")):
            with self.assertRaises(subprocess.CalledProcessError):
                semantic.run_semantic_job(self.args, job)
        self.assertTrue((Path(job["output"]) / "job_identity.json").is_file())
        self.assertFalse((Path(job["output"]) / "_SUCCESS").exists())

    def test_activity_report_for_another_head_does_not_mark_job_success(self):
        job = semantic.build_semantic_job(self.args)
        def run(command, **kwargs):
            if command[0] == "bash":
                self.fake_wrapper(job)
            else:
                report = self.fake_activity_report(job)
                report["activity_evaluation"]["head_sha256"] = "f" * 64
                Path(job["activity_report"]).write_text(json.dumps(report))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="1"), patch.object(semantic.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                semantic.run_semantic_job(self.args, job)
        self.assertTrue((Path(job["cache"]) / "_SUCCESS").is_file())
        self.assertFalse((Path(job["output"]) / "_SUCCESS").exists())

    def test_worker_requires_a_single_selected_gpu_and_stdlib_import(self):
        job = semantic.build_semantic_job(self.args)
        for value in ("", "0,1", "gpu0"):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=value), self.assertRaises(ValueError):
                semantic.run_semantic_job(self.args, job)
        probe = subprocess.run([sys.executable, "-S", "-c",
            "import tools.temporal_downstream_semantic; import sys; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == "__main__":
    unittest.main()
