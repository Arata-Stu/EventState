"""Frozen DSEC-Det comparison planning, isolated cache lifetime and phase routing."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_placement_detection as suite
from event_state.activity_quantiles import build_activity_manifest, training_sequences


class PlacementDetectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_val_and_test_use_same_fitted_quintiles_with_different_sequences(self):
        manifest = suite.read(suite.TRAIN35)
        train, excluded = training_sequences(manifest)
        reference = dict(role="train", name=manifest["name"], excluded_sequences=list(excluded),
                         manifest_sha256=suite.sha256_file(suite.TRAIN35))
        identities = []
        for role, heldout in (("val", "zurich_city_16_a"), ("test", "thun_02_a")):
            names = [*train, heldout]
            counts = list(range(1, len(train) + 1)) + [0 if role == "val" else 1000]
            rows = [dict(sequence=name, timestamp=1000, previous_timestamp=0, event_count=n,
                         window_us=1000, event_rate_hz=n * 1000) for name, n in zip(names, counts)]
            metadata = {name: suite.canonical_sha256(name) for name in names}
            path = self.root / f"{role}.json"
            suite.save(path, build_activity_manifest(rows, train, reference=reference, source_metadata_sha256=metadata))
            sources = {name: dict(metadata=sha, timestamps=suite.canonical_sha256(["1000"]))
                       for name, sha in metadata.items()}
            identities.append(suite.check_activity(path, [heldout], sources))
        self.assertNotEqual(identities[0][0], identities[1][0])
        self.assertEqual(identities[0][1], identities[1][1])

    def test_plan_nine_conditions_with_e0_z_and_all_other_h(self):
        with redirect_stdout(io.StringIO()) as output:
            suite.main(["--output", str(self.root/"new")])
        lines = output.getvalue().splitlines()
        self.assertEqual([line.split("]", 1)[0][1:] for line in lines], list(suite.LABELS))
        self.assertIn("| z | GPU pool 1,2 |", lines[0])
        self.assertTrue(all("| h | GPU pool 1,2 |" in row for row in lines[1:]))
        self.assertFalse((self.root/"new").exists())

    def test_reservation_accounts_for_retained_roles_and_existing_partial_files(self):
        j = dict(frames=dict(train=100, val=3), allocated={
            "train": dict(rectified=99 * suite.RECT_BYTES, benchmark=0),
            "val": dict(rectified=0, benchmark=0)})
        self.assertEqual(suite.remaining(j, ("train", "val")),
                         103 * suite.WARP_BYTES + 3 * suite.RECT_BYTES)
        self.assertEqual(suite.remaining(j, ("val",)), 3 * (suite.WARP_BYTES + suite.RECT_BYTES))

    def test_verified_warp_requires_every_timestamp_and_the_correct_backbone(self):
        seq = "fixture"
        path = self.root/seq
        path.mkdir()
        source = dict(sequence_name=seq)
        info = {seq: dict(metadata=suite.canonical_sha256(source), frames=2,
                          timestamps=suite.canonical_sha256(["20", "40"]))}
        suite.save(path/"metadata.json", dict(official_role="val", sequence_name=seq, benchmark_format_version=3,
                    checkpoint_sha256="backbone", checkpoint_step=99999, state_policy="continuous", detection_scale=1,
                    coordinate_space="dsec_det_distorted", input_size=[430, 640], patch_size=16, features=["h"],
                    frame_count=2, source_event_cache_metadata=source))
        for t in ("20","40"):
            (path/f"{t}.pt").touch()
        suite.check_warp(self.root, "val", [seq], info, "backbone", "h")
        with self.assertRaisesRegex(ValueError, "metadata"):
            suite.check_warp(self.root, "val", [seq], info, "other", "h")
        (path/"40.pt").rename(path/"60.pt")
        with self.assertRaisesRegex(ValueError, "Missing benchmark"):
            suite.check_warp(self.root, "val", [seq], info, "backbone", "h")

    def test_val_fits_once_test_only_scores_the_same_best_head_and_releases_only_rectified(self):
        name, cache = "E1", self.root/"cache"
        backbone, head = self.root/"stage1.pt", self.root/name/"head/best.pt"
        backbone.write_bytes(b"frozen weights")
        head.parent.mkdir(parents=True)
        split = dict(train=["train"], val=["val"], test=["test"])
        source = {seq: {} for seq in ("train", "val", "test")}
        receipts = []
        def invoke(tool, *args, **kwargs):
            receipts.append(tool)
            field = lambda key: args[args.index(key) + 1]
            if tool.startswith("cache_"):
                self.assertEqual(field("--features"), "h")
                self.assertEqual(field("--state-policy"), "continuous")
                dest = Path(field("--output-dir")) / field("--role")
                dest.mkdir(parents=True)
                (dest/"src.pt").write_bytes(b"temporary")
            elif tool.startswith("prepare_"):
                self.assertEqual(field("--scale"), "1")
                dest = Path(field("--output-dir")) / field("--role")
                dest.mkdir(parents=True)
                (dest/"warped.pt").write_bytes(b"keep")
            elif tool.startswith("train_"):
                self.assertEqual(field("--epochs"), "50")
                self.assertEqual(field("--seed"), "0")
                head.write_bytes(b"fixed val best")
            elif tool.startswith("evaluate_"):
                self.assertEqual(Path(field("--checkpoint")), head)
                self.assertTrue(head.is_file())
                Path(field("--output")).write_text("{}")
        def checked(cache_dir, role, *_args):
            self.assertTrue((cache_dir/role/"warped.pt").exists())
        with (self.root/".lock").open("a") as lock, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"2"}), \
                patch.object(suite, "idle_gpus", return_value=("2",)), patch.object(suite, "tool", side_effect=invoke), \
                patch.object(suite, "check_warp", side_effect=checked), patch.object(suite, "verified_report", return_value={}), \
                patch("tools.placement_detection_protocol.prepare_head_resume", return_value=None), \
                patch.object(suite, "head_selection", return_value=dict(epoch_zero_based=4, val_mAP=.4)):
            for phase in ("val", "test"):
                r = suite.roles(phase)
                job = dict(name=name, feature="h", label=suite.LABELS[name], backbone=str(backbone),
                           backbone_sha256=suite.sha256_file(backbone), cache=str(cache), frames={role:1 for role in r},
                           pending_roles=r, allocated={role: dict(rectified=0,benchmark=0) for role in r})
                args = SimpleNamespace(worker_plan=self.root/"plan.json", lock_fd=lock.fileno(), worker_key=name,
                                       worker_status=self.root/"status.json", worker_result=self.root/"result.json")
                suite.save(args.worker_plan, dict(output=str(self.root), phase=phase, split=split, source_info=source,
                           gpus=["1","2"], events="events", raw="raw", labels="labels", activity="activity", jobs=[job]))
                suite.worker(args)
                self.assertEqual(suite.read(args.worker_result)["phase"], phase)
                for role in r:
                    self.assertFalse((cache/"rectified"/role).exists())
                    self.assertTrue((cache/"benchmark"/role/"warped.pt").exists())
        self.assertEqual(receipts.count("train_dsec_detection.py"), 1)
        self.assertEqual(receipts.count("evaluate_dsec_detection.py"), 2)

    def test_summary_groups_only_comparable_support(self):
        a = SimpleNamespace(output=self.root, phase="val")
        row = dict(samples_targets_sha256="samples", activity_assignments_sha256="groups",
                   activity_manifest_sha256="source")
        suite.save(suite.receipt(self.root, "E0", "val"), row)
        suite.summary(a)
        self.assertEqual(suite.read(self.root/"summary_val.json")["results"], {"E0": row})
        suite.save(suite.receipt(self.root, "E1", "val"), dict(row, activity_assignments_sha256="other"))
        with self.assertRaisesRegex(ValueError, "differ"):
            suite.summary(a)
        self.assertFalse((self.root/"_SUCCESS_val").exists())

    def test_test_phase_requires_the_validation_activity_fit_and_train_timeline(self):
        a = SimpleNamespace(phase="test", conditions=["E1"], output=self.root/"out", scratch=self.root/"scratch",
                            placement_root=self.root/"place", baseline_root=self.root/"base", temporal_root=self.root/"temp",
                            events=self.root/"events", raw=self.root/"raw", labels=self.root/"labels",
                            activity_manifest=self.root/"test_activity.json")
        a.activity_manifest.touch()
        suite.save(a.output/"inputs.json", dict(
            **{k:str(getattr(a,k)) for k in ("placement_root", "baseline_root", "temporal_root", "events", "raw",
                                            "labels", "scratch")},
            split_sha256=suite.sha256_file(suite.OFFICIAL), train_manifest_sha256=suite.sha256_file(suite.TRAIN35),
            namespace="dsec-det-fixture"))
        suite.save(suite.receipt(a.output,"E1","val"), {})
        source = {"train":dict(metadata="same", timestamps="same")}
        suite.save(a.output/"inputs_val.json", dict(source_info=source, activity_reference_sha256="valfit"))
        with patch.object(suite, "load_common", return_value={"splits":{"train":["train"]}}), \
                patch.object(suite, "inputs", return_value=source) as load_inputs, \
                patch.object(suite, "check_activity", return_value=("test_manifest", "other_fit")), \
                patch.object(suite, "plan_jobs") as dispatch:
            with self.assertRaisesRegex(ValueError, "thresholds differ"):
                suite.prepare(a, {"test":["test"]})
            load_inputs.return_value = {"train": dict(metadata="same", timestamps="changed")}
            with self.assertRaisesRegex(ValueError, "train35 inputs changed"):
                suite.prepare(a, {"test":["test"]})
            dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
