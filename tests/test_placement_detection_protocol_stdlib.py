import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import digest, sha256_file
from tools.placement_conditions import PLACEMENTS
from tools.placement_detection_protocol import check_backbone, check_report, prepare_head_resume


class DetectionProtocolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model = Path(tmp.name) / "weights.pt"
        self.model.write_bytes(b"detector fixture")
        self.report = Path(tmp.name) / "val.json"
        common = load_common("dsec")
        train = common["splits"]["train"]
        norm = dict(normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])
        stats = dict(norm, protocol_sha256=digest(common), sequences=train)
        self.split = dict(train=train + common["splits"]["val"], val=["zurich_city_16_a"], test=["thun_02_a"])
        self.cfg = dict(seed=0, dataset=dict(name="dsec", train_sequences=train,
            normalization_provenance=stats, representation=norm),
            training=dict(validation_enabled=False, distributed_world_size=1, max_steps=99999),
            model=dict(temporal=dict(type="lstm", num_layers=1, hidden_dim=384, input_dim=384, output_dim=384)),
            loss=dict(z_objective=dict(type="none", weight=0, cosine_weight=1, mse_weight=1),
                      h_distill=dict(enabled=True, cosine_weight=1, mse_weight=1)))

    def backbone(self, condition, *, step=99999):
        training = SimpleNamespace(load_checkpoint_config_metadata=lambda _: SimpleNamespace(
            global_step=step, config=self.cfg))
        with patch.dict(sys.modules, {"event_state.training": training}):
            return check_backbone(self.model, condition, self.split)

    def test_b3_skip_checkpoint_cannot_substitute_for_b3_replacement(self):
        self.cfg["model"]["temporal"].update(type="four_stage_lstm", after_blocks=[3], residual_skip=True)
        with self.assertRaisesRegex(ValueError, "wrong temporal model"):
            self.backbone("E1_b3")

    def test_all_nine_architectures_report_the_actual_weights_and_features(self):
        temporal = self.cfg["model"]["temporal"]
        self.assertEqual(self.backbone("E1")["sha256"], sha256_file(self.model))
        temporal["type"] = "convlstm"
        self.assertEqual(self.backbone("E1_convlstm")["feature"], "h")
        temporal["type"] = "identity"
        self.cfg["loss"]["z_objective"].update(type="direct_dino", weight=1)
        self.cfg["loss"]["h_distill"]["enabled"] = False
        self.assertEqual(self.backbone("E0")["feature"], "z")
        temporal["type"] = "four_stage_lstm"
        self.cfg["loss"]["h_distill"]["enabled"] = True
        self.cfg["loss"]["z_objective"].update(type="none", weight=0)
        for condition, blocks in PLACEMENTS.items():
            temporal["after_blocks"] = list(blocks)
            self.assertEqual(self.backbone(condition)["after_blocks"], list(blocks))

    def test_rejects_wrong_stage_placement_objective_and_holdout(self):
        for condition, step in (("E0", 99999), ("E1_b3", 99999), ("E1", 99)):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                self.backbone(condition, step=step)
        self.cfg["loss"]["h_distill"]["mse_weight"] = 0
        with self.assertRaisesRegex(ValueError, "objective"):
            self.backbone("E1")
        self.cfg["loss"]["h_distill"]["mse_weight"] = 1
        self.cfg["pretraining_history"] = [dict(sequences=self.split["val"])]
        with self.assertRaisesRegex(ValueError, "holdout"):
            self.backbone("E1")

    def sample(self):
        seq = self.split["val"][0]
        cuts = ["p00_20", "p20_40", "p40_60", "p60_80", "p80_100"]
        metrics = dict(mAP=.5, AP50=.6, AP75=.4, AP_car=.5, AP_pedestrian=None,
                       AP_small=None, AP_medium=.5, AP_large=None)

        def group(n):
            return dict(frames=n, gt_objects=n, detections=n, status="ok" if n else "empty",
                        sequence_frames={seq: n} if n else {}, sequences=int(n > 0),
                        gt_objects_per_class={"car": n, "pedestrian": 0},
                        metrics=metrics.copy() if n else dict.fromkeys(metrics))

        groups = {key: group(int(i < 3)) for i, key in enumerate(cuts)}
        groups["all"] = group(3)
        activity = dict(partition="whole_frame_by_scene_event_rate", condition_label="E1 fixture",
                        activity_manifest=dict(manifest_sha256="a" * 64, groups=[dict(name=q) for q in cuts]),
                        state_policy_by_sequence={seq: "continuous"},
                        detector_checkpoint={"sha256": sha256_file(self.model)},
                        samples_targets_sha256="b" * 64, activity_assignments_sha256="c" * 64, groups=groups)
        return dict(role="val", protocol="dsec-det", coordinate_space="dsec_det_distorted",
                    feature="h", activity_evaluation=activity, **metrics)

    def validate(self, record):
        self.report.write_text(json.dumps(record))
        return check_report(self.report, "val", "E1", "h", "E1 fixture", self.model, "a" * 64,
                            self.split["val"])

    def test_preserves_empty_quintiles_and_compares_all_eligible_samples(self):
        result = self.validate(self.sample())
        self.assertEqual(result["samples_targets_sha256"], "b" * 64)
        self.assertIsNone(result["groups"]["p80_100"]["metrics"]["mAP"])
        self.assertEqual(result["mAP"], .5)

    def test_rejects_swapped_heads_roles_inputs_and_damaged_metrics(self):
        for keys, value in [
            (("role",), "test"), (("feature",), "z"), (("mAP",), .99),
            (("activity_evaluation", "detector_checkpoint", "sha256"), "e" * 64),
            (("activity_evaluation", "activity_manifest", "manifest_sha256"), "f" * 64),
            (("activity_evaluation", "state_policy_by_sequence"), {}),
            (("activity_evaluation", "samples_targets_sha256"), "bad"),
            (("activity_evaluation", "groups", "all", "detections"), 2),
            (("activity_evaluation", "groups", "p00_20", "metrics", "AP50"), None),
        ]:
            record = self.sample()
            dest = record
            for key in keys[:-1]:
                dest = dest[key]
            dest[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.validate(record)


class HeadResumeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.head = Path(tmp.name)
        self.log = self.head / "metrics.jsonl"
        self.last, self.best = self.head / "last.pt", self.head / "best.pt"
        (self.head / "config.json").write_text(json.dumps({"epochs": 50, "resume": "/resume/path"}))
        torch = SimpleNamespace(load=lambda p, **_: json.loads(p.read_bytes()))
        self.mocks = patch.dict(sys.modules, {"torch": torch})
        self.mocks.start()
        self.addCleanup(self.mocks.stop)

    @staticmethod
    def checkpoint(path, epoch, best_map):
        path.write_text(json.dumps(dict(epoch=epoch, best_mAP=best_map, config={"epochs": 50},
                                        model={"weights": epoch}, optimizer={}, scheduler={}, scaler={})))

    def history(self, *scores):
        self.log.write_bytes(b"".join((json.dumps({"epoch": i, "val/mAP": s}) + "\n").encode()
                                      for i, s in enumerate(scores)))

    def test_first_epoch_log_without_checkpoint_starts_fresh(self):
        self.assertIsNone(prepare_head_resume(self.head))
        self.history(.2)
        self.assertIsNone(prepare_head_resume(self.head))
        self.assertEqual(self.log.read_bytes(), b"")
        self.assertFalse(self.best.exists())

    def test_atomic_repair_of_partial_tail_and_previous_best_is_idempotent(self):
        self.history(.2, .3)
        expected = self.log.read_bytes()
        self.log.write_bytes(expected + b'{"ep')
        self.checkpoint(self.last, 1, .3)
        self.checkpoint(self.best, 0, .2)
        self.assertEqual(prepare_head_resume(self.head), self.last)
        self.assertEqual(self.log.read_bytes(), expected)
        self.assertEqual(self.best.read_bytes(), self.last.read_bytes())
        self.assertEqual(prepare_head_resume(self.head), self.last)

    def test_missing_selected_last_can_be_restored_and_ties_keep_the_first_best(self):
        self.history(.2)
        self.checkpoint(self.last, 0, .2)
        prepare_head_resume(self.head)
        first_best = self.best.read_bytes()
        self.history(.2, .2, .9)
        self.checkpoint(self.last, 1, .2)
        prepare_head_resume(self.head)
        self.assertEqual(self.best.read_bytes(), first_best)
        self.assertEqual(len(self.log.read_bytes().splitlines()), 2)

    def test_unrecoverable_best_or_committed_score_preserves_all_artifacts(self):
        self.history(.5, .2, .8)
        self.checkpoint(self.last, 1, .5)
        original = self.log.read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot recover"):
            prepare_head_resume(self.head)
        self.assertFalse(self.best.exists())
        self.assertEqual(self.log.read_bytes(), original)
        self.checkpoint(self.last, 1, .9)
        self.checkpoint(self.best, 0, .5)
        before = self.best.read_bytes()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_head_resume(self.head)
        self.assertEqual(self.best.read_bytes(), before)
        self.assertEqual(self.log.read_bytes(), original)
