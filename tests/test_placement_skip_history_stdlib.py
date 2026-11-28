import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tools.evaluate_b3_skip_history import _matched_scores, _model_scales, validate_skip_scales
from tools.placement_conditions import PLACEMENT_BLOCKS, SKIP_CONTROLS


class PlacementSkipHistoryTests(unittest.TestCase):
    def test_checkpoint_gates_are_attributed_to_their_blocks(self):
        for condition in SKIP_CONTROLS:
            blocks = PLACEMENT_BLOCKS[condition]
            values = [block / 10 for block in blocks]
            temporal = SimpleNamespace(after_blocks=blocks, skip_scale=Mock())
            temporal.skip_scale.detach.return_value.cpu.return_value.reshape.return_value.tolist.return_value = values
            with self.subTest(condition=condition):
                result = _model_scales(temporal, condition)
                self.assertEqual(result["learned_skip_scales"], dict(zip(map(str, blocks), values)))
                if len(blocks) == 1:
                    self.assertEqual(result["learned_skip_scale"], values[0])
                else:
                    self.assertNotIn("learned_skip_scale", result)
        with self.assertRaisesRegex(ValueError, "Checkpoint residual placement"):
            _model_scales(SimpleNamespace(after_blocks=(3, 6, 9, 12), skip_scale=None), "E1_b3_6_9_skip")

    def test_retained_reports_must_identify_all_active_finite_gates(self):
        validate_skip_scales({"learned_skip_scale": .675}, "E1_b3_skip")
        cases = [
            ("E1_b6_skip", {"learned_skip_scale": .5}),
            ("E1_b9_skip", {"learned_skip_scales": {"9": .5}, "learned_skip_scale": .1}),
            ("E1_b3_6_9_skip", {"learned_skip_scales": {"3": .5, "6": .1}}),
            ("E1_b3_6_9_skip", {"learned_skip_scales": {"3": .5, "6": .1, "9": float("nan")}}),
            ("E1_b3_6_9_skip", {"learned_skip_scales": {"3": .5, "6": .1, "9": .8}, "learned_skip_scale": .5}),
            ("E1_b3_6_9", {"learned_skip_scales": {"3": .5, "6": .1, "9": .8}}),
        ]
        for condition, record in cases:
            with self.subTest(condition=condition, record=record), self.assertRaises(ValueError):
                validate_skip_scales(record, condition)

    def test_reset_compares_identical_labeled_frames_pixels_and_activity_groups(self):
        group = {"frames": 30, "valid_pixels": 60, "metrics": {"class_support": [19, 41], "mIoU": .5}}
        hashes = {name: name for name in ("samples_sha256", "samples_targets_sha256",
                                           "activity_assignments_sha256", "validity_support_sha256")}
        continuous = {"activity_manifest": {"manifest_sha256": "source"}, **hashes, "groups": {"all": group}}
        reset = copy.deepcopy(continuous)
        reset["groups"]["all"]["metrics"]["mIoU"] = .25
        self.assertEqual(_matched_scores(reset, continuous), ({"all": .25}, hashes))
        for name in hashes:
            with self.subTest(hash=name), self.assertRaisesRegex(ValueError, "different samples"):
                _matched_scores({**reset, name: "different"}, continuous)
        for field, value in (("frames", 31), ("valid_pixels", 61), ("metrics", {"class_support": [41, 19]})):
            changed = copy.deepcopy(reset)
            changed["groups"]["all"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "support changed"):
                _matched_scores(changed, continuous)
