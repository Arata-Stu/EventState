"""Run directly with Python's standard library; does not import torch/numpy."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("split_guard", ROOT / "event_state/data/split_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def sequence_lists(path):
    # These repository manifests use top-level keys and plain sequence lists.
    result, key = {}, None
    for line in path.read_text().splitlines():
        if line and not line.startswith((" ", "#")):
            key = line.split(":", 1)[0]
            result[key] = []
        elif line.startswith("  - "):
            result[key].append(line[4:].strip())
    return result


class JointSplitTests(unittest.TestCase):
    def setUp(self):
        self.det = sequence_lists(ROOT / "tools/manifests/dsec_det_official_split.yaml")
        self.sem = sequence_lists(ROOT / "tools/manifests/dsec_semantic_split.yaml")
        self.config = sequence_lists(ROOT / "configs/dataset/dsec_joint_clean.yaml")
        self.train = self.config["train_sequences"]
        self.val = self.config["val_sequences"]

    def check_split(self, train, val):
        guard.validate_joint_dsec_split(train, val, self.det, self.sem)

    def test_joint_preset_is_clean(self):
        self.check_split(self.train, self.val)

    def test_existing_train41_protocol_is_preserved(self):
        preset = sequence_lists(ROOT / "configs/dataset/dsec_det_train41.yaml")
        guard.validate_official_dsec_pretraining(
            preset["train_sequences"], None, self.det, self.sem, validation_enabled=False,
        )
        self.assertTrue(set(self.sem["val"]) <= set(preset["train_sequences"]))

    def test_final_fit_rejects_validation_and_changed_training_set(self):
        for train, enabled in ((self.det["train"], True), (self.det["train"][:-1], False),
                               (self.det["train"] + self.sem["test"], False)):
            with self.assertRaises(ValueError):
                guard.validate_official_dsec_pretraining(
                    train, None, self.det, self.sem, validation_enabled=enabled,
                )

    def test_semantic_validation_is_rejected(self):
        for name in self.sem["val"] + self.sem["test"]:
            with self.assertRaises(ValueError):
                self.check_split(self.train + [name], self.val)

    def test_detector_holdouts_are_rejected_even_in_pretrain_validation(self):
        for name in self.det["val"] + self.det["test"]:
            with self.assertRaises(ValueError):
                self.check_split(self.train, self.val + [name])

    def test_sibling_of_test_recording_is_rejected(self):
        # interlaken_00_c is nominally det train, but _a/_b are test.
        with self.assertRaises(ValueError):
            self.check_split(self.train + ["interlaken_00_c"], self.val)

    def test_train_val_recording_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_split(self.train + ["zurich_city_11_a"], ["zurich_city_11_b"])

    def test_duplicates_and_implicit_selection_are_rejected(self):
        for train in (None, [], self.train + [self.train[0]]):
            with self.assertRaises(ValueError):
                self.check_split(train, self.val)


if __name__ == "__main__":
    unittest.main()
