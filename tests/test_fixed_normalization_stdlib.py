"""Run with python -m unittest discover -s tests -p test_fixed_normalization_stdlib.py."""

import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "event_state/data/fixed_normalization.py"
spec = importlib.util.spec_from_file_location("fixed_normalization_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FixedNormalizationContractTests(unittest.TestCase):
    def setUp(self):
        self.rep = {"type": "voxel_grid", "normalization": "none", "channels": 2,
                    "channel_order": "positive_then_negative"}
        self.stats = {"format": "eventstate_fixed_normalization_v1",
                      "representation": self.rep, "sensor_size": [4, 5],
                      "includes_zero_pixels": True, "mean": [1., 2.], "std": [2., 3.]}

    def test_coefficients(self):
        self.assertEqual(module.validate_fixed_normalization(self.stats, self.rep, 4, 5),
                         ((1., 2.), (2., 3.)))

    def test_invalid_contracts(self):
        for key, value in [("mean", [1.]), ("std", [0., 1.]), ("std", [float("nan"), 1.]),
                           ("mean", [float("inf"), 1.]), ("sensor_size", [5, 4]),
                           ("includes_zero_pixels", False)]:
            with self.subTest(key=key, value=value):
                stats = copy.deepcopy(self.stats)
                stats[key] = value
                with self.assertRaises(ValueError):
                    module.validate_fixed_normalization(stats, self.rep, 4, 5)

    def test_channel_order_and_double_normalization(self):
        rep = dict(self.rep, channel_order="negative_then_positive")
        with self.assertRaises(ValueError):
            module.validate_fixed_normalization(self.stats, rep, 4, 5)
        rep = dict(self.rep, normalization="nonzero_standardize")
        stats = dict(self.stats, representation=rep)
        with self.assertRaises(ValueError):
            module.validate_fixed_normalization(stats, rep, 4, 5)
