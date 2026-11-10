"""Frame aggregation contracts without importing NumPy, Torch or a dataset."""

import json
import math
import subprocess
import sys
import unittest

from event_state.dense.depth_frame_metrics import F3_FRAME_PROTOCOL, FrameMeanSums, METRIC_NAMES


def frame(value):
    return {key: (value / 10 if key in {"d1", "d2", "d3"} else value) for key in METRIC_NAMES}


class FrameMeanTest(unittest.TestCase):
    def test_unequal_pixel_support_does_not_weight_the_frame_mean(self):
        meter = FrameMeanSums()
        meter.add(100, frame(1))
        meter.add(1, frame(10))
        scores = meter.compute()
        self.assertEqual(scores["frames"], 2)
        self.assertEqual(scores["valid_pixels"], 101)
        self.assertEqual(scores["mae"], 5.5)
        self.assertNotAlmostEqual(scores["mae"], 110 / 101)
        self.assertEqual(scores["d1"], .55)

    def test_rmse_is_averaged_after_each_frames_square_root(self):
        meter = FrameMeanSums()
        meter.add(1, frame(1))
        meter.add(1, frame(3))
        self.assertEqual(meter.compute()["rmse"], 2)
        self.assertNotAlmostEqual(meter.compute()["rmse"], math.sqrt(5))

    def test_empty_frames_are_excluded_and_empty_groups_serialize_nulls(self):
        meter = FrameMeanSums()
        meter.add(0, {})
        scores = meter.compute()
        self.assertEqual(scores["frames"], 0)
        self.assertEqual(scores["valid_pixels"], 0)
        self.assertTrue(all(scores[key] is None for key in METRIC_NAMES))
        json.dumps(scores, allow_nan=False)
        meter.add(2, frame(1))
        meter.add(0, {})
        self.assertEqual(meter.compute()["mae"], 1)
        self.assertEqual(meter.compute()["frames"], 1)

    def test_invalid_inputs_never_partially_mutate_the_meter(self):
        meter = FrameMeanSums()
        meter.add(2, frame(1))
        before = meter.compute()
        invalid_metrics = [dict(frame(1), sq_rel=float("nan")), dict(frame(1), rmse=float("inf")),
                           dict(frame(1), d1=1.01), dict(frame(1), mae=-1), dict(frame(1), mae=True),
                           dict(frame(1), mae="1"), {"mae": 1}]
        for values in invalid_metrics:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    meter.add(3, values)
                self.assertEqual(meter.compute(), before)
        for count in (-1, True, 1.2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                meter.add(count, frame(1))
        with self.assertRaises(ValueError):
            meter.add(0, frame(1))

    def test_protocol_records_formula_precision_and_scope(self):
        self.assertEqual(F3_FRAME_PROTOCOL["arithmetic_dtype"], "float64")
        self.assertIn("target^2", F3_FRAME_PROTOCOL["sq_rel_definition"])
        self.assertIn("no additional F3", F3_FRAME_PROTOCOL["validity"])
        self.assertEqual(F3_FRAME_PROTOCOL["reference"]["revision"], "833e2c65233f9787449ce0675ae41e3f6e9c193a")
        json.dumps(F3_FRAME_PROTOCOL, allow_nan=False)

    def test_import_does_not_load_scientific_libraries(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import event_state.dense.depth_frame_metrics; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
