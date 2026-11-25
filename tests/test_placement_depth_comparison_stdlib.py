import json
from pathlib import Path
import tempfile
import unittest

from tools import placement_depth_comparison as comparison


class PlacementDepthComparisonTest(unittest.TestCase):
    def sample(self, dataset):
        records = json.loads(comparison.ACTIVITY_REFERENCE.read_text())["results"]
        source = next(row for row in records if row["task"] == "depth" and row["condition"] == dataset + "_E1")
        job = dict(dataset=dataset, name=dataset + "_" + comparison.FOUR_STAGE,
                   label=comparison.CONDITIONS[comparison.FOUR_STAGE], activity_report=self.path)
        activity = dict(source, condition_label=job["label"],
                        activity_manifest=dict(manifest_sha256=source["manifest_sha256"]))
        report = dict(task="depth", feature="stages", protocol=dict(dataset=dataset), role="val",
                      state_policy="continuous", samples_targets_sha256=source["samples_targets_sha256"],
                      validity_sha256=source["validity_sha256"], activity_evaluation=activity)
        result = dict(condition=job["label"], activity_groups=activity["groups"])
        if dataset == "dsec":
            result["mae_px"] = 2.0
        return job, result, report

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "activity_continuous.json"

    def test_signed_maes_against_both_published_baselines(self):
        for dataset in ("dsec", "m3ed"):
            with self.subTest(dataset=dataset):
                job, result, report = self.sample(dataset)
                last = result["activity_groups"]["p80_100"]
                last["metrics"]["mae"] += 1.25
                last["metrics_frame_mean_f3"]["mae"] -= 0.75
                self.path.write_text(json.dumps(report))
                summary = comparison.attach_comparison(job, result)["comparison_to_E1"]
                self.assertEqual(set(summary["groups"]), set(result["activity_groups"]))
                self.assertAlmostEqual(summary["groups"]["p80_100"]["pooled_mae_m"]["delta_m"], 1.25)
                self.assertAlmostEqual(summary["groups"]["p80_100"]["frame_mean_mae_m"]["delta_m"], -0.75)
                self.assertEqual(summary["groups"]["all"]["pooled_mae_m"]["delta_m"], 0)
                if dataset == "dsec":
                    self.assertAlmostEqual(summary["disparity_mae_px"]["delta_px"], 2 - 1.8708127091004858)
                else:
                    self.assertNotIn("disparity_mae_px", summary)

    def test_different_manifest_gt_masks_assignments_or_support_fail_closed(self):
        for key in ("manifest_sha256", *comparison.HASHES, "top_level_validity", "sequence_frames"):
            with self.subTest(key=key):
                job, result, report = self.sample("m3ed")
                activity = report["activity_evaluation"]
                if key == "sequence_frames":
                    activity["groups"]["p00_20"]["sequence_frames"] = {"unexpected": 12876}
                elif key == "top_level_validity":
                    report["validity_sha256"] = "stale"
                elif key == "manifest_sha256":
                    activity["activity_manifest"][key] = "stale"
                else:
                    activity[key] = "stale"
                self.path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    comparison.attach_comparison(job, result)

    def test_invalid_score_cannot_be_treated_as_a_delta(self):
        job, result, report = self.sample("m3ed")
        result["activity_groups"]["p40_60"]["metrics_frame_mean_f3"]["mae"] = None
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            comparison.attach_comparison(job, result)
