import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.b3_semantic_timing import semantic_remaining


class SemanticTimingTests(unittest.TestCase):
    now = 100_000

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.outputs = Path(temporary.name)
        self.current = self.outputs / "stage2_b3_skip_semantic_20261126"
        self.old = self.outputs / "stage2_placement_semantic_20261118"

    def file(self, path, when, contents=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents if isinstance(contents, str) else json.dumps(contents))
        os.utime(path, (when, when))
        return path

    def baseline(self, ds):
        root = self.old / f"{ds}_E1_b3"
        self.file(root / "verified.json", 98000, dict(condition="E1_b3", dataset=ds,
                                                       attempt="run-good", cache=str(self.outputs / "cache")))
        reference = root / "run-good"
        self.file(reference / "_SUCCESS", 97000)
        return reference

    def eta(self, ds):
        return semantic_remaining(self.outputs, ds, self.now)

    def test_dsec_historical_then_same_phase_elapsed_but_new_reset_still_unknown(self):
        old = self.baseline("dsec")
        self.file(old / "job_identity.json", 90000)
        self.file(old / "final/config.json", 93900)
        self.assertEqual(self.eta("dsec")[:2], (7000, "履歴リセット評価"))
        job = self.current / "dsec_E1_b3_skip/run-new"
        self.file(job / "final/config.json", 99800)
        self.file(self.current / "logs/dispatch-1/dsec_E1_b3_skip.log", 99000, "fitting")
        self.file(job / "logs/final_fit.log", self.now, "train 2/50: 55%|#####| 44/80 [00:30<00:21, 2batch/s]")
        self.assertEqual(self.eta("dsec")[:2], (2900, "履歴リセット評価"))
        self.assertIn("train 2/50: 55% 44/80", self.eta("dsec")[2])

    def test_stale_run_exceeding_reference_does_not_predict_immediate_completion(self):
        old = self.baseline("dsec")
        self.file(old / "job_identity.json", 96000)
        self.file(self.current / "dsec_E1_b3_skip/run-new/job_identity.json", 98000)
        seconds, unknown, _ = self.eta("dsec")
        self.assertIn("更新待ち", unknown)
        self.file(self.current / "logs/dispatch-1/dsec_E1_b3_skip.log", self.now,
                  "train 1/50: 70%|#####| 70/100 [00:25<00:12, 2batch/s]")
        seconds, unknown, _ = self.eta("dsec")
        self.assertEqual(seconds, 0)
        self.assertIn("過去の所要時間を超えた連続評価", unknown)
        self.assertIn("履歴", unknown)

    def test_missing_or_copied_reference_leaves_unknown_work(self):
        self.assertIn("連続評価", self.eta("dsec")[1])
        self.file(self.current / "logs/dispatch-1/dsec_E1_b3_skip.log", self.now,
                  "cache seq: 20%|#####| 20/100 [00:25<00:12, 2batch/s]")
        self.assertIn("20/100", self.eta("dsec")[2])
        old = self.baseline("dsec")
        self.file(old / "job_identity.json", 97000)
        self.assertEqual(self.eta("dsec")[0], 0)
        self.assertIn("連続評価", self.eta("dsec")[1])

    def test_m3ed_cache_separate_from_head_and_replaced_with_live_rate(self):
        old = self.baseline("m3ed")
        self.file(old / "h/config.json", 93000)
        self.assertEqual(self.eta("m3ed")[:2], (4000, "特徴抽出・履歴リセット評価"))
        self.file(self.outputs / "cache/car_urban_day_penno_big_loop/000001.pt", 90000)
        self.assertEqual(self.eta("m3ed")[:2], (7000, "履歴リセット評価"))
        self.file(self.current / "logs/dispatch-1/m3ed_E1_b3_skip.log", self.now,
                  "M3ED Semantic cache seq: 50%|#####| 150/300 [01:10<01:20, 2.00window/s]")
        self.assertEqual(self.eta("m3ed")[0], 4080)
        self.file(self.current / "m3ed_E1_b3_skip/run-new/h/config.json", 99000)
        self.file(self.current / "logs/dispatch-1/m3ed_E1_b3_skip.log", self.now,
                  "M3ED train 1/50: 55%|#####| 44/80 [00:30<00:21, 2batch/s]")
        self.assertEqual(self.eta("m3ed")[0], 3000)

    def test_reset_uses_latest_line_of_current_dispatch_and_requires_fresh_measurements(self):
        job = self.current / "m3ed_E1_b3_skip"
        self.file(job / "verified.json", self.now-50, {})
        log = self.current / "logs/dispatch-1/m3ed_E1_b3_skip.log"
        self.file(log, self.now, "unrelated\rmodel | reset seq: 40%|###| 500/1250 [00:40<01:15, 9window/s]")
        self.assertEqual(self.eta("m3ed")[:2], (75, ""))
        self.file(log, self.now, "model | reset seq: 40%|###| 500/1250 [00:40<01:15, 9window/s]\nTraceback")
        self.assertNotEqual(self.eta("m3ed")[1], "")
        self.file(log, self.now-300, "model | reset seq: 40%|###| 500/1250 [00:40<01:15, 9window/s]")
        self.assertNotEqual(self.eta("m3ed")[1], "")
        self.file(job / "history.json", self.now)
        self.assertEqual(self.eta("m3ed")[:2], (0, ""))

    def test_old_reset_bar_uses_current_and_later_sequence_window_counts(self):
        job = self.current / "m3ed_E1_b3_skip"
        report = self.file(job / "report.json", 99500,
                           {"activity_evaluation": {"state_policy_by_sequence": {"s1":"continuous", "s2":"continuous"}}})
        self.file(job / "verified.json", 99500, {"report": str(report)})
        data = self.outputs / "data"
        self.file(self.current / "inputs.json", 99000, {"data": str(data)})
        for seq, count in [("s1", 1001), ("s2", 201)]:
            self.file(data / f"m3ed_cache/half_dagr/{seq}/metadata.json", 99000, {"frame_count": count})
        log = self.current / "logs/dispatch-1/m3ed_E1_b3_skip.log"
        self.file(log, self.now, "model | reset s1: 700it [01:10, 10.00it/s]")
        self.assertEqual(self.eta("m3ed")[:2], (50, ""))
