"""Verify review indexing, retry behavior and short native clocks without ML dependencies."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import render_full_evaluation as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scene = dict(id="dsec_depth_sample", sequence="sample", start_us=20000, end_us=140000,
                          duration_us=120000, frames=2, timestamps_us=[20000, 80000])
        self.plan = dict(dataset="dsec", task="depth", scenes=[self.scene])

    def completed(self, run):
        folder = run / self.scene["id"]
        folder.mkdir(parents=True)
        (folder / "comparison.mp4").touch()
        with (folder / "frames.csv").open("w", newline="") as out:
            table = csv.writer(out)
            table.writerow(["source_timestamp_us", "video_time_s", "event_count", "window_us", "event_rate_hz"])
            table.writerows([[20000, 0.0, 12, 20000, 600], [80000, 0.06, 24, 60000, 400]])
        (run / "manifest.json").write_text(json.dumps(dict(
            dataset="dsec", task="depth", status="complete", conditions=list(review.CONDITIONS),
            selection_sha256=review.canonical_sha256(self.plan), videos={self.scene["id"]: {"original_frames": 2}})))
        (run / "_SUCCESS").touch()

    def test_retry_and_resume_publish_local_relative_links(self):
        parent = self.root / "dsec/depth/sample"
        partial = parent / "render-001"
        partial.mkdir(parents=True)
        def renderer(command, **kwargs):
            run = Path(command[command.index("--output")+1])
            self.assertEqual(run, parent / "render-002")
            self.assertEqual(command[command.index("--sequence")+1], "sample")
            self.completed(run)
        with ((self.root/"lock").open("w") as lock,
              patch.object(review, "prepare_plans", return_value=[(self.root/"plan.json", self.plan)]),
              patch.object(review.subprocess, "run", side_effect=renderer) as launch,
              patch("event_state.evaluation.comparison_timing.verify_video_timestamps") as verify):
            args = SimpleNamespace(output=self.root, activity_dir=self.root, device="cuda", font=None)
            review.run_all(args, lock)
            review.run_all(args, lock)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(verify.call_count, 2)
        self.assertTrue(partial.is_dir())
        page = (self.root / "index.html").read_text()
        self.assertNotIn(str(self.root), page)
        self.assertNotIn("__REVIEW_DATA__", page)
        self.assertIn('"rates_hz":[600.0,400.0]', page)
        self.assertIn("render-002/dsec_depth_sample/comparison.mp4", page)

    def test_incomplete_or_different_video_never_enters_index(self):
        run = self.root / "run"
        self.completed(run)
        with self.assertRaisesRegex(ValueError, "source/conditions"):
            review.review_video(self.root, run, {**self.plan, "task": "semantic"}, self.scene)
        with self.assertRaisesRegex(ValueError, "every input frame"):
            review.review_video(self.root, run, self.plan, {**self.scene, "timestamps_us": [20000]})
        with self.assertRaisesRegex(ValueError, "clock disagree"):
            review.review_video(self.root, run, self.plan, {**self.scene, "start_us": 0})


if __name__ == "__main__":
    unittest.main()
