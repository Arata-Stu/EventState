"""Check time origins, native Q1 input windows and replay coverage."""
from copy import deepcopy
import unittest

from event_state.evaluation.ucity_reset_scene import SEQUENCE, make_scene


class UCityResetSceneTests(unittest.TestCase):
    def setUp(self):
        step = 200_000
        # Physical start is 2s; the old video places its first input end (2.2s) at t=0.
        self.records = []
        for i in range(125):
            start = 2_000_000 + i * step
            low = 40 <= i < 95
            self.records.append(dict(sequence=SEQUENCE, previous_timestamp=start, timestamp=start + step,
                                     window_us=step, event_rate_hz=1 if low else 5,
                                     group="p00_20" if low else "p80_100"))
        self.entry = dict(dataset="m3ed", sequence=SEQUENCE, frames=125, q1_upper_events_per_s=1,
                          episodes=[dict(start_us=10_000_000, end_us=21_000_000, duration_s=11,
                                         frames=55, interior=True)],
                          clip=dict(from_video_s=2.8, to_video_s=23.8,
                                    q1_source_start_us=10_000_000, q1_source_end_us=21_000_000))
        self.selection = dict(source="full_sequence_review_20261118", task="semantic",
                              selection=dict(group="p00_20", minimum_continuous_seconds=10),
                              sequences=[self.entry])

    def test_first_low_input_follows_physical_onset_and_exclusive_end_is_held(self):
        scene = make_scene(self.selection, iter(self.records))
        self.assertEqual(scene["q1_start_us"], 10_000_000)
        self.assertEqual(scene["reset_input_us"], 10_200_000)
        self.assertEqual(scene["q1_end_us"], 21_000_000)
        self.assertEqual(scene["timestamps_us"], list(range(5_000_000, 26_000_000, 200_000)))
        self.assertEqual(scene["start_us"], 5_000_000)
        self.assertEqual(scene["end_us"], 26_000_000)
        self.assertEqual(scene["duration_us"], 21_000_000)
        self.assertEqual(scene["terminal_hold_us"], 200_000)
        self.assertEqual(scene["warmup_frames_before_start"], 14)
        self.assertAlmostEqual(scene["low_fraction"], 11/21)
        self.assertAlmostEqual(scene["high_fraction"], 10/21)

    def test_prefix_and_recovery_must_be_contiguous(self):
        with self.assertRaisesRegex(ValueError, "complete native"):
            make_scene(self.selection, self.records[1:])
        for pos in (1, 120):
            records = deepcopy(self.records)
            records[pos]["previous_timestamp"] += 1
            with self.subTest(pos=pos), self.assertRaisesRegex(ValueError, "discontinuous"):
                make_scene(self.selection, records)

    def test_changed_assignment_breaks_match_to_saved_episode(self):
        self.records[60]["group"] = "p80_100"
        with self.assertRaisesRegex(ValueError, "rate threshold"):
            make_scene(self.selection, self.records)
        self.records[60]["event_rate_hz"] = 5
        with self.assertRaisesRegex(ValueError, "No interior"):
            make_scene(self.selection, self.records)

    def test_reject_shifted_bounds_or_wrong_count_in_archive(self):
        self.entry["episodes"][0]["frames"] += 1
        with self.assertRaisesRegex(ValueError, "duration or frame count"):
            make_scene(self.selection, self.records)
        self.entry["episodes"][0]["frames"] -= 1
        self.entry["clip"]["from_video_s"] += 0.2
        with self.assertRaisesRegex(ValueError, "five seconds"):
            make_scene(self.selection, self.records)

    def test_longer_boundary_episode_is_excluded_and_earlier_interior_wins_ties(self):
        step = self.records[-1]["window_us"]
        for i in range(len(self.records), 275):
            start = 2_000_000 + i * step
            low = 135 <= i < 190 or i >= 200
            self.records.append(dict(sequence=SEQUENCE, previous_timestamp=start, timestamp=start + step,
                                     window_us=step, event_rate_hz=1 if low else 5,
                                     group="p00_20" if low else "p80_100"))
        self.entry["frames"] = len(self.records)
        self.assertEqual(make_scene(self.selection, self.records)["q1_start_us"], 10_000_000)


if __name__ == "__main__":
    unittest.main()
