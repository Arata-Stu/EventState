"""Exercise paired replay and the production state fork without ML dependencies."""
import unittest

from event_state.evaluation.comparison_semantic import SemanticBackend
from event_state.evaluation.semantic_reset_replay import replay_pair


class CountingBackend(SemanticBackend):
    def __init__(self):
        self.calls = []
        self.model = object()
        self.head = object()
        self.reset()

    def step(self, frame, predict=True):
        index = frame["frame_index"]
        if self._frame_index + 1 != index or (self._frame_index and (
                self._timestamp != frame["previous_timestamp"] or self._sequence != frame["sequence_name"])):
            raise AssertionError("Lost input position")
        self.calls.append((self, index, predict))
        self._timestamp, self._frame_index, self._sequence = frame["timestamp"], index, frame["sequence_name"]
        self.state = (self.state or 0) + frame["event_count"]
        if predict:
            return (self.state,)


class SemanticResetReplayTest(unittest.TestCase):
    def setUp(self):
        self.backend = CountingBackend()
        self.scene = dict(sequence="fixture", timestamps_us=[20, 30, 40, 60, 70],
                          q1_start_us=30, reset_input_us=40, q1_end_us=60)
        self.frames = [dict(frame_index=i, sequence_name="fixture", timestamp=i * 10,
                            previous_timestamp=(i - 1) * 10, event_count=i) for i in range(1, 8)]
        self.records = [dict(sequence="fixture", timestamp=f["timestamp"], previous_timestamp=f["previous_timestamp"],
                             event_count=f["event_count"], window_us=10) for f in self.frames]

    def test_branch_once_before_first_q1_window_and_continue_after_q1(self):
        def source(items):
            yield from items
            self.fail("Read past the last displayed timestamp")

        rows = list(replay_pair(self.backend, source(self.frames), source(self.records), self.scene))
        self.assertEqual([row[0]["timestamp"] for row in rows], self.scene["timestamps_us"])
        self.assertEqual([(keep, reset) for _, _, (keep,), (reset,) in rows],
                         [(3, 3), (6, 6), (10, 4), (21, 15), (28, 22)])
        self.assertIs(rows[1][2], rows[1][3])
        self.assertIs(rows[0][0], self.frames[1])
        self.assertIs(rows[0][1], self.records[1])

        branches = {backend for backend, _, _ in self.backend.calls} - {self.backend}
        self.assertEqual(len(branches), 1)
        branch = branches.pop()
        self.assertIs(branch.model, self.backend.model)
        self.assertIs(branch.head, self.backend.head)
        self.assertEqual([(index, predict) for b, index, predict in self.backend.calls if b is branch],
                         [(4, True), (5, False), (6, True), (7, True)])
        self.assertEqual([index for b, index, _ in self.backend.calls if b is self.backend], list(range(1, 8)))

    def test_rejects_activity_mismatch_even_in_warmup(self):
        self.records[0]["event_count"] += 1
        with self.assertRaisesRegex(ValueError, "activity window differ"):
            list(replay_pair(self.backend, self.frames, self.records, self.scene))
        self.assertFalse(self.backend.calls)

    def test_rejects_truncated_input_or_activity(self):
        for frames, records in [(self.frames[:-1], self.records), (self.frames, self.records[:-1])]:
            with self.subTest(), self.assertRaisesRegex(ValueError, "ended before"):
                list(replay_pair(self.backend, frames, records, self.scene))

    def test_rejects_noncontiguous_or_wrong_reset_window(self):
        wrong_boundary = dict(self.scene, q1_start_us=20)
        with self.assertRaisesRegex(ValueError, "reset input window"):
            list(replay_pair(self.backend, self.frames, self.records, wrong_boundary))
        with self.assertRaisesRegex(ValueError, "activity window differ"):
            list(replay_pair(self.backend, self.frames[1:], self.records[1:], self.scene))

    def test_rejects_missing_display_timestamp(self):
        scene = dict(self.scene, timestamps_us=[15, 70])
        with self.assertRaisesRegex(ValueError, "Missing display timestamp"):
            list(replay_pair(self.backend, self.frames, self.records, scene))


if __name__ == "__main__":
    unittest.main()
