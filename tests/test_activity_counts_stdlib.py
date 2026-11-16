"""Count/rate units, train assignments, and temporal boundaries without NumPy."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from event_state.activity_quantiles import M3ED_SOURCE_CONTRACT, build_activity_manifest
from event_state.evaluation.activity_counts import summarize_activity_counts
from event_state.evaluation.activity_scenes import eligible_sequences


ROOT = Path(__file__).resolve().parents[1]


def rows(sequence, counts, durations=None, offset=0):
    durations = durations or [100_000] * len(counts)
    result = []
    for count, duration in zip(counts, durations):
        result.append(dict(sequence=sequence, previous_timestamp=offset,
                           timestamp=offset + duration, window_us=duration,
                           event_count=count, event_rate_hz=count * 1e6 / duration))
        offset += duration
    return result


class ActivityCountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.eligible, _ = eligible_sequences("dsec", "semantic")

    def manifest(self, custom=None):
        data = rows("reference", range(10))
        for sequence in self.eligible:
            data += (custom[sequence] if custom and sequence in custom
                     else rows(sequence, [1, 9]))
        payload = build_activity_manifest(data, ["reference"], reference=dict(
            role="train", name="fixture", manifest_sha256="a" * 64,
            excluded_sequences=list(self.eligible)),
            source_metadata_sha256={r["sequence"]: "b" * 64 for r in data})
        path = self.directory / "activity.json"
        path.write_text(json.dumps(payload))
        return path, payload

    def test_actual_count_and_rate_are_distinct_with_variable_intervals(self):
        sequence = self.eligible[0]
        path, original = self.manifest({sequence: rows(sequence, [1, 1], [100_000, 200_000])})
        result = summarize_activity_counts(path, dataset="dsec", task="semantic")
        sequence_result = result["evaluation"]["per_sequence"][sequence]
        self.assertEqual(sequence_result["all"]["event_count"]["min"], 1)
        self.assertEqual(sequence_result["all"]["event_count"]["max"], 1)
        self.assertEqual(sequence_result["all"]["event_rate_hz"]["min"], 5)
        self.assertEqual(sequence_result["all"]["event_rate_hz"]["max"], 10)
        self.assertEqual(sequence_result["all"]["window_us"]["max"], 200_000)
        self.assertAlmostEqual(sequence_result["all"]["time_weighted_event_rate_hz"], 20 / 3)
        self.assertEqual(result["activity_manifest_identity"]["cutpoints"], original["cutpoints"])
        self.assertEqual(result["activity_manifest_identity"]["groups"], original["groups"])
        self.assertEqual(json.loads(path.read_text()), original)
        self.assertIn("not_GT", result["sample_scope"])

    def test_duration_occupancy_differs_from_frame_fraction(self):
        sequence = self.eligible[0]
        path, original = self.manifest({sequence: rows(sequence, [1, 18], [100_000, 200_000])})
        report = summarize_activity_counts(path, dataset="dsec", task="semantic")
        sequence_result = report["evaluation"]["per_sequence"][sequence]
        low, high = original["groups"][0]["name"], original["groups"][-1]["name"]
        self.assertEqual(sequence_result["groups"][low]["frames"], 1)
        self.assertEqual(sequence_result["groups"][high]["frames"], 1)
        self.assertAlmostEqual(sequence_result["groups"][low]["duration_fraction"], 1 / 3)
        self.assertAlmostEqual(sequence_result["groups"][high]["duration_fraction"], 2 / 3)
        self.assertAlmostEqual(sequence_result["all"]["time_weighted_event_rate_hz"], 190 / 3)

    def test_runs_never_join_gaps_long_intervals_or_sequences(self):
        first, second, third = self.eligible
        custom = {first: rows(first, [1, 1]) + rows(first, [1], offset=300_000)
                  + rows(first, [1], [400_000], offset=400_000)
                  + rows(first, [1], offset=800_000),
                  second: rows(second, [1, 1, 1]), third: rows(third, [9])}
        path, _ = self.manifest(custom)
        temporal = summarize_activity_counts(path, dataset="dsec", task="semantic")["evaluation"]["temporal"]
        self.assertEqual(temporal["missing_gap_count"], 1)
        self.assertAlmostEqual(temporal["missing_gap_seconds"], .1)
        self.assertEqual(temporal["excluded_long_intervals"], 1)
        self.assertAlmostEqual(temporal["excluded_long_interval_seconds"], .4)
        low = temporal["run_duration_seconds"]["low"]
        self.assertEqual(low["n"], 4)
        self.assertEqual(low["max"], .3)
        self.assertAlmostEqual(low["mean"], .175)
        self.assertEqual(temporal["run_duration_seconds"]["high"]["max"], .1)

    def test_overlap_is_rejected_instead_of_double_counted(self):
        sequence = self.eligible[0]
        path, _ = self.manifest({sequence: rows(sequence, [1]) + rows(sequence, [2], offset=50_000)})
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            summarize_activity_counts(path, dataset="dsec", task="semantic")

    def test_cli_is_stdlib_only_and_preserves_existing_output(self):
        path, _ = self.manifest()
        output = self.directory / "summary.json"
        command = [sys.executable, "-S", str(ROOT / "tools/summarize_activity_counts.py"),
                   "--activity-manifest", str(path), "--dataset", "dsec", "--task", "semantic",
                   "--output", str(output)]
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("events/RGB interval", run.stdout)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["evaluation"]["all"]["frames"], 6)
        again = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(output.read_bytes(), saved)
        probe = subprocess.run([sys.executable, "-S", "-c",
            "import event_state.evaluation.activity_counts; import sys; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_wrong_dataset_or_tampered_input_is_rejected(self):
        path, payload = self.manifest()
        with self.assertRaisesRegex(ValueError, "Dataset differs"):
            summarize_activity_counts(path, dataset="m3ed", task="semantic")
        payload["records"][0]["event_count"] += 1
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            summarize_activity_counts(path, dataset="dsec", task="semantic")

    def test_m3ed_uses_its_shared_public_val_and_prepared_count_contract(self):
        eligible, _ = eligible_sequences("m3ed", "depth")
        data = rows("reference", range(10))
        for sequence in eligible:
            data += rows(sequence, [1, 9])
        payload = build_activity_manifest(data, ["reference"], reference=dict(
            role="train", name="m3ed_fixture", manifest_sha256="a" * 64,
            excluded_sequences=list(eligible)),
            source_metadata_sha256={r["sequence"]: "b" * 64 for r in data},
            source_contract=M3ED_SOURCE_CONTRACT)
        path = self.directory / "m3ed.json"
        path.write_text(json.dumps(payload))
        result = summarize_activity_counts(path, dataset="m3ed", task="depth")
        self.assertEqual(result["evaluation"]["sequences"], list(eligible))
        self.assertEqual(result["evaluation"]["all"]["frames"], 4)
        self.assertEqual(result["activity_manifest_identity"]["source_contract"], M3ED_SOURCE_CONTRACT)


if __name__ == "__main__":
    unittest.main()
