"""Real-time scene selection, without Torch, predictions, or video libraries."""

import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_state.activity_quantiles import M3ED_SOURCE_CONTRACT, build_activity_manifest
from event_state.evaluation.activity_scenes import (
    continuous_blocks, eligible_sequences, select_full_sequences, select_scenes,
    validate_full_sequence, window_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def rows(sequence, counts, durations=None, offset=0):
    durations = durations or [100_000] * len(counts)
    result = []
    for count, window in zip(counts, durations):
        result.append(dict(sequence=sequence, previous_timestamp=offset, timestamp=offset+window,
                           window_us=window, event_count=count,
                           event_rate_hz=count*1_000_000/window))
        offset += window
    return result


class ActivitySceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def manifest(self, *, dataset="dsec", task="semantic", custom=None):
        eligible, _ = eligible_sequences(dataset, task)
        records = rows("reference_train", range(10))
        for sequence in eligible:
            records += (custom.get(sequence, rows(sequence, [4]*24)) if custom is not None
                        else rows(sequence, [1]*12+[9]*12) if sequence == eligible[0]
                        else rows(sequence, [4]*24))
        payload = build_activity_manifest(records, ["reference_train"], reference=dict(
            role="train", name="fixture", manifest_sha256="a"*64,
            excluded_sequences=list(eligible)), source_metadata_sha256={
                r["sequence"]:"b"*64 for r in records},
            source_contract=M3ED_SOURCE_CONTRACT if dataset == "m3ed" else None)
        path = self.directory / f"{dataset}_{task}_activity.json"
        path.write_text(json.dumps(payload))
        return path, payload, eligible

    def test_low_high_fixed_windows_preserve_all_original_timestamps_and_do_not_overlap(self):
        path, payload, eligible = self.manifest()
        before = path.read_bytes()
        result = select_scenes(path, dataset="dsec", task="semantic", duration_seconds=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"], {"low":1,"high":1})
        self.assertEqual(result["selection"]["interval_convention"], "[start_us,end_us)")
        self.assertEqual(path.read_bytes(), before)
        for scene in result["scenes"]:
            self.assertEqual(scene["sequence"], eligible[0])
            self.assertEqual(scene["end_us"]-scene["start_us"], 1_000_000)
            stamps = [r["timestamp"] for r in payload["records"]
                      if r["sequence"] == scene["sequence"] and
                      scene["start_us"] <= r["timestamp"] < scene["end_us"]]
            self.assertEqual(scene["timestamps_us"], stamps)
            self.assertEqual(scene["timestamps_us"][0], scene["start_us"])
            self.assertEqual(scene["frames"], len(stamps))
            self.assertGreaterEqual(scene[scene["kind"]+"_fraction"],0.8)
            self.assertEqual(scene["state_policy"], "continuous_from_sequence_start")
            self.assertGreaterEqual(scene["warmup_frames_before_start"],0)
        low, high = result["scenes"]
        self.assertTrue(low["end_us"] <= high["start_us"] or high["end_us"] <= low["start_us"])

    def test_mixed_frames_inside_low_scene_are_kept_not_filtered(self):
        eligible, _ = eligible_sequences("dsec","semantic")
        counts = [1]*13
        counts[5] = 4  # A middle-activity frame remains in the video timeline.
        path, payload, _ = self.manifest(custom={eligible[0]:rows(eligible[0],counts)})
        result = select_scenes(path,dataset="dsec",task="semantic",duration_seconds=1,min_fraction=.8)
        scene = next(s for s in result["scenes"] if s["kind"] == "low")
        self.assertIn(600_000,scene["timestamps_us"])
        self.assertEqual(scene["frames"],10)
        self.assertAlmostEqual(scene["low_fraction"],.9)

    def test_full_sequences_cover_all_evaluation_splits_and_preserve_every_frame(self):
        expected_counts = {("dsec", "semantic"): 3, ("dsec", "depth"): 4,
                           ("m3ed", "semantic"): 2, ("m3ed", "depth"): 2}
        for (dataset, task), count in expected_counts.items():
            with self.subTest(dataset=dataset, task=task):
                path, payload, eligible = self.manifest(dataset=dataset, task=task)
                plan = select_full_sequences(path, dataset=dataset, task=task)
                self.assertEqual(plan["status"], "complete")
                self.assertEqual(plan["counts"], {"full_sequence": count})
                self.assertEqual(plan["selection"]["eligible_sequences"], list(eligible))
                self.assertEqual([s["sequence"] for s in plan["scenes"]], list(eligible))
                for scene in plan["scenes"]:
                    records = [r for r in payload["records"] if r["sequence"] == scene["sequence"]]
                    self.assertEqual(scene["timestamps_us"], [r["timestamp"] for r in records])
                    self.assertEqual(scene["warmup_frames_before_start"], 0)
                    self.assertEqual(scene["start_us"], records[0]["timestamp"])
                    self.assertEqual(scene["end_us"], records[-1]["timestamp"] + records[-1]["window_us"])
                    shorter = dict(scene, timestamps_us=scene["timestamps_us"][:-1])
                    with self.assertRaisesRegex(ValueError, "every native input"):
                        validate_full_sequence(shorter, records)

    def test_full_sequence_keeps_long_observed_intervals_and_weights_activity_by_observed_time(self):
        first = eligible_sequences("dsec", "semantic")[0][0]
        data = rows(first, [0, 40, 9], durations=[50_000, 400_000, 100_000])
        path, _, _ = self.manifest(custom={first: data})
        plan = select_full_sequences(path, dataset="dsec", task="semantic")
        scene = plan["scenes"][0]
        self.assertEqual(scene["timestamps_us"], [50_000, 450_000, 550_000])
        self.assertEqual(scene["terminal_hold_us"], 100_000)
        self.assertEqual(scene["end_us"], 650_000)
        self.assertEqual(plan["selection"]["max_gap_seconds"], .4)
        self.assertAlmostEqual(scene["low_fraction"], 50_000/550_000)
        self.assertAlmostEqual(scene["high_fraction"], 500_000/550_000)

    def test_full_sequence_rejects_missing_input_interval(self):
        first = eligible_sequences("dsec", "semantic")[0][0]
        data = rows(first, [1, 2, 3])
        path, _, _ = self.manifest(custom={first: [data[0], data[2]]})
        with self.assertRaisesRegex(ValueError, "Discontinuous input timeline"):
            select_full_sequences(path, dataset="dsec", task="semantic")

    def test_full_sequences_cli_rejects_partial_window_options(self):
        path, _, _ = self.manifest()
        output = self.directory / "full.json"
        command = [sys.executable, "-S", str(ROOT / "tools/select_activity_scenes.py"),
                   "--activity-manifest", str(path), "--dataset", "dsec", "--task", "semantic",
                   "--full-sequences", "--output", str(output)]
        failed = subprocess.run([*command, "--per-kind", "1"], text=True, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(output.exists())
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["counts"]["full_sequence"], 3)

    def test_duration_weighting_differs_from_frame_fraction(self):
        data = rows("sequence",[1,4,4,10,10],[100_000,400_000,400_000,100_000,100_000])
        candidates = list(window_candidates(data,1_000_000,10,70))
        self.assertEqual(len(candidates),1)
        self.assertAlmostEqual(candidates[0]["low_fraction"],.8)
        self.assertAlmostEqual(candidates[0]["high_fraction"],.2)
        self.assertAlmostEqual(candidates[0]["mean_event_rate_hz_interval_weighted"],28)

    def test_partial_last_interval_uses_only_its_duration_overlap(self):
        data = rows("sequence",[1,4,6],[100_000,200_000,300_000])
        candidate = next(window_candidates(data,350_000,10,15))
        # [100000,450000): 200000 us at20 Hz +150000 us at20 Hz.
        self.assertEqual(candidate["start_us"],100_000)
        self.assertEqual(candidate["end_us"],450_000)
        self.assertAlmostEqual(candidate["high_fraction"],1)
        self.assertAlmostEqual(candidate["mean_event_rate_hz_interval_weighted"],20)

    def test_gaps_missing_records_and_sequence_boundaries_split_blocks(self):
        a = rows("a",[1]*7)
        b = rows("b",[1]*7)
        gapped = a[:3]+a[4:]+b
        blocks = continuous_blocks(gapped,250_000)
        self.assertEqual([len(b) for b in blocks],[3,3,7])
        self.assertFalse(any(list(window_candidates(b,1_000_000,10,70)) for b in blocks))
        long = rows("a",[1,1,10,1,1],[100_000,100_000,1_000_000,100_000,100_000])
        self.assertEqual([len(b) for b in continuous_blocks(long,250_000)],[2,2])

    def test_requested_ten_seconds_are_not_shortened_when_candidates_are_insufficient(self):
        path, _, _ = self.manifest()
        result = select_scenes(path,dataset="dsec",task="semantic")
        self.assertEqual(result["scenes"],[])
        self.assertEqual(result["status"],"insufficient_candidates")
        self.assertEqual(result["selection"]["duration_seconds"],10)
        self.assertEqual(len(result["shortfalls"]),2)

    def test_split_contracts_use_dsec_semantic_test_and_depth_val_m3ed_shared_val(self):
        ds_sem, _ = eligible_sequences("dsec","semantic")
        ds_depth, _ = eligible_sequences("dsec","depth")
        self.assertEqual(ds_sem,('zurich_city_13_a','zurich_city_14_c','zurich_city_15_a'))
        self.assertFalse(set(ds_sem)&set(ds_depth))
        self.assertEqual(len(ds_depth),4)
        self.assertEqual(eligible_sequences("m3ed","semantic")[0],
                         eligible_sequences("m3ed","depth")[0])
        path, _, _ = self.manifest(dataset="m3ed")
        result = select_scenes(path,dataset="m3ed",task="semantic",duration_seconds=1)
        self.assertTrue(result["scenes"])
        with self.assertRaises(ValueError):
            select_scenes(path,dataset="dsec",task="semantic",duration_seconds=1)

    def test_manifest_tampering_or_missing_task_sequence_fails_closed(self):
        path, payload, eligible = self.manifest()
        payload["records"][0]["event_count"]+=1
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            select_scenes(path,dataset="dsec",task="semantic",duration_seconds=1)
        path, payload, eligible = self.manifest()
        data=[r for r in payload["records"] if r["sequence"]!=eligible[0]]
        rebuilt=build_activity_manifest(data,["reference_train"],reference=payload["reference"],
            source_metadata_sha256={k:v for k,v in payload["source_metadata_sha256"].items() if k!=eligible[0]})
        path.write_text(json.dumps(rebuilt))
        with self.assertRaisesRegex(ValueError,"sequence missing"):
            select_scenes(path,dataset="dsec",task="semantic",duration_seconds=1)

    def test_bad_selection_arguments_do_not_create_output(self):
        path, _, _ = self.manifest()
        for change in ({"duration_seconds":0},{"duration_seconds":float('nan')},
                       {"min_fraction":0},{"min_fraction":1.1},{"per_kind":0},
                       {"per_kind":True},{"max_gap_seconds":0}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                select_scenes(path,dataset="dsec",task="semantic",**change)

    def test_cli_is_stdlib_only_and_preserves_existing_scene_manifest(self):
        path, _, _ = self.manifest()
        output = self.directory/'scenes.json'
        command=[sys.executable,'-S',str(ROOT/'tools/select_activity_scenes.py'),
                 '--activity-manifest',str(path),'--dataset','dsec','--task','semantic',
                 '--duration-seconds','1','--output',str(output)]
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)['counts'],{'low':1,'high':1})
        again = subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(again.returncode,0)
        self.assertEqual(output.read_bytes(),saved)
        probe = subprocess.run([sys.executable,'-S','-c',
            'import event_state.evaluation.activity_scenes; import sys; '
            'assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
            cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(probe.returncode,0,probe.stderr)

    def test_evaluation_lazy_exports_preserve_from_import_interface(self):
        module=importlib.import_module('event_state.evaluation')
        name='joint_pca_rgb'
        missing=object(); saved=module.__dict__.pop(name,missing)
        function=lambda value:value
        try:
            with patch.object(module,'import_module',return_value=SimpleNamespace(**{name:function})):
                scope={}
                exec('from event_state.evaluation import joint_pca_rgb',scope)
                self.assertIs(scope[name],function)
        finally:
            module.__dict__.pop(name,None)
            if saved is not missing:
                module.__dict__[name]=saved


if __name__ == '__main__':
    unittest.main()
