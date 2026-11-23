"""Verify both causal passes and native-timestamp output without checkpoints or a GPU."""
import csv
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from event_state.activity_quantiles import build_activity_manifest, canonical_sha256
from event_state.evaluation.activity_scenes import eligible_sequences, select_full_sequences
from tools import render_placement_feature_sequence as worker


class FeatureWorkerTests(unittest.TestCase):
    def test_worker_rechecks_its_assigned_gpu_before_rendering(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"):
            root = Path(folder)
            command = ["--scenes", str(root/"plan"), "--sequence", "sample", "--output", str(root/"video"),
                       "--worker-result", str(root/"result")]
            with (patch("tools.run_temporal_downstream.idle_gpus", return_value=()) as idle,
                  patch("tools.run_temporal_downstream.shared_gpus", side_effect=[[], ["2"]]) as shared,
                  patch.object(worker.time, "sleep") as sleep,
                  patch.object(worker, "render") as render):
                with self.assertRaises(RuntimeError):
                    worker.main(command)
                idle.assert_called_once_with(("2",))
                shared.assert_not_called()
                render.assert_not_called()
                worker.main([*command, "--share-gpus", "8192"])
                self.assertEqual(shared.call_args_list, [call(("2",), min_free_mib=8192)] * 2)
                sleep.assert_called_once_with(5)
                render.assert_called_once()
                self.assertEqual(idle.call_count, 1)

    def test_input_stream_cannot_skip_reorder_or_substitute_frames(self):
        records = [dict(sequence="S", timestamp=i+1, previous_timestamp=i, event_count=i) for i in range(3)]
        frames = [(dict(row, sequence_name="S"), {}) for row in records]
        for changed in (frames[:-1], frames[1:], frames+[frames[0]], list(reversed(frames)),
                        [(dict(frames[0][0], event_count=99), {}), *frames[1:]]):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Feature and activity"):
                list(worker.aligned_frames(SimpleNamespace(frames=lambda seq: iter(changed)), "S", records))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    def test_full_recording_is_encoded_with_all_inputs_and_missing_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seqs, _ = eligible_sequences("dsec", "semantic")
            sequence = seqs[0]
            stamps = [40000, 90000, 158000]
            all_records = [dict(sequence=seq, timestamp=t, previous_timestamp=prev, event_count=i,
                                window_us=t-prev, event_rate_hz=i*1e6/(t-prev))
                           for seq in ("train", *seqs)
                           for i, (prev, t) in enumerate(zip([0, *stamps[:-1]], stamps))]
            metadata = {"sequence": sequence}
            activity = root / "activity.json"
            worker.save(activity, build_activity_manifest(all_records, ["train"],
                reference=dict(role="train", name="fixture", manifest_sha256="a"*64, excluded_sequences=seqs),
                source_metadata_sha256={r["sequence"]: canonical_sha256(metadata) for r in all_records}))
            worker.save(root/sequence/"metadata.json", metadata)
            args = SimpleNamespace(scenes=root/"plan.json", sequence=sequence, output=root/"attempt",
                                   project_root=root, device="cpu", font=None, worker_result=root/"result.json")
            worker.save(args.scenes, select_full_sequences(activity, dataset="dsec", task="semantic"))
            conditions = {"E1": "baseline", "E1_b3_6_9_12": "four"}
            seg = [row for row in all_records if row["sequence"] == sequence]
            inputs = [(dict(row, events=None, sequence_name=sequence), row["timestamp"]) for row in seg]
            pair = SimpleNamespace(cfg=None, prepared_root=root, frames=Mock(side_effect=lambda seq: iter(inputs)),
                                   identity={k: {"backbone_sha256": str(i)*64} for i,k in enumerate(conditions)})
            runtime = SimpleNamespace(PlacementFeaturePair=Mock(return_value=pair), CONDITIONS=conditions,
                                      BLOCKS=(3, 6, 9, 12))
            projection = SimpleNamespace(to_dict=lambda: {"axes": "fixture"}, render=Mock(side_effect=lambda x: x))
            calibration = SimpleNamespace(add=Mock(), fit=lambda: projection)
            canvas = SimpleNamespace(size=(16, 16), tobytes=lambda: bytes(16*16*3))
            panels = SimpleNamespace(FeatureCalibration=lambda: calibration, calibration_indices=lambda n: (0, 2),
                                     draw_feature_frame=Mock(return_value=canvas), select_panel_font=lambda _: None)
            with patch.dict(sys.modules, {"event_state.evaluation.placement_features": runtime,
                                          "event_state.evaluation.placement_feature_panels": panels}), \
                 patch.object(worker, "event_frame"), \
                 patch.object(worker, "rgb_frame", side_effect=[object(), None, object()]):
                worker.render(args)
            self.assertEqual(pair.frames.call_count, 2)
            self.assertEqual([c.args for c in calibration.add.call_args_list], [(stamps[0], 0), (stamps[2], 2)])
            self.assertEqual([c.args[0] for c in projection.render.call_args_list], stamps)
            scene_id = f"dsec_semantic_{sequence}"
            dest = args.output / scene_id
            manifest = worker.read(args.output/"manifest.json")
            self.assertTrue((args.output/"_SUCCESS").is_file())
            self.assertEqual(worker.read(args.worker_result)["run"], str(args.output.resolve()))
            self.assertEqual(manifest["videos"][scene_id]["timing"]["encoded_packets"], len(stamps)+1)
            self.assertEqual(manifest["videos"][scene_id]["rgb_available_frames"], 2)
            self.assertEqual(manifest["video_sha256"], worker.sha256_file(dest/"comparison.mp4"))
            self.assertEqual(manifest["projection_sha256"], worker.sha256_file(dest/"projection.json"))
            with (dest/"frames.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r["source_timestamp_us"]) for r in rows], stamps)
            self.assertEqual([r["rgb_available"] for r in rows], ["True", "False", "True"])
