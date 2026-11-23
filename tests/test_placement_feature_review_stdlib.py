"""The review catalog must only publish complete, correctly identified paired feature movies."""
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import render_placement_feature_review as review
from tools import render_placement_feature_sequence as worker
from event_state.activity_quantiles import canonical_sha256


class PlacementReviewTests(unittest.TestCase):
    def test_resumes_valid_movies_but_rebuilds_changed_artifacts(self):
        for with_semantic in (True, False):
            with self.subTest(with_semantic=with_semantic):
                self.check_resume(with_semantic)

    def check_resume(self, with_semantic):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "videos"
            out.mkdir()
            args = SimpleNamespace(project_root=root, output=out, activity_dir=root,
                                   gpus=("1", "2"), font=None, share_gpus=None, with_semantic=with_semantic)
            checkpoints = review.checkpoint_paths("dsec", root)
            for key, path in checkpoints.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode())
            heads = {key: root / (key + "-head.pt") for key in review.CONDITIONS}
            for key, path in heads.items():
                path.write_bytes((key + "head").encode())
            scene = dict(id="dsec_semantic_sample", sequence="sample", start_us=100, end_us=120,
                         duration_us=20, timestamps_us=[100, 110])
            plan = dict(dataset="dsec", task="semantic", scenes=[scene])
            created = []

            def dispatch(jobs, gpus, *, on_complete, **kwargs):
                self.assertEqual(gpus, ("1", "2"))
                self.assertEqual(len(jobs), 1)
                job = jobs[0]
                cmd = kwargs["worker_command"](job, root/"status", root/"result")
                self.assertIn("--worker-result", cmd)
                # Exercise the actual worker CLI with the parent's options and physical GPU assignment.
                with (patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"),
                      patch("tools.run_temporal_downstream.idle_gpus",
                            return_value=("2",) if args.share_gpus is None else ()) as idle,
                      patch("tools.run_temporal_downstream.shared_gpus", return_value=["2"]) as shared,
                      patch.object(worker, "render") as render):
                    worker.main(cmd[3:])
                    render.assert_called_once()
                    self.assertEqual(render.call_args.args[0].with_semantic, with_semantic)
                    if args.share_gpus is None:
                        idle.assert_called_once_with(("2",))
                        shared.assert_not_called()
                    else:
                        shared.assert_called_once_with(("2",), min_free_mib=args.share_gpus)
                        idle.assert_not_called()
                run = job["run"]
                folder = run / scene["id"]
                folder.mkdir(parents=True)
                (folder/"comparison.mp4").write_bytes(b"test only")
                (folder/"projection.json").write_text("{}")
                with (folder/"frames.csv").open("w", newline="") as stream:
                    table = csv.writer(stream)
                    table.writerow(["source_timestamp_us", "video_time_s", "event_count", "window_us", "event_rate_hz"])
                    table.writerows([(100, 0, 2, 10, 200000), (110, .00001, 1, 10, 100000)])
                manifest = dict(feature_review_version=1, status="complete", task="semantic", dataset="dsec",
                                conditions=list(review.CONDITIONS), selection_sha256=canonical_sha256(plan),
                                checkpoint_sha256={k: review.sha256_file(p) for k,p in checkpoints.items()},
                                videos={scene["id"]: dict(original_frames=2)})
                if with_semantic:
                    manifest.update(feature_review_version=2, semantic_head_sha256={
                        k: review.sha256_file(p) for k, p in heads.items()})
                    manifest["videos"][scene["id"]].update(semantic_prediction_frames=2, gt_available_frames=1)
                for name, key in (("comparison.mp4", "video"), ("frames.csv", "frames_csv"), ("projection.json", "projection")):
                    manifest[key+"_sha256"] = review.sha256_file(folder/name)
                (run/"manifest.json").write_text(json.dumps(manifest))
                (run/"_SUCCESS").touch()
                on_complete(job["name"], dict(run=str(run)))
                created.append(folder)

            with ((out/".render.lock").open("a") as lock,
                  patch.object(review, "TASKS", (("dsec", "semantic"),)),
                  patch.object(review, "semantic_head_paths", return_value=heads),
                  patch.object(review, "prepare_plans", return_value=[(root/"plan.json", plan)]),
                  patch("event_state.evaluation.comparison_timing.verify_video_timestamps"),
                  patch.object(review, "run_parallel", side_effect=dispatch) as launch):
                review.render(args, lock)
                self.assertIs(launch.call_args.kwargs["idle_devices"], review.idle_gpus)
                args.share_gpus = 4096
                review.render(args, lock)
                self.assertEqual(launch.call_count, 1)
                (created[0]/"projection.json").write_text('{"tampered":true}')
                review.render(args, lock)
                self.assertEqual(launch.call_count, 2)
                self.assertIs(launch.call_args.kwargs["idle_devices"].func, review.shared_gpus)
                self.assertEqual(launch.call_args.kwargs["idle_devices"].keywords, {"min_free_mib": 4096})
                self.assertTrue((out/"_SUCCESS").exists())
                page = (out/"index.html").read_text()
                self.assertIn(created[-1].relative_to(out).as_posix(), page)
                self.assertNotIn(created[0].relative_to(out).as_posix(), page)
                (heads if with_semantic else checkpoints)["E1"].write_bytes(b"different run")
                with self.assertRaisesRegex(ValueError, "Inputs changed"):
                    review.render(args, lock)

    def test_plan_lists_five_semantic_recordings_without_output_or_gpu(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as out:
            review.main(["--project-root", tmp])
            for ds, task in review.TASKS:
                for seq in review.eligible_sequences(ds, task)[0]:
                    self.assertIn(seq, out.getvalue())
            self.assertFalse((Path(tmp)/"outputs").exists())
            review.main(["--project-root", tmp, "--with-semantic"])
            self.assertIn("placement_feature_semantic_review_20261123", out.getvalue())
            self.assertFalse((Path(tmp)/"outputs").exists())

    def test_shared_gpus_recheck_free_vram_and_only_consider_requested_devices(self):
        inventories = ["0, 10000\n1, 4095\n2, 4096", "0, 10000\n1, 7050\n2, 4096", "1, 14\n2, 20"]
        requested = ("2", "1")
        with patch.object(review.subprocess, "run", side_effect=[SimpleNamespace(stdout=s) for s in inventories]) as smi:
            for expected in (["2"], ["1", "2"], []):
                self.assertEqual(review.shared_gpus(requested, min_free_mib=4096), expected)
            self.assertTrue(all("--query-gpu=index,memory.free" in call.args[0] for call in smi.call_args_list))
        with patch.object(review.subprocess, "run", return_value=SimpleNamespace(stdout="1, 10000")):
            with self.assertRaisesRegex(ValueError, "Unknown physical GPU"):
                review.shared_gpus(requested, min_free_mib=4096)
