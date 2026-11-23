#!/usr/bin/env python3
"""Render one native-timestamp Semantic evaluation recording for two frozen LSTM placements."""
import argparse
import csv
from itertools import chain, zip_longest
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_quantiles import ActivityIndex, canonical_sha256
from event_state.activity_reporting import validate_activity_source
from event_state.dense.protocol import sha256_file
from event_state.evaluation.activity_scenes import eligible_sequences, validate_full_sequence
from event_state.evaluation.comparison_timing import ffmpeg_executable, verify_video_timestamps
from event_state.evaluation.full_video import encode_native_video
from tools.render_activity_comparison import event_frame, rgb_frame
from tools.run_placement_semantic import read, save


def select_recording(plan, sequence):
    dataset = plan["dataset"]
    allowed, split = eligible_sequences(dataset, "semantic")
    if (plan["task"] != "semantic" or plan["status"] != "complete" or plan["shortfalls"]
            or plan["selection"]["mode"] != "full_sequences"
            or plan["selection"]["split_source_sha256"] != sha256_file(split)
            or sorted(scene["sequence"] for scene in plan["scenes"]) != sorted(allowed)
            or sequence not in allowed):
        raise ValueError("Expected a complete, current full-sequence Semantic evaluation plan")
    scene = next(s for s in plan["scenes"] if s["sequence"] == sequence)
    if scene["id"] != f"{dataset}_semantic_{sequence}":
        raise ValueError(f"Unexpected scene identity: {scene['id']}")
    activity = ActivityIndex.load(plan["activity_manifest"])
    identity = activity.identity
    if (identity["manifest_sha256"] != plan["activity_manifest_sha256"]
            or identity["source_contract"]["dataset"] != dataset):
        raise ValueError("Selection and event activity manifest do not match")
    records = [activity.lookup(sequence, t) for t in activity.timestamps(sequence)]
    validate_full_sequence(scene, records)
    return scene, activity, records


def aligned_frames(pair, sequence, records):
    """Restart both models at the recording origin; reject missing, swapped or extra inputs."""
    for i, (sample, row) in enumerate(zip_longest(pair.frames(sequence), records)):
        if sample is None or row is None:
            raise ValueError(f"Feature and activity frame counts differ at {sequence} frame {i}")
        frame, maps = sample
        if (frame["sequence_name"] != sequence or any(frame[k] != row[k] for k in
                ("timestamp", "previous_timestamp", "event_count"))):
            raise ValueError(f"Feature and activity inputs differ at {sequence} frame {i}")
        yield i, frame, maps, row


def render(args):
    from event_state.evaluation.placement_features import PlacementFeaturePair, CONDITIONS, BLOCKS
    from event_state.evaluation.placement_feature_panels import (
        calibration_indices, draw_feature_frame, FeatureCalibration, select_panel_font,
    )

    plan = read(args.scenes)
    dataset, sequence = plan["dataset"], args.sequence
    scene, activity, records = select_recording(plan, sequence)
    font = select_panel_font(args.font)
    ffmpeg = ffmpeg_executable()
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / scene["id"]
    destination.mkdir()
    pair = PlacementFeaturePair(dataset, args.project_root, device=args.device)
    validate_activity_source(activity, sequence, read(pair.prepared_root / sequence / "metadata.json"))
    del activity
    print(f"[{sequence}] {len(records)} inputs | " + " / ".join(CONDITIONS.values()), flush=True)

    indices = set(calibration_indices(len(records)))
    calibration = FeatureCalibration()
    for i, frame, maps, row in aligned_frames(pair, sequence, records):
        if i in indices:
            calibration.add(maps, i)
        if (i + 1) % 500 == 0:
            print(f"[{sequence}] PCA calibration {i+1}/{len(records)}", flush=True)
    projection = calibration.fit()
    del calibration
    projection_path = destination / "projection.json"
    save(projection_path, projection.to_dict())

    rgb_count = 0
    csv_path = destination / "frames.csv"
    video = destination / "comparison.mp4"
    with csv_path.open("x", encoding="utf-8", newline="") as out:
        table = csv.writer(out)
        table.writerow(("source_timestamp_us", "video_time_s", "event_count", "window_us",
                        "event_rate_hz", "activity_group", "rgb_available"))

        def canvases():
            nonlocal rgb_count
            for i, frame, maps, row in aligned_frames(pair, sequence, records):
                timestamp = row["timestamp"]
                elapsed = (timestamp - scene["start_us"]) / 1e6
                rgb = rgb_frame(dataset, "semantic", pair.cfg, pair.prepared_root, sequence, timestamp,
                                missing_ok=True)
                rgb_count += int(rgb is not None)
                canvas = draw_feature_frame(
                    projection.render(maps), dataset=dataset, sequence=sequence, timestamp_us=timestamp,
                    elapsed_s=elapsed, event_rate_hz=row["event_rate_hz"], rgb=rgb, font_path=font,
                    events=event_frame(frame["events"], dataset, "semantic"),
                )
                table.writerow((timestamp, elapsed, row["event_count"], row["window_us"],
                                row["event_rate_hz"], row["group"], rgb is not None))
                if (i + 1) % 100 == 0:
                    print(f"[{sequence}] rendering {i+1}/{len(records)}", flush=True)
                yield canvas

        frames = canvases()
        first = next(frames)
        encode_native_video(ffmpeg, video, scene["timestamps_us"], scene["end_us"], first.size,
                            (canvas.tobytes() for canvas in chain((first,), frames)))

    timing = verify_video_timestamps(video, scene["timestamps_us"], scene["end_us"])
    manifest = dict(
        status="complete", feature_review_version=1, dataset=dataset, task="semantic",
        conditions=list(CONDITIONS), blocks=list(BLOCKS), models=pair.identity,
        checkpoint_sha256={k: pair.identity[k]["backbone_sha256"] for k in CONDITIONS},
        selection_sha256=canonical_sha256(plan), activity_manifest_sha256=plan["activity_manifest_sha256"],
        projection_sha256=sha256_file(projection_path), frames_csv_sha256=sha256_file(csv_path),
        panel_language="ja" if font else "en",
        video_sha256=sha256_file(video),
        videos={scene["id"]: dict(original_frames=len(records), video_size=list(first.size),
                                  rgb_available_frames=rgb_count, timing=timing)},
    )
    save(args.output / "manifest.json", manifest)
    (args.output / "_SUCCESS").touch(exist_ok=False)
    if args.worker_result:
        save(args.worker_result, {"run": str(args.output.resolve())})
    print(f"[{sequence}] verified {len(records)} frames: {video}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--font", type=Path)
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Existing render attempt: {args.output}; choose a fresh output")
    if args.worker_result:
        from tools.run_temporal_downstream import idle_gpus
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if args.device != "cuda" or not gpu.isdecimal() or not idle_gpus((gpu,)):
            raise RuntimeError(f"Assigned physical GPU is unavailable: {gpu}")
    render(args)


if __name__ == "__main__":
    main()
