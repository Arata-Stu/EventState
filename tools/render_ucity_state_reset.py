#!/usr/bin/env python3
"""Compare retained and once-reset E1 Semantic predictions on UCity's 50.64 s Q1 episode."""
import argparse
import csv
from contextlib import closing
from itertools import chain
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_quantiles import ActivityIndex
from event_state.activity_reporting import validate_activity_source
from event_state.evaluation.comparison_semantic import SemanticBackend
from event_state.evaluation.comparison_timing import ffmpeg_executable, verify_video_timestamps
from event_state.evaluation.full_video import encode_native_video
from event_state.evaluation.semantic_reset_replay import replay_pair
from event_state.evaluation.ucity_reset_scene import SEQUENCE, make_scene
from tools.common_depth_parallel import parse_gpus
from tools.render_activity_comparison import activity_chart, cjk_font, draw_frame, event_frame, read, rgb_frame, save
from tools.run_temporal_downstream import idle_gpus

SELECTION = ROOT / "docs/evaluation_records/20261125_semantic_low_activity_intervals.json"
CONDITION = "E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM"
LABELS = (CONDITION + "・状態保持", CONDITION + "・低活動開始時に1回リセット")


def paired_canvases(backend, records, scene, cutpoints, font, table):
    """Replay to the end of the displayed clip; encode both predictions on one clock."""
    import numpy as np
    from event_state.dense.source import prepared_frames

    height, width = backend.frame_size
    scored = backend.scored_frame_indices.get(SEQUENCE)
    ymax = 1.05 * max(cutpoints[-1], max(r["event_rate_hz"] for r in records
                                         if scene["start_us"] <= r["timestamp"] < scene["end_us"]))
    chart = activity_chart(3 * width, scene, records, cutpoints, ymax)
    table.writerow(["source_timestamp_us", "video_time_s", "window_start_us", "event_count",
                    "window_us", "event_rate_hz", "activity_group", "reset_before_this_input",
                    "gt_available", "valid_pixels", "continuous_frame_mIoU", "reset_frame_mIoU",
                    "prediction_disagreement_fraction"])

    def inputs():
        for frame in prepared_frames(backend.prepared_root, SEQUENCE, backend.cfg, "m3ed"):
            if frame["frame_index"] % 500 == 0:
                print(f"{CONDITION} | 系列先頭から {frame['frame_index']} 入力、時刻 {frame['timestamp']} µs", flush=True)
            yield frame

    for frame, row, continuous, reset in replay_pair(backend, inputs(), records, scene):
        stamp = row["timestamp"]
        if continuous.shape != (height, width) or reset.shape != (height, width):
            raise ValueError(f"Semantic prediction geometry differs at {stamp}")
        target = backend.ground_truth(SEQUENCE, stamp)
        if scored is not None and (frame["frame_index"] in scored) != (target is not None):
            raise ValueError(f"Pseudo-label coverage differs from the original evaluation at {stamp}")
        valid = target != 255 if target is not None else np.zeros((height, width), dtype=bool)
        different = continuous != reset
        fraction = float(different.mean())
        at_reset = stamp == scene["reset_input_us"]
        if stamp < scene["reset_input_us"]:
            phase = "介入前・共通の履歴"
        elif at_reset:
            phase = "この入力窓の直前にリセット"
        elif stamp <= scene["q1_end_us"]:
            phase = "対象の低活動区間"
        else:
            phase = "対象区間終了後・両方とも状態を継続"
        diff_panel = (f"予測クラスの不一致（白） {fraction:.1%} | {phase}",
                      np.repeat((different.astype(np.uint8) * 255)[:, :, None], 3, axis=2))
        canvas, metrics = draw_frame(
            "semantic", rgb_frame("m3ed", "semantic", backend.cfg, backend.prepared_root, SEQUENCE, stamp),
            event_frame(frame["events"], "m3ed", "semantic"), target, valid, (continuous, reset), LABELS,
            scene, stamp, row, records, cutpoints, font, None, ymax, chart=chart, extra_panel=diff_panel,
            target_label="InternImage 疑似ラベル（無効画素は灰色）")
        table.writerow([stamp, (stamp - scene["start_us"]) / 1e6, row["previous_timestamp"],
                        row["event_count"], row["window_us"], row["event_rate_hz"], row["group"],
                        at_reset, target is not None, int(valid.sum()), *metrics, fraction])
        yield canvas


def render(args):
    output = args.output.expanduser().resolve()
    selection = read(SELECTION)
    expected = selection["e1_reference"]
    activity = ActivityIndex.load(args.activity_manifest)
    identity = activity.identity
    if identity["manifest_sha256"] != expected["activity_manifest_sha256"]:
        raise ValueError("Activity manifest differs from the original UCity review")
    records = [activity.lookup(SEQUENCE, t) for t in activity.timestamps(SEQUENCE)]
    scene = make_scene(selection, records)
    cutpoints = [c["event_rate_hz"] for c in identity["cutpoints"]]
    if not args.execute:
        print(f"{CONDITION} | {SEQUENCE} | 表示 {scene['frames']} コマ、"
              f"リセットする入力窓 ({scene['q1_start_us']}, {scene['reset_input_us']}] µs")
        return
    if output.exists():
        raise FileExistsError(f"既存の出力があります。別の --output を指定してください: {output}")
    ffmpeg, font = ffmpeg_executable(), cjk_font(args.font)
    available = idle_gpus(args.gpus)
    if not available:
        print(f"空きGPU待ち: {', '.join(args.gpus)}", flush=True)
    while not available:
        time.sleep(10)
        available = idle_gpus(args.gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = available[0]
    print(f"GPU {available[0]} | {CONDITION} | 表示前の {scene['warmup_frames_before_start']} 入力から再生", flush=True)
    with closing(SemanticBackend("m3ed", "E1", ROOT, args.teacher_checkpoint, "cuda")) as backend:
        if any(backend.identity[k] != expected[k] for k in ("backbone_sha256", "head_sha256")):
            raise ValueError("The E1 backbone or Semantic head differs from the reference video")
        validate_activity_source(activity, SEQUENCE, read(backend.prepared_root / SEQUENCE / "metadata.json"))
        stamps = scene["timestamps_us"]
        if not set(stamps).issubset(backend.targets.labeled_timestamps(SEQUENCE)):
            raise ValueError("Missing M3ED Semantic pseudo-label timestamps in the UCity clip")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".ucity-reset-", dir=output.parent) as tmp:
            dest = Path(tmp) / "result"
            dest.mkdir()
            video = dest / "comparison.mp4"
            with (dest / "frames.csv").open("x", newline="", encoding="utf-8") as stream:
                canvases = paired_canvases(backend, records, scene, cutpoints, font, csv.writer(stream))
                first = next(canvases)
                rgb = (image.tobytes() for image in chain((first,), canvases))
                encode_native_video(ffmpeg, video, stamps, scene["end_us"], first.size, rgb)
            timing = verify_video_timestamps(video, stamps, scene["end_us"])
            save(dest / "manifest.json", dict(sequence=SEQUENCE, split="M3ED public validation",
                 model=backend.identity, selection=str(SELECTION), source_activity_manifest_sha256=identity["manifest_sha256"],
                 scene=scene, timing=timing, state_policies=["continuous_from_sequence_start",
                 "shared_prefix_then_clear_h_and_c_once_before_reset_input"], comparison_video=video.name))
            (dest / "_SUCCESS").touch()
            dest.rename(output)
    print(f"Completed: {output / 'comparison.mp4'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-manifest", type=Path,
                        default=ROOT / "outputs/semantic_depth_activity_20261110/manifests/m3ed.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ucity_e1_onset_reset_20261125")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2"),
                        help="Physical GPU IDs to choose one idle GPU from")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true", help="Render the video (default: inspect the scene)")
    args = parser.parse_args()
    if args.execute:
        try:
            args.gpus = parse_gpus(args.gpus)
        except ValueError as error:
            parser.error(str(error))
    render(args)


if __name__ == "__main__":
    main()
