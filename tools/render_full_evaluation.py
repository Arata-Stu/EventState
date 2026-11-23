#!/usr/bin/env python3
"""Render every common-evaluation recording and build an offline video review page."""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_quantiles import canonical_sha256
from event_state.evaluation.activity_scenes import select_full_sequences


CONDITIONS = {
    "E0": "E0 Encoder（z蒸留）",
    "E1": "E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM",
    "E1_convlstm": "E1 Encoder＋LSTM（h蒸留）・ConvLSTM",
}
TASKS = (("dsec", "semantic"), ("dsec", "depth"), ("m3ed", "semantic"), ("m3ed", "depth"))


def publish(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_plans(output, activity_dir, tasks=TASKS):
    directory = output / "plans"
    directory.mkdir(exist_ok=True)
    plans = []
    for dataset, task in tasks:
        path = directory / f"{dataset}_{task}.json"
        plan = select_full_sequences(activity_dir / f"{dataset}.json", dataset=dataset, task=task)
        if path.exists():
            if json.loads(path.read_text()) != plan:
                raise ValueError(f"The source timeline changed since the previous render: {path}")
        else:
            publish(path, json.dumps(plan, ensure_ascii=False, allow_nan=False) + "\n")
        plans.append((path, plan))
    return plans


def review_video(root, run, plan, scene, conditions=CONDITIONS):
    from event_state.evaluation.comparison_timing import verify_video_timestamps

    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["dataset"] != plan["dataset"]
            or manifest["task"] != plan["task"] or manifest["conditions"] != list(conditions)
            or manifest["selection_sha256"] != canonical_sha256(plan)
            or set(manifest["videos"]) != {scene["id"]} or not (run / "_SUCCESS").is_file()):
        raise ValueError(f"Incomplete video or mismatched source/conditions: {run}")
    folder = run / scene["id"]
    video, table = folder / "comparison.mp4", folder / "frames.csv"
    counts, windows, rates, stamps = [], [], [], []
    with table.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stamp, count, window = map(int, (row["source_timestamp_us"], row["event_count"], row["window_us"]))
            rate = float(row["event_rate_hz"])
            if count < 0 or window <= 0 or not math.isclose(rate, count * 1e6 / window, rel_tol=1e-9):
                raise ValueError(f"Inconsistent event activity at {table}, timestamp {stamp}")
            if not math.isclose(float(row["video_time_s"]), (stamp-scene["start_us"])/1e6, abs_tol=0.0000005, rel_tol=0):
                raise ValueError(f"CSV and native video clock disagree at {stamp}")
            stamps.append(stamp)
            counts.append(count)
            windows.append(window)
            rates.append(rate)
    if stamps != scene["timestamps_us"] or manifest["videos"][scene["id"]]["original_frames"] != len(stamps):
        raise ValueError(f"Expected every input frame in source order: {table}")
    verify_video_timestamps(video, stamps, scene["end_us"])
    return dict(dataset=plan["dataset"], task=plan["task"], sequence=scene["sequence"],
                src=video.relative_to(root).as_posix(), frames_csv=table.relative_to(root).as_posix(),
                duration_seconds=scene["duration_us"]/1e6, timestamps_us=stamps,
                event_counts=counts, window_us=windows, rates_hz=rates)


def review_page(root, template, videos, conditions=CONDITIONS, *, title=None, note=None):
    # Inline the data so a downloaded directory opens directly from file://.
    data = json.dumps(dict(conditions=list(conditions.values()), videos=videos, title=title, note=note),
                      ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    publish(root / "index.html", template.replace("__REVIEW_DATA__", data))


def run_all(args, lock):
    template = (ROOT / "tools/templates/full_sequence_review.html").read_text(encoding="utf-8")
    if template.count("__REVIEW_DATA__") != 1:
        raise ValueError("Missing review data slot in HTML template")
    plans = prepare_plans(args.output, args.activity_dir)
    videos = []
    total = sum(len(plan["scenes"]) for _, plan in plans)
    print(f"全{total}本 | " + " / ".join(CONDITIONS.values()), flush=True)
    for path, plan in plans:
        for scene in plan["scenes"]:
            parent = args.output / plan["dataset"] / plan["task"] / scene["sequence"]
            parent.mkdir(exist_ok=True, parents=True)
            attempts = sorted(parent.glob("render-*"))
            entry = None
            for previous in reversed(attempts):
                if not (previous / "_SUCCESS").is_file():
                    continue
                try:
                    entry = review_video(args.output, previous, plan, scene)
                    break
                except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
                    print(f"Previously generated video failed validation: {previous}: {error}", flush=True)
            if entry is None:
                number = max((int(p.name.removeprefix("render-")) for p in attempts), default=0) + 1
                run = parent / f"render-{number:03d}"
                cmd = [sys.executable, "-u", str(ROOT / "tools/render_activity_comparison.py"),
                       "--scenes", str(path), "--task", plan["task"], "--sequence", scene["sequence"],
                       "--conditions", ",".join(CONDITIONS), "--output", str(run), "--device", args.device]
                if args.font:
                    cmd.extend(("--font", str(args.font)))
                print(f"[{len(videos)+1}/{total}] {scene['id']} | {scene['frames']} frames", flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
                entry = review_video(args.output, run, plan, scene)
            videos.append(entry)
            review_page(args.output, template, videos)
            print(f"[{len(videos)}/{total}] 検証済み {scene['id']} | {args.output/'index.html'}", flush=True)
    print(f"Completed: {args.output/'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-dir", type=Path, default=ROOT/"outputs/semantic_depth_activity_20261110/manifests")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/full_sequence_review_20261118")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "index.html").exists() and not (args.output / "plans/dsec_semantic.json").is_file():
        parser.exit(1, f"An existing index.html belongs to a different output: {args.output}\n")
    with (args.output/".render.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"Another renderer is using {args.output}\n")
        run_all(args, lock)


if __name__ == "__main__":
    main()
