#!/usr/bin/env python3
"""Render synchronized block features for the two trained Semantic placements."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.dense.protocol import sha256_file
from event_state.evaluation.activity_scenes import eligible_sequences
from event_state.evaluation.placement_features import CONDITIONS, checkpoint_paths
from tools.common_depth_parallel import parse_gpus, run_parallel
from tools.render_full_evaluation import prepare_plans, publish, review_page, review_video
from tools.run_temporal_downstream import idle_gpus

TASKS = (("dsec", "semantic"), ("m3ed", "semantic"))
TITLE = "LSTM配置と特徴の変化"
NOTE = ("特徴の色は同じ系列・同じ層の2モデルで共通のPCAです。層や系列が変わると色の対応も変わります。"
        "色は意味クラスを表しません。")


def checked_entry(output, run, plan, scene, checkpoints):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("feature_review_version") != 1 or manifest.get("checkpoint_sha256") != checkpoints:
        raise ValueError(f"Different feature protocol or checkpoint: {run}")
    folder = run / scene["id"]
    for file, key in (("projection.json", "projection_sha256"), ("frames.csv", "frames_csv_sha256"),
                      ("comparison.mp4", "video_sha256")):
        if sha256_file(folder / file) != manifest[key]:
            raise ValueError(f"Changed or incomplete artifact: {folder / file}")
    return review_video(output, run, plan, scene, CONDITIONS)


def render(args, lock):
    output = args.output
    identity_path = output / "placement_features.json"
    if not identity_path.exists() and any(p.name != ".render.lock" for p in output.iterdir()):
        raise ValueError(f"Choose a fresh directory for the placement feature review: {output}")
    checkpoints = {ds: {key: sha256_file(path) for key, path in checkpoint_paths(ds, args.project_root).items()}
                   for ds, _ in TASKS}
    identity = dict(feature_review_version=1, checkpoint_sha256=checkpoints, activity_dir=str(args.activity_dir))
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"Inputs changed; choose another --output: {output}")
    else:
        publish(identity_path, json.dumps(identity, indent=2))
    plans = prepare_plans(output, args.activity_dir, TASKS)
    template = (ROOT / "tools/templates/full_sequence_review.html").read_text()
    entries, sources, pending = {}, {}, []
    for path, plan in plans:
        ds = plan["dataset"]
        for scene in plan["scenes"]:
            key = scene["id"]
            sources[key] = (plan, scene)
            parent = output / ds / scene["sequence"]
            parent.mkdir(parents=True, exist_ok=True)
            attempts = sorted(parent.glob("render-*"))
            for run in reversed(attempts):
                if (run / "_SUCCESS").is_file():
                    try:
                        entries[key] = checked_entry(output, run, plan, scene, checkpoints[ds])
                        break
                    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                        print(f"Previous video {run}: {error}", flush=True)
            if key not in entries:
                dest = parent / ("render-" + uuid4().hex)
                pending.append(dict(name=key, kind="features", remaining_cache_bytes=0,
                                    label=" / ".join(CONDITIONS.values()), plan=path,
                                    run=dest, sequence=scene["sequence"]))

    def write_index():
        review_page(output, template, [entries[k] for k in sources if k in entries], CONDITIONS,
                    title=TITLE, note=NOTE)

    (output / "_SUCCESS").unlink(missing_ok=True)
    write_index()
    if pending:
        def command(job, _status, result):
            cmd = [sys.executable, "-u", str(ROOT / "tools/render_placement_feature_sequence.py"),
                   "--scenes", str(job["plan"]), "--sequence", job["sequence"], "--output", str(job["run"]),
                   "--project-root", str(args.project_root), "--device", "cuda", "--worker-result", str(result)]
            if args.font:
                cmd += ["--font", str(args.font)]
            return cmd

        def completed(key, result):
            plan, scene = sources[key]
            entries[key] = checked_entry(output, Path(result["run"]), plan, scene, checkpoints[plan["dataset"]])
            write_index()
            print(f"[{len(entries)}/{len(sources)}] {scene['sequence']} → {output / 'index.html'}", flush=True)

        run_parallel(pending, args.gpus, scratch=output, state_dir=output / "logs" / uuid4().hex,
                     worker_command=command, on_complete=completed, idle_devices=idle_gpus,
                     lock_fd=lock.fileno(), poll_interval=5)
    if set(entries) != set(sources):
        raise RuntimeError("Incomplete full-sequence feature review")
    publish(output / "_SUCCESS", f"{len(entries)} verified videos\n")
    print(f"Completed {len(entries)} videos: {output / 'index.html'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--activity-dir", type=Path)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--execute", action="store_true", help="Generate the full videos on the learning server")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    outputs = args.project_root / "outputs"
    args.output = (args.output or outputs / "placement_feature_review_20261123").expanduser().resolve()
    args.activity_dir = (args.activity_dir or outputs / "semantic_depth_activity_20261110/manifests").expanduser().resolve()
    args.gpus = parse_gpus(args.gpus)
    for ds, task in TASKS:
        names, _ = eligible_sequences(ds, task)
        print(f"{ds} {task} | {', '.join(names)}", flush=True)
    print(" / ".join(CONDITIONS.values()), flush=True)
    if not args.execute:
        print(f"Output: {args.output} | use --execute to render")
        return
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".render.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"A renderer already owns {args.output}\n")
        render(args, lock)


if __name__ == "__main__":
    main()
