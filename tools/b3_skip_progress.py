"""Report remaining training and evaluation work for the residual placement experiments."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.b3_semantic_timing import semantic_remaining
from tools.common_depth_parallel import parse_gpus
from tools.placement_conditions import SKIP_CONDITION, SKIP_CONTROLS, SKIP_PLACEMENT_SUITE, FINAL_SKIP_SUITE, placement_label
from tools.progress_eta import format_eta
from tools.stage1_progress import read_stage1_progress


PROJECT = Path(__file__).resolve().parents[1]
RATE_RECORD = PROJECT / "docs/evaluation_records/20261123_stage1_placement_train_loss.json"
DATASETS = ("dsec", "m3ed")


def historical_update_times():
    records = json.loads(RATE_RECORD.read_text())["results"]
    clips = {"dsec": 8, "m3ed": 4}  # Joined random and stream batches on one GPU.
    return {(r["dataset"], r["condition"]): clips[r["dataset"]] / r["samples_per_second"] for r in records}


def stage1_remaining(root, dataset, target, reference_time, now, condition=SKIP_CONDITION, reference="E1_b3"):
    name = f"{dataset}_{condition}_w1"
    checkpoint = root / name / "checkpoints" / f"step_{target:08d}.pt"
    if (root / "_SUCCESS").is_file() or (checkpoint.is_file() and checkpoint.stat().st_size):
        return 0, "更新完了"
    log = root / "logs" / f"{name}.log"
    progress = read_stage1_progress(log)
    step = progress.step if progress else 0
    rate = progress.seconds_per_update if progress else None
    detail = f"{step:,}/{target:,}更新"
    if progress and progress.loss is not None:
        detail += f" loss={progress.loss:.4g}"
    if progress:
        age = now - log.stat().st_mtime
        if age > 600:
            return None, f"{detail}、ログ更新から{int(age / 60)}分（残り時間を再計測待ち）"
    source = "今回の速度" if rate else f"過去の{reference}の速度"
    remaining = max(target - step, 0) * (rate or reference_time)
    return remaining, f"{detail}、{source}で残り{remaining / 3600:.2f}時間"


def queued_seconds(durations, slots):
    """Project the launcher's ordered, one-GPU-per-job queue with the requested slots."""
    loads = [0.] * slots
    for seconds in durations:
        free = loads.index(min(loads))
        loads[free] += seconds
    return max(loads)


def snapshot(outputs, gpu_count, now, *, suite=False, final=False):
    conditions = SKIP_PLACEMENT_SUITE if suite else (SKIP_CONDITION,)
    tag, date = ("skip_placement", "20261128") if suite else ("b3_skip", "20261126")
    if final:
        conditions, tag, date = FINAL_SKIP_SUITE, "final_skip", "20261128"
    semantic_root = outputs / f"stage2_{tag}_semantic_{date}"
    title = "[全工程] 6・9・3+6+9層後のスキップ接続と3+6+9層後の置換対照" if suite else (
        f"[全工程] {SKIP_CONDITION} {placement_label(SKIP_CONDITION)}")
    if final:
        title = "[全工程] 12層後と3+6+9+12層後のスキップ接続"
    if (semantic_root / "_SUCCESS").is_file():
        return [f"{title} | 完了 | {semantic_root / 'summary.json'}"], True

    lines = [title]
    stages, unknown = [], []
    slots = min(len(DATASETS) * len(conditions), gpu_count)
    references = historical_update_times()
    for stage, target in (("smoke", 99), ("full", 99999)):
        root = outputs / f"stage1_{tag}_{stage}_{date}"
        durations = []
        for ds in DATASETS:
            for cond in conditions:
                comparable = SKIP_CONTROLS.get(cond, cond)
                if (ds, comparable) not in references:
                    comparable = "E1_b6_9_12"  # Three LSTMs on the same GPU and dataset.
                seconds, detail = stage1_remaining(root, ds, target, references[ds, comparable], now, cond, comparable)
                durations.append(seconds or 0)
                if seconds is None:
                    unknown.append(f"Stage 1 {stage} {ds.upper()} {cond}")
                lines.append(f"  Stage 1 {stage} {ds.upper()} | {cond} {placement_label(cond)} | {detail}")
        stages.append(durations)
    stage1_end = sum(queued_seconds(stage, slots) for stage in stages)
    if stage1_end > 0 and not unknown:
        lines.append(f"  Stage 1本学習終了まで（GPU待ちを除く） {format_eta(stage1_end)}")
    semantic = []
    for cond in conditions:
        for dataset in DATASETS:
            seconds, missing, detail = semantic_remaining(outputs, dataset, now, condition=cond, root=semantic_root)
            estimate = f"、見積可能分の残り{seconds / 3600:.2f}時間" if seconds > 0 else ""
            lines.append(f"  Semantic {dataset.upper()} | {cond} {placement_label(cond)} | {detail}{estimate}")
            semantic.append(seconds)
            if missing:
                unknown.append(f"{dataset.upper()} {missing}" + (f" ({cond})" if suite else ""))
    stages.append(semantic)

    # The shell waits for both datasets before advancing to the next stage.
    parallel = sum(queued_seconds(stage, slots) for stage in stages)
    serial = sum(sum(stage) for stage in stages)
    scope = "見積可能な工程の残り" if unknown else "全工程の残り"
    seconds = parallel
    projection = f"約{seconds / 3600:.1f}時間" if unknown else format_eta(seconds)
    lines.append(f"  {scope}（{slots} GPU使用、待ち時間を除く） {projection}")
    if slots > 1 and serial > parallel:
        lines.append(f"  以降1 GPUで順次処理する場合は約{serial / 3600:.1f}時間")
    if unknown:
        lines.append("  全体の終了時刻は未確定。計測待ちの工程：" + " / ".join(unknown))
    return lines, False


def launch_and_follow(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as writer, log.open("rb") as reader:
        reader.seek(0, os.SEEK_END)
        job = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=writer,
                               stderr=subprocess.STDOUT, start_new_session=True)
        print(f"PID={job.pid}  log={log}  (Ctrl+Cでログ表示を終了、学習は継続)", flush=True)
        while True:
            chunk = reader.read(65536)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                continue
            code = job.poll()
            if code is not None:
                sys.stdout.buffer.write(reader.read())
                sys.stdout.buffer.flush()
                return code if code >= 0 else 128 - code
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2"))
    parser.add_argument("--skip-placement-suite", action="store_true")
    parser.add_argument("--final-skip-suite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="Refresh until the completion marker appears")
    mode.add_argument("--follow", action="store_true", help="Launch in a separate session and follow new output")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    if args.follow:
        script, log = ("run_skip_placements.sh", "skip_placement_20261128.log") if args.skip_placement_suite else (
            "run_b3_skip.sh", "b3_skip_20261126.log")
        if args.final_skip_suite:
            script, log = "run_final_skip.sh", "final_skip_20261128.log"
        command = ["bash", f"tools/{script}", ",".join(args.gpus)]
        return launch_and_follow(command, PROJECT / "outputs" / log)
    while args.parent_pid is None or os.getppid() == args.parent_pid:
        now = time.time()
        try:
            lines, done = snapshot(PROJECT / "outputs", len(args.gpus), now, suite=args.skip_placement_suite, final=args.final_skip_suite)
        except (OSError, ValueError, KeyError) as error:
            if not args.watch:
                raise
            lines, done = [f"[全工程] 進捗を取得できませんでした。次回再確認します ({error})"], False
        stamp = datetime.fromtimestamp(now, timezone(timedelta(hours=9)))
        print(f"{stamp:%m/%d %H:%M JST} " + "\n".join(lines), flush=True)
        if done or not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
