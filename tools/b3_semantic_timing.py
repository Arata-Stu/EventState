"""Estimate Semantic work from comparable placement runs and current worker logs."""

import json
import math
from pathlib import Path
import re

from tools.progress_eta import tqdm_remaining_seconds
from tools.placement_conditions import SKIP_CONDITION, SKIP_CONTROLS


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _stamp(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _worker_line(root, name, now, attempt):
    dispatch = max((root / "logs").glob("dispatch-*"), key=_stamp, default=None)
    if dispatch is None:
        return ""
    log = dispatch / f"{name}.log"
    if not log.is_file():
        return ""
    if attempt:
        # In older DSEC runs the trainer's output only reached these inner logs.
        log = max([log, *(attempt / "logs").glob("*.log")], key=_stamp)
    if not 0 <= now - _stamp(log) <= 180:
        return ""
    try:
        with log.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 16384))
            lines = stream.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    return next((line.strip() for line in reversed(lines) if line.strip()), "")


def _progress(line):
    if not re.search(r"\d+/\d+\s*\[|\d+it\s*\[", line):
        return ""
    return re.sub(r"(\d+%)\|[^|]*\|\s*", r"\1 ", line)[-250:]


def _reset_time(line, root, ds, receipt):
    if " | reset " not in line:
        return None
    remaining = tqdm_remaining_seconds(line)
    if remaining is not None:
        return remaining
    # Original runs report a rate per sequence and no denominator.
    old = re.search(r" \| reset ([^:]+):\s*(\d+)it\s*\[[^\]]+,\s*([\d.]+)(it/s|s/it)\]", line)
    if old is None:
        return None
    try:
        report = _json(Path(receipt["report"]))["activity_evaluation"]
        sequences = list(report["state_policy_by_sequence"])
        rest = sequences[sequences.index(old[1]):]
        data = Path(_json(root / "inputs.json")["data"])
        source = data / ("DSEC_cache/events/gep_rgb" if ds == "dsec" else "m3ed_cache/half_dagr")
        counts = [int(_json(source / seq / "metadata.json")["frame_count"]) - int(ds == "m3ed") for seq in rest]
        done, rate = int(old[2]), float(old[3])
        if not 0 < done <= counts[0] or min(counts) <= 0 or not math.isfinite(rate) or rate <= 0:
            return None
        seconds_per_window = 1 / rate if old[4] == "it/s" else rate
        seconds = (sum(counts) - done) * seconds_per_window
        return seconds if math.isfinite(seconds) else None
    except (KeyError, TypeError, ValueError):
        return None


def semantic_remaining(outputs: Path, ds: str, now: float, *, condition=SKIP_CONDITION,
                       root=None) -> tuple[float, str, str]:
    """Return seconds for known work, unknown components and the evidence for the estimate."""
    root = root or outputs / "stage2_b3_skip_semantic_20261126"
    name = f"{ds}_{condition}"
    directory = root / name
    head = directory / "verified.json"
    needs_history = condition in SKIP_CONTROLS
    unknown = ["履歴リセット評価"] if needs_history else []
    attempts = list(directory.glob("run-*"))
    # If several attempts were interrupted we cannot safely use an old start time.
    current = attempts[0] if len(attempts) == 1 else None
    line = _worker_line(root, name, now, current if not head.is_file() else None)
    progress = _progress(line)
    if head.is_file():
        if not needs_history or (directory / "history.json").is_file():
            return 0, "", "Semantic完了"
        seconds = _reset_time(line, root, ds, _json(head))
        if seconds is not None and seconds > 0:
            return seconds, "", f"履歴リセット評価の現在速度 {progress}"
        return 0, unknown[0], f"連続評価完了、履歴リセット評価の計測待ち {progress}".strip()

    # The old three-insertion run supplies timing only until the new control is measured.
    comparable = SKIP_CONTROLS.get(condition, condition)
    if comparable == "E1_b3_6_9" and (root / f"{ds}_{comparable}/verified.json").is_file():
        control = root / f"{ds}_{comparable}"
    else:
        comparable = "E1_b6_9_12" if comparable == "E1_b3_6_9" else comparable
        control = outputs / "stage2_placement_semantic_20261118" / f"{ds}_{comparable}"
    receipt = _json(control / "verified.json")
    attempt = receipt.get("attempt", "")
    if (not isinstance(attempt, str) or not re.fullmatch(r"run-[\w-]+", attempt)
            or receipt.get("condition") != comparable or receipt.get("dataset") != ds):
        return 0, "・".join(["連続評価", *unknown]), f"参考にできる実測時間なし {progress}".strip()
    reference = control / attempt
    end = _stamp(reference / "_SUCCESS")
    markers = ("final/config.json", "development/config.json", "job_identity.json") if ds == "dsec" else ("h/config.json",)
    known = 0.0
    sources = []
    for marker in markers:
        started = _stamp(current / marker) if current else 0
        if not started and marker != markers[-1]:
            continue
        past = _stamp(reference / marker)
        if not (0 < past and past + 60 <= end <= now):
            unknown.insert(0, "連続評価")
            break
        if started and not progress:
            unknown.insert(0, "連続評価の進捗更新待ち")
            break
        known = end - past - (now - started if 0 < started <= now else 0)
        if known <= 0:
            unknown.insert(0, "過去の所要時間を超えた連続評価")
            known = 0.0
        else:
            sources.append(f"{comparable}の連続評価実績")
        break

    if ds == "m3ed" and not (current and (current / "h/config.json").is_file()):
        cache_seconds = tqdm_remaining_seconds(line) if line.startswith("M3ED Semantic cache ") else None
        if cache_seconds is None:
            # This is the first sequence/window in the 3-train/2-val protocol.
            cache = receipt.get("cache")
            first = _stamp(Path(cache) / "car_urban_day_penno_big_loop/000001.pt") if cache else 0
            finished = _stamp(reference / "h/config.json")
            if 0 < first and first + 60 <= finished <= end <= now:
                cache_seconds = finished - first
        if cache_seconds is None:
            unknown.insert(0, "特徴抽出")
        else:
            known += cache_seconds
            sources.append("特徴抽出の現在速度" if line.startswith("M3ED Semantic cache ") else "過去の特徴抽出時間")
    evidence = "、".join(sources) or "連続評価の計測待ち"
    return known, "・".join(unknown), f"{evidence} {progress}".strip()
