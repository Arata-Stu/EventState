"""Read Stage 1 progress from the existing worker logs without importing PyTorch."""

from collections import deque
import math
from pathlib import Path
from statistics import median
from typing import NamedTuple

from tools.progress_eta import format_eta
from tools.summarize_training_logs import LINE_RE, VALUE_RE


class TrainingProgress(NamedTuple):
    step: int
    loss: float | None
    seconds_per_update: float | None


def read_stage1_progress(path: Path) -> TrainingProgress | None:
    """Read the most recent complete train record and median recent update time."""
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 128 * 1024)
            stream.seek(start)
            data = stream.read()
    except FileNotFoundError:
        return None
    lines = data.split(b"\n")[:-1]  # The last record may still be being written.
    if start:
        lines = lines[1:]
    latest = None
    durations = []
    for line in reversed(lines):
        match = LINE_RE.match(line.decode(errors="replace").strip())
        if match is None or match[1] != "train":
            continue
        values = dict(VALUE_RE.findall(match[3]))
        if latest is None:
            latest = (int(match[2]), float(values["loss"]) if "loss" in values else None)
        clips = float(values.get("clips_per_update", 0))
        rate = float(values.get("samples_per_second", 0))
        if clips > 0 and rate > 0 and math.isfinite(clips) and math.isfinite(rate):
            durations.append(clips / rate)
        if len(durations) == 9:
            break
    if latest is not None:
        return TrainingProgress(*latest, median(durations) if durations else None)
    return None


class Stage1JobProgress:
    """Emit the first visible update, then at most one line per minute per job."""

    def __init__(self):
        self.next_poll = 0.0
        self.observations = deque(maxlen=10)

    def status(self, path, total, now):
        if now < self.next_poll:
            return None
        progress = read_stage1_progress(path)
        if progress is None:
            self.next_poll = now + 2
            return None
        self.next_poll = now + 60
        step, loss, seconds_per_update = progress
        result = f"step {step:,}/{total:,} ({100 * step / total:.1f}%)"
        if loss is not None:
            result += f" | loss={loss:.5g}"
        if self.observations and self.observations[-1][1] == step:
            age = now - self.observations[-1][0]
            return f"{result} | 最終stepから約{age / 60:.0f}分、残り時間を再計測中"
        self.observations.append((now, step))
        first_time, first_step = self.observations[0]
        source = "直近の学習ログから暫定"
        if step > first_step:
            # Includes loader stalls, optimizer skips, logging and checkpoints.
            seconds_per_update = (now - first_time) / (step - first_step)
            source = "実測更新速度"
        if step >= total:
            return f"{result} | 学習更新終了、最終checkpointを確認中"
        if seconds_per_update is None:
            return result + " | 残り時間を計測中"
        seconds = (total - step) * seconds_per_update
        return f"{result} | この学習の残り{format_eta(seconds)}（{source}）"
