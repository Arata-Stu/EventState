"""Pure-Python temporal contract for a trailing RGB video teacher."""

from bisect import bisect_right
from fractions import Fraction
import math


def validate_context(mode, context_frames, sample_fps):
    if mode not in {"image", "video"}:
        raise ValueError("teacher.input_mode must be image or video")
    if mode == "image":
        if context_frames != 1 or sample_fps is not None:
            raise ValueError("Image teacher requires context_frames=1 and sample_fps=null")
    elif context_frames < 2 or context_frames % 2:
        raise ValueError("Video context_frames must be even and >=2")
    if sample_fps is not None and (not math.isfinite(float(sample_fps)) or not 0 < float(sample_fps) <= 1_000_000):
        raise ValueError("sample_fps must be finite, positive and <=1MHz")


def sampled_history(timestamps, start, length, *, mode, context_frames, sample_fps):
    """Return source indices, requested microsecond times and prefix flags.

    The caller validates the entire timeline once per sequence. Each target is
    anchored independently. Bisect selects the newest RGB <= requested time;
    only requests before sequence start repeat the first RGB. No interpolation.
    Rational arithmetic avoids floating-point drift on microsecond timestamps.
    """
    validate_context(mode, context_frames, sample_fps)
    if start < 0 or length < 1 or start + length > len(timestamps):
        raise ValueError("Target interval is outside the sequence")
    period = Fraction(1_000_000, 1) / Fraction(str(sample_fps)) if sample_fps is not None else None
    positions, requests, prefix = [], [], []
    for index in range(start, start + length):
        target = int(timestamps[index])
        if mode == "image":
            row, wanted, padded = [index], [target], [False]
        elif period is None:
            raw = [index - context_frames + 1 + j for j in range(context_frames)]
            row = [max(0, i) for i in raw]
            first_interval = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1
            wanted = [int(timestamps[i]) if i >= 0 else int(timestamps[0] + i * first_interval) for i in raw]
            padded = [i < 0 for i in raw]
        else:
            wanted = [math.floor(Fraction(target) - j * period) for j in reversed(range(context_frames))]
            padded = [time < timestamps[0] for time in wanted]
            row = [max(0, bisect_right(timestamps, time, 0, index + 1) - 1) for time in wanted]
        positions.append(row)
        requests.append(wanted)
        prefix.append(padded)
    return positions, requests, prefix


def history_positions(start: int, length: int, context_frames: int) -> list[int]:
    """One overlapping history buffer; repeat frame zero only at sequence start.

    Window j is buffer[j:j+K], ending at source frame start+j. This is
    independent of student clip boundaries, batch order, and stream state.
    """
    if start < 0 or length < 1 or context_frames < 2 or context_frames % 2:
        raise ValueError("Require start>=0, length>=1 and an even context_frames>=2")
    return [max(0, start - context_frames + 1 + j)
            for j in range(length + context_frames - 1)]


def validate_history(positions, timestamps, target_timestamps, context_frames):
    if len(positions) != len(target_timestamps) + context_frames - 1:
        raise ValueError("History length does not match target length")
    if len(timestamps) != len(positions):
        raise ValueError("Missing history timestamps")
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1] == 0:
            if timestamps[i] != timestamps[i - 1]:
                raise ValueError("Prefix padding must repeat the first RGB frame")
        elif positions[i] != positions[i - 1] + 1 or timestamps[i] <= timestamps[i - 1]:
            raise ValueError("RGB history must be strictly chronological, without skipped records")
    for j, target in enumerate(target_timestamps):
        if timestamps[j + context_frames - 1] != target:
            raise ValueError("RGB history endpoint differs from student target timestamp")
