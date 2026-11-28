"""User-facing remaining time and projected finish time for training launchers."""

from datetime import datetime, timedelta, timezone
import math
import re


def tqdm_remaining_seconds(line):
    """Read the remaining duration in tqdm's default [elapsed<remaining, rate] field."""
    match = re.search(r'\[\d+:\d{2}(?::\d{2})?<((?:\d+:)?\d+:\d{2}),', line)
    if match is None:
        return None
    seconds = 0
    for part in match[1].split(':'):
        seconds = seconds * 60 + int(part)
    return seconds


def format_eta(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "見積もり待ち"
    duration = f"{seconds / 3600:.1f}時間" if seconds >= 3600 else f"{math.ceil(seconds / 60)}分"
    jst = timezone(timedelta(hours=9))
    try:
        finish = datetime.now(jst) + timedelta(seconds=seconds)
    except OverflowError:
        return "見積もり待ち"
    return f"約{duration}、終了見込み {finish:%m/%d %H:%M} JST"
