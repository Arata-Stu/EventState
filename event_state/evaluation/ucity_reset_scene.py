"""Recover the archived UCity clip and the first Q1 input on its native clock."""
from bisect import bisect_left
from itertools import groupby

from event_state.evaluation.comparison_timing import validate_scene


SEQUENCE = "car_urban_day_ucity_small_loop"


def make_scene(selection_dict, records):
    """Match the saved clip against all ordered UCity ActivityIndex records."""
    if (selection_dict["source"] != "full_sequence_review_20261118"
            or selection_dict["task"] != "semantic"
            or selection_dict["selection"]["group"] != "p00_20"
            or selection_dict["selection"]["minimum_continuous_seconds"] != 10):
        raise ValueError("Expected the archived full-sequence Semantic Q1 selection")
    entries = [row for row in selection_dict["sequences"]
               if row["dataset"] == "m3ed" and row["sequence"] == SEQUENCE]
    if len(entries) != 1:
        raise ValueError("Expected one M3ED UCity selection")
    entry, = entries
    records = list(records)
    if not records or len(records) != entry["frames"]:
        raise ValueError("Expected the complete native input timeline for UCity")
    previous = records[0]["previous_timestamp"]
    for row in records:
        if (row["sequence"] != SEQUENCE or row["previous_timestamp"] != previous
                or row["timestamp"] <= previous or row["window_us"] != row["timestamp"] - previous):
            raise ValueError("Invalid or discontinuous UCity input timeline")
        previous = row["timestamp"]
        if (row["event_rate_hz"] <= entry["q1_upper_events_per_s"]) != (row["group"] == "p00_20"):
            raise ValueError("UCity Q1 assignments disagree with the archived rate threshold")

    stamps = [row["timestamp"] for row in records]
    episodes = []
    for low, group in groupby(records, key=lambda row: row["group"] == "p00_20"):
        if low:
            rows = list(group)
            first, last = rows[0], rows[-1]
            start, end = first["previous_timestamp"], last["timestamp"]
            if first is not records[0] and last is not records[-1] and end - start >= 10_000_000:
                episodes.append((start, end, len(rows)))
    if not episodes:
        raise ValueError("No interior UCity Q1 episode of at least 10 seconds")
    # Episodes are chronological, so max retains the earlier one in a duration tie.
    q1_start, q1_end, q1_frames = max(episodes, key=lambda span: span[1] - span[0])
    clip = entry["clip"]
    if (clip["q1_source_start_us"], clip["q1_source_end_us"]) != (q1_start, q1_end):
        raise ValueError("Longest interior Q1 episode differs from the archived clip")
    matched = [episode for episode in entry["episodes"]
               if episode["start_us"] == q1_start and episode["end_us"] == q1_end]
    if (len(matched) != 1 or not matched[0]["interior"] or matched[0]["frames"] != q1_frames
            or round(matched[0]["duration_s"] * 1_000_000) != q1_end - q1_start):
        raise ValueError("Archived UCity Q1 episode duration or frame count differs")

    start = stamps[0] + round(clip["from_video_s"] * 1_000_000)
    end = stamps[0] + round(clip["to_video_s"] * 1_000_000)
    if (start, end) != (q1_start - 5_000_000, q1_end + 5_000_000):
        raise ValueError("Archived UCity clip must cover five seconds on each side of Q1")
    first = bisect_left(stamps, start)
    stop = bisect_left(stamps, end)
    if first >= len(stamps) or stop >= len(stamps) or stamps[first] != start or stamps[stop] != end:
        raise ValueError("UCity clip endpoints must match native input timestamps")

    # Each input measures the preceding physical window. The input at 'end'
    # contributes to the activity duration but is outside the displayed clip.
    windows = records[first + 1:stop + 1]
    duration = end - start
    scene = dict(id="m3ed_semantic_ucity_onset_reset", sequence=SEQUENCE, kind="Q1 onset reset",
                 timestamps_us=stamps[first:stop], start_us=start, end_us=end,
                 duration_us=duration, frames=stop - first, terminal_hold_us=end - stamps[stop - 1],
                 warmup_frames_before_start=first, q1_start_us=q1_start, q1_end_us=q1_end,
                 reset_input_us=stamps[bisect_left(stamps, q1_start) + 1],
                 low_fraction=sum(r["window_us"] for r in windows if r["group"] == "p00_20") / duration,
                 high_fraction=sum(r["window_us"] for r in windows if r["group"] == "p80_100") / duration)
    validate_scene(scene)
    return scene
