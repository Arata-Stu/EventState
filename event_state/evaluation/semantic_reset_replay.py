"""Paired Semantic inference with a single state reset at an event-window boundary."""


def replay_pair(backend, frames, records, scene):
    """Yield (frame, activity record, continuous prediction, reset prediction) for display.

    Both iterables start at the first event window of the sequence. Before the
    intervention a single forward pass supplies both predictions. The original
    backend owns the weights and target reader shared by the two trajectories.
    """
    sequence = scene["sequence"]
    stamps = scene["timestamps_us"]
    reset_stamp = scene["reset_input_us"]
    q1_start = scene["q1_start_us"]
    if (not stamps or any(b <= a for a, b in zip(stamps, stamps[1:]))
            or not q1_start < reset_stamp <= stamps[-1]):
        raise ValueError("Invalid display timestamps or reset boundary")

    backend.reset()
    reset_branch = None
    display_index = 0
    previous_end = None
    for index, (frame, row) in enumerate(zip(frames, records), start=1):
        stamp = frame["timestamp"]
        previous = frame["previous_timestamp"]
        if (frame["sequence_name"] != sequence or row["sequence"] != sequence
                or frame["frame_index"] != index or stamp <= previous
                or (previous_end is not None and previous != previous_end)
                or row["window_us"] != stamp - previous
                or any(frame[key] != row[key] for key in ("timestamp", "previous_timestamp", "event_count"))):
            raise ValueError(f"Prepared input and activity window differ at {sequence} frame {index}")
        previous_end = stamp

        if reset_branch is None and stamp >= reset_stamp:
            if stamp != reset_stamp or previous != q1_start:
                raise ValueError(f"Missing reset input window ({q1_start}, {reset_stamp}] for {sequence}")
            reset_branch = backend.fork_without_history()

        next_display = stamps[display_index]
        if stamp > next_display:
            raise ValueError(f"Missing display timestamp {next_display} for {sequence}")
        display = stamp == next_display
        kept = backend.step(frame, predict=display)
        reset = kept if reset_branch is None else reset_branch.step(frame, predict=display)
        if display:
            if kept is None or reset is None:
                raise ValueError(f"Missing Semantic prediction at {sequence}/{stamp}")
            yield frame, row, kept, reset
            display_index += 1
            if display_index == len(stamps):
                return

    raise ValueError(f"Prepared inputs or activity records ended before {sequence}/{stamps[display_index]}")
