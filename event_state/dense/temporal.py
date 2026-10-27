"""Low-event duration measured on every causal input frame, including unlabeled ones."""


class LowActivityClock:
    def __init__(self):
        self.previous_end = None
        self.duration_us = 0

    def update(self, previous_timestamp, timestamp, low):
        if timestamp <= previous_timestamp:
            raise ValueError("Nonpositive input interval")
        if self.previous_end is not None and previous_timestamp != self.previous_end:
            raise ValueError("Input gap invalidates continuous low-activity duration")
        self.duration_us = self.duration_us + timestamp - previous_timestamp if low else 0
        self.previous_end = timestamp
        return self.duration_us
