"""Deterministic, non-overlapping stream segments and full chronological batches.

Pure Python for portable coverage checks. Shorter lanes repeat within an epoch;
each segment restart must reset state. This is explicit sampling, not padding.
"""
import random


def segments(records, parts):
    if parts < 1:
        raise ValueError('stream_segments_per_sequence must be positive')
    groups = {}
    for i, (name, start, _) in enumerate(records):
        groups.setdefault(str(name), []).append((start, i))
    result = {}
    for name, pairs in sorted(groups.items()):
        indices = [i for _, i in sorted(pairs)]
        if len(indices) < parts:
            raise ValueError(f'{name}: too few clips for {parts} stream segments')
        for part in range(parts):
            result[f'{name}::segment{part}'] = indices[len(indices)*part//parts:len(indices)*(part+1)//parts]
    return result


class FullStreamBatchSampler:
    def __init__(self, groups, batch_size, seed, rank=0, world=1):
        if batch_size < 1 or not 0 <= rank < world:
            raise ValueError('Invalid stream batch/rank')
        names = sorted(groups)
        if len(names) < batch_size * world:
            raise ValueError('Need at least world_size * stream batch independent segments')
        random.Random(seed).shuffle(names)
        # Fixed rank ownership; balance frame counts across lanes without splitting a segment.
        slots = [[] for _ in range(batch_size * world)]
        loads = [0] * len(slots)
        for name in sorted(names, key=lambda x: -len(groups[x])):
            slot = min(range(len(slots)), key=lambda i: loads[i])
            slots[slot].append(name)
            loads[slot] += len(groups[name])
        self.slots = slots[rank*batch_size:(rank+1)*batch_size]
        self.groups, self.seed, self.epoch = groups, seed, 0
        self.length = max(loads[rank*batch_size:(rank+1)*batch_size])

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.length

    def __iter__(self):
        streams = []
        for slot in self.slots:
            names = list(slot)
            random.Random(self.seed+self.epoch).shuffle(names)
            streams.append([i for name in names for i in self.groups[name]])
        for step in range(self.length):
            yield [(stream[step % len(stream)], self.epoch) for stream in streams]
