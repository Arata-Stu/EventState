from __future__ import annotations

from event_state.training.data import StatefulStreamBatchSampler


def test_stream_sampler_preserves_order_and_covers_every_clip() -> None:
    records = (
        ("a", 1, 2),
        ("a", 3, 2),
        ("a", 5, 2),
        ("b", 1, 2),
        ("b", 3, 2),
        ("c", 1, 2),
        ("c", 3, 2),
        ("c", 5, 2),
        ("c", 7, 2),
    )
    sampler = StatefulStreamBatchSampler(
        records,
        batch_size=2,
        seed=3,
        shuffle_sequences=False,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index, _epoch in batch]

    assert sorted(flattened) == list(range(len(records)))
    assert len(batches) == len(sampler)
    assert all(epoch == 0 for batch in batches for _index, epoch in batch)
    for batch in batches:
        names = [records[index][0] for index, _epoch in batch]
        assert len(names) == len(set(names))
    for sequence_name in {record[0] for record in records}:
        starts = [
            records[index][1]
            for index in flattened
            if records[index][0] == sequence_name
        ]
        assert starts == sorted(starts)
