import numpy as np
import pytest
from types import SimpleNamespace
from event_state.data.dsec import DSECSequenceDataset, fixed_event_window_start
from event_state.data.event_representation import GEPEventFrame


def test_exact_window_not_fraction():
    assert fixed_event_window_start(0, 50000, 20000) == 30000
    assert fixed_event_window_start(0, 51000, 20000) == 31000
    with pytest.raises(ValueError, match='shorter'):
        fixed_event_window_start(0, 19000, 20000)
    for value in (0, -1, True, 20.5):
        with pytest.raises(ValueError): fixed_event_window_start(0, 50000, value)


def test_raw_gep_uses_exact_tail_and_end_timestamp():
    dataset = DSECSequenceDataset.__new__(DSECSequenceDataset)
    dataset.event_cache_dir = None
    dataset.event_window_fraction = 1.0
    dataset.event_window_us = 20000
    dataset.event_representation = GEPEventFrame(height=1, width=1)
    calls = []
    def sliced(start, end):
        calls.append((start,end))
        return {key: np.array([], dtype=np.int64) for key in ('x','y','t','p')}
    dataset._reader = lambda sequence: SimpleNamespace(slice=sliced)
    record = SimpleNamespace(previous_timestamp=0,timestamp=51000,sequence_name='seq')
    tensor, count = dataset._load_event_frame(record)
    assert calls == [(31000,51000)]
    assert count == 0 and tuple(tensor.shape) == (3,1,1)
    dataset.event_window_us = None
    dataset._load_event_frame(record)
    assert calls[-1] == (0,51000)
