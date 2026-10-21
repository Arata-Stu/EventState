"""Server-side fixed-window source contracts; no real DSEC files required."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from event_state.data import dsec_feature_source as source


def config(window=20000):
    return OmegaConf.create(dict(dataset=dict(name='dsec', event_window_us=window,
        event_window_fraction=1., rectify_events=True, sensor_height=480, sensor_width=640,
        representation=dict(type='gep_rgb', channels=3, percentile=90.))))


def test_source_contract_rejects_old_cache():
    cfg = config()
    source.validate_source(cfg, None, Path('/raw'))
    with pytest.raises(ValueError): source.validate_source(cfg, Path('/old_cache'), None)
    with pytest.raises(ValueError): source.validate_source(cfg, Path('/old_cache'), Path('/raw'))
    cfg.dataset.event_window_us = None
    source.validate_source(cfg, Path('/old_cache'), None)
    with pytest.raises(ValueError): source.validate_source(cfg, None, Path('/raw'))


def test_source_keeps_timestamp_index_and_calls_training_extractor(tmp_path, monkeypatch):
    for name in ('train_events', 'train_images'):
        (tmp_path/name/'seq').mkdir(parents=True)
    calls = []
    records = [SimpleNamespace(frame_index=i,timestamp=t,previous_timestamp=max(0,t-50000))
               for i,t in enumerate((0,50000,100000))]
    class Dataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self._frames_by_sequence = {'seq': records}
            self.event_window_us = kwargs['event_window_us']
        def _load_event_frame(self, record):
            calls.append(record)
            return torch.ones(3,480,640), 7
        def close(self): calls.append('closed')
    monkeypatch.setattr(source, 'DSECSequenceDataset', Dataset)
    raw = source.RawGEPFeatureSource(tmp_path,'seq',config())
    assert [p.name for p in raw.paths] == ['50000.pt','100000.pt']
    assert calls[0]['event_cache_dir'] is None and calls[0]['feature_cache_dir'] is None
    assert calls[0]['event_window_us'] == 20000 and calls[0]['rectify_events']
    assert raw.metadata['event_window_us'] == 20000
    assert raw.metadata['frame_indices'] == [1,2]
    payload = raw.load(raw.paths[1])
    assert payload['frame_index'] == 2 and payload['timestamp'] == 100000
    assert payload['event_count'] == 7
    assert calls[-1] is records[2]
    raw.close()
    assert calls[-1] == 'closed'
