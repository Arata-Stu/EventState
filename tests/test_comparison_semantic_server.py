"""Server-only Semantic video checks; tiny CPU fixtures, no teacher weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
from torch import nn

from event_state.evaluation.comparison_semantic import SemanticBackend, SemanticTargets


def test_m3ed_targets_use_exact_timestamps_and_frame_validity(tmp_path):
    sequence = 'fixture'
    prepared = tmp_path / 'prepared' / sequence
    prepared.mkdir(parents=True)
    (prepared / 'metadata.json').write_text(json.dumps({'timestamps': [100, 200, 300]}))
    targets = tmp_path / 'targets' / sequence
    targets.mkdir(parents=True)
    with h5py.File(targets / 'targets.h5', 'w') as source:
        source['timestamps'] = [100, 200, 300]
        source['semantics_frame_valid'] = [False, True, True]
        labels = np.zeros((3, 352, 640), dtype=np.uint8)
        labels[1] = 3
        labels[2] = 8
        source['semantics_11'] = labels
    reader = SemanticTargets('m3ed', {'target_cache_dir': str(targets.parent)},
                             None, tmp_path, prepared.parent, (352, 640), [sequence])
    try:
        assert reader.get(sequence, 100) is None
        assert reader.get(sequence, 150) is None
        assert np.all(reader.get(sequence, 200) == 3)
        assert np.all(reader.get(sequence, 300) == 8)
        assert reader.get(sequence, 301) is None
        # Returned arrays must not be shared mutable buffers.
        reader.get(sequence, 200)[0, 0] = 9
        assert reader.get(sequence, 200)[0, 0] == 3
    finally:
        reader.close()


def test_m3ed_targets_reject_a_shifted_prepared_timeline(tmp_path):
    sequence = 'fixture'
    prepared = tmp_path / 'prepared' / sequence
    prepared.mkdir(parents=True)
    (prepared / 'metadata.json').write_text(json.dumps({'timestamps': [100, 201]}))
    targets = tmp_path / 'targets' / sequence
    targets.mkdir(parents=True)
    with h5py.File(targets / 'targets.h5', 'w') as source:
        source['timestamps'] = [100, 200]
        source['semantics_frame_valid'] = [True, True]
        source['semantics_11'] = np.zeros((2, 352, 640), dtype=np.uint8)
    reader = SemanticTargets('m3ed', {'target_cache_dir': str(targets.parent)},
                             None, tmp_path, prepared.parent, (352, 640), [sequence])
    with pytest.raises(ValueError, match='timeline mismatch'):
        reader.get(sequence, 200)
    reader.close()


def test_dsec_targets_do_not_hold_a_sparse_label_between_native_rgb_timestamps(tmp_path):
    sequence = 'fixture'
    directory = tmp_path / 'labels/test' / sequence / '11classes'
    directory.mkdir(parents=True)
    Image.fromarray(np.full((440, 640), 8, dtype=np.uint8)).save(directory / '000001.png')
    timeline = tmp_path / 'DSEC/test_images' / sequence / 'images/timestamps.txt'
    timeline.parent.mkdir(parents=True)
    timeline.write_text('100\n200\n300\n')
    reader = SemanticTargets('dsec', {'labels_root': str(tmp_path / 'labels')},
                             SimpleNamespace(dataset=SimpleNamespace(root=str(tmp_path / 'DSEC'))),
                             tmp_path, tmp_path / 'events', (440, 640))
    assert reader.get(sequence, 100) is None
    assert reader.get(sequence, 199) is None
    assert np.all(reader.get(sequence, 200) == 8)
    assert reader.get(sequence, 201) is None
    assert reader.get(sequence, 300) is None
    assert reader.get('without_labels', 200) is None


class TinyRecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_encoder = SimpleNamespace(patch_size=16)
        self.inputs = []

    def forward(self, inputs, state=None):
        self.inputs.append(inputs.clone())
        value = (0.0 if state is None else float(state)) + 1.0
        batch, time, _, height, width = inputs.shape
        features = inputs.new_full((batch, time, height // 16 * (width // 16), 4), value + 0.0004)
        return {'z': features, 'h': features, 'state': inputs.new_tensor(value)}


class RecordingHead(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.features = []

    def forward(self, features):
        self.features.append(features.clone())
        logits = features.new_zeros((1, 11, *self.size))
        logits[:, 8] = 1
        return logits


def fake_backend(dataset):
    backend = SemanticBackend.__new__(SemanticBackend)
    backend.dataset = dataset
    backend._torch, backend._np = torch, np
    backend.cfg = OmegaConf.create({'dataset': {
        'input_height': 448 if dataset == 'dsec' else 352, 'input_width': 640,
        'representation': {'normalize_mean': [0, 0, 0], 'normalize_std': [1, 1, 1]}}})
    backend.device = torch.device('cpu')
    backend.model = TinyRecurrentModel()
    backend.frame_size = (440, 640) if dataset == 'dsec' else (352, 640)
    backend.head = RecordingHead(backend.frame_size)
    backend.feature = 'h'
    backend.use_amp = False
    backend.reset()
    return backend


def frame(dataset, index):
    events = torch.full((3, 480 if dataset == 'dsec' else 360, 640), 0.25)
    if dataset == 'dsec':
        events[:, 440:] = 0.9
    else:
        events[:, :4] = 0.9
        events[:, -4:] = 0.9
    return dict(sequence_name='fixture', timestamp=index * 100,
                previous_timestamp=(index - 1) * 100, frame_index=index, events=events)


@pytest.mark.parametrize('dataset', ['dsec', 'm3ed'])
def test_semantic_warmup_updates_state_without_head_and_matches_cache_rounding(dataset):
    backend = fake_backend(dataset)
    assert backend.step(frame(dataset, 1), predict=False) is None
    assert backend.step(frame(dataset, 2), predict=False) is None
    assert not backend.head.features
    prediction = backend.step(frame(dataset, 3))
    assert len(backend.model.inputs) == 3 and len(backend.head.features) == 1
    assert prediction.shape == backend.frame_size and prediction.dtype == np.uint8
    assert np.all(prediction == 8)
    assert backend.state.item() == 3
    assert backend.head.features[0].dtype == torch.float32
    expected = torch.tensor(3.0004).half().float()
    torch.testing.assert_close(backend.head.features[0], torch.full_like(backend.head.features[0], expected.item()))
    value = backend.model.inputs[0][0, 0]
    if dataset == 'dsec':
        assert value.shape == (3, 448, 640)
        assert torch.all(value[:, :440] == 0.25)
        assert torch.all(value[:, 440:] == 0)
    else:
        assert value.shape == (3, 352, 640)
        assert torch.all(value == 0.25)


def test_semantic_inference_rejects_missing_prefix_gaps_and_sequence_switches():
    backend = fake_backend('dsec')
    with pytest.raises(ValueError, match='sequence start'):
        backend.step(frame('dsec', 2))
    backend.step(frame('dsec', 1), predict=False)
    with pytest.raises(ValueError, match='contiguous'):
        backend.step(frame('dsec', 3))
    changed = frame('dsec', 2)
    changed['sequence_name'] = 'other'
    with pytest.raises(ValueError, match='history changed'):
        backend.step(changed)
    backend.reset()
    assert backend.step(frame('dsec', 1)).shape == (440, 640)


def test_history_fork_keeps_the_input_cursor_and_independent_recurrent_states():
    keep = fake_backend('m3ed')
    keep.step(frame('m3ed', 1), predict=False)
    branch = keep.fork_without_history()
    assert branch.model is keep.model and branch.head is keep.head
    assert branch.state is None and keep.state.item() == 1
    for index in (2, 3):
        sample = frame('m3ed', index)
        keep.step(sample, predict=False)
        branch.step(sample, predict=False)
        assert keep.state.item() == index and branch.state.item() == index - 1
    assert len(keep.model.inputs) == 5


def test_semantic_constructor_rejects_a_cache_with_a_different_backbone(tmp_path):
    from tools import run_semantic_depth_activity

    run = run_semantic_depth_activity.semantic_run('dsec', 'E1').relative_to(run_semantic_depth_activity.ROOT)
    directory = tmp_path / run / 'final'
    directory.mkdir(parents=True)
    (directory / 'last.pt').write_bytes(b'not loaded: provenance check must fail first')
    cache = tmp_path / 'features' / 'fixture'
    cache.mkdir(parents=True)
    backbone = tmp_path / 'backbone.pt'
    backbone.write_bytes(b'not loaded: provenance check must fail first')
    (directory / 'config.json').write_text(json.dumps({
        'feature': 'h', 'feature_cache_dir': str(cache.parent), 'label_size': [440, 640]}))
    (cache / 'metadata.json').write_text(json.dumps({
        'task': 'dsec_semantic', 'state_policy': 'continuous', 'label_size': [440, 640],
        'patch_size': 16, 'features': ['h'], 'checkpoint': str(backbone),
        'checkpoint_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='backbone SHA256 mismatch'):
        SemanticBackend('dsec', 'E1', tmp_path, device='cpu')
