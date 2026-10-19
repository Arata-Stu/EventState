"""Server Torch checks: exact interventions and stratified confusion conservation."""
import torch
from event_state.segmentation.state_probe import forward_frames, ActivityEvaluator


class MemoryModel:
    def __call__(self, events, state=None):
        old_h, old_c = (torch.zeros_like(events), torch.zeros_like(events)) if state is None else state
        c = old_c + events
        h = old_h + c
        return {'z': events, 'h': h, 'state': (h, c)}


def test_reset_every_frame_and_continuous_clip_carry():
    events = torch.ones(1, 3, 1)
    model = MemoryModel()
    continuous = forward_frames(model, events, None, 'continuous')
    reset = forward_frames(model, events, None, 'frame')
    assert continuous['h'].flatten().tolist() == [1, 3, 6]
    assert reset['h'].flatten().tolist() == [1, 1, 1]
    assert torch.equal(continuous['z'], reset['z'])
    carried = forward_frames(model, events[:, :1], continuous['state'], 'continuous')
    assert carried['h'].item() == 10
    # Sequence boundary: caller clears BOTH h and c.
    assert forward_frames(model, events[:, :1], None, 'continuous')['h'].item() == 1


def test_strata_partition_and_ignore():
    evaluator = ActivityEvaluator(2, 0.25)
    labels = torch.tensor([[[0, 1], [1, 255]], [[0, 0], [1, 1]]])
    predictions = torch.tensor([[[0, 0], [1, 1]], [[1, 0], [0, 1]]])
    activity = torch.tensor([[[True, False], [False, False]], [[True, True], [True, True]]])
    evaluator.update(predictions, labels, activity)
    groups = evaluator.groups
    assert torch.equal(groups['all'].confusion, groups['active'].confusion + groups['inactive'].confusion)
    assert torch.equal(groups['all'].confusion, groups['low_activity_frames'].confusion + groups['other_frames'].confusion)
    result = evaluator.compute()
    assert result['all']['evaluated_pixels'] == 7
    assert result['low_activity_frames']['frames'] == 1
    assert result['inactive']['class_support'] == [0, 2]


def test_empty_stratum_has_no_support():
    evaluator = ActivityEvaluator(2)
    value = torch.zeros(1, 2, 2, dtype=torch.long)
    evaluator.update(value, value, value.bool())
    assert evaluator.compute()['active']['evaluated_pixels'] == 0


def test_m3ed_activity_expansion(tmp_path):
    import json
    import h5py
    import numpy as np
    from event_state.segmentation.m3ed import M3EDSemanticFeatureDataset
    from event_state.losses.activity_weighting import ACTIVITY_FORMAT
    cache = tmp_path / 'features' / 'seq'
    targets = tmp_path / 'targets' / 'seq'
    cache.mkdir(parents=True); targets.mkdir(parents=True)
    (cache / 'metadata.json').write_text(json.dumps(dict(task='m3ed_semantic', features=['z'],
        label_size=[4, 4], patch_size=2, cached_frame_indices=[1])))
    torch.save(dict(sequence_name='seq', frame_index=1, features={'z':torch.ones(3,2,2)},
        activity_format=ACTIVITY_FORMAT, event_activity=torch.tensor([[True,False],[False,False]])), cache/'000001.pt')
    with h5py.File(targets/'targets.h5', 'w') as f:
        f['semantics_11'] = np.zeros((2,4,4), dtype=np.uint8)
        f['semantics_frame_valid'] = np.ones(2, dtype=np.bool_)
        f['timestamps'] = [0, 10]
    dataset = M3EDSemanticFeatureDataset(feature_cache_dir=cache.parent,
        target_cache_dir=targets.parent, sequences=['seq'], feature='z', load_activity=True)
    mask = dataset[0]['event_activity']
    assert mask.shape == (4,4)
    assert mask[:2,:2].all() and mask.sum() == 4
