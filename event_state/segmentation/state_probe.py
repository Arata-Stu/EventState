"""Inference-only state interventions and activity-stratified semantic metrics."""
import torch
from .metrics import SemanticSegmentationEvaluator


def forward_frames(model, events, state, policy):
    """Reset both recurrent tensors before EVERY frame, not just every clip."""
    if policy not in {'continuous', 'frame'}:
        raise ValueError('Unknown state policy')
    values = {'z': [], 'h': []}
    for t in range(events.shape[1]):
        out = model(events[:, t:t + 1], state=None if policy == 'frame' else state)
        state = out.get('state')
        for key in values:
            values[key].append(out[key])
    return {**{k: torch.cat(v, dim=1) for k, v in values.items()}, 'state': state}


class ActivityEvaluator:
    def __init__(self, num_classes=11, low_threshold=0.1):
        if not 0 <= low_threshold <= 1:
            raise ValueError('low threshold must lie in [0,1]')
        self.threshold = low_threshold
        self.groups = {k: SemanticSegmentationEvaluator(num_classes) for k in
                       ('all', 'active', 'inactive', 'low_activity_frames', 'other_frames')}
        self.frames = {k: 0 for k in self.groups}

    def update(self, predictions, labels, activity):
        if activity.dtype != torch.bool or activity.shape != labels.shape:
            raise ValueError('Activity must be bool and aligned with labels')
        predictions, labels, activity = [v.detach().cpu() for v in (predictions, labels, activity)]
        # Fraction uses the complete sensor label area, independent of class/ignore labels.
        low = activity.flatten(1).float().mean(1) <= self.threshold
        masks = {'all': torch.ones_like(activity), 'active': activity, 'inactive': ~activity,
                 'low_activity_frames': low[:, None, None].expand_as(activity),
                 'other_frames': (~low)[:, None, None].expand_as(activity)}
        for name, mask in masks.items():
            self.groups[name].update(predictions, labels.masked_fill(~mask, 255))
            self.frames[name] += int(mask.flatten(1).any(1).sum())

    def compute(self):
        return {k: {**e.compute(), 'confusion_matrix': e.confusion.tolist(),
                    'class_support': e.confusion.sum(1).tolist(), 'frames': self.frames[k]}
                for k, e in self.groups.items()}
