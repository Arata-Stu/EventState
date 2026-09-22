"""Run on the ML host; baseline equivalence and activity gradient isolation."""

import pytest
import torch
import torch.nn.functional as F

from event_state.losses.activity import ActivityDistillationLoss
from event_state.losses.distillation import DistillationLoss
from event_state.losses.activity_weighting import spatial_activity_weights
from event_state.segmentation.loss import activity_cross_entropy, multiclass_dice_loss
from event_state.detection.yolox import EventStateYOLOX


def test_dense_activity_all_ones_matches_baseline_and_empty_is_zero():
    torch.manual_seed(4)
    p = torch.randn(2, 3, 5, 7, requires_grad=True)
    q = torch.randn_like(p, requires_grad=True)
    baseline = DistillationLoss()(p, q)
    criterion = ActivityDistillationLoss()
    full = criterion(p, q, torch.ones(2, 3, 5, dtype=torch.bool))
    torch.testing.assert_close(full, baseline)
    torch.testing.assert_close(torch.autograd.grad(full, p, retain_graph=True)[0],
                               torch.autograd.grad(baseline, p, retain_graph=True)[0])
    empty = criterion(p, q, torch.zeros(2, 3, 5, dtype=torch.bool))
    empty.backward()
    assert empty.item() == 0
    assert torch.count_nonzero(p.grad) == 0
    assert q.grad is None


def test_activity_selection_only_supervises_selected_tokens():
    p = torch.randn(1, 3, 4, requires_grad=True)
    q = torch.randn_like(p)
    ActivityDistillationLoss()(p, q, torch.tensor([[False, True, False]])).backward()
    assert torch.count_nonzero(p.grad[:, [0, 2]]) == 0
    assert torch.count_nonzero(p.grad[:, 1]) > 0


def test_semantic_weights_preserve_baseline_and_ignore_void():
    logits = torch.randn(2, 3, 4, 6, requires_grad=True)
    targets = torch.randint(0, 3, (2, 4, 6))
    targets[:, 0] = 255
    weights = torch.ones_like(targets, dtype=torch.float32)
    torch.testing.assert_close(activity_cross_entropy(logits, targets, weights),
                               F.cross_entropy(logits, targets, ignore_index=255))
    torch.testing.assert_close(multiclass_dice_loss(logits, targets, ignore_index=255,
                                                   pixel_weights=weights),
                               multiclass_dice_loss(logits, targets, ignore_index=255))
    weights[:, :, :3] = 0
    loss = activity_cross_entropy(logits, targets, weights)
    loss = loss + multiclass_dice_loss(logits, targets, ignore_index=255, pixel_weights=weights)
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., :3]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 0]) == 0


@pytest.mark.parametrize("has_boxes", [True, False])
def test_detector_weights_preserve_assignment_and_baseline(has_boxes):
    torch.manual_seed(7)
    model = EventStateYOLOX(in_channels=16, num_classes=2, width=32)
    model.eval()  # Fix BN statistics while comparing identical forwards.
    features = torch.randn(1, 16, 6, 8)
    targets = [{"boxes": torch.tensor([[12., 10., 35., 42.]]) if has_boxes
                else torch.empty(0, 4),
                "labels": torch.tensor([0]) if has_boxes else torch.empty(0, dtype=torch.long)}]
    base = model(features, targets)
    ones = model(features, targets, loss_weights=torch.ones(1, 96, 128))
    zeros = model(features, targets, loss_weights=torch.zeros(1, 96, 128))
    for key in base:
        torch.testing.assert_close(base[key], ones[key])
    assert zeros["loss"].item() == 0
    assert zeros["positive_anchors"].item() == base["positive_anchors"].item()
    zeros["loss"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_activity_weights_are_complementary_and_validate_configuration():
    activity = torch.tensor([True, False])
    torch.testing.assert_close(spatial_activity_weights(activity, 1, 0), torch.tensor([1., 0.]))
    torch.testing.assert_close(spatial_activity_weights(activity, 0, 1), torch.tensor([0., 1.]))
    for active, inactive in ((0, 0), (-1, 1), (float("nan"), 1)):
        with pytest.raises(ValueError):
            spatial_activity_weights(activity, active, inactive)


def test_semantic_cached_activity_uses_native_stride_crop_and_shared_flip(tmp_path):
    import json
    import numpy as np
    from PIL import Image

    from event_state.losses.activity_weighting import ACTIVITY_FORMAT
    from event_state.segmentation.data import DSECSemanticFeatureDataset, segmentation_collate

    sequence = "zurich_city_00_a"
    cache = tmp_path / "features" / sequence
    labels = tmp_path / "labels" / "train" / sequence / "11classes"
    cache.mkdir(parents=True)
    labels.mkdir(parents=True)
    (cache / "metadata.json").write_text(json.dumps({
        "task": "dsec_semantic", "features": ["h"], "patch_size": 16,
        "label_size": [440, 640],
    }))
    mask = torch.zeros(28, 40, dtype=torch.bool)
    mask[27, 0] = True
    feature = torch.zeros(4, 28, 40)
    feature[:, 27, 0] = 1
    torch.save({"sequence_name": sequence, "frame_index": 3, "timestamp": 100,
                "features": {"h": feature}, "event_activity": mask,
                "activity_format": ACTIVITY_FORMAT}, cache / "000003.pt")
    Image.fromarray(np.zeros((440, 640), dtype=np.uint8)).save(labels / "000003.png")
    dataset = DSECSemanticFeatureDataset(
        feature_cache_dir=tmp_path / "features", labels_root=tmp_path / "labels",
        sequences=[sequence], role="train", feature="h", load_activity=True,
        horizontal_flip_probability=1,
    )
    sample = dataset[0]
    expected = torch.zeros(440, 640, dtype=torch.bool)
    expected[432:440, 624:640] = True
    assert torch.equal(sample["event_activity"], expected)
    assert torch.equal(sample["feature"][:, 27, -1], torch.ones(4))
    assert torch.equal(segmentation_collate([sample])["event_activity"][0], expected)


def test_activity_cache_is_required_only_when_enabled(tmp_path):
    from event_state.losses.activity_weighting import require_activity, ACTIVITY_FORMAT

    with pytest.raises(ValueError, match="include-activity"):
        require_activity({}, (2, 3))
    with pytest.raises(ValueError):
        require_activity({"activity_format": ACTIVITY_FORMAT,
                          "event_activity": torch.ones(2, 3)}, (2, 3))
