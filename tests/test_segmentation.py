import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from event_state.segmentation import (
    DSECSemanticFeatureDataset,
    EventStateSegmentationHead,
    SemanticSegmentationEvaluator,
    load_dsec_semantic_split,
    multiclass_dice_loss,
)


def test_linear_semantic_probe_has_only_one_learned_layer() -> None:
    head = EventStateSegmentationHead(
        in_channels=8,
        num_classes=11,
        output_size=(440, 640),
        head_type="linear",
    )
    learned_modules = [
        module for module in head.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    assert learned_modules == [head.classifier]
    assert head.classifier.kernel_size == (1, 1)


def test_gep_patch_head_removes_bottom_padding_without_resizing() -> None:
    head = EventStateSegmentationHead(
        in_channels=8,
        num_classes=11,
        output_size=(440, 640),
        head_type="gep_patch",
    )
    logits = head(torch.zeros((2, 8, 28, 40)))
    assert tuple(logits.shape) == (2, 11, 440, 640)


def test_multiclass_dice_loss_ignores_void_pixels() -> None:
    logits = torch.tensor([[[[8.0, -8.0]], [[-8.0, 8.0]]]])
    targets = torch.tensor([[[0, 255]]])
    changed_ignored_logits = logits.clone()
    changed_ignored_logits[0, :, 0, 1] = torch.tensor([100.0, -100.0])
    loss = multiclass_dice_loss(logits, targets, ignore_index=255)
    changed_loss = multiclass_dice_loss(
        changed_ignored_logits, targets, ignore_index=255
    )
    torch.testing.assert_close(loss, changed_loss)


def test_official_dsec_semantic_split_is_6_2_3() -> None:
    split = load_dsec_semantic_split("tools/manifests/dsec_semantic_split.yaml")
    assert (len(split.train), len(split.val), len(split.test)) == (6, 2, 3)
    assert len(split.official_train) == 8
    assert set(split.train).isdisjoint(split.val)
    assert set(split.official_train).isdisjoint(split.test)


def test_semantic_metrics_ignore_void_pixels() -> None:
    evaluator = SemanticSegmentationEvaluator(3, ignore_index=255)
    predictions = torch.tensor([[[0, 1], [1, 2]]])
    targets = torch.tensor([[[0, 1], [2, 255]]])
    evaluator.update(predictions, targets)
    metrics = evaluator.compute()
    assert metrics["evaluated_pixels"] == 3
    assert metrics["pixel_accuracy"] == 2 / 3
    np.testing.assert_allclose(metrics["class_IoU"], [1.0, 0.5, 0.0])
    assert metrics["mIoU"] == 0.5


def test_semantic_feature_dataset_pairs_sparse_cache_and_labels(tmp_path: Path) -> None:
    sequence = "example_00_a"
    cache = tmp_path / "cache" / sequence
    labels = tmp_path / "labels" / "train" / sequence / "11classes"
    cache.mkdir(parents=True)
    labels.mkdir(parents=True)
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "task": "dsec_semantic",
                "features": ["h", "z"],
                "label_size": [4, 6],
                "patch_size": 16,
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "sequence_name": sequence,
            "frame_index": 3,
            "timestamp": 123,
            "features": {
                "z": torch.zeros((4, 2, 3), dtype=torch.float16),
                "h": torch.ones((4, 2, 3), dtype=torch.float16),
            },
        },
        cache / "000003.pt",
    )
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8)).save(labels / "000003.png")
    dataset = DSECSemanticFeatureDataset(
        feature_cache_dir=tmp_path / "cache",
        labels_root=tmp_path / "labels",
        sequences=[sequence],
        role="train",
        feature="concat",
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert tuple(sample["feature"].shape) == (8, 2, 3)
    assert tuple(sample["label"].shape) == (4, 6)
    assert sample["frame_index"] == 3
