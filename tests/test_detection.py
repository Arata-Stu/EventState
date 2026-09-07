from pathlib import Path

import numpy as np
import pytest
import torch

from event_state.detection.data import rectify_xywh_boxes, transform_xywh_boxes_to_input
from event_state.detection.split import load_dsec_detection_split
from event_state.detection.yolox import EventStateYOLOX


def test_official_dsec_detection_split_is_41_6_13() -> None:
    manifest = Path("tools/manifests/dsec_det_official_split.yaml")
    split = load_dsec_detection_split(manifest)
    assert (len(split.train), len(split.val), len(split.test)) == (41, 6, 13)
    assert "interlaken_00_c" in split.train
    assert "zurich_city_16_a" in split.val
    assert "zurich_city_13_b" in split.test


def test_rectify_boxes_is_identity_for_identity_map() -> None:
    y, x = np.mgrid[:48, :64]
    rectify_map = np.stack((x, y), axis=-1).astype(np.float32)
    boxes = np.array([[10.0, 12.0, 20.0, 18.0]], dtype=np.float32)
    np.testing.assert_allclose(rectify_xywh_boxes(boxes, rectify_map), boxes)


def test_detection_boxes_follow_eventstate_center_crop() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    transformed = transform_xywh_boxes_to_input(
        boxes, source_size=(480, 640), target_size=(448, 640)
    )
    np.testing.assert_allclose(
        transformed, np.array([[10.0, 4.0, 30.0, 40.0]], dtype=np.float32)
    )


def test_yolox_probe_returns_losses_and_detections() -> None:
    model = EventStateYOLOX(in_channels=16, num_classes=2, width=32)
    features = torch.randn(2, 16, 6, 8)
    targets = [
        {
            "boxes": torch.tensor([[12.0, 10.0, 35.0, 42.0]]),
            "labels": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[20.0, 16.0, 50.0, 55.0]]),
            "labels": torch.tensor([1]),
        },
    ]
    losses = model(features, targets)
    assert losses["loss"].isfinite()
    losses["loss"].backward()
    model.eval()
    with torch.no_grad():
        detections = model(features)
    assert len(detections) == 2
    assert set(detections[0]) == {"boxes", "scores", "labels"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_yolox_assignment_is_safe_under_cuda_autocast() -> None:
    model = EventStateYOLOX(in_channels=16, num_classes=2, width=32).cuda()
    features = torch.randn(1, 16, 6, 8, device="cuda")
    targets = [
        {
            "boxes": torch.tensor([[12.0, 10.0, 35.0, 42.0]], device="cuda"),
            "labels": torch.tensor([0], device="cuda"),
        }
    ]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        losses = model(features, targets)
    assert losses["loss"].isfinite()


def test_split_loader_rejects_noncanonical_counts(tmp_path: Path) -> None:
    path = tmp_path / "split.yaml"
    path.write_text("train: [a]\nval: [b]\ntest: [c]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 41"):
        load_dsec_detection_split(path)
