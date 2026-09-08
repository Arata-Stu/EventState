from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from event_state.detection.data import (
    DSECDetectionEventDataset,
    build_dagr_sampling_grid,
    load_dagr_sequence_targets,
    rectify_xywh_boxes,
    transform_xywh_boxes_to_input,
)
from event_state.detection.metrics import COCODetectionEvaluator
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


def test_dagr_sampling_grid_has_official_geometry() -> None:
    y, x = np.mgrid[:480, :640]
    rectify_map = np.stack((x, y), axis=-1).astype(np.float32)
    grid = build_dagr_sampling_grid(
        rectify_map,
        source_input_size=(448, 640),
        source_stride=16,
    )
    assert tuple(grid.shape) == (27, 40, 2)
    assert grid.dtype == torch.float32
    half_grid = build_dagr_sampling_grid(
        rectify_map,
        source_input_size=(448, 640),
        source_stride=16,
        scale=2,
    )
    assert tuple(half_grid.shape) == (27, 40, 2)


def test_dagr_targets_use_consecutive_valid_frames(tmp_path: Path) -> None:
    sequence = "example_00_a"
    labels = tmp_path / "labels" / "train" / sequence / "object_detections" / "left"
    images = tmp_path / "dataset" / "train_images" / sequence / "images"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    (images / "timestamps.txt").write_text("100\n200\n300\n400\n", encoding="utf-8")
    dtype = np.dtype(
        [
            ("t", "<i8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "u1"),
        ]
    )
    tracks = np.array(
        [
            (100, 20, 40, 40, 40, 2),
            (200, 24, 44, 40, 40, 2),
            (300, 24, 44, 40, 40, 1),
            (400, 28, 48, 40, 40, 2),
        ],
        dtype=dtype,
    )
    np.save(labels / "tracks.npy", tracks)
    targets = load_dagr_sequence_targets(
        labels_root=tmp_path / "labels",
        dataset_root=tmp_path / "dataset",
        sequence=sequence,
    )
    assert set(targets) == {200}
    torch.testing.assert_close(
        targets[200]["boxes"], torch.tensor([[24.0, 44.0, 64.0, 84.0]])
    )
    assert targets[200]["labels"].tolist() == [0]


def test_detection_event_dataset_uses_clips_for_train_and_stream_for_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = "example_00_a"
    event_dir = tmp_path / "events" / sequence
    rectify_dir = (
        tmp_path / "dataset" / "train_events" / sequence / "events" / "left"
    )
    event_dir.mkdir(parents=True)
    rectify_dir.mkdir(parents=True)
    (event_dir / "metadata.json").write_text(
        """{
  "coordinate_space": "rectified_event",
  "event_window": "rgb_interval",
  "height": 480,
  "width": 640,
  "representation": {"channels": 3}
}\n""",
        encoding="utf-8",
    )
    for frame_index, timestamp in enumerate((100, 200, 300, 400), start=1):
        torch.save(
            {
                "events": torch.zeros((3, 480, 640), dtype=torch.uint8),
                "timestamp": timestamp,
                "frame_index": frame_index,
                "sequence_name": sequence,
            },
            event_dir / f"{timestamp}.pt",
        )
    y, x = np.mgrid[:480, :640]
    with h5py.File(rectify_dir / "rectify_map.h5", "w") as handle:
        handle.create_dataset(
            "rectify_map", data=np.stack((x, y), axis=-1).astype(np.float32)
        )
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 50.0]]),
        "labels": torch.tensor([0]),
    }
    monkeypatch.setattr(
        "event_state.detection.data.load_dagr_sequence_targets",
        lambda **_: {300: target},
    )

    common = {
        "event_cache_dir": tmp_path / "events",
        "labels_root": tmp_path / "labels",
        "dataset_root": tmp_path / "dataset",
        "sequences": [sequence],
        "sequence_length": 2,
        "event_mean": (0.0, 0.0, 0.0),
        "event_std": (1.0, 1.0, 1.0),
    }
    train = DSECDetectionEventDataset(continuous=False, **common)
    stream = DSECDetectionEventDataset(continuous=True, **common)

    assert len(train) == 1
    assert tuple(train[0]["events"].shape) == (2, 3, 448, 640)
    assert train[0]["timestamp"] == 300
    assert len(stream) == 4
    assert [stream[index]["evaluate"] for index in range(4)] == [False, False, True, False]
    assert stream[0]["is_sequence_start"] is True


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


def test_yolox_probe_accepts_odd_dsec_det_grid() -> None:
    model = EventStateYOLOX(
        in_channels=16,
        num_classes=2,
        width=32,
        input_stride=16,
        image_size=(430, 640),
    )
    model.eval()
    with torch.no_grad():
        detections = model(torch.randn(1, 16, 27, 40))
    assert len(detections) == 1


def test_coco_evaluator_reports_perfect_detection() -> None:
    pytest.importorskip("pycocotools")
    evaluator = COCODetectionEvaluator()
    box = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    evaluator.update(
        [{"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}],
        [{"boxes": box, "labels": torch.tensor([0])}],
        sequence_names=["example_00_a"],
        timestamps=[100],
        image_size=(215, 320),
    )
    metrics = evaluator.compute()
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)


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
