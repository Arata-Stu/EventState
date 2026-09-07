from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from event_state.data.cache_metadata import (
    ALIGNMENT_FORMAT_VERSION,
    SUCCESS_MARKER_NAME,
    build_input_fingerprint,
    frame_manifest_sha256,
    success_marker_payload,
)
from event_state.data.dsec import (
    DSECEventReader,
    DSECSequenceDataset,
    event_window_contract,
    event_window_start_timestamp,
)
from event_state.data.event_representation import GEPEventFrame
from event_state.data.transforms import PairedSequenceTransform


def _write_sequence(root: Path, sequence: str, *, split: str = "train") -> None:
    image_base = root / f"{split}_images" / sequence / "images"
    image_dir = image_base / "left" / "aligned_event"
    source_image_dir = image_base / "left" / "rectified"
    event_dir = root / f"{split}_events" / sequence / "events" / "left"
    calibration_dir = root / f"{split}_calibration" / sequence / "calibration"
    image_dir.mkdir(parents=True)
    source_image_dir.mkdir(parents=True)
    event_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)

    timestamps = np.array([10_000, 11_000, 12_000, 13_000], dtype=np.int64)
    np.savetxt(image_base / "timestamps.txt", timestamps, fmt="%d")
    for timestamp in timestamps:
        image = Image.fromarray(np.full((4, 5, 3), timestamp % 255, dtype=np.uint8))
        image.save(source_image_dir / f"{timestamp}.png")
        image.save(image_dir / f"{timestamp}.png")
    calibration_path = calibration_dir / "cam_to_cam.yaml"
    calibration_path.write_text("synthetic: true\n", encoding="utf-8")

    alignment_metadata = {
        "format_version": ALIGNMENT_FORMAT_VERSION,
        "dataset": "DSEC",
        "split": split,
        "sequence_name": sequence,
        "source_image_directory": (
            f"{split}_images/{sequence}/images/left/rectified"
        ),
        "frame_count": len(timestamps),
        "timestamp_manifest_sha256": frame_manifest_sha256(
            (
                index,
                int(timestamps[index - 1]) if index else int(timestamp),
                int(timestamp),
            )
            for index, timestamp in enumerate(timestamps)
        ),
        "input_fingerprint": build_input_fingerprint(
            root,
            files={
                "rgb_timestamps": image_base / "timestamps.txt",
                "camera_calibration": calibration_path,
            },
            file_sets={
                "rectified_rgb_frames": sorted(source_image_dir.glob("*.png")),
            },
        ),
        "interpolation": "synthetic",
        "border_mode": "synthetic",
        "calibration": {"height": 4, "width": 5},
    }
    alignment_metadata["output_fingerprint"] = build_input_fingerprint(
        root,
        files={},
        file_sets={"aligned_rgb_frames": sorted(image_dir.glob("*.png"))},
    )
    (image_dir / "metadata.json").write_text(
        json.dumps(alignment_metadata), encoding="utf-8"
    )
    (image_dir / SUCCESS_MARKER_NAME).write_text(
        json.dumps(success_marker_payload(alignment_metadata)), encoding="utf-8"
    )

    local_t = np.array([200, 600, 1_000, 1_400, 1_600, 2_500, 3_000], dtype=np.int64)
    with h5py.File(event_dir / "events.h5", "w") as h5_file:
        events = h5_file.create_group("events")
        events.create_dataset("t", data=local_t)
        events.create_dataset("x", data=np.arange(len(local_t)) % 5)
        events.create_dataset("y", data=np.arange(len(local_t)) % 4)
        events.create_dataset("p", data=np.arange(len(local_t)) % 2)
        h5_file.create_dataset("t_offset", data=np.array(10_000, dtype=np.int64))
        h5_file.create_dataset(
            "ms_to_idx",
            data=np.array(
                [
                    np.searchsorted(local_t, millisecond * 1_000, side="left")
                    for millisecond in range(5)
                ],
                dtype=np.int64,
            ),
        )
    yy, xx = np.mgrid[:4, :5]
    rectify_map = np.stack((xx, yy), axis=-1).astype(np.float32)
    with h5py.File(event_dir / "rectify_map.h5", "w") as h5_file:
        h5_file.create_dataset("rectify_map", data=rectify_map)


def test_timestamp_alignment_uses_open_closed_rgb_interval(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "zurich_city_00_a")
    event_path = (
        tmp_path
        / "train_events"
        / "zurich_city_00_a"
        / "events"
        / "left"
    )
    reader = DSECEventReader(
        event_path / "events.h5",
        rectify_map_path=event_path / "rectify_map.h5",
        output_height=4,
        output_width=5,
    )

    sliced = reader.slice(11_000, 12_000)

    assert sliced["t"].tolist() == [11_400, 11_600]


def test_timestamp_alignment_includes_exact_end_and_handles_stream_edges(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path, "zurich_city_00_a")
    event_path = (
        tmp_path
        / "train_events"
        / "zurich_city_00_a"
        / "events"
        / "left"
    )
    reader = DSECEventReader(
        event_path / "events.h5",
        rectify_map_path=event_path / "rectify_map.h5",
        output_height=4,
        output_width=5,
    )

    assert reader.slice(12_000, 13_000)["t"].tolist() == [12_500, 13_000]
    assert reader.slice(9_000, 10_000)["t"].size == 0
    assert reader.slice(13_000, 14_000)["t"].size == 0


def test_causal_tail_event_window_uses_the_end_of_rgb_interval() -> None:
    assert event_window_start_timestamp(10_000, 11_000, 1.0) == 10_000
    assert event_window_start_timestamp(10_000, 11_000, 0.5) == 10_500
    assert event_window_start_timestamp(10_000, 11_001, 0.25) == 10_750
    assert event_window_contract(0.25)["event_window"] == (
        "rgb_interval_tail_fraction"
    )


def test_event_reader_counts_without_loading_event_payload(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "zurich_city_00_a")
    event_path = (
        tmp_path
        / "train_events"
        / "zurich_city_00_a"
        / "events"
        / "left"
    )
    reader = DSECEventReader(
        event_path / "events.h5",
        rectify_map_path=event_path / "rectify_map.h5",
        output_height=4,
        output_width=5,
    )

    assert reader.count(10_000, 11_000) == 3
    assert reader.count(10_500, 11_000) == 2


def test_rectification_quantizes_before_bounds_and_event_count(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "zurich_city_00_a")
    event_path = (
        tmp_path
        / "train_events"
        / "zurich_city_00_a"
        / "events"
        / "left"
    )
    with h5py.File(event_path / "rectify_map.h5", "r+") as h5_file:
        h5_file["rectify_map"][0, 4] = np.array([4.6, 0.0], dtype=np.float32)
    reader = DSECEventReader(
        event_path / "events.h5",
        rectify_map_path=event_path / "rectify_map.h5",
        output_height=4,
        output_width=5,
    )

    sliced = reader.slice(11_000, 12_000)
    assert sliced["t"].tolist() == [11_400]
    assert sliced["x"].tolist() == [3]


def test_sequence_dataset_never_crosses_sequence_boundary(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "sequence_a")
    _write_sequence(tmp_path, "sequence_b")
    dataset = DSECSequenceDataset(
        tmp_path,
        split="train",
        sequences=["sequence_a", "sequence_b"],
        sequence_length=2,
        clip_stride=1,
        image_directory="aligned_event",
        rectify_events=True,
        event_representation=GEPEventFrame(height=4, width=5),
        transform=PairedSequenceTransform(height=4, width=5),
    )

    assert len(dataset) == 4
    for sample in dataset:
        assert sample["events"].shape == (2, 3, 4, 5)
        assert sample["images"].shape == (2, 3, 4, 5)
        assert torch.all(torch.diff(sample["frame_indices"]) == 1)
        assert sample["sequence_name"] in {"sequence_a", "sequence_b"}


def test_sequence_dataset_can_use_causal_tail_event_windows(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "sequence_a")
    dataset = DSECSequenceDataset(
        tmp_path,
        split="train",
        sequences=["sequence_a"],
        sequence_length=2,
        clip_stride=2,
        image_directory="aligned_event",
        rectify_events=True,
        event_representation=GEPEventFrame(height=4, width=5),
        event_window_fraction=0.5,
        transform=PairedSequenceTransform(height=4, width=5),
    )

    assert dataset[0]["event_counts"].tolist() == [2, 1]


def test_evaluation_clips_cover_each_frame_once_with_boundary_flags(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path, "sequence_a")
    dataset = DSECSequenceDataset(
        tmp_path,
        split="train",
        sequences=["sequence_a"],
        sequence_length=2,
        clip_stride=2,
        include_incomplete_clips=True,
        image_directory="aligned_event",
        rectify_events=True,
        event_representation=GEPEventFrame(height=4, width=5),
        transform=PairedSequenceTransform(height=4, width=5),
    )

    assert len(dataset) == 2
    assert dataset[0]["frame_indices"].tolist() == [1, 2]
    assert dataset[1]["frame_indices"].tolist() == [3]
    assert dataset[0]["is_sequence_start"] is True
    assert dataset[0]["is_sequence_end"] is False
    assert dataset[1]["is_sequence_start"] is False
    assert dataset[1]["is_sequence_end"] is True
