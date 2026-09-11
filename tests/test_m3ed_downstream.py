from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from event_state.data import GEPEventFrame, M3EDSequenceDataset, PairedSequenceTransform
from event_state.data.m3ed import M3ED_PREPARED_FORMAT_VERSION
from event_state.data.m3ed_downstream import (
    M3ED_DOWNSTREAM_FORMAT_VERSION,
    camera_relative_motion,
    map_cityscapes_19_to_dsec_11,
    match_nearest_timestamps,
)


def _write_prepared(root: Path, name: str) -> list[int]:
    sequence = root / name
    images = sequence / "aligned_rgb"
    events = sequence / "events"
    images.mkdir(parents=True)
    events.mkdir()
    timestamps = [1_000, 2_000, 3_000, 4_000]
    for index, timestamp in enumerate(timestamps):
        Image.new("RGB", (4, 4), color=(index, index, index)).save(
            images / f"{timestamp}.png"
        )
        if index:
            torch.save(
                {
                    "format_version": M3ED_PREPARED_FORMAT_VERSION,
                    "dataset": "M3ED",
                    "sequence_name": name,
                    "frame_index": index,
                    "timestamp": timestamp,
                    "previous_timestamp": timestamps[index - 1],
                    "events": torch.full((3, 4, 4), float(index)),
                    "event_count": index,
                },
                events / f"{timestamp}.pt",
            )
    (sequence / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": M3ED_PREPARED_FORMAT_VERSION,
                "dataset": "M3ED",
                "sequence_name": name,
                "frame_count": len(timestamps),
                "timestamps": timestamps,
                "event_size": [4, 4],
                "recommended_model_input_size": [4, 4],
                "downsampling": "dagr_stateful_signed_event_downsampling",
                "coordinate_space": "rectified_left_event",
                "representation": {"type": "gep_rgb", "channels": 3},
            }
        ),
        encoding="utf-8",
    )
    return timestamps


def _write_targets(root: Path, name: str, timestamps: list[int]) -> None:
    sequence = root / name
    sequence.mkdir(parents=True)
    frame_count = len(timestamps)
    with h5py.File(sequence / "targets.h5", "w") as target:
        target.create_dataset("timestamps", data=timestamps)
        target.create_dataset(
            "depth_m", data=np.ones((frame_count, 4, 4), dtype=np.float32)
        )
        target.create_dataset(
            "depth_valid", data=np.ones((frame_count, 4, 4), dtype=np.bool_)
        )
        target.create_dataset("depth_frame_valid", data=np.ones(frame_count, dtype=np.bool_))
        target.create_dataset(
            "semantics_19", data=np.full((frame_count, 4, 4), 13, dtype=np.uint8)
        )
        target.create_dataset(
            "semantics_11", data=np.full((frame_count, 4, 4), 8, dtype=np.uint8)
        )
        target.create_dataset(
            "semantics_frame_valid", data=np.ones(frame_count, dtype=np.bool_)
        )
    (sequence / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": M3ED_DOWNSTREAM_FORMAT_VERSION,
                "dataset": "M3ED",
                "sequence_name": name,
                "prepared_format_version": M3ED_PREPARED_FORMAT_VERSION,
                "frame_count": frame_count,
                "timestamps": timestamps,
                "tasks": ["depth", "semantics"],
                "coordinate_space": "rectified_left_event",
                "target_size": [4, 4],
            }
        ),
        encoding="utf-8",
    )
    (sequence / "_SUCCESS").write_text("complete\n", encoding="utf-8")


def test_nearest_timestamp_matching_respects_tolerance() -> None:
    matches = match_nearest_timestamps(
        np.asarray([100, 200, 300]),
        np.asarray([90, 205, 450]),
        max_delta_us=20,
    )
    assert matches.indices.tolist() == [0, 1, -1]
    assert matches.deltas_us.tolist() == [10, -5, -150]
    assert matches.valid.tolist() == [True, True, False]


def test_cityscapes_mapping_preserves_ignore_label() -> None:
    labels = np.asarray([[0, 10, 13, 18, 255]], dtype=np.uint8)
    assert map_cityscapes_19_to_dsec_11(labels).tolist() == [[5, 0, 8, 8, 255]]
    with pytest.raises(ValueError, match="outside Cityscapes"):
        map_cityscapes_19_to_dsec_11(np.asarray([[19]], dtype=np.uint8))


def test_relative_motion_uses_m3ed_camera_convention() -> None:
    previous = np.eye(4)
    current = np.eye(4)
    current[0, 3] = 2.0
    relative, linear, angular = camera_relative_motion(
        previous, current, delta_seconds=2.0
    )
    assert relative[0, 3] == pytest.approx(-2.0)
    assert linear.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert angular.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_m3ed_dataset_loads_aligned_downstream_targets(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    target_root = tmp_path / "targets"
    timestamps = _write_prepared(prepared_root, "traffic_stop")
    _write_targets(target_root, "traffic_stop", timestamps)
    dataset = M3EDSequenceDataset(
        root=tmp_path,
        prepared_root=prepared_root,
        split="train",
        sequences=["traffic_stop"],
        sequence_length=2,
        clip_stride=1,
        event_representation=GEPEventFrame(height=4, width=4),
        transform=PairedSequenceTransform(height=4, width=4),
        event_cache_dir=prepared_root,
        load_events=True,
        load_images=False,
        target_cache_dir=target_root,
        target_tasks=["depth", "semantics"],
    )
    sample = dataset[0]
    assert sample["depth_m"].shape == (2, 4, 4)
    assert sample["depth_valid"].all()
    assert sample["semantics_11"].unique().tolist() == [8]
    assert sample["frame_indices"].tolist() == [1, 2]
