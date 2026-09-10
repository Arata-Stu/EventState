from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from event_state.data import GEPEventFrame, M3EDSequenceDataset, PairedSequenceTransform
from event_state.data.m3ed import M3ED_PREPARED_FORMAT_VERSION


def _write_prepared_sequence(root: Path, name: str, frame_count: int = 4) -> None:
    sequence = root / name
    images = sequence / "aligned_rgb"
    events = sequence / "events"
    images.mkdir(parents=True)
    events.mkdir()
    timestamps = [1000 * (index + 1) for index in range(frame_count)]
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
                "frame_count": frame_count,
                "timestamps": timestamps,
                "event_size": [4, 4],
                "downsampling": "dagr_stateful_signed_event_downsampling",
                "coordinate_space": "rectified_left_event",
                "representation": {"type": "gep_rgb", "channels": 3},
            }
        ),
        encoding="utf-8",
    )


def test_m3ed_prepared_dataset_matches_training_contract(tmp_path: Path) -> None:
    _write_prepared_sequence(tmp_path, "traffic_stop")
    dataset = M3EDSequenceDataset(
        root=tmp_path,
        prepared_root=tmp_path,
        split="train",
        sequences=["traffic_stop"],
        sequence_length=2,
        clip_stride=1,
        event_representation=GEPEventFrame(height=4, width=4),
        transform=PairedSequenceTransform(height=4, width=4),
        feature_cache_dir=None,
        event_cache_dir=tmp_path,
        load_events=True,
        load_images=True,
    )
    sample = dataset[0]
    assert sample["events"].shape == (2, 3, 4, 4)
    assert sample["images"].shape == (2, 3, 4, 4)
    assert sample["event_counts"].tolist() == [1, 2]
    assert sample["timestamps"].tolist() == [2000, 3000]
    assert sample["is_sequence_start"] is True


def test_m3ed_validation_can_emit_short_final_clip(tmp_path: Path) -> None:
    _write_prepared_sequence(tmp_path, "traffic_stop", frame_count=5)
    dataset = M3EDSequenceDataset(
        root=tmp_path,
        prepared_root=tmp_path,
        split="validation",
        sequences=["traffic_stop"],
        sequence_length=3,
        clip_stride=3,
        include_incomplete_clips=True,
        event_representation=GEPEventFrame(height=4, width=4),
        transform=PairedSequenceTransform(height=4, width=4),
        event_cache_dir=tmp_path,
        load_events=True,
        load_images=False,
    )
    assert len(dataset) == 2
    assert dataset[1]["events"].shape[0] == 1
    assert dataset[1]["is_sequence_end"] is True
