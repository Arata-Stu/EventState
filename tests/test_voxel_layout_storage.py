from pathlib import Path

import pytest
import torch

from event_state.data.dsec import (
    EVENT_CACHE_FORMAT_VERSION,
    event_representation_metadata,
    event_window_contract,
    validate_event_cache_payload,
)
from event_state.data.event_representation import EventVoxelizer


def test_polarity_major_is_positive_first_permutation() -> None:
    args = dict(num_bins=3, height=1, width=2, normalization="none")
    events = dict(
        x=torch.tensor([0, 1, 0, 1]), y=torch.zeros(4),
        t=torch.tensor([2, 4, 6, 10]), p=torch.tensor([1, 0, 1, 0]),
        start_time=0, end_time=10,
    )
    legacy = EventVoxelizer(**args)(**events)
    new = EventVoxelizer(**args, channel_layout="polarity_major")(**events)
    assert torch.equal(new, legacy[[1, 3, 5, 0, 2, 4]])
    assert new[:3].sum() == 2
    assert new[3:].sum() == 2
    assert event_representation_metadata(EventVoxelizer(**args)) != (
        event_representation_metadata(EventVoxelizer(**args, channel_layout="polarity_major"))
    )
    with pytest.raises(ValueError, match="channel layout"):
        EventVoxelizer(channel_layout="invalid")


def test_fp16_cache_roundtrip_and_dtype_contract(tmp_path: Path) -> None:
    representation = event_representation_metadata(
        EventVoxelizer(num_bins=2, height=1, width=1, channel_layout="polarity_major")
    )
    expected = dict(
        frame_index=1, timestamp=10, previous_timestamp=0, sequence_name="synthetic",
        split="train", height=1, width=1, representation=representation,
        rectified=True, input_fingerprint_digest="input", manifest_digest="manifest",
    )
    payload = dict(
        expected, format_version=EVENT_CACHE_FORMAT_VERSION, dataset="DSEC",
        event_window_boundary=event_window_contract(1.0)["event_window_boundary"],
        event_count=2, events=torch.tensor([0, 1.25, 0.5, 0.25]).view(4, 1, 1).half(),
    )
    path = tmp_path / "10.pt"
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=True)
    tensor, count = validate_event_cache_payload(
        loaded, cache_path=path, cache_dtype="float16", **expected,
    )
    assert loaded["events"].dtype == torch.float16
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, payload["events"].float())
    assert count == 2
    with pytest.raises(ValueError, match="float32"):
        validate_event_cache_payload(loaded, cache_path=path, **expected)
    loaded["representation"] = event_representation_metadata(EventVoxelizer(num_bins=2))
    with pytest.raises(ValueError, match="representation"):
        validate_event_cache_payload(loaded, cache_path=path, cache_dtype="float16", **expected)
