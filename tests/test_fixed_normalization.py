import json

import pytest
import torch
from omegaconf import OmegaConf

from event_state.data.dsec import event_representation_metadata
from event_state.data.fixed_normalization import resolve_fixed_normalization
from event_state.training.data import _build_transform, build_event_representation


def test_embed_portable_coefficients_and_normalize_zero_pixels(tmp_path):
    config = OmegaConf.create({"dataset": {
        "sensor_height": 1, "sensor_width": 2, "input_height": 1, "input_width": 2,
        "rectify_events": True, "representation": {
            "type": "voxel_grid", "channels": 2, "event_bins": 1,
            "channel_layout": "polarity_major", "voxel_normalization": "none",
            "fixed_normalization_required": True,
        },
    }})
    stats = {
        "format": "eventstate_fixed_normalization_v1",
        "representation": event_representation_metadata(build_event_representation(config.dataset)),
        "sensor_size": [1, 2], "includes_zero_pixels": True,
        "event_window_fraction": 1.0, "event_window_boundary": "(previous_timestamp,timestamp]",
        "coordinate_space": "rectified_event", "mean": [1., 2.], "std": [2., 4.],
    }
    from event_state.data.dsec import event_window_contract
    stats["event_window_boundary"] = event_window_contract(1.0)["event_window_boundary"]
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    config.dataset.representation.fixed_normalization_file = str(path)
    resolve_fixed_normalization(config)
    saved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    path.unlink()
    resolve_fixed_normalization(saved)  # Embedded coefficients need no original file.
    transform = _build_transform(saved.dataset, training=False, channels=2)
    result, _ = transform(torch.tensor([[[[0., 3.]], [[0., 6.]]]]), None)
    assert torch.equal(result, torch.tensor([[[[-.5, 1.]], [[-.5, 1.]]]]))
    saved.dataset.representation.voxel_normalization = "nonzero_standardize"
    with pytest.raises(ValueError):
        _build_transform(saved.dataset, training=False, channels=2)
