"""Run on the ML host: prepared M3ED activity and split integration."""
from pathlib import Path

import pytest
import torch

from event_state.data import GEPEventFrame, M3EDSequenceDataset, PairedSequenceTransform
from test_m3ed import _write_prepared_sequence


def test_m3ed_activity_uses_unnormalized_cache_and_preserves_inputs(tmp_path):
    _write_prepared_sequence(tmp_path, "traffic_stop")
    # White inactive followed by black active. Normalization turns white to zero,
    # which must never change its activity classification.
    for timestamp, value in ((2000, 1.0), (3000, 0.0)):
        path = tmp_path / "traffic_stop/events" / f"{timestamp}.pt"
        payload = torch.load(path, weights_only=True)
        payload["events"].fill_(value)
        torch.save(payload, path)
    kwargs = dict(root=tmp_path, split="train", sequences=["traffic_stop"],
                  sequence_length=2, event_representation=GEPEventFrame(height=4, width=4),
                  transform=PairedSequenceTransform(height=16, width=16,
                                                   event_mean=(1., 1., 1.),
                                                   event_std=(0.5, 0.5, 0.5)),
                  load_images=False)
    ordinary = M3EDSequenceDataset(**kwargs)[0]
    active = M3EDSequenceDataset(**kwargs, activity_mask=True)[0]
    assert active["event_activity"].tolist() == [[False], [True]]
    assert "event_activity" not in ordinary
    for key in ("events", "event_counts", "timestamps", "frame_indices"):
        torch.testing.assert_close(active[key], ordinary[key])
    torch.testing.assert_close(active["events"][0], torch.zeros(3, 16, 16))


@pytest.mark.parametrize("experiment", ["h_distill_lstm_zloss", "activity_dual", "activity_z_h_soft"])
def test_m3ed_activity_configs_and_split_guard(experiment):
    from hydra import compose, initialize_config_dir
    from event_state.training.factory import validate_config
    from event_state.training.data import _dataset_options

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(config_name="config", overrides=[
            "dataset=m3ed_half_dagr", "model=lstm", f"experiment={experiment}",
            "+dataset.pretraining_protocol=m3ed_activity", "training.validation_enabled=true",
            "training.batch_size=4", "training.sampling.random_batch_size=4",
        ])
    validate_config(config)
    options = _dataset_options(config.dataset, config.teacher, None)
    assert options["activity_mask"] == (experiment != "h_distill_lstm_zloss")
    original = list(config.dataset.train_sequences)
    for forbidden in ("car_urban_day_ucity_small_loop", "car_urban_day_rittenhouse"):
        config.dataset.train_sequences = original[:-1] + [forbidden]
        with pytest.raises(ValueError, match="established train"):
            validate_config(config)
    config.dataset.train_sequences = original
    config.dataset.val_sequences = ["car_urban_day_rittenhouse"]
    with pytest.raises(ValueError, match="established validation"):
        validate_config(config)
