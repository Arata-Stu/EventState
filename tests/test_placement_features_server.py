"""Semantic placement feature streaming. Run with the server's PyTorch CPU environment."""
import json

from omegaconf import OmegaConf
import pytest
import torch

from event_state.dense import source
from event_state.evaluation.placement_features import (
    BLOCKS, CONDITIONS, FOUR_STAGE, PlacementFeaturePair, _check_semantic_provenance, checkpoint_paths,
)


@pytest.mark.parametrize("dataset", ["dsec", "m3ed"])
def test_feature_pass_replays_all_frames_with_fresh_state_and_semantic_geometry(monkeypatch, tmp_path, dataset):
    from event_state.evaluation.activity_scenes import eligible_sequences

    pair = PlacementFeaturePair.__new__(PlacementFeaturePair)
    pair.dataset, pair.device, pair.prepared_root = dataset, torch.device("cpu"), tmp_path
    seq = eligible_sequences(dataset, "semantic")[0][0]
    height = 448 if dataset == "dsec" else 352
    pair.input_size = (height, 640)
    pair.mean, pair.std = torch.zeros(3, 1, 1), torch.ones(3, 1, 1)
    pair.cfg = OmegaConf.create({"dataset": {"input_height": height, "input_width": 640,
        "representation": {"normalize_mean": [0, 0, 0], "normalize_std": [1, 1, 1]}}})
    image = torch.ones(3, 480 if dataset == "dsec" else 360, 640)
    if dataset == "m3ed":
        image[:, :4] = 0
        image[:, 356:] = 0
    samples = [dict(timestamp=t, events=image, frame_index=i) for i, t in enumerate((100, 201, 325), 1)]
    monkeypatch.setattr(source, "prepared_frames", lambda *args: iter(samples))

    class Model:
        def __init__(self, condition):
            self.condition = condition

        def forward_depth_stages(self, value, state):
            assert value.shape == (1, 1, 3, height, 640)
            if dataset == "dsec":
                assert torch.all(value[0, 0, :, :440] == 1) and torch.all(value[0, 0, :, 440:] == 0)
            else:
                assert torch.all(value == 1)
            # Unique nested states expose sharing across models or calibration and render passes.
            if state is None:
                number = 1
            else:
                assert state[0][0] == self.condition
                number = state[0][1] + 1
            levels = [torch.full((1, 1, 1, 1), float(block + number)).expand(1, 1, height // 16 * 40, 384)
                      for block in BLOCKS]
            return levels, ((self.condition, number),)

    pair.models = {key: Model(key) for key in CONDITIONS}
    for _ in range(2):
        times = []
        for index, (frame, maps) in enumerate(pair.frames(seq), 1):
            times.append(frame["timestamp"])
            for key in CONDITIONS:
                for block in BLOCKS:
                    assert maps[key][block].shape == (height // 16, 40, 384)
                    assert maps[key][block][0, 0, 0] == block + index
        assert times == [100, 201, 325]


def test_only_matched_stage1_and_retained_semantic_receipts_are_accepted(tmp_path):
    paths = checkpoint_paths("dsec", tmp_path)
    assert all(path.name == "step_00099999.pt" for path in paths.values())
    assert "stage1_joined_baseline_full_20261107/dsec_E1_w1/" in paths["E1"].as_posix()
    assert "stage1_placement_full_20261118/dsec_E1_b3_6_9_12_w1/" in paths[FOUR_STAGE].as_posix()
    receipt = tmp_path / "outputs/stage2_placement_semantic_20261118" / f"dsec_{FOUR_STAGE}" / "verified.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(dataset="dsec", condition=FOUR_STAGE, after_blocks=list(BLOCKS),
                                       global_step=99999, seed=0, backbone_sha256="evaluated")))
    assert _check_semantic_provenance(tmp_path, "dsec", FOUR_STAGE, "evaluated") == [str(receipt)]
    with pytest.raises(ValueError, match="checkpoint differs"):
        _check_semantic_provenance(tmp_path, "dsec", FOUR_STAGE, "other")
