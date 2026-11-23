"""Small CPU fixtures for the sparse four-block M3ED Semantic readout."""

import json
from importlib.util import find_spec
from pathlib import Path

import h5py
import pytest
import torch

from event_state.dense.common_features import CommonFeatures
from event_state.segmentation.m3ed import M3EDSemanticFeatureDataset
from event_state.segmentation.stage_readout import STAGE_BLOCKS


class CountingModel:
    def forward_depth_stages(self, inputs, state):
        state = inputs.mean() + (0 if state is None else state)
        levels = [torch.full((1, 1, 2, 384), block) + state for block in STAGE_BLOCKS]
        return levels, state


def test_sparse_stage_readout_preserves_history_and_native_targets(tmp_path):
    net = CommonFeatures.__new__(CommonFeatures)
    net.device = "cpu"
    net.mean = net.std = torch.full((3, 1, 1), 0.5)
    net.model, net.state, net.stage_readout = CountingModel(), None, True
    frame = torch.ones(3, 16, 32)
    assert net.step(frame, ()) == {}
    maps = net.step(frame, ["stages"])
    staged = maps["stages"]
    assert staged.dtype == torch.float16 and staged.shape == (1536, 1, 2)
    for block, channels in zip(STAGE_BLOCKS, staged.chunk(4)):
        assert channels.eq(block + 2).all()
    net.state = None
    assert net.step(frame, ["stages"])["stages"][-1, 0, 0] == 13

    features = tmp_path / "features" / "seq"
    targets = tmp_path / "targets" / "seq"
    features.mkdir(parents=True)
    targets.mkdir(parents=True)
    metadata = dict(stage_blocks=list(STAGE_BLOCKS), checkpoint_sha256="a" * 64,
                    state_policy="continuous", sequence_name="seq")
    (features.parent / "identity.json").write_text(json.dumps(metadata))
    (features / "metadata.json").write_text(json.dumps(dict(
        metadata, task="m3ed_semantic", features=["stages"], cached_frame_indices=[2],
        label_size=[16, 32], patch_size=16)))
    payload = dict(metadata, frame_index=2, features=maps, timestamp=300)
    path = features / "000002.pt"
    torch.save(payload, path)
    with h5py.File(targets / "targets.h5", "w") as source:
        source["semantics_11"] = torch.full((3, 16, 32), 7, dtype=torch.uint8).numpy()
        source["semantics_frame_valid"] = [False, False, True]
        source["timestamps"] = [100, 200, 300]
    dataset = M3EDSemanticFeatureDataset(feature_cache_dir=features.parent,
                                         target_cache_dir=targets.parent,
                                         sequences=["seq"], feature="stages")
    sample = dataset[0]
    assert len(dataset) == 1 and sample["timestamp"] == 300
    assert sample["label"].eq(7).all() and sample["label"].shape == (16, 32)
    torch.testing.assert_close(sample["feature"], staged.float())
    torch.save(dict(payload, stage_blocks=[12, 9, 6, 3]), path)
    with pytest.raises(ValueError, match="feature mismatch"):
        dataset[0]


@pytest.mark.parametrize("condition", ["E1", "E1_b3_6_9_12"])
def test_frozen_final_channels_match_h_cache_on_both_placements(condition):
    """Use the real DINO intermediate/forward_features APIs on 2x3 patch grids, offline."""
    from event_state.models import (
        DINO_V3_DEFAULT_REPOSITORY, EventEncoder, EventStateModel, build_temporal_backbone,
    )
    owner_repo, revision = DINO_V3_DEFAULT_REPOSITORY.split(":")
    checkout = Path(torch.hub.get_dir()) / f"{owner_repo.replace('/', '_')}_{revision}"
    if checkout.is_dir():
        backend = dict(source="torch_hub", repository=checkout, hub_source="local")
    elif find_spec("dinov3") is not None:
        backend = dict(source="package")
    else:
        pytest.skip("Requires the installed DINOv3 package or the project's cached pinned checkout")

    torch.manual_seed(34)
    encoder = EventEncoder(pretrained=False, **backend)
    memory = (build_temporal_backbone("lstm") if condition == "E1" else
              build_temporal_backbone("four_stage_lstm", after_blocks=STAGE_BLOCKS))
    model = EventStateModel(encoder, memory).eval()
    streams = []
    for stage_readout in (False, True):
        net = CommonFeatures.__new__(CommonFeatures)
        net.model, net.state, net.stage_readout = model, None, stage_readout
        net.device = "cpu"
        net.mean = torch.tensor([0.1, 0.2, 0.3])[:, None, None]
        net.std = torch.tensor([0.7, 0.8, 0.9])[:, None, None]
        streams.append(net)
    for i, frame in enumerate(torch.rand(3, 3, 32, 48)):
        # The middle input has no label but must advance both recurrent streams.
        historical = streams[0].step(frame, ["h"] if i != 1 else ())
        multi = streams[1].step(frame, ["stages"] if i != 1 else ())
        if i != 1:
            torch.testing.assert_close(multi["stages"][-384:], historical["h"], atol=1e-5, rtol=1e-3)
