"""Exercise sparse DSEC four-block extraction with the training server's PyTorch on CPU."""

import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from event_state.segmentation import DSECSemanticFeatureDataset, EventStateSegmentationHead
from event_state.segmentation.stage_readout import stage_feature_map
from tools import cache_dsec_semantic_features as cache


def test_four_block_linear_head_contains_final_only_readout():
    grid = (2, 3)
    generator = torch.Generator().manual_seed(0)
    levels = [torch.randn(1, 1, 6, 2, generator=generator) for _ in range(4)]
    final_map = levels[-1].reshape(2, 3, 2).permute(2, 0, 1).half().float()
    stages = stage_feature_map(levels, grid).float()
    final_head = EventStateSegmentationHead(in_channels=2, num_classes=3, output_size=(8, 12))
    stage_head = EventStateSegmentationHead(in_channels=8, num_classes=3, output_size=(8, 12))
    with torch.no_grad():
        stage_head.classifier.weight.zero_()
        stage_head.classifier.weight[:, -2:].copy_(final_head.classifier.weight)
        stage_head.classifier.bias.copy_(final_head.classifier.bias)
        torch.testing.assert_close(stage_head(stages[None]), final_head(final_map[None]))


def test_stages_run_on_every_frame_and_preserve_native_label_geometry(tmp_path, monkeypatch):
    split = cache.load_dsec_semantic_split(cache.DEFAULT_SPLIT)
    seq = split.val[0]
    events = tmp_path / "events" / seq
    labels = tmp_path / "labels" / "train" / seq / "11classes"
    events.mkdir(parents=True)
    labels.mkdir(parents=True)
    (events / "metadata.json").write_text("{}")
    image = torch.ones((3, 480, 640))
    for index in range(1, 4):
        torch.save(dict(events=image, frame_index=index, timestamp=100 * index), events / f"{index:06}.pt")
    Image.fromarray(np.zeros((440, 640), dtype=np.uint8)).save(labels / "000003.png")
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"frozen fixture")
    monkeypatch.setattr(sys, "argv", ["cache", "--checkpoint", str(checkpoint),
        "--event-cache-dir", str(events.parent), "--labels-root", str(tmp_path / "labels"),
        "--output-dir", str(tmp_path / "cache"), "--role", "val", "--num-shards", "2",
        "--features", "stages", "--device", "cpu"])
    cfg = {"dataset": {"input_width": 640, "input_height": 448,
                       "representation": {"normalize_mean": [0] * 3, "normalize_std": [1] * 3}}}
    monkeypatch.setattr(cache, "load_checkpoint_config_metadata", lambda path: SimpleNamespace(config=cfg))
    monkeypatch.setattr(cache, "load_checkpoint", lambda *args, **kwargs: SimpleNamespace(global_step=99999))

    class Backbone(torch.nn.Module):
        event_encoder = SimpleNamespace(patch_size=16)

        def forward_depth_stages(self, value, state):
            assert value.shape == (1, 1, 3, 448, 640)
            assert (value[..., :440, :] == 1).all() and (value[..., 440:, :] == 0).all()
            step = (0 if state is None else int(state[0][0])) + 1
            # Each patch and level differ to expose swapped blocks or grid axes.
            patches = torch.arange(28 * 40).view(1, 1, -1, 1)
            return [(patches + step + block).expand(-1, -1, -1, 384) for block in (3, 6, 9, 12)], ((torch.tensor(step),),)

    monkeypatch.setattr(cache, "build_model", lambda *args, **kwargs: Backbone())
    cache.main()
    destination = tmp_path / "cache" / seq
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["stage_blocks"] == [3, 6, 9, 12]
    assert metadata["state_policy"] == "continuous"
    assert sorted(p.name for p in destination.glob("*.pt")) == ["000003.pt"]
    frame = cache._safe_load(destination / "000003.pt")
    assert frame["features"]["stages"].dtype == torch.float16
    data = DSECSemanticFeatureDataset(feature_cache_dir=destination.parent, labels_root=tmp_path / "labels",
                                      sequences=[seq], role="val", feature="stages")
    sample = data[0]
    assert sample["feature"].shape == (1536, 28, 40)
    assert data.readout_provenance == {"stage_blocks": [3, 6, 9, 12], "backbone_sha256": cache._sha256(checkpoint)}
    assert sample["timestamp"] == 300
    torch.testing.assert_close(sample["feature"][::384, 2, 5], torch.tensor([91, 94, 97, 100]).float())
    head = EventStateSegmentationHead(in_channels=1536, num_classes=11, output_size=data.label_size)
    assert head(sample["feature"][None]).shape == (1, 11, 440, 640)
    frame["checkpoint_sha256"] = "a" * 64
    torch.save(frame, destination / "000003.pt")
    with pytest.raises(ValueError, match="feature mismatch"):
        data[0]
