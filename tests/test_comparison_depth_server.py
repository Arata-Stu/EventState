"""CPU/server Depth video contracts; no datasets, pretrained weights or GPU."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from event_state.dense.protocol import digest, load_protocol, sha256_file
from event_state.evaluation.comparison_depth import DepthBackend, DepthTargets


def target_fixture(tmp_path, dataset="dsec"):
    protocol = load_protocol(dataset + "_f3")
    sources, records = {}, []
    for sequence in protocol["splits"]["val"]:
        folder = tmp_path / sequence
        folder.mkdir()
        source = folder / "metadata.json"
        source.write_text("source fixture")
        sources[str(source)] = sha256_file(source)
        records.append(dict(sequence=sequence, timestamp=100,
                            target_path=str(folder / "target"), encoding="fixture"))
    index = dict(protocol=protocol, protocol_sha256=digest(protocol), dataset=dataset,
                 role="val", labeled=True, task="depth", source_identity=sources, records=records)
    folder = tmp_path / "targets"
    folder.mkdir()
    path = folder / f"{dataset}_val.json"
    path.write_text(json.dumps(index))
    return protocol["splits"]["val"][0], path, index


def test_depth_target_exact_timestamps_and_existing_mask(tmp_path):
    sequence, _, _ = target_fixture(tmp_path)
    targets = DepthTargets(tmp_path, "dsec", sequence)
    depth = np.ones((480, 640), dtype=np.float32)
    depth[0, :6] = [0.1, 0.1001, 80, 80.01, np.nan, 3]
    mask = np.ones(depth.shape, dtype=bool)
    mask[0, 5] = False
    with patch("event_state.dense.targets.read_target", return_value=(depth, mask)) as read:
        assert targets.get(99) is None
        assert targets.get(101) is None
        read.assert_not_called()
        returned, valid = targets.get(100)
    assert returned is depth
    assert valid[0, :6].tolist() == [False, True, True, False, False, False]
    assert mask[0, 0]  # The source's validity array was not modified.
    assert read.call_args.args[1] == "depth"


@pytest.mark.parametrize("field,value", [("role", "test"), ("labeled", False),
                                          ("task", "disparity"), ("protocol_sha256", "wrong")])
def test_depth_target_rejects_other_evaluation_contract(tmp_path, field, value):
    sequence, path, index = target_fixture(tmp_path)
    index[field] = value
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="public-val"):
        DepthTargets(tmp_path, "dsec", sequence)


def test_depth_target_rejects_duplicate_membership_and_changed_source(tmp_path):
    sequence, path, index = target_fixture(tmp_path)
    duplicated = copy.deepcopy(index)
    duplicated["records"].append(duplicated["records"][0])
    path.write_text(json.dumps(duplicated))
    with pytest.raises(ValueError, match="Duplicate"):
        DepthTargets(tmp_path, "dsec", sequence)
    path.write_text(json.dumps(index))
    source = next(p for p in index["source_identity"] if sequence in p)
    with open(source, "w") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="provenance changed"):
        DepthTargets(tmp_path, "dsec", sequence)


@pytest.mark.parametrize("dataset,native_height,height", [("dsec", 480, 480), ("m3ed", 360, 352)])
def test_depth_backend_reuses_evaluated_predictor_and_full_state_updates(tmp_path, dataset, native_height, height):
    prepared = tmp_path / "events"
    prepared.mkdir()
    predictor = SimpleNamespace(cfg=OmegaConf.create({"dataset": {"event_cache_dir": str(prepared)}}),
                                identity={"head_sha256": "head", "backbone_sha256": "encoder", "feature": "stages"},
                                pretraining_audit={"status": "declared_inputs_disjoint"}, state="carried")
    calls = []
    def step(events, predict=True):
        calls.append((events.clone(), predict))
        return torch.ones((height, 640)) if predict else None
    predictor.step = step
    with patch("event_state.dense.benchmark_inference.DepthPredictor", return_value=predictor) as factory:
        backend = DepthBackend(dataset, "E1", tmp_path, "run", teacher_checkpoint="teacher.pt", device="cpu")
    args, kwargs = factory.call_args
    assert args[0] == tmp_path / "run"
    assert args[1:] == (dataset + "_E1", "stages", "cpu")
    assert kwargs["backbone"] == tmp_path / "outputs/stage1_joined_baseline_full_20261107" / (dataset + "_E1_w1") / "checkpoints/step_00099999.pt"
    assert kwargs["teacher_weights"] == tmp_path / "teacher.pt"
    backend.reset()
    assert predictor.state is None
    image = torch.arange(native_height, dtype=torch.float32)[None, :, None].expand(3, native_height, 640)
    assert backend.step({"events": image}, predict=False) is None
    result = backend.step({"events": image})
    assert isinstance(result, np.ndarray) and result.shape == (height, 640)
    assert [predict for _, predict in calls] == [False, True]
    assert calls[0][0][0, 0, 0].item() == (4 if dataset == "m3ed" else 0)
    assert backend.frame_size == (height, 640)
    assert backend.identity["feature"] == "stages"
