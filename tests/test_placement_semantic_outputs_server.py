"""Synthetic checkpoints and sparse native labels for the two placement heads (server CPU)."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest
import torch

from event_state.activity_quantiles import canonical_sha256
from event_state.dense.protocol import sha256_file
from event_state.evaluation import placement_semantic_outputs as outputs
from event_state.segmentation import (DSEC_SEMANTIC_11_CLASSES, EventStateSegmentationHead,
                                     load_dsec_semantic_split, load_m3ed_semantic_split)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("dataset", ["dsec", "m3ed"])
def test_load_exact_heads_and_decode_on_unlabeled_frames(tmp_path, monkeypatch, dataset):
    dsec = dataset == "dsec"
    role = "test" if dsec else "validation"
    manifest = outputs.ROOT / "tools/manifests" / ("dsec_semantic_split.yaml" if dsec else "m3ed_semantic_common_v1.yaml")
    split = load_dsec_semantic_split(manifest) if dsec else load_m3ed_semantic_split(manifest)
    evaluated = split.sequences(role)
    sequence = evaluated[0]
    frame_size = (440 if dsec else 352, 640)
    prepared = tmp_path / "prepared"
    labels = tmp_path / "labels"
    rgb_root = tmp_path / "rgb"
    for seq in evaluated:
        stamps = [100, 110, 121]
        if dsec:
            timeline = rgb_root / "test_images" / seq / "images/timestamps.txt"
            timeline.parent.mkdir(parents=True)
            timeline.write_text("100\n110\n121\n")
            dest = labels / "test" / seq / "11classes"
            dest.mkdir(parents=True)
            for idx in (0, 1):
                Image.fromarray(np.full(frame_size, 255, dtype=np.uint8)).save(dest / f"{idx:06}.png")
        else:
            write_json(prepared / seq / "metadata.json", dict(timestamps=stamps))
            folder = labels / seq
            folder.mkdir(parents=True)
            with h5py.File(folder / "targets.h5", "w") as h5:
                h5["timestamps"] = stamps
                h5["semantics_frame_valid"] = [True, True, False]
                h5.create_dataset("semantics_11", (3, *frame_size), dtype="u1", fillvalue=255)
    expected = {k: "same" for k in outputs.COMPARISON_HASHES}
    expected.update(samples_sha256=canonical_sha256(sorted((seq, 110) for seq in evaluated)),
                    state_policy_by_sequence={seq: "continuous" for seq in evaluated},
                    groups={"all": {"sequence_frames": {seq: 1 for seq in evaluated}}})
    pair = SimpleNamespace(dataset=dataset, cfg=SimpleNamespace(dataset=SimpleNamespace(root=str(rgb_root))),
                           device=torch.device("cpu"), frame_size=frame_size, prepared_root=prepared,
                           input_size=(448 if dsec else 352, 640),
                           identity={outputs.FOUR_STAGE: {"backbone_sha256": "backbone"}})
    receipt = dict(attempt="run-chosen", dataset=dataset, condition=outputs.FOUR_STAGE, role=role,
                   backbone_sha256="backbone")
    path = outputs._receipt(tmp_path, dataset)
    write_json(path, receipt)
    paths = outputs.semantic_head_paths(dataset, tmp_path)
    cfg = dict(split_manifest=str(manifest), training_sequences=list(split.official_train if dsec else split.train),
               validation_sequences=list(evaluated), protocol="official-final", labels_root=str(labels),
               target_cache_dir=str(labels), feature="h", in_channels=384, head_type="linear", head_width=192,
               num_classes=11, classes=list(DSEC_SEMANTIC_11_CLASSES), ignore_index=255, label_size=list(frame_size),
               precision="fp32")
    for i, (key, head_path) in enumerate(paths.items()):
        head = EventStateSegmentationHead(in_channels=384, output_size=frame_size)
        with torch.no_grad():
            head.classifier.weight.zero_()
            head.classifier.bias.zero_()
            # Class 3 wins only if a 1.0001 feature goes through the original fp16 cache rounding.
            head.classifier.weight[1, 0] = 1
            head.classifier.bias[3 + i] = 1.00005
        head_path.parent.mkdir(parents=True)
        torch.save(dict(config=cfg, model=head.state_dict()), head_path)
    baseline = dict(expected, head_sha256=sha256_file(paths["E1"]))
    receipt["head_sha256"] = sha256_file(paths[outputs.FOUR_STAGE])
    write_json(path, receipt)
    activity = dict(expected, head_sha256=receipt["head_sha256"])
    report_path = paths[outputs.FOUR_STAGE].parents[1] / ("activity_test.json" if dsec else "h/activity_groups.json")
    write_json(report_path, dict(role=role, feature="h", activity_evaluation=activity))
    monkeypatch.setattr(outputs, "baseline_results", lambda: {dataset: baseline})

    runtime = outputs.PlacementSemanticOutputs(dataset, tmp_path, pair, sequence)
    try:
        assert runtime.expected_gt_frames == 1
        assert runtime.target(100) is None
        assert (runtime.target(110) == 255).all()
        assert runtime.target(121) is None
        sample = torch.full((*runtime.grid_size, 384), 1.0001)
        predictions = runtime.predict({key: {12: sample} for key in paths})
        for i, key in enumerate(paths):
            assert predictions[key].shape == frame_size
            assert predictions[key].dtype == np.uint8
            assert (predictions[key] == 3 + i).all()
            assert runtime.identity[key]["head_sha256"] == sha256_file(paths[key])
    finally:
        runtime.close()

    # Same label count but different timestamp must invalidate both comparisons.
    if dsec:
        src = labels / "test" / sequence / "11classes/000001.png"
        src.rename(src.with_name("000002.png"))
    else:
        with h5py.File(labels / sequence / "targets.h5", "a") as h5:
            h5["semantics_frame_valid"][:] = [True, False, True]
    with pytest.raises(ValueError, match="GT timestamps"):
        outputs.PlacementSemanticOutputs(dataset, tmp_path, pair, sequence)

    paths["E1"].write_bytes(b"replaced head")
    with pytest.raises(ValueError, match="Semantic head or state policy"):
        outputs.PlacementSemanticOutputs(dataset, tmp_path, pair, sequence)
