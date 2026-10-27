#!/usr/bin/env python3
"""Score public labels or export official DSEC hidden-test disparity predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Evaluation JSON, or a new prediction directory with --export")
    p.add_argument("--export", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Existing evaluation/predictions are preserved")
    import torch
    from event_state.dense.data import DenseFeatureDataset
    from event_state.dense.head import DenseRegressionHead
    from event_state.dense.protocol import sha256_file
    from event_state.dense.runtime import evaluate
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    dataset = DenseFeatureDataset(a.cache, config["feature"], require_labels=not a.export,
                                  minimum=config["minimum"], maximum=config["maximum"])
    for key, expected in (("checkpoint_sha256", config["backbone_sha256"]),
                          ("protocol_sha256", config["protocol_sha256"]),
                          ("task", config["task"]), ("geometry", config["geometry"]),
                          ("low_activity_threshold", config["low_activity_threshold"]),
                          ("low_activity_metric", config["low_activity_metric"]),
                          ("normalization_audit", config["normalization_audit"])):
        if dataset.meta[key] != expected:
            raise ValueError(f"Head/evaluation feature provenance mismatch: {key}")
    model = DenseRegressionHead(config["in_channels"], config["width"]).to(a.device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    report = dict(protocol=dataset.protocol, role=dataset.meta["role"], task=config["task"],
                  feature=config["feature"], state_policy=dataset.meta["state_policy"],
                  head_sha256=sha256_file(a.checkpoint), backbone_sha256=config["backbone_sha256"],
                  cache_metadata_sha256=sha256_file(a.cache / "metadata.json"),
                  protocol_sha256=config["protocol_sha256"], minimum=config["minimum"],
                  maximum=config["maximum"], scale_alignment="none", epoch=payload["epoch"],
                  selection=config["selection"], low_activity_threshold=config["low_activity_threshold"],
                  low_activity_metric=config["low_activity_metric"],
                  normalization_audit=config["normalization_audit"], geometry=config["geometry"],
                  metric_aggregation="valid-pixel pooled; lower errors are better",
                  units="meters" if config["task"] == "depth" else "native disparity pixels")
    if not a.export:
        report.update(evaluate(model, dataset, torch.device(a.device)))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["metrics"], indent=2))
        return
    if dataset.meta["dataset"] != "dsec" or config["task"] != "disparity":
        raise ValueError("Official DSEC PNG export uses a disparity-trained head; "
                         "metric-depth scores are a separate local probe")
    import numpy as np
    from PIL import Image
    a.output.mkdir(parents=True)
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample = dataset[i]
            prediction = model(sample["feature"][None].to(a.device), sample["target"].shape)[0].cpu().numpy()
            if (not np.isfinite(prediction).all() or (prediction <= 0).any()
                    or (prediction * 256 > 65535).any()):
                raise ValueError("Prediction cannot be represented faithfully as uint16 disparity*256")
            destination = a.output / sample["sequence"]
            destination.mkdir(exist_ok=True)
            Image.fromarray(np.round(prediction * 256).astype(np.uint16)).save(
                destination / f"{sample['output_index']:06d}.png")
    # Keep metadata beside the directory: the official submission zip needs only SEQ/*.png.
    a.output.with_name(a.output.name + ".metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Predictions exported; run the official DSEC submission checker before submitting")


if __name__ == "__main__":
    main()
