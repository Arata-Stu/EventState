#!/usr/bin/env python3
"""Reevaluate an existing M3ED semantic head, optionally by activity quintile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.activity_reporting import (
    add_activity_arguments, load_activity_options, write_activity_csv,
)


def evaluate_m3ed_semantic(*, checkpoint, feature_cache_dir, target_cache_dir, split_manifest,
                           feature, device="cuda", batch_size=None, num_workers=0,
                           precision=None, activity_manifest=None, condition_label=None,
                           prepared_root=None):
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from event_state.dense.protocol import sha256_file
    from event_state.segmentation import (
        EventStateSegmentationHead, M3EDSemanticFeatureDataset,
        SemanticSegmentationEvaluator, load_m3ed_semantic_split, segmentation_collate,
    )
    from event_state.segmentation.activity_evaluation import (
        ActivitySemanticEvaluator, _safe, validate_semantic_cache,
    )

    checkpoint, feature_cache_dir = Path(checkpoint), Path(feature_cache_dir)
    target_cache_dir, split_manifest = Path(target_cache_dir), Path(split_manifest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if config.get("feature", feature) != feature:
        raise ValueError("Semantic head feature differs from --feature")
    split = load_m3ed_semantic_split(split_manifest)
    dataset = M3EDSemanticFeatureDataset(
        feature_cache_dir=feature_cache_dir, target_cache_dir=target_cache_dir,
        sequences=split.validation, feature=feature, horizontal_flip_probability=0,
    )
    index = load_activity_options(activity_manifest, condition_label, dataset="m3ed")
    classes, ignore = int(config["num_classes"]), int(config.get("ignore_index", 255))
    states = {}
    if index is not None:
        if not (feature_cache_dir / "_SUCCESS").is_file():
            raise ValueError("Incomplete M3ED common semantic feature cache")
        keys, states = validate_semantic_cache(
            index, dataset="m3ed", feature_cache_dir=feature_cache_dir,
            sequences=split.validation, samples=dataset.samples, prepared_root=prepared_root,
            load_payload=lambda path: torch.load(path, map_location="cpu", weights_only=True, mmap=True),
        )
        for sequence in split.validation:
            source = json.loads((Path(prepared_root) / sequence / "metadata.json").read_text())
            with h5py.File(target_cache_dir / sequence / "targets.h5") as targets:
                if targets["timestamps"][:].tolist() != source["timestamps"]:
                    raise ValueError(f"M3ED semantic target/prepared timeline mismatch: {sequence}")
        evaluator = ActivitySemanticEvaluator(
            index, dataset="m3ed", expected_keys=keys, condition_label=condition_label,
            num_classes=classes, ignore_index=ignore,
        )
    else:
        evaluator = SemanticSegmentationEvaluator(classes, ignore_index=ignore)
    model = EventStateSegmentationHead(
        in_channels=int(config["in_channels"]), num_classes=classes,
        width=int(config.get("head_width", 192)), output_size=tuple(dataset.label_size),
        head_type=str(config.get("head_type", "linear")),
    ).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    batch_size = int(config.get("batch_size", 8) if batch_size is None else batch_size)
    precision = precision or config.get("precision", "fp16")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid semantic evaluation batch size/worker count")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=segmentation_collate)
    durations = {}
    for sequence in split.validation:
        path = feature_cache_dir / sequence / "low_duration_us.json"
        if path.is_file():
            durations[sequence] = json.loads(path.read_text())
    legacy_names = ["all", *split.validation]
    if len(durations) == len(split.validation):
        legacy_names.extend(["low", "other", "low_500ms_plus"])
    legacy = {name: SemanticSegmentationEvaluator(classes, ignore_index=ignore) for name in legacy_names}
    support = {name: 0 for name in legacy}
    desc = f"{condition_label} | M3ED Semantic validation" if index else "M3ED Semantic validation"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, unit="batch"):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=precision == "fp16" and torch.device(device).type == "cuda"):
                predictions = model(batch["features"].to(device)).argmax(1).cpu()
            if index is None:
                evaluator.update(predictions, batch["labels"])
            else:
                evaluator.update(predictions, batch["labels"], sequence_names=batch["sequence_names"],
                                 timestamps=batch["timestamps"])
            for j, (sequence, frame_index) in enumerate(zip(batch["sequence_names"], batch["frame_indices"])):
                names = ["all", sequence]
                if "low" in legacy:
                    duration = durations[sequence].get(str(frame_index))
                    if duration is None:
                        raise ValueError(f"M3ED low-activity timeline missing: {sequence}/{frame_index}")
                    names.append("low" if duration else "other")
                    if duration >= 500000:
                        names.append("low_500ms_plus")
                for name in names:
                    legacy[name].update(predictions[j:j + 1], batch["labels"][j:j + 1])
                    support[name] += 1
    result = dict(dataset="m3ed", role="validation", feature=feature,
                  label_source="InternImage pseudo-labels mapped to DSEC-11",
                  split_sha256=sha256_file(split_manifest), head_sha256=sha256_file(checkpoint),
                  **_safe(evaluator.compute()))
    result["continuous_groups"] = {
        name: dict(frames=support[name], valid_pixels=int(meter.confusion.sum()),
                   mIoU=_safe(meter.compute()["mIoU"])) for name, meter in legacy.items()
    }
    if index is not None:
        result["activity_evaluation"]["state_policy_by_sequence"] = states
        result["activity_evaluation"]["head_sha256"] = result["head_sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "feature-cache-dir", "target-cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        default=ROOT / "tools/manifests/m3ed_semantic_common_v1.yaml")
    parser.add_argument("--feature", choices=("z", "h", "concat"), required=True)
    parser.add_argument("--prepared-root", type=Path,
                        help="Original M3ED prepared events; required for activity provenance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"))
    add_activity_arguments(parser)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError("Existing semantic evaluation is preserved; choose a new --output")
    result = evaluate_m3ed_semantic(
        checkpoint=args.checkpoint, feature_cache_dir=args.feature_cache_dir,
        target_cache_dir=args.target_cache_dir, split_manifest=args.split_manifest,
        feature=args.feature, device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers, precision=args.precision,
        activity_manifest=args.activity_manifest, condition_label=args.condition_label,
        prepared_root=args.prepared_root,
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)
    if args.activity_manifest is not None:
        write_activity_csv(output, result["activity_evaluation"])
    print(encoded, end="")


if __name__ == "__main__":
    main()
