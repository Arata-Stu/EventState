"""Score a frozen residual-placement Semantic head with all LSTM states cleared each window."""

import json
import math
from pathlib import Path

from event_state.dense.protocol import sha256_file
from tools.placement_conditions import PLACEMENT_BLOCKS, SKIP_CONDITION, SKIP_CONTROLS


def validate_skip_scales(evaluation, condition):
    """Check the gain associated with each insertion, including the original scalar b3 reports."""
    if condition not in SKIP_CONTROLS:
        raise ValueError(f"No residual placement for {condition}")
    blocks = [str(block) for block in PLACEMENT_BLOCKS[condition]]
    scales = evaluation.get("learned_skip_scales")
    if condition == SKIP_CONDITION and "learned_skip_scales" not in evaluation:
        scales = {"3": evaluation.get("learned_skip_scale")}
    if (not isinstance(scales, dict) or set(scales) != set(blocks)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scales.values())):
        raise ValueError(f"Invalid per-block residual scales for {condition}")
    if "learned_skip_scale" in evaluation:
        scalar = evaluation["learned_skip_scale"]
        if len(blocks) != 1 or type(scalar) not in (int, float) or scalar != scales[blocks[0]]:
            raise ValueError(f"Inconsistent scalar residual scale for {condition}")


def _model_scales(temporal, condition):
    blocks = PLACEMENT_BLOCKS[condition]
    if temporal.after_blocks != blocks or temporal.skip_scale is None:
        raise ValueError(f"Checkpoint residual placement differs from {condition}")
    values = temporal.skip_scale.detach().cpu().reshape(-1).tolist()
    scales = {str(block): value for block, value in zip(blocks, values, strict=True)}
    result = {"learned_skip_scales": scales}
    if len(blocks) == 1:
        result["learned_skip_scale"] = values[0]
    validate_skip_scales(result, condition)
    return result


def _matched_scores(report, continuous):
    hashes = ("samples_sha256", "samples_targets_sha256", "activity_assignments_sha256",
              "validity_support_sha256")
    if (report["activity_manifest"]["manifest_sha256"] != continuous["activity_manifest"]["manifest_sha256"]
            or any(report[key] != continuous[key] for key in hashes)
            or report["groups"].keys() != continuous["groups"].keys()):
        raise ValueError("Reset evaluation used different samples, labels, validity or activity groups")
    scores = {}
    for name, group in report["groups"].items():
        reference = continuous["groups"][name]
        if (group["metrics"]["class_support"] != reference["metrics"]["class_support"]
                or group["frames"] != reference["frames"] or group["valid_pixels"] != reference["valid_pixels"]):
            raise ValueError(f"Reset evaluation support changed for {name}")
        score = group["metrics"]["mIoU"]
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid reset mIoU for {name}")
        scores[name] = score
    return scores, {key: report[key] for key in hashes}


def evaluate_history(dataset, receipt, activity_path, *, data_root=None):
    """Reuse verified continuous scores and stream reset features without creating a feature cache."""
    import torch
    from tqdm import tqdm

    from event_state.activity_reporting import load_activity_options, validate_activity_source
    from event_state.dense.common_features import CommonFeatures
    from event_state.dense.source import input_geometry, prepared_frames
    from event_state.evaluation.comparison_semantic import SemanticTargets
    from event_state.models.tokens import tokens_to_map
    from event_state.segmentation import EventStateSegmentationHead, find_semantic_label_dir, load_semantic_label
    from event_state.segmentation.activity_evaluation import ActivitySemanticEvaluator

    condition = receipt["condition"]
    if dataset not in ("dsec", "m3ed") or condition not in SKIP_CONTROLS or receipt["dataset"] != dataset:
        raise ValueError("History control requires a verified residual-placement condition")
    head_path, checkpoint = Path(receipt["head"]), Path(receipt["backbone"])
    if sha256_file(head_path) != receipt["head_sha256"] or sha256_file(checkpoint) != receipt["backbone_sha256"]:
        raise ValueError("Weights changed since continuous evaluation")
    continuous = json.loads(Path(receipt["report"]).read_text())["activity_evaluation"]
    sequences = tuple(continuous["state_policy_by_sequence"])
    if (not sequences or set(continuous["state_policy_by_sequence"].values()) != {"continuous"}
            or continuous["head_sha256"] != receipt["head_sha256"]):
        raise ValueError("Expected an evaluation with the same head and uninterrupted sequence histories")
    index = load_activity_options(activity_path, receipt["label"], dataset=dataset)
    if index.identity["manifest_sha256"] != continuous["activity_manifest"]["manifest_sha256"]:
        raise ValueError("Activity manifest changed since continuous evaluation")
    archived_scores = {name: row["metrics"]["mIoU"] for name, row in continuous["groups"].items()}
    if archived_scores != receipt["mIoU"]:
        raise ValueError("Verified continuous scores differ from the archived report")

    device = torch.device("cuda")
    net = CommonFeatures(checkpoint, dataset, device)
    scales = _model_scales(net.model.temporal_model, condition)
    head_checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    cfg = head_checkpoint["config"]
    size = (440 if dataset == "dsec" else 352, 640)
    if (tuple(cfg["label_size"]) != size or cfg["feature"] != "h" or cfg["in_channels"] != 384
            or cfg["num_classes"] != 11 or cfg["head_type"] != "linear" or cfg["precision"] != "fp16"):
        raise ValueError("Unexpected frozen Semantic readout")
    head = EventStateSegmentationHead(in_channels=384, num_classes=11, width=cfg["head_width"],
                                      head_type="linear", output_size=size).to(device).eval()
    head.load_state_dict(head_checkpoint["model"], strict=True)
    project = Path(__file__).resolve().parents[1]
    if data_root is not None:
        source = Path(data_root) / ("DSEC_cache/events/gep_rgb" if dataset == "dsec" else "m3ed_cache/half_dagr")
    else:
        source = project / (net.cfg.dataset.event_cache_dir if dataset == "dsec" else net.cfg.dataset.prepared_root)
    targets = SemanticTargets(dataset, cfg, net.cfg, project, source, size, sequences)
    label_files = {}
    try:
        keys = []
        windows = {}
        for sequence in sequences:
            metadata = json.loads((source / sequence / "metadata.json").read_text())
            validate_activity_source(index, sequence, metadata)
            windows[sequence] = int(metadata["frame_count"]) - int(dataset == "m3ed")
            if dataset == "dsec":
                # Cache generation uses the original RGB frame index; frame 0 has no preceding event window.
                directory = find_semantic_label_dir(project / cfg["labels_root"], role="test", sequence=sequence)
                stamps = index.timestamps(sequence)
                for path in directory.glob("*.png"):
                    if path.stem.isdecimal() and int(path.stem) > 0:
                        label_files[sequence, stamps[int(path.stem) - 1]] = path
                keys.extend(key for key in label_files if key[0] == sequence)
            else:
                keys.extend((sequence, stamp) for stamp in targets.labeled_timestamps(sequence))
        labeled = set(keys)
        meter = ActivitySemanticEvaluator(index, dataset=dataset, expected_keys=keys,
            condition_label=receipt["label"] + "（毎窓リセット）", num_classes=11)
        batch_size = 8 if dataset == "dsec" else int(cfg["batch_size"])
        batch = []

        def score_batch():
            features, labels, names, stamps = zip(*batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = head(torch.stack(features).to(device).float())
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite reset Semantic logits")
            meter.update(logits.argmax(1).cpu(), torch.stack(labels), sequence_names=names, timestamps=stamps)
            batch.clear()

        with torch.inference_mode():
            completed_windows = 0
            total_windows = sum(windows.values())
            for sequence in sequences:
                frames = prepared_frames(source, sequence, net.cfg, dataset)
                description = f"{dataset} {receipt['condition']} {receipt['label']} | reset {sequence}"
                for frame in tqdm(frames, desc=description, total=total_windows, initial=completed_windows, unit="window"):
                    stamp = int(frame["timestamp"])
                    if (sequence, stamp) not in labeled:
                        continue  # No state is carried between any two windows in this control.
                    if dataset == "dsec":
                        from tools.cache_dsec_semantic_features import _semantic_transform
                        value = _semantic_transform(frame["events"], net.cfg).to(device)
                        output = net.model(value[:, None], state=None)["h"][0, 0]
                        grid = tuple(v // net.model.event_encoder.patch_size for v in value.shape[-2:])
                        features = tokens_to_map(output, grid).half().cpu()
                        label = load_semantic_label(label_files[sequence, stamp])
                    else:
                        net.state = None
                        features = net.step(input_geometry(frame["events"], dataset), ("h",))["h"]
                        label = torch.from_numpy(targets.get(sequence, stamp)).long()
                    batch.append((features, label, sequence, stamp))
                    if len(batch) == batch_size:
                        score_batch()
                completed_windows += windows[sequence]
            if batch:
                score_batch()
        scores, hashes = _matched_scores(meter.compute()["activity_evaluation"], continuous)
    finally:
        targets.close()
    return dict(state_policy="reset_each_window", **scales,
                head_sha256=receipt["head_sha256"], backbone_sha256=receipt["backbone_sha256"],
                activity_manifest_sha256=index.identity["manifest_sha256"], comparison_hashes=hashes,
                continuous_report=receipt["report"], continuous_mIoU=archived_scores,
                reset_each_window_mIoU=scores,
                continuous_minus_reset_pp={k: 100 * (archived_scores[k] - v) for k, v in scores.items()})
