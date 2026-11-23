"""Match the existing h-only Semantic controls and compute the 2 × 2 readout comparison."""

import math
from pathlib import Path

from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import sha256_file
from event_state.evaluation.placement_features import BLOCKS, CONDITIONS, FOUR_STAGE
from tools.run_placement_semantic import COMPARISON_HASHES, REFERENCE, baseline_results, read, verified_metrics
from tools.run_semantic_depth_activity import ROOT, semantic_run
from tools.temporal_downstream_semantic import _split

GROUPS = ("all", "p00_20", "p20_40", "p40_60", "p60_80", "p80_100")


def verify_head_recipe(dataset, feature, head, project_root):
    """Return the comparable training and evaluation contract from an actual head run."""
    project_root = Path(project_root)
    config_path = Path(head).parent / "config.json"
    config = read(config_path)
    if dataset not in ("dsec", "m3ed") or feature not in ("h", "stages"):
        raise ValueError(f"Unsupported Semantic cell: {dataset} {feature}")
    dsec = dataset == "dsec"
    manifest = project_root / "tools/manifests" / (
        "dsec_semantic_split.yaml" if dsec else "m3ed_semantic_common_v1.yaml")
    manifest_sha = sha256_file(manifest)
    if sha256_file(project_root / config["split_manifest"]) != manifest_sha:
        raise ValueError(f"Semantic split differs: {config_path}")
    expected = dict(seed=0, num_classes=11, head_type="linear", loss="ce", precision="fp16",
                    batch_size=8, learning_rate=2e-4, weight_decay=0.05,
                    horizontal_flip_probability=0.5, ignore_index=255,
                    label_size=[440 if dsec else 352, 640])
    if dsec:
        split = _split(manifest)
        expected.update(protocol="official-final", training_sequences=split["train"] + split["val"],
                        activity_active_weight=1.0, activity_inactive_weight=1.0)
        epoch = config["selected_development_epoch"]
        if type(epoch) is not int or not 0 <= epoch < 50 or config["epochs"] != epoch + 1:
            raise ValueError(f"DSEC refit duration differs from development selection: {config_path}")
    else:
        split = load_common("m3ed")["splits"]
        expected.update(epochs=50, validate_every=5, evaluation_role="validation",
                        training_sequences=split["train"], validation_sequences=split["val"])
    for name, value in dict(expected, feature=feature, in_channels=384 if feature == "h" else 1536).items():
        if config.get(name) != value:
            raise ValueError(f"Semantic head recipe differs ({name}): {config_path}")
    # Counts and class ordering also have to match across the two independently trained probes.
    return dict(expected, classes=config["classes"], train_frames=config["train_frames"],
                validation_frames=config["validation_frames"], split_sha256=manifest_sha)


def load_final_control(dataset, condition, *, project_root, placement_root, backbone_sha256):
    """Verify and reuse a completed final-layer probe without rerunning its training."""
    if dataset not in ("dsec", "m3ed") or condition not in CONDITIONS:
        raise ValueError(f"Unknown Semantic control: {dataset} {condition}")
    root = Path(project_root).resolve()
    reference = baseline_results()[dataset]
    role = "test" if dataset == "dsec" else "validation"
    head_name = "final/last.pt" if dataset == "dsec" else "h/best.pt"
    if condition == FOUR_STAGE:
        receipt = Path(placement_root) / f"{dataset}_{condition}" / "verified.json"
        saved = read(receipt)
        attempt = saved["attempt"]
        if not attempt.startswith("run-") or Path(attempt).name != attempt:
            raise ValueError(f"Invalid Semantic attempt in {receipt}")
        if (saved["dataset"] != dataset or saved["condition"] != condition or saved["seed"] != 0
                or saved["global_step"] != 99999 or saved["backbone_sha256"] != backbone_sha256
                or saved["after_blocks"] != list(BLOCKS)):
            raise ValueError(f"Frozen backbone differs from the existing Semantic control: {receipt}")
        attempt_dir = receipt.parent / attempt
        verified = verified_metrics(dataset, condition, attempt_dir, reference)
        for key in ("role", "head_sha256", "mIoU"):
            if saved[key] != verified[key]:
                raise ValueError(f"Control receipt no longer matches the measured {key}: {receipt}")
        head = attempt_dir / head_name
        result = dict(saved, head=str(head), report=verified["report"])
    else:
        head = root / semantic_run(dataset, condition).relative_to(ROOT) / head_name
        actual_head_sha = sha256_file(head)
        if actual_head_sha != reference["head_sha256"]:
            raise ValueError(f"Head differs from the archived Semantic control: {head}")
        result = dict(dataset=dataset, condition=condition, role=role, seed=0, global_step=99999,
                      after_blocks=[12], backbone_sha256=backbone_sha256, head=str(head),
                      head_sha256=actual_head_sha, report=reference["report"], archive=str(REFERENCE),
                      mIoU={group: reference["groups"][group]["metrics"]["mIoU"] for group in GROUPS})
    recipe = verify_head_recipe(dataset, "h", head, root)
    if condition == "E1":
        # The archived baseline contains the head digest but no Stage 1 digest.
        # Only small per-sequence metadata is needed to verify which backbone produced its inputs.
        cache = root / read(head.parent / "config.json")["feature_cache_dir"]
        for sequence in (*recipe["training_sequences"], *reference["state_policy_by_sequence"]):
            path = cache / sequence / "metadata.json"
            meta = read(path)
            if (meta["checkpoint_sha256"] != backbone_sha256 or meta["sequence_name"] != sequence
                    or meta["state_policy"] != "continuous" or "h" not in meta["features"]):
                raise ValueError(f"Backbone or state differs from the archived Semantic control: {path}")
    return dict(result, label=CONDITIONS[condition], feature="h", head_recipe=recipe)


def compare_readouts(dataset, controls, stages):
    """Compute signed changes in percentage points from all four verified cell receipts."""
    if dataset not in ("dsec", "m3ed") or set(controls) != set(CONDITIONS) or set(stages) != set(CONDITIONS):
        raise ValueError("A readout comparison requires both placements and both readouts for one dataset")
    cells, gain, gap = {}, {}, {}
    recipe = controls["E1"]["head_recipe"]
    role = "test" if dataset == "dsec" else "validation"
    for condition in CONDITIONS:
        cells[condition] = {"h": controls[condition], "stages": stages[condition]}
        for feature, row in cells[condition].items():
            if (row["dataset"] != dataset or row["condition"] != condition or row["feature"] != feature
                    or row["seed"] != 0 or row["global_step"] != 99999 or row["role"] != role
                    or row["head_recipe"] != recipe or not row["head_sha256"] or not row["backbone_sha256"]
                    or set(row["mIoU"]) != set(GROUPS)):
                raise ValueError(f"Cell differs from the matched readout protocol: {dataset} {condition} {feature}")
            for group, score in row["mIoU"].items():
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid mIoU for {dataset} {condition} {feature} {group}")
        if controls[condition]["backbone_sha256"] != stages[condition]["backbone_sha256"]:
            raise ValueError(f"Two readouts use different Stage 1 checkpoints: {dataset} {condition}")
        gain[condition] = {g: 100 * (stages[condition]["mIoU"][g] - controls[condition]["mIoU"][g])
                           for g in GROUPS}
    if controls["E1"]["backbone_sha256"] == controls[FOUR_STAGE]["backbone_sha256"]:
        raise ValueError(f"Different LSTM placements reference the same Stage 1 checkpoint: {dataset}")
    for feature in ("h", "stages"):
        gap[feature] = {g: 100 * (cells[FOUR_STAGE][feature]["mIoU"][g] - cells["E1"][feature]["mIoU"][g])
                        for g in GROUPS}
    reference = baseline_results()[dataset]
    evaluation = {key: reference[key] for key in (*COMPARISON_HASHES, "manifest_sha256", "state_policy_by_sequence")}
    evaluation["reference"] = str(REFERENCE)
    evaluation["support"] = {
        group: dict(frames=row["frames"], valid_pixels=row["valid_pixels"],
                    class_support=row["metrics"]["class_support"])
        for group, row in reference["groups"].items()}
    return dict(dataset=dataset, role=role, evaluation=evaluation, cells=cells, gain_pp=gain, placement_gap_pp=gap,
                interaction_pp={g: gain[FOUR_STAGE][g] - gain["E1"][g] for g in GROUPS})
