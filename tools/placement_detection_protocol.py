"""Verify the frozen DSEC backbone and matched inputs for each Detection report."""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import tempfile

from event_state.dense.common_protocol import audit_common
from event_state.dense.protocol import audit_pretraining, sha256_file
from tools.placement_conditions import PLACEMENTS


def _replace_from_stream(destination, source):
    temporary = tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".resume-", delete=False)
    path = Path(temporary.name)
    try:
        with temporary:
            shutil.copyfileobj(source, temporary)
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)


def _head_checkpoint(path, config):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    fields = ("model", "optimizer", "scheduler", "scaler", "config")
    if any(not isinstance(checkpoint.get(key), dict) for key in fields):
        raise ValueError(f"Incomplete Detection checkpoint: {path}")
    saved_config = {k: v for k, v in checkpoint["config"].items() if k != "resume"}
    if saved_config != config:
        raise ValueError(f"Detection checkpoint and head config differ: {path}")
    epoch, score = checkpoint.get("epoch"), checkpoint.get("best_mAP")
    if (type(epoch) is not int or not 0 <= epoch < config["epochs"]
            or type(score) not in (float, int) or not 0 <= score <= 1):
        raise ValueError(f"Invalid Detection checkpoint epoch or mAP: {path}")
    return epoch, score


def prepare_head_resume(head):
    """Reconcile the trainer's log → last → best writes while holding the suite's run lock."""
    head = Path(head)
    last, best, log = (head / name for name in ("last.pt", "best.pt", "metrics.jsonl"))
    if not any(p.exists() for p in (last, best, log)):
        return None
    config = json.loads((head / "config.json").read_text(encoding="utf-8"))
    config.pop("resume", None)
    epoch, best_score = _head_checkpoint(last, config) if last.exists() else (-1, None)
    rows = log.read_bytes().splitlines(keepends=True) if log.exists() else []
    committed = epoch + 1
    if not committed <= len(rows) <= min(config["epochs"], committed + 1):
        raise ValueError(f"Detection metrics history does not match last.pt: {log}")
    scores = []
    for i, row in enumerate(rows):
        try:
            record = json.loads(row)
        except ValueError:
            if i == committed and row.startswith(b"{") and not row.endswith(b"\n"):
                break  # The next epoch's log append was interrupted before its checkpoint.
            raise ValueError(f"Invalid Detection metrics at epoch {i}: {log}") from None
        if (not isinstance(record, dict) or type(record.get("epoch")) is not int or record["epoch"] != i
                or type(record.get("val/mAP")) not in (float, int) or not 0 <= record["val/mAP"] <= 1
                or (i < committed and not row.endswith(b"\n"))):
            raise ValueError(f"Invalid Detection epoch sequence or mAP: {log}")
        if i < committed:
            scores.append(record["val/mAP"])
    stored_best = _head_checkpoint(best, config) if best.exists() else None
    restore_best = False
    if committed:
        winner = max(range(committed), key=scores.__getitem__)
        if scores[winner] != best_score:
            raise ValueError(f"Detection best mAP in last.pt disagrees with the committed log: {head}")
        if stored_best != (winner, best_score):
            previous = max(range(epoch), key=scores.__getitem__) if epoch else None
            if winner != epoch or (stored_best is not None and
                    (previous is None or stored_best != (previous, scores[previous]))):
                raise ValueError(f"Cannot recover the selected Detection head: {best}")
            restore_best = True
    elif stored_best is not None:
        raise ValueError(f"Detection best.pt exists without last.pt: {head}")

    # Validate the complete state first. Both atomic repairs can be retried independently.
    if restore_best:
        with last.open("rb") as source:
            _replace_from_stream(best, source)
    if len(rows) > committed:
        _replace_from_stream(log, BytesIO(b"".join(rows[:committed])))
    return last if committed else None


def check_backbone(path, condition, split):
    from event_state.training import load_checkpoint_config_metadata

    metadata = load_checkpoint_config_metadata(path)
    config = metadata.config
    training, dataset = config["training"], config["dataset"]
    if (metadata.global_step != 99999 or training.get("max_steps") != 99999
            or config.get("seed") != 0 or training.get("distributed_world_size") != 1
            or dataset.get("name") != "dsec" or dataset.get("event_window_us") is not None):
        raise ValueError(f"{path}: expected the full DSEC single-GPU Stage 1 checkpoint")
    audit_common(config)
    audit = audit_pretraining(config, {"dataset": "dsec", "splits": split},
                               statistics_sequences=dataset["normalization_provenance"]["sequences"])
    temporal = config["model"]["temporal"]
    expected = {"E0": "identity", "E1": "lstm", "E1_convlstm": "convlstm"}
    kind = "four_stage_lstm" if condition in PLACEMENTS else expected.get(condition)
    blocks = list(PLACEMENTS[condition]) if condition in PLACEMENTS else None
    dims = ("input_dim", "output_dim") if condition == "E0" else ("input_dim", "hidden_dim", "output_dim")
    if (not kind or temporal.get("type") != kind or temporal.get("after_blocks") != blocks
            or temporal.get("residual_skip", False) is not False
            or any(temporal.get(k) != 384 for k in dims)
            or (condition != "E0" and temporal.get("num_layers") != 1)):
        raise ValueError(f"{path}: wrong temporal model or placement for {condition}")
    z, h = config["loss"]["z_objective"], config["loss"]["h_distill"]
    use_h = condition != "E0"
    objective = h if use_h else z
    if (h.get("enabled") is not use_h
            or z.get("type") != ("none" if use_h else "direct_dino")
            or z.get("weight") != (0 if use_h else 1)
            or any(objective.get(key) != 1 for key in ("cosine_weight", "mse_weight"))):
        raise ValueError(f"{path}: wrong distillation objective for {condition}")
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path), global_step=metadata.global_step,
                condition=condition, feature="h" if use_h else "z", seed=config["seed"],
                temporal=kind, after_blocks=blocks, pretraining_audit=audit)


def _support(group, sequences):
    totals = [group["frames"], group["gt_objects"], group["detections"]]
    by_sequence, by_class = group["sequence_frames"], group["gt_objects_per_class"]
    if (any(type(n) is not int or n < 0 for n in [*totals, *by_sequence.values(), *by_class.values()])
            or not set(by_sequence) <= sequences or set(by_class) != {"car", "pedestrian"}
            or sum(by_sequence.values()) != totals[0] or sum(by_class.values()) != totals[1]
            or group["sequences"] != len(by_sequence)):
        raise ValueError("Inconsistent Detection group support")
    frames, objects, _ = totals
    status = "empty" if frames == 0 else "no_ground_truth" if objects == 0 else "ok"
    if group["status"] != status or (not frames and (objects or group["detections"])):
        raise ValueError("Inconsistent Detection group status")
    scores = group["metrics"]
    keys = {"mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AP_car", "AP_pedestrian"}
    if keys - scores.keys():
        raise ValueError("Incomplete Detection group metrics")
    for key in keys:
        value = scores[key]
        if value is None:
            if objects and (key in {"mAP", "AP50", "AP75"} or by_class.get(key.removeprefix("AP_"), 0)):
                raise ValueError(f"Missing Detection score for supported {key}")
        elif (not objects or type(value) not in (float, int) or not 0 <= value <= 1
              or by_class.get(key.removeprefix("AP_")) == 0):
            raise ValueError(f"Invalid Detection score for {key}")


def check_report(path, role, condition, feature, label, head, activity_hash, sequences):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    activity = report["activity_evaluation"]
    manifest = activity["activity_manifest"]
    head_hash = sha256_file(head)
    sequences = set(sequences)
    if (role not in {"val", "test"} or report["role"] != role or report["feature"] != feature
            or report["protocol"] != "dsec-det" or report["coordinate_space"] != "dsec_det_distorted"
            or activity["partition"] != "whole_frame_by_scene_event_rate"
            or activity["condition_label"] != label or manifest["manifest_sha256"] != activity_hash
            or activity["detector_checkpoint"]["sha256"] != head_hash
            or activity["state_policy_by_sequence"] != {seq: "continuous" for seq in sequences}):
        raise ValueError(f"{path}: Detection evaluation identity differs for {condition}")
    hashes = {key: activity[key] for key in ("samples_targets_sha256", "activity_assignments_sha256")}
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (activity_hash, *hashes.values())):
        raise ValueError(f"{path}: invalid evaluation digest")
    groups = activity["groups"]
    partitions = [entry["name"] for entry in manifest["groups"]]
    if (partitions != ["p00_20", "p20_40", "p40_60", "p60_80", "p80_100"]
            or set(groups) != {"all", *partitions}):
        raise ValueError(f"{path}: missing or unexpected Detection activity groups")
    for group in groups.values():
        _support(group, sequences)
    overall = groups["all"]
    if (overall["status"] != "ok" or set(overall["sequence_frames"]) != sequences
            or any(key not in report or report[key] != value for key, value in overall["metrics"].items())):
        raise ValueError(f"{path}: inconsistent overall Detection scores or sequence coverage")
    for field in ("frames", "gt_objects", "detections"):
        if sum(groups[name][field] for name in partitions) != overall[field]:
            raise ValueError(f"{path}: Detection group totals differ for {field}")
    for field in ("sequence_frames", "gt_objects_per_class"):
        total = Counter()
        for name in partitions:
            total.update(groups[name][field])
        if total != Counter(overall[field]):
            raise ValueError(f"{path}: Detection group counts differ for {field}")
    return dict(condition=condition, label=label, feature=feature, role=role, report=str(Path(path).resolve()),
                head_sha256=head_hash, activity_manifest_sha256=activity_hash,
                groups=groups, **hashes, **overall["metrics"])
