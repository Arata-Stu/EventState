"""Published sequence splits and fail-closed representation-training audits.

Do not infer a split from files present on a particular machine. A public
validation set stays validation, even when the official test labels are hidden.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL_ROOT = Path(__file__).resolve().parents[2] / "tools" / "manifests" / "dense"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol(name):
    if name not in {"dsec_f3", "dsec_official", "m3ed_f3", "m3ed_official_car_day"}:
        raise ValueError(f"Unknown published protocol: {name}")
    protocol = json.loads((PROTOCOL_ROOT / f"{name}.json").read_text())
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    if protocol.get("dataset") not in {"dsec", "m3ed"} or not protocol.get("sources"):
        raise ValueError("Dataset and published split provenance are required")
    seen = set()
    for role in ("train", "val", "test"):
        names = protocol["splits"][role]
        if not isinstance(names, list) or any(not isinstance(s, str) or not s or "/" in s for s in names):
            raise ValueError(f"Invalid {role} sequence list")
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("Published train/val/test sequence lists overlap or contain duplicates")
        seen.update(names)
    if not protocol["splits"]["train"]:
        raise ValueError("Empty training split")
    if set(protocol["labels_available"]) != {"train", "val", "test"}:
        raise ValueError("Explicit label availability for every role is required")
    if protocol["labels_available"]["test"]:
        raise ValueError("These protocols do not distribute official test ground truth")


def sequences_for(protocol, role, *, labels=False):
    if role not in {"train", "val", "test"}:
        raise ValueError("role must be train, val, or test")
    names = protocol["splits"][role]
    if not names:
        raise ValueError(f"{protocol['name']} has no {role}; do not invent a validation split")
    if labels and not protocol["labels_available"][role]:
        raise ValueError(f"{role} ground truth is not public; use prediction export, not local scoring")
    return tuple(names)


def audit_pretraining(config, protocol, *, statistics_sequences=None):
    """Check all declared event pretraining and model-selection inputs.

    stats provenance is supplied separately because legacy GEP checkpoints only
    store coefficients. Its absence must not silently certify a clean split.
    """
    dataset = config.get("dataset", {})
    if str(dataset.get("name", "")).lower() != protocol["dataset"]:
        raise ValueError("Checkpoint dataset does not match downstream protocol")
    train = dataset.get("train_sequences")
    if not isinstance(train, list) or not train or len(train) != len(set(train)):
        raise ValueError("Checkpoint must declare its actual pretraining train sequences")
    enabled = config.get("training", {}).get("validation_enabled", True)
    validation = dataset.get("val_sequences") or []
    if enabled and not validation:
        raise ValueError("Pretraining validation enabled but sequence provenance is missing")
    used_val = validation if enabled else []
    if config.get("training", {}).get("resume") and not config.get("pretraining_history"):
        raise ValueError("Resumed pretraining needs explicit earlier-stage sequence provenance")
    if set(train) & set(used_val):
        raise ValueError("Pretraining train and model-selection validation overlap")
    holdout = set(protocol["splits"]["val"]) | set(protocol["splits"]["test"])
    stages = [("train", train), ("validation/model selection", used_val)]
    # Chained representations can carry explicit earlier-training provenance.
    for i, stage in enumerate(config.get("pretraining_history", [])):
        if not isinstance(stage.get("sequences"), list) or not stage["sequences"]:
            raise ValueError("Earlier pretraining stage has incomplete provenance")
        stages.append((f"history[{i}]", stage["sequences"]))
    for stage, names in stages:
        overlap = sorted(holdout.intersection(names))
        if overlap:
            raise ValueError(f"Pretraining {stage} overlaps downstream holdout: {overlap}. "
                             "Use a clean pretrained checkpoint; no leakage bypass is provided.")
    if not isinstance(statistics_sequences, list) or not statistics_sequences:
        raise ValueError("Normalization statistics need an explicit source sequence list")
    if not set(statistics_sequences) <= set(protocol["splits"]["train"]):
        raise ValueError("Normalization statistics include sequences outside downstream train")
    overlap = sorted(holdout.intersection(statistics_sequences))
    if overlap:
        raise ValueError(f"Normalization statistics overlap holdout: {overlap}")
    return {"protocol_sha256": digest(protocol), "pretrain_train": train,
            "pretrain_validation_used": used_val, "statistics_sequences": statistics_sequences,
            "holdout": sorted(holdout), "status": "declared_inputs_disjoint",
            "scope": "EventState stages and declared statistics; upstream RGB foundation data unknown"}


def load_statistics_provenance(path, config):
    """Require both sequence provenance and the exact coefficients used by the model."""
    value = json.loads(Path(path).read_text())
    representation = config["dataset"]["representation"]
    for key in ("normalize_mean", "normalize_std"):
        if value.get(key) != representation.get(key):
            raise ValueError(f"Statistics provenance {key} does not match checkpoint coefficients")
    if not (value.get("source") or value.get("sources")):
        raise ValueError("Statistics provenance must identify the original computation/source")
    embedded = config["dataset"].get("normalization_provenance")
    if embedded and (embedded.get("sequences") != value.get("sequences")
                     or embedded.get("protocol_sha256") != value.get("protocol_sha256")):
        raise ValueError("External statistics provenance differs from the checkpoint's embedded provenance")
    return value
