"""Project-level Stage 1 pool; deliberately separate from official task splits."""
import json
from pathlib import Path

from event_state.dense.protocol import audit_pretraining, digest, validate_protocol


def load_common(dataset):
    if dataset not in {"dsec", "m3ed"}:
        raise ValueError("Expected dsec or m3ed")
    path = Path(__file__).resolve().parents[2] / "tools/manifests" / f"{dataset}_stage1_common_v1.json"
    value = json.loads(path.read_text())
    validate_protocol(value)
    return value


def audit_common(config):
    protocol = load_common(config["dataset"]["name"])
    ds = config["dataset"]
    if ds["train_sequences"] != protocol["splits"]["train"]:
        raise ValueError("Stage 1 common train manifest mismatch")
    stats = ds["normalization_provenance"]
    if stats.get("protocol_sha256") != digest(protocol):
        raise ValueError("Stage 1 normalization belongs to a different common manifest")
    if stats.get("sequences") != protocol["splits"]["train"]:
        raise ValueError("Stage 1 normalization must use exactly the common train pool")
    for key in ("normalize_mean", "normalize_std"):
        if stats.get(key) != ds["representation"].get(key):
            raise ValueError("Stage 1 normalization coefficient mismatch")
    if config["training"].get("validation_enabled", True):
        raise ValueError("Common Stage 1 uses fixed-step selection, no task validation")
    return audit_pretraining(config, protocol, statistics_sequences=stats["sequences"])
