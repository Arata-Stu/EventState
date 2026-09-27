"""Portable fixed event normalization contract (validation uses stdlib only)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def validate_fixed_normalization(stats, representation, height, width):
    if stats.get("format") != "eventstate_fixed_normalization_v1":
        raise ValueError("Unsupported fixed normalization format")
    if stats.get("representation") != representation:
        raise ValueError("Fixed normalization representation does not match")
    if representation.get("type") != "voxel_grid" or representation.get("normalization") != "none":
        raise ValueError("Fixed normalization requires unnormalized voxel counts")
    if stats.get("sensor_size") != [height, width]:
        raise ValueError("Fixed normalization sensor size does not match")
    if stats.get("includes_zero_pixels") is not True:
        raise ValueError("Fixed normalization must include zero pixels")
    channels = representation["channels"]
    mean, std = stats.get("mean", []), stats.get("std", [])
    if len(mean) != channels or len(std) != channels:
        raise ValueError("Fixed normalization channel count does not match")
    if not all(math.isfinite(x) for x in list(mean) + list(std)) or not all(x > 0 for x in std):
        raise ValueError("Fixed normalization coefficients must be finite with positive std")
    return tuple(mean), tuple(std)


def resolve_fixed_normalization(config):
    """Embed coefficients before logging/checkpointing; no file dependency at inference."""
    from omegaconf import OmegaConf, open_dict
    from event_state.data.dsec import event_representation_metadata
    from event_state.training.data import build_event_representation

    rep = config.dataset.representation
    path = OmegaConf.select(rep, "fixed_normalization_file")
    embedded = OmegaConf.select(rep, "fixed_normalization")
    if not path and embedded is None:
        if OmegaConf.select(rep, "fixed_normalization_required"):
            raise ValueError("Set dataset.representation.fixed_normalization_file before training")
        return
    if path:
        raw = Path(str(path)).expanduser().read_bytes()
        stats = json.loads(raw)
        stats["file_sha256"] = hashlib.sha256(raw).hexdigest()
        if embedded is not None and OmegaConf.to_container(embedded, resolve=True) != stats:
            raise ValueError("Normalization file differs from embedded coefficients")
    else:
        stats = OmegaConf.to_container(embedded, resolve=True)
    mean, std = validate_fixed_normalization(
        stats, event_representation_metadata(build_event_representation(config.dataset)),
        int(config.dataset.sensor_height), int(config.dataset.sensor_width),
    )
    if int(rep.channels) != len(mean):
        raise ValueError("Configured channels differ from fixed normalization")
    with open_dict(rep):
        rep.fixed_normalization = stats
        rep.fixed_normalization_file = None
        rep.normalize_mean = list(mean)
        rep.normalize_std = list(std)
