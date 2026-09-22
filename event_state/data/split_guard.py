"""Sequence and recording-group checks without ML runtime dependencies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def validate_official_dsec_pretraining(
    train: Sequence[str] | None,
    validation: Sequence[str] | None,
    detection: Mapping[str, Sequence[str]],
    semantic: Mapping[str, Sequence[str]],
    *,
    validation_enabled: bool,
) -> None:
    """Preserve the established 41-sequence final-fit protocol exactly.

    Semantic development-val inputs belong to official train and are allowed.
    Official benchmark splits are sequence-based, not recording-prefix-based.
    """
    if not train or len(train) != len(set(train)) or set(train) != set(detection["train"]):
        raise ValueError("DSEC final fit requires exactly the canonical 41 training sequences")
    if validation_enabled or validation:
        raise ValueError("DSEC 41-sequence final fit must disable pretraining validation")
    held_out = set(detection["val"]) | set(detection["test"]) | set(semantic["test"])
    if set(train) & held_out:
        raise ValueError("Official downstream holdout overlaps pretraining")


def recording_group(name: str) -> str:
    if not re.fullmatch(r"(?:zurich_city|interlaken|thun)_\d{2}_[a-z]", name):
        raise ValueError(f"Unrecognized DSEC sequence: {name!r}")
    return name.rsplit("_", 1)[0]


def validate_joint_dsec_split(
    train: Sequence[str] | None,
    validation: Sequence[str] | None,
    detection: Mapping[str, Sequence[str]],
    semantic: Mapping[str, Sequence[str]],
) -> None:
    """Exclude both task holdouts, including sibling recording segments.

    Pretraining validation influences model selection, so the same exclusion
    applies to it. Only canonical detection-training sequences are eligible.
    """
    if not train or not validation:
        raise ValueError("Joint DSEC pretraining requires explicit non-empty train/val lists")
    allowed = set(detection["train"])
    forbidden = {
        recording_group(name)
        for manifest in (detection, semantic)
        for role in ("val", "test")
        for name in manifest[role]
    }
    groups = []
    for role, names in (("train", train), ("validation", validation)):
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sequences in pretraining {role}")
        invalid = sorted(name for name in names
                         if name not in allowed or recording_group(name) in forbidden)
        if invalid:
            raise ValueError(f"Joint DSEC leakage in {role}: {', '.join(invalid)}")
        groups.append({recording_group(name) for name in names})
    overlap = sorted(groups[0] & groups[1])
    if overlap:
        raise ValueError(f"Pretraining train/validation recording groups overlap: {overlap}")
