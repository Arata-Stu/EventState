"""Canonical DSEC-Detection split loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DSECDetectionSplit:
    """The official 41/6/13 sequence split used by DSEC-Det and DAGR."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def sequences(self, role: str) -> tuple[str, ...]:
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train, val, or test")
        return getattr(self, role)


def load_dsec_detection_split(path: str | Path) -> DSECDetectionSplit:
    """Load and strictly validate a DSEC-Detection split manifest."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DSEC-Detection split manifest not found: {manifest_path}")
    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"train", "val", "test"}:
        raise ValueError("DSEC-Detection manifest must contain exactly train, val, and test")
    sections: dict[str, tuple[str, ...]] = {}
    all_sequences: list[str] = []
    for role in ("train", "val", "test"):
        entries = value[role]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"DSEC-Detection {role} split must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in entries):
            raise ValueError(f"DSEC-Detection {role} split contains an invalid name")
        if len(entries) != len(set(entries)):
            raise ValueError(f"DSEC-Detection {role} split contains duplicates")
        sections[role] = tuple(entries)
        all_sequences.extend(entries)
    if len(all_sequences) != len(set(all_sequences)):
        raise ValueError("DSEC-Detection split sections overlap")
    expected_counts = {"train": 41, "val": 6, "test": 13}
    for role, expected in expected_counts.items():
        if len(sections[role]) != expected:
            raise ValueError(
                f"DSEC-Detection {role} split has {len(sections[role])} sequences; "
                f"expected {expected}"
            )
    return DSECDetectionSplit(**sections)


__all__ = ["DSECDetectionSplit", "load_dsec_detection_split"]
