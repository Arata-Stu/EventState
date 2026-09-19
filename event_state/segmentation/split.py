"""Sequence-level split handling for DSEC-Semantic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DSECSemanticSplit:
    """Development 6/2 split plus the untouched official three-sequence test."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def sequences(self, role: str) -> tuple[str, ...]:
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train, val, or test")
        return getattr(self, role)

    @property
    def official_train(self) -> tuple[str, ...]:
        return self.train + self.val


def load_dsec_semantic_split(path: str | Path) -> DSECSemanticSplit:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DSEC-Semantic split manifest not found: {manifest_path}")
    value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"train", "val", "test"}:
        raise ValueError("DSEC-Semantic manifest must contain train, val, and test")
    sections: dict[str, tuple[str, ...]] = {}
    all_sequences: list[str] = []
    for role in ("train", "val", "test"):
        entries = value[role]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"DSEC-Semantic {role} split must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in entries):
            raise ValueError(f"DSEC-Semantic {role} contains an invalid sequence")
        if len(entries) != len(set(entries)):
            raise ValueError(f"DSEC-Semantic {role} contains duplicates")
        sections[role] = tuple(entries)
        all_sequences.extend(entries)
    if len(all_sequences) != len(set(all_sequences)):
        raise ValueError("DSEC-Semantic split sections overlap")
    expected = {"train": 6, "val": 2, "test": 3}
    for role, count in expected.items():
        if len(sections[role]) != count:
            raise ValueError(
                f"DSEC-Semantic {role} has {len(sections[role])} sequences; expected {count}"
            )
    return DSECSemanticSplit(**sections)


__all__ = ["DSECSemanticSplit", "load_dsec_semantic_split"]

