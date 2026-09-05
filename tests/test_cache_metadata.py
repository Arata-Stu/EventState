from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_state.data.cache_metadata import (
    SUCCESS_MARKER_NAME,
    build_input_fingerprint,
    dinov3_checkpoint_identity,
    dinov3_repository_identity,
    normalize_dinov3_checkpoint,
    reject_untracked_outputs,
    success_marker_payload,
    validate_success_marker,
)


def test_cache_marker_binds_manifest_and_output_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    item = cache_directory / "1.pt"
    item.write_bytes(b"cached tensor bytes")
    input_fingerprint = build_input_fingerprint(
        tmp_path,
        files={"source": source},
    )
    output_fingerprint = build_input_fingerprint(
        cache_directory,
        files={},
        file_sets={"cache_items": [item]},
    )
    metadata = {
        "format_version": 2,
        "dataset": "synthetic",
        "split": "train",
        "sequence_name": "sequence",
        "frame_count": 1,
        "timestamp_manifest_sha256": "0" * 64,
        "input_fingerprint": input_fingerprint,
    }
    marker = success_marker_payload(
        metadata,
        output_fingerprint=output_fingerprint,
    )
    (cache_directory / SUCCESS_MARKER_NAME).write_text(
        json.dumps(marker),
        encoding="utf-8",
    )

    validate_success_marker(cache_directory, metadata, output_paths=[item])
    item.write_bytes(b"modified tensor bytes")
    with pytest.raises(ValueError, match="completion marker"):
        validate_success_marker(cache_directory, metadata, output_paths=[item])


def test_untracked_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    metadata_path = cache_directory / "metadata.json"
    output_path = cache_directory / "1.pt"
    output_path.write_bytes(b"untracked output")

    with pytest.raises(RuntimeError, match="provenance cannot be verified"):
        reject_untracked_outputs(
            metadata_path,
            [output_path],
            overwrite=False,
            description="Synthetic cache",
        )

    reject_untracked_outputs(
        metadata_path,
        [output_path],
        overwrite=True,
        description="Synthetic cache",
    )
    metadata_path.write_text("{}", encoding="utf-8")
    reject_untracked_outputs(
        metadata_path,
        [output_path],
        overwrite=False,
        description="Synthetic cache",
    )


def test_dinov3_identity_rejects_disposable_reference_checkout() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository = project_root / "reference_repo" / "dinov3"
    with pytest.raises(ValueError, match="disposable reference_repo"):
        dinov3_repository_identity(
            backend="torch_hub",
            source="local",
            repository=repository,
        )


def test_dinov3_checkpoint_file_uri_matches_plain_local_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights with spaces.pt"
    checkpoint.write_bytes(b"frozen teacher checkpoint")

    assert normalize_dinov3_checkpoint(checkpoint.as_uri()) == str(checkpoint.resolve())
    assert dinov3_checkpoint_identity(
        checkpoint.as_uri(),
        pretrained=True,
    ) == dinov3_checkpoint_identity(checkpoint, pretrained=True)


def test_dinov3_checkpoint_file_uri_rejects_disposable_reference_checkout() -> None:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint = project_root / "reference_repo" / "weights.pt"

    with pytest.raises(ValueError, match="disposable reference_repo"):
        normalize_dinov3_checkpoint(checkpoint.as_uri())
