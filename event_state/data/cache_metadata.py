"""Content fingerprints and completion markers for deterministic caches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


SUCCESS_MARKER_NAME = "_SUCCESS"
FINGERPRINT_ALGORITHM = "sha256"
ALIGNMENT_FORMAT_VERSION = 2
EVENT_CACHE_FORMAT_VERSION = 2
TEACHER_CACHE_FORMAT_VERSION = 2


def _reject_disposable_reference_tree(path: Path, *, description: str) -> None:
    """Keep cache identities independent from the disposable reference checkout."""

    reference_root = (Path(__file__).resolve().parents[2] / "reference_repo").resolve()
    try:
        path.resolve().relative_to(reference_root)
    except ValueError:
        return
    raise ValueError(
        f"{description} is inside disposable reference_repo; copy or install it elsewhere: "
        f"{path.resolve()}"
    )


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with a stable encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_dinov3_checkpoint(checkpoint: str | Path | None) -> str | None:
    """Normalize local DINOv3 checkpoints while preserving remote identifiers.

    Existing plain paths and ``file:///`` URIs are returned as the same resolved
    filesystem path.  HTTP(S) URLs and builder-specific weight identifiers are
    passed through unchanged.
    """

    if checkpoint is None:
        return None
    value = str(checkpoint)
    parsed = urlparse(value)
    if parsed.scheme.lower() == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(
                "DINOv3 checkpoint file URI must refer to the local host, "
                f"got {checkpoint!r}"
            )
        if parsed.query or parsed.fragment:
            raise ValueError(
                "DINOv3 checkpoint file URI must not contain a query or fragment, "
                f"got {checkpoint!r}"
            )
        candidate = Path(unquote(parsed.path)).expanduser().resolve()
        _reject_disposable_reference_tree(candidate, description="DINOv3 checkpoint")
        if not candidate.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint file URI not found: {checkpoint}")
        return str(candidate)
    if parsed.scheme:
        return value

    candidate = Path(value).expanduser()
    resolved = candidate.resolve()
    _reject_disposable_reference_tree(resolved, description="DINOv3 checkpoint")
    return str(resolved) if candidate.is_file() else value


def dinov3_checkpoint_identity(
    checkpoint: str | Path | None,
    *,
    pretrained: bool,
) -> dict[str, Any]:
    """Return a location-independent identity for DINOv3 checkpoint weights."""

    normalized = normalize_dinov3_checkpoint(checkpoint)
    if normalized is None:
        return {
            "kind": "model_default" if pretrained else "random_initialization",
            "value": "official_default_pretrained_weights" if pretrained else None,
            "sha256": None,
        }
    candidate = Path(normalized)
    if candidate.is_file():
        return {
            "kind": "local_file",
            "value": candidate.name,
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return {
        "kind": "url_or_identifier",
        "value": normalized,
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def source_tree_fingerprint(root: str | Path) -> dict[str, Any]:
    """Hash code/config files in a local model checkout without storing its path."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Source tree not found: {root_path}")
    ignored_directories = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    included_suffixes = {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    included_names = {"LICENSE", "NOTICE"}
    paths = sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file()
            and not ignored_directories.intersection(path.relative_to(root_path).parts)
            and (path.suffix.lower() in included_suffixes or path.name in included_names)
        ),
        key=lambda path: path.relative_to(root_path).as_posix(),
    )
    if not paths:
        raise ValueError(f"Source tree has no code/config files: {root_path}")
    digest = hashlib.sha256()
    total_size = 0
    for path in paths:
        relative_id = path.relative_to(root_path).as_posix()
        size = path.stat().st_size
        digest.update(relative_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        total_size += size
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "file_count": len(paths),
        "total_size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def dinov3_repository_identity(
    *,
    backend: str,
    source: str,
    repository: str | Path,
) -> dict[str, Any]:
    """Return a portable, content-sensitive identity for DINOv3 source code."""

    normalized_backend = str(backend).lower()
    normalized_source = str(source).lower()
    if normalized_backend == "torch_hub":
        if normalized_source == "local":
            repository_path = Path(repository).expanduser().resolve()
            _reject_disposable_reference_tree(
                repository_path,
                description="Local DINOv3 repository",
            )
            return {
                "kind": "local_source_tree",
                "name": repository_path.name,
                "tree": source_tree_fingerprint(repository_path),
            }
        if normalized_source != "github":
            raise ValueError("Torch Hub source must be 'github' or 'local'")
        repository_value = str(repository)
        if ":" not in repository_value:
            raise ValueError(
                "A GitHub DINOv3 repository must include a pinned revision, "
                "for example owner/repo:<commit>"
            )
        return {
            "kind": "github_repository_revision",
            "value": repository_value,
            "sha256": hashlib.sha256(repository_value.encode("utf-8")).hexdigest(),
        }
    if normalized_backend != "package":
        raise ValueError("DINOv3 backend must be 'torch_hub' or 'package'")

    spec = importlib.util.find_spec("dinov3")
    if spec is None:
        raise ImportError("The package DINOv3 backend was requested, but dinov3 is unavailable")
    if spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin:
        package_root = Path(spec.origin).resolve().parent
    else:
        raise ImportError("Cannot locate the installed dinov3 package source")
    _reject_disposable_reference_tree(
        package_root,
        description="Installed DINOv3 package",
    )
    try:
        version = importlib.metadata.version("dinov3")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "kind": "installed_package",
        "distribution": "dinov3",
        "version": version,
        "tree": source_tree_fingerprint(package_root),
    }


def relative_file_id(root: str | Path, path: str | Path) -> str:
    """Return a portable dataset-relative identifier and reject outside paths."""

    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Input file is outside the declared dataset root: {resolved_path}"
        ) from error


def build_input_fingerprint(
    root: str | Path,
    *,
    files: Mapping[str, str | Path],
    file_sets: Mapping[str, Sequence[str | Path]] | None = None,
) -> dict[str, Any]:
    """Hash source files without recording host-specific absolute paths.

    Individual, usually small, inputs are listed separately. Large collections
    such as an aligned image sequence are represented by one ordered tree hash.
    The files themselves are still read completely, so an unchanged digest
    means that every byte used to build the cache is unchanged.
    """

    root_path = Path(root).resolve()
    file_entries: list[dict[str, Any]] = []
    for role, source in sorted(files.items()):
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Fingerprint input not found: {source_path}")
        file_entries.append(
            {
                "role": role,
                "path": relative_file_id(root_path, source_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )

    set_entries: list[dict[str, Any]] = []
    for role, sources in sorted((file_sets or {}).items()):
        paths = sorted((Path(source) for source in sources), key=lambda item: str(item))
        if not paths:
            raise ValueError(f"Fingerprint file set {role!r} is empty")
        digest = hashlib.sha256()
        total_size = 0
        for source_path in paths:
            if not source_path.is_file():
                raise FileNotFoundError(f"Fingerprint input not found: {source_path}")
            relative_id = relative_file_id(root_path, source_path)
            size = source_path.stat().st_size
            content_digest = sha256_file(source_path)
            digest.update(relative_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(content_digest.encode("ascii"))
            digest.update(b"\n")
            total_size += size
        set_entries.append(
            {
                "role": role,
                "file_count": len(paths),
                "total_size_bytes": total_size,
                "sha256": digest.hexdigest(),
            }
        )

    fingerprint: dict[str, Any] = {
        "algorithm": FINGERPRINT_ALGORITHM,
        "files": file_entries,
        "file_sets": set_entries,
    }
    fingerprint["digest"] = canonical_json_sha256(fingerprint)
    return fingerprint


def validate_input_fingerprint(
    fingerprint: Any,
    *,
    required_file_roles: Iterable[str] = (),
    required_file_set_roles: Iterable[str] = (),
) -> str:
    """Validate fingerprint structure and return its aggregate digest."""

    if not isinstance(fingerprint, dict):
        raise ValueError("Cache input_fingerprint must be a mapping")
    if fingerprint.get("algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("Cache input_fingerprint must use sha256")
    files = fingerprint.get("files")
    file_sets = fingerprint.get("file_sets")
    if not isinstance(files, list) or not isinstance(file_sets, list):
        raise ValueError("Cache input_fingerprint must contain files and file_sets lists")

    seen_file_roles: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Cache input_fingerprint file entries must be mappings")
        role = entry.get("role")
        path_value = entry.get("path")
        if not isinstance(role, str) or not role or role in seen_file_roles:
            raise ValueError("Cache input_fingerprint contains an invalid or duplicate file role")
        if not _is_relative_identifier(path_value):
            raise ValueError(f"Cache fingerprint path must be dataset-relative: {path_value!r}")
        _validate_size_and_digest(entry)
        seen_file_roles.add(role)

    seen_set_roles: set[str] = set()
    for entry in file_sets:
        if not isinstance(entry, dict):
            raise ValueError("Cache input_fingerprint file-set entries must be mappings")
        role = entry.get("role")
        if not isinstance(role, str) or not role or role in seen_set_roles:
            raise ValueError("Cache input_fingerprint contains an invalid or duplicate set role")
        if not isinstance(entry.get("file_count"), int) or entry["file_count"] <= 0:
            raise ValueError("Cache fingerprint file_count must be a positive integer")
        total_size = entry.get("total_size_bytes")
        if not isinstance(total_size, int) or total_size < 0:
            raise ValueError("Cache fingerprint total_size_bytes must be non-negative")
        _validate_sha256(entry.get("sha256"))
        seen_set_roles.add(role)

    missing_files = set(required_file_roles) - seen_file_roles
    missing_sets = set(required_file_set_roles) - seen_set_roles
    if missing_files or missing_sets:
        raise ValueError(
            "Cache input_fingerprint is missing roles: "
            + ", ".join(sorted(missing_files | missing_sets))
        )

    recorded_digest = fingerprint.get("digest")
    _validate_sha256(recorded_digest)
    unsigned = {key: value for key, value in fingerprint.items() if key != "digest"}
    if canonical_json_sha256(unsigned) != recorded_digest:
        raise ValueError("Cache input_fingerprint aggregate digest is inconsistent")
    return str(recorded_digest)


def frame_manifest_sha256(entries: Iterable[tuple[int, int, int]]) -> str:
    """Hash ordered ``(frame_index, previous_timestamp, timestamp)`` rows."""

    rows = [
        {
            "frame_index": int(frame_index),
            "previous_timestamp": int(previous_timestamp),
            "timestamp": int(timestamp),
        }
        for frame_index, previous_timestamp, timestamp in entries
    ]
    return canonical_json_sha256(rows)


def success_marker_payload(
    metadata: Mapping[str, Any],
    *,
    output_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact marker written only after every sequence item exists."""

    fingerprint = metadata.get("input_fingerprint")
    fingerprint_digest = validate_input_fingerprint(fingerprint)
    frame_count = metadata.get("frame_count")
    timestamp_digest = metadata.get("timestamp_manifest_sha256")
    if not isinstance(frame_count, int) or frame_count < 0:
        raise ValueError("Cache metadata frame_count must be a non-negative integer")
    _validate_sha256(timestamp_digest)
    marker = {
        "marker_format_version": 1,
        "cache_format_version": metadata.get("format_version"),
        "dataset": metadata.get("dataset"),
        "split": metadata.get("split"),
        "sequence_name": metadata.get("sequence_name"),
        "frame_count": frame_count,
        "timestamp_manifest_sha256": timestamp_digest,
        "input_fingerprint_digest": fingerprint_digest,
        "manifest_sha256": canonical_json_sha256(metadata),
    }
    if output_fingerprint is not None:
        validate_input_fingerprint(
            output_fingerprint,
            required_file_set_roles={"cache_items"},
        )
        marker["output_fingerprint"] = dict(output_fingerprint)
    return marker


def validate_success_marker(
    cache_directory: str | Path,
    metadata: Mapping[str, Any],
    *,
    output_paths: Sequence[str | Path] | None = None,
) -> None:
    cache_path = Path(cache_directory)
    marker_path = cache_path / SUCCESS_MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Cache is incomplete because its completion marker is missing: {marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid cache completion marker: {marker_path}") from error
    output_fingerprint = None
    if output_paths is not None:
        output_fingerprint = build_input_fingerprint(
            cache_path,
            files={},
            file_sets={"cache_items": output_paths},
        )
    expected = success_marker_payload(metadata, output_fingerprint=output_fingerprint)
    if marker != expected:
        raise ValueError(f"Cache completion marker does not match its manifest: {marker_path}")


def reject_untracked_outputs(
    metadata_path: str | Path,
    output_paths: Iterable[str | Path],
    *,
    overwrite: bool,
    description: str,
) -> None:
    """Reject existing outputs whose provenance cannot be established.

    A missing completion marker can be repaired from a valid manifest and its
    recorded output fingerprint. Missing metadata is different: accepting an
    existing file based only on its name, shape, or payload structure could
    relabel output from an older preprocessing implementation as current.
    """

    manifest_path = Path(metadata_path)
    if overwrite or manifest_path.is_file():
        return
    existing = [Path(path) for path in output_paths if Path(path).exists()]
    if not existing:
        return
    preview = ", ".join(path.name for path in existing[:5])
    raise RuntimeError(
        f"{description} contains {len(existing)} existing outputs but its metadata "
        f"is missing: {manifest_path}. Their provenance cannot be verified; use "
        f"--overwrite. First files: {preview}"
    )


def _validate_size_and_digest(entry: Mapping[str, Any]) -> None:
    size = entry.get("size_bytes")
    if not isinstance(size, int) or size < 0:
        raise ValueError("Cache fingerprint size_bytes must be non-negative")
    _validate_sha256(entry.get("sha256"))


def _validate_sha256(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Expected a hexadecimal SHA-256 digest, got {value!r}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"Expected a hexadecimal SHA-256 digest, got {value!r}") from error


def _is_relative_identifier(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts
