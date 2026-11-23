"""Portable DSEC Semantic job for the remaining ConvLSTM downstream comparison.

The worker delegates feature extraction, development fitting, and official-train
refitting to the existing wrapper.  No training dependencies are imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from event_state.activity_quantiles import ActivityIndex, canonical_sha256
from event_state.activity_reporting import validate_activity_source


LABEL = "E1 Encoder＋LSTM（h蒸留）・ConvLSTM"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    return path


def _resolve(project, value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else project / value).resolve()


def _split(path):
    """Read the repository's simple YAML block lists without a YAML dependency."""
    result = {"train": [], "val": [], "test": []}
    role = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            role = line.removesuffix(":")
            if role not in result or line != role + ":":
                raise ValueError(f"Unsupported Semantic split structure: {path}")
        else:
            item = line.strip()
            name = item.removeprefix("- ")
            if role is None or item == name or not name or Path(name).name != name:
                raise ValueError(f"Invalid Semantic split item: {path}")
            result[role].append(name)
    names = [name for values in result.values() for name in values]
    if [len(result[k]) for k in ("train", "val", "test")] != [6, 2, 3] or len(set(names)) != 11:
        raise ValueError("DSEC Semantic comparison requires the existing disjoint 6/2/3 split")
    return result


def _labels(root, role, sequence):
    physical = "test" if role == "test" else "train"
    base = root / physical / sequence
    candidates = [base / "11classes", base / "semantic/left/11classes/data",
                  base / "semantic/left/11classes"]
    matches = [p for p in candidates if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one DSEC 11-class label directory: {base}")
    files = [p for p in matches[0].glob("*.png") if p.stem.isdecimal()]
    if not files or any(not p.stat().st_size for p in files):
        raise FileNotFoundError(f"Missing or empty DSEC Semantic labels: {matches[0]}")
    return matches[0], len(files)


def build_semantic_job(a, *, checkpoint=None, name="dsec_semantic", label=LABEL, feature="h"):
    """Validate a fresh job and return its plan without writing any files."""
    project = Path(a.project_root).resolve()
    config_path = _file(_resolve(project, a.baseline_semantic_config))
    config = _read(config_path)
    # Values not exposed by the wrapper must match its existing defaults.
    expected = dict(feature="h", head_type="linear", loss="ce", precision="fp16",
                    protocol="official-final", batch_size=8, seed=0,
                    learning_rate=2e-4, weight_decay=.05,
                    horizontal_flip_probability=.5, activity_active_weight=1.0,
                    activity_inactive_weight=1.0, num_classes=11)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Patchwise DSEC Semantic recipe differs ({key}): {config_path}")
    if getattr(a, "seed", 0) != 0:
        raise ValueError("The matched ConvLSTM comparison uses the existing seed 0")
    workers = config.get("num_workers")
    width = config.get("head_width")
    if (isinstance(workers, bool) or not isinstance(workers, int) or workers < 0
            or isinstance(width, bool) or not isinstance(width, int) or width <= 0):
        raise ValueError("Invalid previous Semantic worker count or head width")
    split_file = _file(project / "tools/manifests/dsec_semantic_split.yaml")
    baseline_split = _file(_resolve(project, config["split_manifest"]))
    if _sha(split_file) != _sha(baseline_split):
        raise ValueError("Previous DSEC Semantic head used a different split")
    split = _split(split_file)
    if config.get("training_sequences", split["train"] + split["val"]) != split["train"] + split["val"]:
        raise ValueError("Previous final DSEC Semantic head did not use the official eight train sequences")
    labels_root = _resolve(project, config["labels_root"])
    events = _resolve(project, Path(a.data) / "DSEC_cache/events/gep_rgb")
    checkpoint = _file(_resolve(project, checkpoint or Path(a.stage1_root) /
        "dsec_E1_convlstm_w1/checkpoints/step_00099999.pt"))
    activity_path = _file(_resolve(project, Path(a.activity_root) / "dsec.json"))
    activity = ActivityIndex.load(activity_path)
    if activity.identity["source_contract"]["dataset"] != "dsec":
        raise ValueError("Expected the shared DSEC activity manifest")
    output = _resolve(project, Path(a.output) / name)
    cache = _resolve(project, Path(a.scratch) / name)
    if output.exists() or cache.exists():
        raise FileExistsError(f"Preserving existing Semantic output/cache; choose fresh roots: {output}, {cache}")
    sources, count = {}, 0
    inputs = [config_path, baseline_split, split_file, checkpoint, activity_path]
    inputs.extend(_file(project / "tools" / name) for name in (
        "run_dsec_semantic_frozen.sh", "cache_dsec_semantic_features.py",
        "train_dsec_semantic.py", "evaluate_dsec_semantic.py"))
    for role, sequences in split.items():
        for sequence in sequences:
            label_dir, n = _labels(labels_root, role, sequence)
            metadata_path = _file(events / sequence / "metadata.json")
            metadata = _read(metadata_path)
            if (str(metadata.get("dataset", "")).lower() != "dsec"
                    or metadata.get("sequence_name") != sequence
                    or metadata.get("coordinate_space") != "rectified_event"
                    or metadata.get("height") != 480 or metadata.get("width") != 640
                    or metadata.get("representation", {}).get("type") != "gep_rgb"
                    or metadata.get("event_window") != "rgb_interval"
                    or metadata.get("event_window_us") is not None
                    or float(metadata.get("event_window_fraction", 1)) != 1):
                raise ValueError(f"Unexpected DSEC event source contract: {metadata_path}")
            if not any(p.stem.isdecimal() for p in metadata_path.parent.glob("*.pt")):
                raise FileNotFoundError(f"No prepared event frames: {metadata_path.parent}")
            if role == "test":
                validate_activity_source(activity, sequence, metadata)
                activity.timestamps(sequence)
            inputs.append(metadata_path)
            sources[sequence] = dict(role=role, labels=str(label_dir), label_files=n,
                                     event_metadata_sha256=canonical_sha256(metadata))
            count += n
    channels = {"h": 384, "stages": 1536}[feature]
    cache_bytes = count * (channels * 28 * 40 * 2 + 16_384)
    identity = dict(backbone_sha256=_sha(checkpoint), baseline_config_sha256=_sha(config_path),
                    split_sha256=_sha(split_file), activity_manifest_sha256=activity.identity["manifest_sha256"],
                    activity_group_names=list(activity.groups),
                    labels_root=str(labels_root), sources=sources, seed=0,
                    development_epochs=50, feature=feature, head_type="linear", loss="ce")
    return dict(name=name, kind="semantic", task="semantic", dataset="dsec", label=label,
                backbone=str(checkpoint), checkpoint=str(checkpoint), output=str(output), head=str(output),
                cache=str(cache), events=str(events), labels_root=str(labels_root), split=str(split_file),
                activity_manifest=str(activity_path), activity_report=str(output / "activity_test.json"),
                report=str(output / "test_metrics.json"), num_workers=workers, head_width=width,
                cache_bytes=cache_bytes, remaining_cache_bytes=cache_bytes,
                inputs=sorted(set(map(str, inputs))), identity=identity)


def _save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_semantic_job(a, job, status_callback=None):
    """Run the unchanged two-fit wrapper, then the existing activity evaluator."""
    project = Path(a.project_root).resolve()
    output, cache = Path(job["output"]), Path(job["cache"])
    feature = job["identity"]["feature"]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdecimal():
        raise ValueError("Semantic worker requires exactly one numeric CUDA_VISIBLE_DEVICES GPU ID")
    if output.exists() or cache.exists():
        raise FileExistsError("Existing Semantic outputs/cache are preserved; use fresh roots")
    fd = getattr(a, "parent_lock_fd", None)
    pass_fds = () if fd is None else (fd,)

    def phase(value, remaining=job["cache_bytes"]):
        print(f"[{job['name']}] {job['label']} | {value}", flush=True)
        if status_callback:
            status_callback(dict(name=job["name"], condition=job["label"], phase=value,
                                 remaining_cache_bytes=remaining, cache_dir=str(cache)))

    output.mkdir(parents=True)
    _save(output / "job_identity.json", job["identity"])
    phase("extracting features; fitting development and official-final heads")
    command = ["bash", str(project / "tools/run_dsec_semantic_frozen.sh"),
               "--checkpoint", job["backbone"], "--event-cache-dir", job["events"],
               "--labels-root", job["labels_root"], "--feature-cache-dir", str(cache),
               "--output-dir", str(output), "--feature", feature, "--gpu", gpu,
               "--seed", "0", "--batch-size", "8", "--epochs", "50",
               "--num-workers", str(job["num_workers"]), "--head-width", str(job["head_width"]),
               "--head-type", "linear", "--loss", "ce",
               "--activity-active-weight", "1.0", "--activity-inactive-weight", "1.0"]
    subprocess.run(command, cwd=project, check=True, pass_fds=pass_fds)
    _file(output / "final/last.pt")
    _file(output / "test_metrics.json")
    for sequence in job["identity"]["sources"]:
        meta = _read(_file(cache / sequence / "metadata.json"))
        indices = meta.get("cached_frame_indices", [])
        if (meta.get("state_policy") != "continuous" or meta.get("features") != [feature]
                or (feature == "stages" and meta.get("stage_blocks") != [3, 6, 9, 12])
                or meta.get("checkpoint_sha256") != job["identity"]["backbone_sha256"]
                or not indices or len(indices) != len(set(indices))):
            raise ValueError(f"Incomplete/different Semantic feature cache: {sequence}")
        for index in indices:
            _file(cache / sequence / f"{index:06d}.pt")
    (cache / "_SUCCESS").write_text("All Semantic cache sequences verified after wrapper completion.\n")
    phase("evaluating official test activity quintiles", remaining=0)
    command = [sys.executable, "-u", str(project / "tools/evaluate_dsec_semantic.py"),
               "--checkpoint", str(output / "final/last.pt"), "--feature-cache-dir", str(cache),
               "--labels-root", job["labels_root"], "--split-manifest", job["split"],
               "--role", "test", "--feature", feature, "--batch-size", "8",
               "--num-workers", str(job["num_workers"]), "--precision", "fp16", "--device", "cuda",
               "--activity-manifest", job["activity_manifest"], "--condition-label", job["label"],
               "--output", job["activity_report"]]
    subprocess.run(command, cwd=project, check=True, pass_fds=pass_fds)
    main, report = _read(output / "test_metrics.json"), _read(job["activity_report"])
    if any(not isinstance(r.get("mIoU"), (float, int)) or isinstance(r["mIoU"], bool)
           or not math.isfinite(r["mIoU"]) or not 0 <= r["mIoU"] <= 1
           or r.get("role") != "test" or r.get("feature") != feature for r in (main, report)):
        raise ValueError("Incomplete/different DSEC Semantic test report")
    activity = report.get("activity_evaluation", {})
    if (activity.get("condition_label") != job["label"]
            or activity.get("head_sha256") != _sha(output / "final/last.pt")
            or activity.get("activity_manifest", {}).get("manifest_sha256") !=
               job["identity"]["activity_manifest_sha256"]):
        raise ValueError("Semantic activity report provenance differs")
    groups = activity.get("groups", {})
    if (set(groups) != {"all", *job["identity"]["activity_group_names"]}
            or groups["all"].get("frames", 0) <= 0 or groups["all"].get("valid_pixels", 0) <= 0):
        raise ValueError("Semantic activity report has no evaluated samples")
    for key in ("frames", "valid_pixels"):
        if sum(g.get(key, 0) for name, g in groups.items() if name != "all") != groups["all"][key]:
            raise ValueError("Semantic activity support does not sum to all")
    expected_states = {s: "continuous" for s, source in job["identity"]["sources"].items()
                       if source["role"] == "test"}
    if activity.get("state_policy_by_sequence") != expected_states:
        raise ValueError("Semantic activity evaluation did not retain continuous test state")
    result = dict(condition=job["label"], dataset="dsec", task="semantic",
                  semantic_mIoU=report["mIoU"], main_report=str(output / "test_metrics.json"),
                  activity_report=job["activity_report"], activity_groups=groups,
                  head_sha256=activity["head_sha256"], backbone_sha256=job["identity"]["backbone_sha256"])
    _save(output / "result.json", result)
    (output / "_SUCCESS").write_text("DSEC Semantic official final fit/test and activity evaluation complete.\n")
    phase("completed", remaining=0)
    return result


__all__ = ["build_semantic_job", "run_semantic_job"]
