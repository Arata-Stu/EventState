#!/usr/bin/env python3
"""Reevaluate completed Semantic and Depth probes with shared activity quintiles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    ("E0", "E0 Encoder（z蒸留）・単GPU", "z"),
    ("E1", "E1 Encoder＋LSTM（h蒸留）・単GPU", "h"),
    ("E2", "E2 Encoder＋LSTM（z+h蒸留）・単GPU", "h"),
    ("E1_old_ddp", "E1 Encoder＋LSTM（h蒸留）・旧型DDP", "h"),
    ("E1_new_ddp", "E1 Encoder＋LSTM（h蒸留）・新型DDP", "h"),
)
DSEC_TEST = ("zurich_city_13_a", "zurich_city_14_c", "zurich_city_15_a")
M3ED_VAL = ("car_urban_day_rittenhouse", "car_urban_day_ucity_small_loop")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path(value):
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def require_file(value, *, nonempty=False):
    value = Path(value)
    if not value.is_file() or (nonempty and not value.stat().st_size):
        raise FileNotFoundError(f"Required file missing or empty: {value}")
    return value


def sha256(value):
    digest = hashlib.sha256()
    with require_file(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(tool, *args):
    return [sys.executable, "-u", str(ROOT / "tools" / tool), *map(str, args)]


def launch(cmd):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=environment, check=True)


@contextmanager
def depth_available(depth_run):
    """Check the existing advisory lock without creating or editing any file."""
    lock = depth_run / "run.lock"
    if not lock.exists():
        yield
        return
    with lock.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Depth run is still active; wait for it to finish") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def semantic_run(dataset, condition):
    if dataset == "m3ed":
        return ROOT / "outputs/stage2_m3ed_semantic_linear_20261109" / condition
    if condition.startswith("E1_"):
        return ROOT / "outputs/stage2_dsec_semantic_e1_ddp_compare_20261109" / condition[3:]
    return ROOT / "outputs/stage2_dsec_semantic_linear_20261109" / condition


def check_semantic_cache(cache, dataset, sequences, feature):
    if dataset == "m3ed":
        require_file(cache / "_SUCCESS")
        require_file(cache / "identity.json", nonempty=True)
    for sequence in sequences:
        folder = cache / sequence
        meta = read(require_file(folder / "metadata.json", nonempty=True))
        if meta.get("task") != dataset + "_semantic" or feature not in meta.get("features", []):
            raise ValueError(f"Wrong Semantic feature cache: {folder}")
        if meta.get("state_policy") != "continuous":
            raise ValueError(f"Expected continuous Semantic feature cache: {folder}")
        if not any(p.stem.isdecimal() for p in folder.glob("*.pt")):
            raise FileNotFoundError(f"No Semantic features in {folder}")
        for index in meta.get("cached_frame_indices", []):
            require_file(folder / f"{int(index):06d}.pt", nonempty=True)


def semantic_job(args, dataset, condition, label, expected_feature, manifest):
    run = semantic_run(dataset, condition)
    head_dir = run / (expected_feature if dataset == "m3ed" else "final")
    head = require_file(head_dir / ("best.pt" if dataset == "m3ed" else "last.pt"), nonempty=True)
    config_path = require_file(head_dir / "config.json", nonempty=True)
    config = read(config_path)
    if config.get("feature") != expected_feature:
        raise ValueError(f"Unexpected feature for {dataset}/{condition}: {config_path}")
    split = path(config["split_manifest"])
    official_split = ROOT / "tools/manifests" / (
        "m3ed_semantic_common_v1.yaml" if dataset == "m3ed" else "dsec_semantic_split.yaml")
    if sha256(split) != sha256(official_split):
        raise ValueError(f"Different Semantic split; this batch requires the common protocol: {split}")
    sequences = M3ED_VAL if dataset == "m3ed" else DSEC_TEST
    cache = path(config["feature_cache_dir"])
    check_semantic_cache(cache, dataset, sequences, expected_feature)
    cmd = command("evaluate_" + dataset + "_semantic.py", "--checkpoint", head,
                  "--feature-cache-dir", cache, "--split-manifest", split,
                  "--feature", expected_feature, "--batch-size", config.get("batch_size", 8),
                  "--num-workers", config.get("num_workers", 4),
                  "--precision", config.get("precision", "fp16"), "--device", "cuda")
    if dataset == "dsec":
        labels = path(config["labels_root"])
        for sequence in sequences:
            base = labels / "test" / sequence
            candidates = [base / "11classes", base / "semantic/left/11classes/data",
                          base / "semantic/left/11classes"]
            matches = [p for p in candidates if p.is_dir()]
            if len(matches) != 1 or not any(matches[0].glob("*.png")):
                raise FileNotFoundError(f"Missing/ambiguous DSEC Semantic labels: {base}")
            for feature_path in (cache / sequence).glob("*.pt"):
                if feature_path.stem.isdecimal():
                    require_file(matches[0] / f"{int(feature_path.stem):06d}.png", nonempty=True)
        cmd += ["--labels-root", str(labels), "--role", "test"]
    else:
        targets = path(config["target_cache_dir"])
        for sequence in sequences:
            require_file(targets / sequence / "targets.h5", nonempty=True)
        cmd += ["--target-cache-dir", str(targets), "--prepared-root",
                str(args.data / "m3ed_cache/half_dagr")]
    output = args.output / "semantic" / f"{dataset}_{condition}.json"
    cmd += ["--activity-manifest", str(manifest), "--condition-label", label, "--output", str(output)]
    return dict(dataset=dataset, condition=condition, label=label, head=head, config=config_path,
                split=split, output=output, command=cmd)


def source_sequences(dataset):
    reference = ROOT / "tools/manifests" / f"{dataset}_stage1_common_v1.json"
    train = read(reference)["splits"]["train"]
    evaluation = (sorted(set(DSEC_TEST) | set(read(ROOT / "tools/manifests/dense/dsec_f3.json")["splits"]["val"]))
                  if dataset == "dsec" else list(M3ED_VAL))
    return reference, sorted(set(train) | set(evaluation)), evaluation


def activity_index(manifest, dataset, event_root, sequences, reference):
    if not manifest.exists():
        return None
    from event_state.activity_quantiles import ActivityIndex, canonical_sha256
    index = ActivityIndex.load(manifest)
    if index.identity["source_contract"]["dataset"] != dataset:
        raise ValueError(f"Wrong dataset activity manifest: {manifest}")
    if index.identity["reference"].get("manifest_sha256") != sha256(reference):
        raise ValueError(f"Activity train reference changed: {manifest}")
    if sorted(index.identity["reference"]["sequences"]) != sorted(read(reference)["splits"]["train"]):
        raise ValueError(f"Activity training sequence pool differs: {manifest}")
    for sequence in sequences:
        if index.source_metadata_sha256.get(sequence) != canonical_sha256(read(event_root / sequence / "metadata.json")):
            raise ValueError(f"Activity source metadata changed: {sequence}")
        index.timestamps(sequence)
    return index


def signature(job, index):
    return dict(format_version=1, head_sha256=sha256(job["head"]),
                config_sha256=sha256(job["config"]), split_sha256=sha256(job["split"]),
                activity_manifest_sha256=index.identity["manifest_sha256"], args=job["command"])


def already_complete(job, index):
    output = job["output"]
    sidecar = output.with_suffix(".signature.json")
    csv = output.with_suffix(".csv")
    if not any(p.exists() for p in (output, sidecar, csv)):
        return False
    if index is None or not all(p.is_file() for p in (output, sidecar, csv)):
        raise ValueError(f"Incomplete existing evaluation is preserved: {output}")
    saved = read(sidecar)
    expected = signature(job, index)
    if (saved.get("signature") != expected or saved.get("report_sha256") != sha256(output)
            or saved.get("csv_sha256") != sha256(csv)):
        raise ValueError(f"Different/stale existing evaluation is preserved: {output}")
    report = read(output).get("activity_evaluation", {})
    if report.get("activity_manifest") != index.identity or report.get("condition_label") != job["label"]:
        raise ValueError(f"Existing activity evaluation identity differs: {output}")
    return True


def prepare_plan(args):
    conditions = CONDITIONS[:3] if args.baselines_only else CONDITIONS
    manifests, generators, indexes, jobs = {}, [], {}, []
    for dataset in ("dsec", "m3ed"):
        reference, sequences, evaluation = source_sequences(dataset)
        event_root = args.data / ("DSEC_cache/events/gep_rgb" if dataset == "dsec" else "m3ed_cache/half_dagr")
        for sequence in sequences:
            base = event_root / sequence
            require_file(base / "metadata.json", nonempty=True)
            if dataset == "dsec":
                require_file(base / "_SUCCESS")
            folder = base if dataset == "dsec" else base / "events"
            if not any(p.stem.isdecimal() for p in folder.glob("*.pt")):
                raise FileNotFoundError(f"Prepared events missing: {folder}")
        manifest = args.output / "manifests" / f"{dataset}.json"
        manifests[dataset] = manifest
        indexes[dataset] = activity_index(manifest, dataset, event_root, sequences, reference)
        generators.append(command("prepare_" + dataset + "_activity_quantiles.py",
            "--event-cache-dir" if dataset == "dsec" else "--prepared-root", event_root,
            "--train-manifest", reference, "--sequences", *evaluation, "--output", manifest))
        target_index = read(require_file(args.depth_run / "targets" / f"{dataset}_val.json", nonempty=True))
        if (target_index.get("dataset") != dataset or target_index.get("role") != "val"
                or target_index.get("task") != "depth" or not target_index.get("labeled")
                or not target_index.get("records")):
            raise ValueError(f"Invalid existing Depth validation targets: {dataset}")
        for record in target_index["records"]:
            require_file(record["target_path"], nonempty=True)
            if record["sequence"] not in evaluation:
                raise ValueError(f"Depth validation sequence outside the expected split: {record['sequence']}")
            if indexes[dataset] is not None:
                indexes[dataset].lookup(record["sequence"], record["timestamp"])
        require_file(ROOT / f"outputs/stage1_common_20261105/statistics/{dataset}_common.json", nonempty=True)
        for condition, label, feature in conditions:
            head = args.depth_run / "heads" / f"{dataset}_{condition}" / "stages"
            require_file(head / "_SUCCESS")
            require_file(head / "best.pt", nonempty=True)
            folder, model, step = ("stage1_joined_baseline_full_20261107", condition + "_w1", 99999)
            if condition.startswith("E1_"):
                folder = ("stage1_rvt_ddp_full_20261105" if condition == "E1_old_ddp"
                          else "stage1_joined_ddp_full_new_20261106")
                model, step = "E1_w3", 33333
            require_file(ROOT / "outputs" / folder / f"{dataset}_{model}" / "checkpoints" / f"step_{step:08d}.pt", nonempty=True)
            job = semantic_job(args, dataset, condition, label, feature, manifest)
            job["skip"] = already_complete(job, indexes[dataset])
            jobs.append(job)
    depth_command = command("run_common_depth.py", "--activity-only", "--data", args.data,
        "--output", args.depth_run, "--dsec-activity-manifest", manifests["dsec"],
        "--m3ed-activity-manifest", manifests["m3ed"])
    if args.baselines_only:
        depth_command.append("--baselines-only")
    if args.scratch:
        depth_command += ["--scratch", str(args.scratch)]
    return dict(manifests=manifests, indexes=indexes, generators=generators, jobs=jobs,
                depth_command=depth_command, conditions=conditions)


def execute(args):
    # Check every model and both datasets before launching any expensive operation.
    with depth_available(args.depth_run):
        plan = prepare_plan(args)
        if args.preflight:
            for dataset, cmd in zip(("dsec", "m3ed"), plan["generators"]):
                print(("[reuse] " if plan["indexes"][dataset] else "[generate] ") + shlex.join(cmd))
            for job in plan["jobs"]:
                print(("[skip verified] " if job["skip"] else "[evaluate] ") + shlex.join(job["command"]))
            print("[evaluate Depth] " + shlex.join(plan["depth_command"]))
            print("Preflight passed; no files were created or modified.")
            return
        args.output.mkdir(parents=True, exist_ok=True)
        for dataset, cmd in zip(("dsec", "m3ed"), plan["generators"]):
            if plan["indexes"][dataset] is None:
                launch(cmd)
                reference, sequences, _ = source_sequences(dataset)
                event_root = args.data / ("DSEC_cache/events/gep_rgb" if dataset == "dsec" else "m3ed_cache/half_dagr")
                plan["indexes"][dataset] = activity_index(plan["manifests"][dataset], dataset, event_root, sequences, reference)
        for job in plan["jobs"]:
            index = plan["indexes"][job["dataset"]]
            if already_complete(job, index):
                print(f"[skip verified] {job['dataset']} | {job['label']}", flush=True)
                continue
            launch(job["command"])
            output = require_file(job["output"], nonempty=True)
            csv = require_file(output.with_suffix(".csv"), nonempty=True)
            report = read(output).get("activity_evaluation", {})
            if report.get("activity_manifest") != index.identity or report.get("condition_label") != job["label"]:
                raise ValueError(f"Evaluation returned a different activity identity: {output}")
            sidecar = dict(signature=signature(job, index), report_sha256=sha256(output), csv_sha256=sha256(csv))
            output.with_suffix(".signature.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    # The Depth child acquires the same lock itself.
    launch(plan["depth_command"])
    depth_reports = {}
    depth_summary_path = args.depth_run / "activity_summary.json"
    depth_summary = read(require_file(depth_summary_path, nonempty=True))
    for dataset in ("dsec", "m3ed"):
        for condition, _, _ in plan["conditions"]:
            name = f"{dataset}_{condition}"
            report = path(depth_summary[name]["activity_report"])
            require_file(report, nonempty=True)
            require_file(report.with_suffix(".csv"), nonempty=True)
            depth_reports[name] = str(report)
    summary = dict(manifests={k: str(v) for k, v in plan["manifests"].items()},
        semantic={job["dataset"] + "_" + job["condition"]: str(job["output"]) for job in plan["jobs"]},
        depth=depth_reports, depth_summary=str(depth_summary_path))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Completed Semantic + Depth activity evaluation: {args.output / 'summary.json'}")


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/home/iASL/Arata_repo/dataset"))
    parser.add_argument("--depth-run", type=Path, default=Path("outputs/stage2_common_depth_vkitti_20261110"))
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_depth_activity_20261110"))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check inputs and show commands without writing or running inference")
    args = parser.parse_args()
    for name in ("data", "depth_run", "output", "scratch"):
        if getattr(args, name) is not None:
            setattr(args, name, path(getattr(args, name)))
    execute(args)


if __name__ == "__main__":
    main()
