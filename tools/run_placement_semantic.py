#!/usr/bin/env python3
"""Fit frozen Semantic probes for Patchwise LSTM placements, one job per idle GPU."""
import argparse
from copy import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.dense.common_protocol import audit_common, load_common
from event_state.dense.protocol import sha256_file
from tools.common_depth_parallel import parse_gpus, run_parallel
from tools.placement_conditions import PLACEMENTS, PLACEMENT_BLOCKS, SKIP_CONDITION, placement_label
from tools.run_temporal_downstream import idle_gpus
from tools.temporal_downstream_semantic import build_semantic_job, run_semantic_job

DATASETS = ("dsec", "m3ed")
PLACEMENT_OUTPUT = ROOT / "outputs/stage2_placement_semantic_20261118"
REFERENCE = ROOT / "docs/evaluation_records/20261115_semantic_depth_activity_metrics.json"
COMPARISON_HASHES = ("samples_sha256", "samples_targets_sha256",
                     "activity_assignments_sha256", "validity_support_sha256")
INPUT_PATHS = ("stage1_root", "activity_root", "data", "statistics",
               "baseline_semantic_config", "m3ed_target_root")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def baseline_results():
    records = read(REFERENCE)["results"]
    return {ds: next(r for r in records if r["task"] == "semantic" and r["condition"] == ds + "_E1")
            for ds in DATASETS}


def validate_config(config, dataset, condition):
    temporal = config["model"]["temporal"]
    loss = config["loss"]
    if (config["dataset"]["name"] != dataset or config["seed"] != 0
            or config["training"]["max_steps"] != 99999
            or temporal["type"] != "four_stage_lstm"
            or temporal.get("after_blocks") != list(PLACEMENT_BLOCKS[condition])
            or temporal.get("residual_skip", False) is not (condition == SKIP_CONDITION)
            or any(temporal[k] != 384 for k in ("input_dim", "hidden_dim", "output_dim"))
            or not loss["h_distill"]["enabled"] or loss["h_distill"]["cosine_weight"] != 1
            or loss["h_distill"]["mse_weight"] != 1 or loss["z_objective"]["type"] != "none"
            or loss["z_objective"]["weight"] != 0):
        raise ValueError(f"Stage 1 checkpoint recipe/placement mismatch: {dataset}, {condition}")
    audit_common(config)


def verified_metrics(dataset, condition, output, baseline, *, feature="h", label=None, backbone_sha256=None):
    """Compare evaluation inputs/GT/support with the completed, single-GPU S4 reference."""
    if not (output / "_SUCCESS").is_file():
        raise ValueError(f"Semantic worker has not completed: {output}")
    report_path, head_path, role = (
        ("activity_test.json", "final/last.pt", "test") if dataset == "dsec" else
        (f"{feature}/activity_groups.json", f"{feature}/best.pt", "validation"))
    label = label or placement_label(condition)
    report = read(output / report_path)
    activity = report["activity_evaluation"]
    if (report["role"] != role or report["feature"] != feature
            or activity["condition_label"] != label
            or activity["head_sha256"] != sha256_file(output / head_path)
            or activity["activity_manifest"]["manifest_sha256"] != baseline["manifest_sha256"]
            or activity["state_policy_by_sequence"] != baseline["state_policy_by_sequence"]
            or any(activity[key] != baseline[key] for key in COMPARISON_HASHES)
            or activity["groups"].keys() != baseline["groups"].keys()):
        raise ValueError(f"Head or evaluation set differs from the matched Semantic protocol: {output}")
    if feature == "stages":
        config = read((output / head_path).parent / "config.json")
        if (not backbone_sha256 or any(source.get("backbone_sha256") != backbone_sha256
                or source.get("stage_blocks") != [3, 6, 9, 12] for source in (config, report))):
            raise ValueError(f"Frozen checkpoint or block order differs for the four-layer readout: {output}")
    scores = {}
    deltas = {}
    for group, source in baseline["groups"].items():
        row = activity["groups"][group]
        if (row["frames"] != source["frames"] or row["valid_pixels"] != source["valid_pixels"]
                or row["metrics"]["class_support"] != source["metrics"]["class_support"]):
            raise ValueError(f"Semantic support differs in {group}: {output}")
        score = row["metrics"]["mIoU"]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid Semantic mIoU in {group}: {output}")
        scores[group] = score
        deltas[group] = 100 * (score - source["metrics"]["mIoU"])
    result = dict(dataset=dataset, condition=condition, label=label,
                  after_blocks=[12] if condition == "E1" else list(PLACEMENT_BLOCKS[condition]), role=role, head=str(output / head_path),
                  head_sha256=activity["head_sha256"], report=str(output / report_path),
                  mIoU=scores)
    if feature == "stages":
        result.update(feature=feature, stage_blocks=config["stage_blocks"])
    else:
        result["delta_from_final_block_only_pp"] = deltas
    if dataset == "dsec":
        epoch = read(output / "final/config.json")["selected_development_epoch"]
        metrics_path = output / "development/metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        for index, record in enumerate(records):
            val = record["val/mIoU"]
            if (record["epoch"] != index or type(val) not in (int, float)
                    or not math.isfinite(val) or not 0 <= val <= 1):
                raise ValueError(f"Invalid DSEC development metrics at epoch {index}: {metrics_path}")
        if (type(epoch) is not int or not records
                or epoch != max(records, key=lambda row: row["val/mIoU"])["epoch"]):
            raise ValueError(f"DSEC refit epoch differs from the development selection: {metrics_path}")
        result["development_selection"] = dict(epoch_zero_based=epoch, val_mIoU=records[epoch]["val/mIoU"],
                                                metrics=str(metrics_path))
    return result


def m3ed_command(a, ckpt, cache, output, condition, *, feature="h", label=None):
    return [sys.executable, "-u", str(ROOT / "tools/run_common_m3ed_semantic.py"),
            "--checkpoint", str(ckpt), "--prepared-root", str(a.data / "m3ed_cache/half_dagr"),
            "--target-root", str(a.m3ed_target_root), "--statistics", str(a.statistics),
            "--cache", str(cache), "--output", str(output), "--features", feature,
            "--activity-manifest", str(a.activity_root / "m3ed.json"), "--condition-label", label or placement_label(condition)]


def replacement_control(root, dataset, baseline):
    """Reuse the trained block-3 replacement head on exactly the same scored samples."""
    from event_state.training import load_checkpoint_config_metadata

    receipt = root / f"{dataset}_E1_b3/verified.json"
    saved = read(receipt)
    attempt = saved["attempt"]
    if not attempt.startswith("run-") or Path(attempt).name != attempt:
        raise ValueError(f"Invalid replacement control attempt: {receipt}")
    checkpoint = Path(saved["backbone"])
    metadata = load_checkpoint_config_metadata(checkpoint)
    validate_config(metadata.config, dataset, "E1_b3")
    if (metadata.global_step != 99999 or saved["global_step"] != 99999 or saved["seed"] != 0
            or sha256_file(checkpoint) != saved["backbone_sha256"]):
        raise ValueError(f"Unexpected Stage 1 replacement control: {receipt}")
    measured = verified_metrics(dataset, "E1_b3", receipt.parent / attempt, baseline)
    if any(saved[k] != v for k, v in measured.items()):
        raise ValueError(f"Archived replacement control differs from its report: {receipt}")
    return dict(condition="E1_b3", label=measured["label"], mIoU=measured["mIoU"],
                head_sha256=measured["head_sha256"], backbone_sha256=saved["backbone_sha256"],
                receipt=str(receipt))


def skip_comparison(scores, control):
    return dict(residual_skip=True, replacement_control=control,
                delta_from_b3_replacement_pp={group: 100 * (score - control["mIoU"][group])
                                              for group, score in scores.items()})


def run_condition(a, ds, condition, baseline, lock_fd, status_callback=None, *, feature="h", checkpoint=None, label=None):
    from event_state.training import load_checkpoint_config_metadata

    key = f"{ds}_{condition}"
    label = label or placement_label(condition)
    checkpoint = checkpoint or a.stage1_root / f"{key}_w1/checkpoints/step_00099999.pt"
    metadata = load_checkpoint_config_metadata(checkpoint)
    if metadata.global_step != 99999:
        raise ValueError(f"Expected the final 99,999-update Stage 1 checkpoint: {checkpoint}")
    if feature == "stages":
        from event_state.evaluation.placement_features import placement_input_contract
        placement_input_contract(metadata.config, ds, condition, metadata.global_step, ROOT)
    else:
        validate_config(metadata.config, ds, condition)
    checkpoint_sha = sha256_file(checkpoint)
    control = replacement_control(a.placement_semantic_root, ds, baseline) if condition == SKIP_CONDITION else None
    directory = a.output / key
    receipt = directory / "verified.json"
    if receipt.is_file():
        saved = read(receipt)
        if saved["backbone_sha256"] != checkpoint_sha:
            raise ValueError(f"Existing run used another Stage 1 checkpoint: {receipt}; use a new --output")
        measured = verified_metrics(ds, condition, directory / saved["attempt"], baseline,
                                    feature=feature, label=label, backbone_sha256=checkpoint_sha)
        if control:
            measured.update(skip_comparison(measured["mIoU"], control))
        if any(saved.get(k) != v for k, v in measured.items()):
            raise ValueError(f"Retained metrics no longer match the saved report: {receipt}")
        print(f"[{key}] {label} | verified existing head", flush=True)
        return saved
    # Distinct attempts preserve previous failures and any data from a differently named suite.
    attempt = "run-" + uuid4().hex[:12]
    output = directory / attempt
    cache = a.scratch / key / attempt
    if cache.exists() or output.exists():
        raise FileExistsError(f"Attempt already exists: {output}, {cache}")
    if ds == "dsec":
        settings = copy(a)
        settings.output, settings.scratch = directory, cache.parent
        job = build_semantic_job(settings, checkpoint=checkpoint, name=attempt, label=label, feature=feature)
        job["name"] = key
        if a.execute:
            a.scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(a.scratch).free < 1.1 * job["cache_bytes"]:
                raise OSError(f"Insufficient disk for DSEC sparse {feature} features: {cache}")
            settings.parent_lock_fd = lock_fd
            run_semantic_job(settings, job, status_callback=status_callback)
    else:
        command = m3ed_command(a, checkpoint, cache, output, condition, feature=feature, label=label)
        if not a.execute:
            command.append("--preflight")
        if lock_fd is not None:
            command.extend(("--parent-lock-fd", str(lock_fd)))
        if getattr(a, "worker_status", None) is not None:
            command.extend(("--worker-status", str(a.worker_status)))
        subprocess.run(command, cwd=ROOT, check=True, pass_fds=() if lock_fd is None else (lock_fd,))
    if a.execute:
        result = verified_metrics(ds, condition, output, baseline, feature=feature, label=label,
                                  backbone_sha256=checkpoint_sha)
        result.update(attempt=attempt, seed=0, global_step=metadata.global_step,
                      backbone=str(checkpoint), backbone_sha256=checkpoint_sha, cache=str(cache))
        if control:
            result.update(skip_comparison(result["mIoU"], control))
        save(receipt, result)
        print(f"[{key}] {label} | {result['role']} mIoU={100*result['mIoU']['all']:.3f}%", flush=True)
        return result


def skip_history(a, dataset, result, *, execute=False):
    directory = a.output / f"{dataset}_{SKIP_CONDITION}"
    path = directory / "history.json"
    activity = a.activity_root / f"{dataset}.json"
    identity = dict(head_receipt_sha256=sha256_file(directory / "verified.json"),
                    activity_manifest_sha256=sha256_file(activity))
    if path.is_file():
        saved = read(path)
        if saved["identity"] != identity:
            raise ValueError(f"Different backbone, head, or activity manifest for history evaluation: {path}")
        evaluation = saved["evaluation"]
    elif execute:
        from tools.evaluate_b3_skip_history import evaluate_history
        evaluation = evaluate_history(dataset, result, activity, data_root=a.data)
    else:
        return None
    reference = baseline_results()[dataset]
    scale = evaluation["learned_skip_scale"]
    reset = evaluation["reset_each_window_mIoU"]
    deltas = evaluation["continuous_minus_reset_pp"]
    expected = dict(head_sha256=result["head_sha256"], backbone_sha256=result["backbone_sha256"],
                    activity_manifest_sha256=reference["manifest_sha256"], state_policy="reset_each_window",
                    continuous_report=result["report"], continuous_mIoU=result["mIoU"],
                    comparison_hashes={key: reference[key] for key in COMPARISON_HASHES})
    if (any(evaluation.get(k) != v for k, v in expected.items())
            or type(scale) not in (float, int) or not math.isfinite(scale)
            or reset.keys() != result["mIoU"].keys() or deltas.keys() != reset.keys()
            or any(not 0 <= value <= 1 or not math.isclose(deltas[group],
                   100 * (result["mIoU"][group] - value), abs_tol=1e-8) for group, value in reset.items())):
        raise ValueError(f"History scores differ from the verified continuous head or protocol: {path}")
    if not path.is_file():
        save(path, dict(identity=identity, evaluation=evaluation))
    return evaluation


def write_summary(a, baselines):
    skip = SKIP_CONDITION in getattr(a, "conditions", ())
    suite_conditions = (SKIP_CONDITION,) if skip else tuple(PLACEMENTS)
    complete = {}
    for ds in DATASETS:
        for condition in suite_conditions:
            key = f"{ds}_{condition}"
            receipt = a.output / key / "verified.json"
            if receipt.is_file():
                result = read(receipt)
                if skip:
                    history = skip_history(a, ds, result)
                    if history is None:
                        continue
                    result["history"] = history
                complete[key] = result
    refs = {ds: dict(label="E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（12層後）",
                     mIoU={name: g["metrics"]["mIoU"] for name, g in baseline["groups"].items()})
            for ds, baseline in baselines.items()}
    save(a.output / "summary.json", dict(baseline_source=str(REFERENCE), baseline=refs,
        mIoU_unit="fraction [0,1]", delta_unit="percentage points",
        splits={"dsec": "6 train/2 dev to select epochs, refit 8 train, score 3 official test",
                "m3ed": "3 train, 2 public validation (InternImage pseudo-labels; validation-best head)"},
        semantic=complete))
    if len(complete) == len(DATASETS) * len(suite_conditions):
        (a.output / "_SUCCESS").touch()
    elif skip:
        (a.output / "_SUCCESS").unlink(missing_ok=True)


def cache_bytes(a, dataset, condition, *, feature="h", checkpoint=None, label=None):
    """Reserve one fp16 feature map per labeled frame on the native Semantic grid."""
    if dataset == "dsec":
        checkpoint = checkpoint or a.stage1_root / f"dsec_{condition}_w1/checkpoints/step_00099999.pt"
        job = build_semantic_job(a, checkpoint=checkpoint, name="estimate-" + uuid4().hex,
                                 label=label or placement_label(condition), feature=feature)
        return job["cache_bytes"]
    import h5py
    import hdf5plugin  # Register the compressed filters used by the prepared M3ED targets.
    import numpy as np

    split = load_common("m3ed")["splits"]
    frames = 0
    for seq in (*split["train"], *split["val"]):
        with h5py.File(a.m3ed_target_root / seq / "targets.h5") as targets:
            frames += int(np.count_nonzero(targets["semantics_frame_valid"][1:]))
    channels = {"h": 384, "stages": 1536}[feature]
    return frames * (channels * 22 * 40 * 2 + 4096)


def worker_command(a, job, status_path, result_path, lock_fd):
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--execute",
           "--datasets", job["dataset"], "--conditions", job["condition"],
           "--worker-result", str(result_path), "--worker-status", str(status_path),
           "--parent-lock-fd", str(lock_fd)]
    for name in (*INPUT_PATHS, "output", "scratch"):
        cmd.extend(("--" + name.replace("_", "-"), str(getattr(a, name))))
    if job["condition"] == SKIP_CONDITION:
        cmd.extend(("--placement-semantic-root", str(a.placement_semantic_root)))
    return cmd


def execute_jobs(a, jobs, gpus, baselines, lock_fd):
    pending = []
    for ds, condition in jobs:
        key = f"{ds}_{condition}"
        directory = a.output / key
        head_done = (directory / "verified.json").is_file()
        if head_done and (condition != SKIP_CONDITION or (directory / "history.json").is_file()):
            run_condition(a, ds, condition, baselines[ds], lock_fd)
        else:
            pending.append(dict(name=key, dataset=ds, condition=condition, kind="semantic",
                                label=placement_label(condition), history_only=head_done))
    if not pending:
        write_summary(a, baselines)
        return
    sizes = {}
    for job in pending:
        ds = job["dataset"]
        if job["history_only"]:
            job["remaining_cache_bytes"] = 0
            continue
        if ds not in sizes:
            sizes[ds] = cache_bytes(a, ds, job["condition"])
        job["remaining_cache_bytes"] = sizes[ds]
    skip = SKIP_CONDITION in a.conditions
    available = gpus if skip else idle_gpus(gpus)
    if not available:
        raise RuntimeError(f"No idle GPU among {', '.join(gpus)}")
    a.scratch.mkdir(parents=True, exist_ok=True)
    state_dir = a.output / "logs" / ("dispatch-" + uuid4().hex)
    print(f"GPU pool: {', '.join(available)} | worker logs: {state_dir}", flush=True)
    total = sum(job["remaining_cache_bytes"] for job in pending)
    print(f"Estimated retained features for {len(pending)} pending jobs: {total / 2**30:.1f} GiB | "
          f"available: {shutil.disk_usage(a.scratch).free / 2**30:.1f} GiB at {a.scratch}", flush=True)
    try:
        run_parallel(pending, available, scratch=a.scratch, state_dir=state_dir, lock_fd=lock_fd,
                     worker_command=lambda job, sp, rp: worker_command(a, job, sp, rp, lock_fd),
                     on_complete=lambda _name, _result: write_summary(a, baselines),
                     idle_devices=idle_gpus if skip else None)
    finally:
        # A worker can save its verified receipt just before its group is interrupted.
        write_summary(a, baselines)


def run_worker(a, ds, condition):
    if not a.execute or a.worker_status is None or a.parent_lock_fd is None:
        raise ValueError("Internal Semantic worker requires an execute invocation and inherited output lock")
    held = os.fstat(a.parent_lock_fd)
    lock = (a.output / ".lock").stat()
    if (held.st_dev, held.st_ino) != (lock.st_dev, lock.st_ino):
        raise ValueError("Internal Semantic worker inherited a different run lock")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdecimal() or idle_gpus((gpu,)) != (gpu,):
        raise RuntimeError(f"Assigned GPU {gpu} is unavailable before starting {ds}_{condition}")
    result = run_condition(a, ds, condition, baseline_results()[ds], a.parent_lock_fd,
                           status_callback=lambda row: save(a.worker_status, row))
    if condition == SKIP_CONDITION:
        save(a.worker_status, dict(name=f"{ds}_{condition}", condition=placement_label(condition),
                                   phase="scoring retained and reset history", remaining_cache_bytes=0))
        result["history"] = skip_history(a, ds, result, execute=True)
    save(a.worker_result, result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--placement-semantic-root", type=Path, default=PLACEMENT_OUTPUT,
                   help="Existing six-placement Semantic suite supplying the block-3 replacement control")
    devices = p.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int, help="Physical GPU ID for sequential jobs")
    devices.add_argument("--gpus", help="Comma-separated physical GPU IDs, e.g. 0,1,2. One job per idle GPU")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--conditions", nargs="+", choices=tuple(PLACEMENT_BLOCKS), default=tuple(PLACEMENTS))
    p.add_argument("--data", type=Path, default=Path("/home/iASL/Arata_repo/dataset"))
    p.add_argument("--scratch", type=Path)
    p.add_argument("--activity-root", type=Path, default=ROOT/"outputs/semantic_depth_activity_20261110/manifests")
    p.add_argument("--statistics", type=Path, help="M3ED common stats (default: stage1-root/m3ed_common.json)")
    p.add_argument("--baseline-semantic-config", type=Path,
                   default=ROOT/"outputs/stage2_dsec_semantic_linear_20261109/E1/final/config.json")
    p.add_argument("--m3ed-target-root", type=Path,
                   default=ROOT/"outputs/stage2_m3ed_semantic_linear_20261109/E1/targets")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Check real checkpoints and input data on server")
    modes.add_argument("--execute", action="store_true", help="Fit and evaluate the requested Semantic probes")
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-status", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--parent-lock-fd", type=int, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    skip = SKIP_CONDITION in a.conditions
    gpu_selection = a.gpus if a.gpus is not None else a.gpu
    try:
        gpus = parse_gpus(str(gpu_selection) if gpu_selection is not None else ("0,1,2" if skip else "0"))
    except ValueError as error:
        p.error(str(error))
    if len(set(a.datasets)) != len(a.datasets) or len(set(a.conditions)) != len(a.conditions):
        p.error("Expected unique datasets and conditions")
    if skip and len(a.conditions) != 1:
        p.error(f"Run --conditions {SKIP_CONDITION} separately from the original placement suite")
    a.stage1_root = a.stage1_root or ROOT / "outputs" / (
        "stage1_b3_skip_full_20261126" if skip else "stage1_placement_full_20261118")
    a.output = a.output or (ROOT / "outputs/stage2_b3_skip_semantic_20261126" if skip else PLACEMENT_OUTPUT)
    if skip and a.output.resolve() == a.placement_semantic_root.resolve():
        p.error("Use a distinct output for the residual experiment and its replacement control")
    a.project_root = ROOT
    a.statistics = a.statistics or a.stage1_root / "m3ed_common.json"
    a.scratch = a.scratch or a.data / "downstream_scratch" / a.output.name
    for key, value in vars(a).copy().items():
        if isinstance(value, Path):
            setattr(a, key, value.expanduser().resolve())
    jobs = [(ds, cond) for cond in a.conditions for ds in a.datasets]
    if a.worker_result is not None:
        if len(jobs) != 1:
            p.error("Internal Semantic worker requires exactly one dataset and condition")
        run_worker(a, *jobs[0])
        return
    for ds, cond in jobs:
        print(f"[{ds}_{cond}] {placement_label(cond)} | Semantic | GPU pool {','.join(gpus)}", flush=True)
    if not a.preflight and not a.execute:
        print(f"{len(jobs)} planned; use --preflight or --execute on the learning server.")
        return
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    baselines = baseline_results()
    if a.preflight:
        for ds, cond in jobs:
            run_condition(a, ds, cond, baselines[ds], None)
        print(f"Preflight passed ({len(jobs)} Semantic jobs).")
        return
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Subset invocations can contribute to a common summary only for the same input runs.
        identity = {name: str(getattr(a, name)) for name in INPUT_PATHS}
        if skip:
            identity.update(conditions=[SKIP_CONDITION], placement_semantic_root=str(a.placement_semantic_root))
        identity_path = a.output / "inputs.json"
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError(f"Different input runs are already recorded at {a.output}; choose a new --output")
        if not identity_path.exists():
            save(identity_path, identity)
        if skip:
            (a.output / "_SUCCESS").unlink(missing_ok=True)
        execute_jobs(a, jobs, gpus, baselines, lock.fileno())
    print(f"Completed {len(jobs)} requested Semantic jobs: {a.output / 'summary.json'}")


if __name__ == "__main__":
    main()
