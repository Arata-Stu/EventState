#!/usr/bin/env python3
"""Compare final-only and four-block linear Semantic heads on two frozen LSTM placements."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.dense.protocol import sha256_file
from event_state.evaluation.placement_features import CONDITIONS, checkpoint_paths
from tools.common_depth_parallel import parse_gpus, run_parallel
from tools.placement_semantic_readout_results import compare_readouts, load_final_control, verify_head_recipe
from tools.run_placement_semantic import DATASETS, baseline_results, cache_bytes, read, run_condition, save
from tools.run_temporal_downstream import idle_gpus


OUTPUTS = ROOT / "outputs"
PLACEMENT_HEADS = OUTPUTS / "stage2_placement_semantic_20261118"
JOBS = [(ds, condition) for condition in CONDITIONS for ds in DATASETS]


def label(condition):
    return CONDITIONS[condition] + "・3/6/9/12層連結"


def new_probe(a, ds, condition, reference, lock_fd):
    status_callback = (lambda status: save(a.worker_status, status)) if a.worker_status else None
    result = run_condition(a, ds, condition, reference, lock_fd, feature="stages",
                           checkpoint=checkpoint_paths(ds, ROOT)[condition], label=label(condition),
                           status_callback=status_callback)
    if result is not None:
        return dict(result, head_recipe=verify_head_recipe(ds, "stages", result["head"], ROOT))


def controls_for_jobs():
    controls = {ds: {} for ds in DATASETS}
    for ds, condition in JOBS:
        checkpoint = checkpoint_paths(ds, ROOT)[condition]
        controls[ds][condition] = load_final_control(ds, condition, project_root=ROOT,
            placement_root=PLACEMENT_HEADS, backbone_sha256=sha256_file(checkpoint))
    for ds, pair in controls.items():
        if any(row["head_recipe"] != pair["E1"]["head_recipe"] for row in pair.values()):
            raise ValueError(f"{ds}: the two final-only control heads used different training recipes")
    return controls


def write_summary(output, controls, fitted):
    marker = output / "_SUCCESS"
    if len(fitted) < len(JOBS):
        marker.unlink(missing_ok=True)
    comparisons = {}
    for ds in DATASETS:
        if all(f"{ds}_{cond}" in fitted for cond in CONDITIONS):
            stages = {cond: fitted[f"{ds}_{cond}"] for cond in CONDITIONS}
            comparisons[ds] = compare_readouts(ds, controls[ds], stages)
    save(output / "summary.json", dict(mIoU_unit="fraction", effect_unit="percentage points",
        final_only_controls=controls, four_layer_heads=fitted, comparisons=comparisons))
    if len(comparisons) == len(DATASETS):
        marker.touch()
    return comparisons


def worker_command(a, job, status, result, fd):
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--execute",
            "--data", str(a.data), "--output", str(a.output), "--scratch", str(a.scratch),
            "--worker", job["dataset"], job["condition"], "--parent-lock-fd", str(fd),
            "--worker-status", str(status), "--worker-result", str(result)]


def dispatch(a, gpus, controls, lock):
    (a.output / "_SUCCESS").unlink(missing_ok=True)
    references = baseline_results()
    fitted, sizes, pending = {}, {}, []
    for ds, condition in JOBS:
        key = f"{ds}_{condition}"
        if (a.output / key / "verified.json").is_file():
            fitted[key] = new_probe(a, ds, condition, references[ds], lock.fileno())
        else:
            if ds not in sizes:
                sizes[ds] = cache_bytes(a, ds, condition, feature="stages", label=label(condition),
                                        checkpoint=checkpoint_paths(ds, ROOT)[condition])
            pending.append(dict(name=key, label=label(condition), dataset=ds, condition=condition,
                                kind="semantic", remaining_cache_bytes=sizes[ds]))
    write_summary(a.output, controls, fitted)

    def completed(key, receipt):
        fitted[key] = receipt
        write_summary(a.output, controls, fitted)

    if pending:
        a.scratch.mkdir(parents=True, exist_ok=True)
        gib = 2**30
        print(f"Retained feature caches for {len(pending)} new jobs: "
              f"{sum(j['remaining_cache_bytes'] for j in pending)/gib:.1f} GiB; "
              f"{shutil.disk_usage(a.scratch).free/gib:.1f} GiB available at {a.scratch}", flush=True)
        run_parallel(pending, gpus, scratch=a.scratch, state_dir=a.output / "logs" / uuid4().hex,
                     lock_fd=lock.fileno(), idle_devices=idle_gpus,
                     worker_command=lambda job, sp, rp: worker_command(a, job, sp, rp, lock.fileno()),
                     on_complete=completed)
    summary = read(a.output / "summary.json")
    for ds in DATASETS:
        comparison = summary["comparisons"][ds]
        print(f"\n{ds} {comparison['role']} | mIoU, all frames")
        for condition, cells in comparison["cells"].items():
            gain = comparison["gain_pp"][condition]["all"]
            print(f"  {CONDITIONS[condition]} | final {100*cells['h']['mIoU']['all']:.3f}%"
                  f" -> four layers {100*cells['stages']['mIoU']['all']:.3f}% ({gain:+.3f} pp)")
        print(f"  Four-stage gain minus 12-only gain = {comparison['interaction_pp']['all']:+.3f} pp")
    print(f"Completed 4 new Semantic heads and 4 matched controls: {a.output / 'summary.json'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1,2", help="One Semantic job per idle physical GPU")
    p.add_argument("--data", type=Path, default=Path("/home/iASL/Arata_repo/dataset"))
    p.add_argument("--output", type=Path, default=OUTPUTS / "stage2_placement_semantic_readout_20261123")
    p.add_argument("--scratch", type=Path, help="Root for retained sparse feature maps")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Verify controls, checkpoints and data on server")
    mode.add_argument("--execute", action="store_true", help="Fit four heads, then compare both readouts")
    p.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    p.add_argument("--worker-status", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--parent-lock-fd", type=int, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    try:
        gpus = parse_gpus(a.gpus)
    except ValueError as error:
        p.error(str(error))
    a.data, a.output = a.data.expanduser().resolve(), a.output.expanduser().resolve()
    a.scratch = (a.scratch or a.data / "downstream_scratch" / a.output.name).expanduser().resolve()
    a.activity_root = OUTPUTS / "semantic_depth_activity_20261110/manifests"
    a.baseline_semantic_config = OUTPUTS / "stage2_dsec_semantic_linear_20261109/E1/final/config.json"
    a.m3ed_target_root = OUTPUTS / "stage2_m3ed_semantic_linear_20261109/E1/targets"
    a.statistics = OUTPUTS / "stage1_joined_baseline_full_20261107/m3ed_common.json"
    a.project_root = ROOT
    if a.worker:
        if (tuple(a.worker) not in JOBS or not a.execute or a.worker_status is None
                or a.worker_result is None or a.parent_lock_fd is None):
            p.error("Expected a Semantic worker assigned by the parent launcher")
        if not os.path.samestat(os.fstat(a.parent_lock_fd), (a.output / ".lock").stat()):
            raise ValueError("Worker received another suite's lock")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not gpu.isdecimal() or idle_gpus((gpu,)) != (gpu,):
            raise RuntimeError(f"Assigned GPU {gpu} is busy or unavailable")
        ds, condition = a.worker
        save(a.worker_result, new_probe(a, ds, condition, baseline_results()[ds], a.parent_lock_fd))
        return
    for ds, condition in JOBS:
        print(f"[{ds}_{condition}] {label(condition)} | reuse final-only head, fit 4-layer head")
    if not (a.execute or a.preflight):
        print(f"4 new Semantic jobs; GPU pool {','.join(gpus)}. Use --preflight or --execute on sig-gpu5.")
        return
    # The DSEC shell wrapper launches 'python'. Keep it in this launcher's environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    controls = controls_for_jobs()
    if a.preflight:
        references = baseline_results()
        for ds, condition in JOBS:
            new_probe(a, ds, condition, references[ds], None)
        print("Verified four final-only controls and all four proposed Semantic jobs.")
        return
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(study="placement_semantic_readout", data=str(a.data),
                        controls={ds: {key: [row["backbone_sha256"], row["head_sha256"]]
                                       for key, row in rows.items()} for ds, rows in controls.items()})
        path = a.output / "inputs.json"
        if path.is_file():
            if read(path) != identity:
                raise ValueError(f"Different data or control heads at {a.output}; choose a fresh --output")
        elif any(child.name != ".lock" for child in a.output.iterdir()):
            raise FileExistsError(f"Existing files at {a.output}; choose a fresh --output")
        else:
            save(path, identity)
        dispatch(a, gpus, controls, lock)


if __name__ == "__main__":
    main()
