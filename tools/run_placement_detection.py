#!/usr/bin/env python3
"""Run nine frozen DSEC-Det comparisons on available GPU 1/2. Plan by default."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from event_state.activity_quantiles import ActivityIndex, canonical_sha256
from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import sha256_file
from tools.common_depth_parallel import parse_gpus, run_parallel
from tools.placement_conditions import PLACEMENTS, placement_label
from tools.run_placement_semantic import read, save
from tools.run_temporal_downstream import idle_gpus

OFFICIAL = ROOT / "tools/manifests/dsec_det_official_split.yaml"
TRAIN35 = ROOT / "tools/manifests/dsec_stage1_common_v1.json"
LABELS = {"E0": "E0 Encoder（z蒸留）", "E1": "E1 Encoder＋LSTM（h蒸留）・Patchwise LSTM（12層後）",
          "E1_convlstm": "E1_convlstm E1 Encoder＋LSTM（h蒸留）・ConvLSTM",
          **{key: f"{key} {placement_label(key)}" for key in PLACEMENTS}}
RECT_BYTES = 384 * 28 * 40 * 2 + 8192
WARP_BYTES = 384 * 27 * 40 * 2 + 8192


def args_for(condition, a):
    root = a.placement_root if condition in PLACEMENTS else (
        a.temporal_root if condition == "E1_convlstm" else a.baseline_root)
    return root / f"dsec_{condition}_w1/checkpoints/step_00099999.pt", "z" if condition == "E0" else "h"


def roles(phase):
    return ("train", "val") if phase == "val" else ("test",)


def tool(name, *args, lock_fd=None, cpu=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-u", str(ROOT / "tools" / name), *map(str, args)],
                   env=env, cwd=ROOT, check=True, pass_fds=() if lock_fd is None else (lock_fd,))


def inputs(a, split):
    """Hash source metadata, calibration and labels; require the complete prepared timelines."""
    from event_state.detection.data import find_tracks_file, find_rectify_map, find_image_timestamps

    info = {}
    common = set(load_common("dsec")["splits"]["train"])
    if not common <= set(split["train"]):
        raise ValueError("The common Stage 1 training pool crosses the DSEC-Det holdout")
    names = {seq for role in roles(a.phase) for seq in split[role]}
    for seq in sorted(names | common):
        source = a.events / seq
        metadata = read(source / "metadata.json")
        sha = canonical_sha256(metadata)
        marker = read(source / "_SUCCESS")
        if (marker.get("manifest_sha256") != sha or metadata["sequence_name"] != seq
                or str(metadata["dataset"]).lower() != "dsec" or metadata["coordinate_space"] != "rectified_event"
                or metadata["height"] != 480 or metadata["width"] != 640
                or metadata["representation"]["type"] != "gep_rgb" or metadata["event_window"] != "rgb_interval"
                or metadata.get("event_window_us") is not None or metadata.get("event_window_fraction", 1) != 1):
            raise ValueError(f"Incorrect source geometry, event window or completion marker: {source}")
        count = metadata["frame_count"]
        paths = sorted((p.stem for p in source.glob("*.pt") if p.stem.isdecimal()), key=int)
        if len(paths) != count or count <= 0:
            raise ValueError(f"Incomplete event source: {source}")
        row = dict(frames=count, metadata=sha, timestamps=canonical_sha256(paths))
        if seq in names:
            for key, path in (("tracks", find_tracks_file(a.labels, seq)), ("rectify_map", find_rectify_map(a.raw, seq)),
                              ("image_timestamps", find_image_timestamps(a.raw, seq))):
                row[key] = sha256_file(path)
        info[seq] = row
    return info


def check_activity(path, sequences, source_info):
    index = ActivityIndex.load(path)
    identity = index.identity
    reference = identity["reference"]
    if (reference["manifest_sha256"] != sha256_file(TRAIN35)
            or set(reference["sequences"]) != set(load_common("dsec")["splits"]["train"])
            or len(index.groups) != 5):
        raise ValueError(f"Expected five activity groups fitted to common DSEC train35: {path}")
    for seq in set(reference["sequences"]) | set(sequences):
        if index.source_metadata_sha256.get(seq) != source_info[seq]["metadata"]:
            raise ValueError(f"Activity source mismatch/missing coverage for {seq}: {path}")
        stamps = [str(t) for t in index.timestamps(seq)]
        if canonical_sha256(stamps) != source_info[seq]["timestamps"]:
            raise ValueError(f"Activity timeline mismatch for {seq}: {path}")
    fit = {k: identity[k] for k in ("reference", "cutpoints", "groups", "source_contract")}
    return identity["manifest_sha256"], canonical_sha256(fit)


def check_warp(cache, role, sequences, source_info, backbone_sha, feature):
    """Require every source timestamp and a complete compact, native warped cache before releasing scratch."""
    for seq in sequences:
        folder = cache / seq
        meta = read(folder / "metadata.json")
        required = dict(official_role=role, sequence_name=seq, benchmark_format_version=3,
                        checkpoint_sha256=backbone_sha, checkpoint_step=99999, state_policy="continuous",
                        detection_scale=1, coordinate_space="dsec_det_distorted", input_size=[430, 640],
                        patch_size=16, features=[feature], frame_count=source_info[seq]["frames"])
        if (any(meta.get(k) != v for k, v in required.items())
                or canonical_sha256(meta["source_event_cache_metadata"]) != source_info[seq]["metadata"]):
            raise ValueError(f"Wrong benchmark feature metadata: {folder}")
        paths = sorted((p.stem for p in folder.glob("*.pt") if p.stem.isdecimal()), key=int)
        if canonical_sha256(paths) != source_info[seq]["timestamps"]:
            raise ValueError(f"Missing benchmark features: {folder}")


def remaining(job, pending_roles):
    # Each role extracts then warps; only its own rectified directory is temporary.
    retained = 0
    peak = 0
    for role in pending_roles:
        n = job["frames"][role]
        allocated = job["allocated"][role]
        retained += max(0, n * WARP_BYTES - allocated["benchmark"])
        peak = max(peak, retained + max(0, n * RECT_BYTES - allocated["rectified"]))
    return peak


def allocated_tensor_bytes(folder, sequences):
    return sum(p.stat().st_blocks * 512 for seq in sequences for p in (folder / seq).glob("*.pt")
               if p.stem.isdecimal())


def head_selection(head):
    import torch
    best = torch.load(head / "best.pt", map_location="cpu", weights_only=False, mmap=True)
    rows = [json.loads(s) for s in (head / "metrics.jsonl").read_text().splitlines()]
    if [r["epoch"] for r in rows] != list(range(50)):
        raise ValueError(f"Expected the complete 50-epoch validation history: {head}")
    scores = [r["val/mAP"] for r in rows]
    epoch = max(range(50), key=scores.__getitem__)
    if best["epoch"] != epoch or best["best_mAP"] != scores[epoch]:
        raise ValueError(f"best.pt differs from the first highest validation mAP: {head}")
    return dict(epoch_zero_based=epoch, val_mAP=scores[epoch])


def receipt(output, condition, phase):
    return output / condition / (phase + ".verified.json")


def verified_report(job, plan):
    from tools.placement_detection_protocol import check_report
    base = Path(plan["output"]) / job["name"]
    phase = plan["phase"]
    return check_report(base / f"{phase}.json", phase, job["name"], job["feature"], job["label"],
                        base / "head/best.pt", plan["activity_hash"], plan["split"][phase])


def verify_saved(job, plan, phase):
    base = Path(plan["output"]) / job["name"]
    path = receipt(Path(plan["output"]), job["name"], phase)
    if not path.is_file():
        raise FileNotFoundError(f"Finish the {phase} phase for {job['name']} first: {path}")
    saved = read(path)
    if (saved["backbone_sha256"] != job["backbone_sha256"] or saved["phase"] != phase
            or saved["head_sha256"] != sha256_file(base / "head/best.pt")
            or saved["report_sha256"] != sha256_file(base / f"{phase}.json")):
        raise ValueError(f"Existing {phase} result or weights differ for {job['name']}")
    return saved


def worker(a):
    plan = read(a.worker_plan)
    root = Path(plan["output"])
    held, expected = os.fstat(a.lock_fd), (root / ".lock").stat()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if ((held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino)
            or gpu not in plan["gpus"] or idle_gpus((gpu,)) != (gpu,)):
        raise ValueError("Worker lost its output lock or assigned idle GPU")
    job = next(j for j in plan["jobs"] if j["name"] == a.worker_key)
    condition, phase = job["name"], plan["phase"]
    feature = job["feature"]
    base = root / condition
    cache = Path(job["cache"])
    head = base / "head"
    if sha256_file(job["backbone"]) != job["backbone_sha256"]:
        raise ValueError(f"Stage 1 backbone changed after planning: {condition}")
    def run(name, *args):
        tool(name, *args, lock_fd=a.lock_fd)
    if phase == "test":
        validation = verify_saved(job, plan, "val")["validation_selection"]
    pending = list(job["pending_roles"])
    for role in roles(phase):
        sequences = plan["split"][role]
        rectified = cache / "rectified" / role
        benchmark = cache / "benchmark"
        marker = base / f"{role}.features.json"
        if role in pending:
            run("cache_dsec_detection_features.py", "--checkpoint", job["backbone"], "--event-cache-dir",
                plan["events"], "--output-dir", rectified, "--role", role, "--features", feature,
                "--split-manifest", OFFICIAL, "--state-policy", "continuous", "--device", "cuda")
            run("prepare_dsec_detection_benchmark_features.py", "--input-dir", rectified,
                "--output-dir", benchmark, "--dataset-root", plan["raw"], "--role", role,
                "--split-manifest", OFFICIAL, "--features", feature, "--scale", "1", "--device", "cuda")
        check_warp(benchmark, role, sequences, plan["source_info"], job["backbone_sha256"], feature)
        save(marker, dict(cache=str(benchmark), role=role, backbone_sha256=job["backbone_sha256"]))
        # This directory belongs to this suite's random scratch namespace and to this role only.
        if rectified.exists():
            shutil.rmtree(rectified)
        if role in pending:
            pending.remove(role)
        save(a.worker_status, dict(phase=f"{role} benchmark cache verified", remaining_cache_bytes=remaining(job, pending)))
    if phase == "val":
        from tools.placement_detection_protocol import prepare_head_resume
        cmd = ["--feature-cache-dir", benchmark, "--labels-root", plan["labels"], "--dataset-root", plan["raw"],
               "--split-manifest", OFFICIAL, "--feature", feature, "--protocol", "dsec-det", "--output-dir", head,
               "--epochs", "50", "--batch-size", "16", "--seed", "0", "--precision", "fp16", "--device", "cuda"]
        resume = prepare_head_resume(head)
        completed = 0
        if resume is not None:
            cmd.extend(["--resume", resume])
            completed = len((head / "metrics.jsonl").read_text().splitlines())
            if completed < 50:
                history = base / "head_resumes.json"
                prior = read(history) if history.exists() else []
                save(history, [*prior, completed])
        if completed < 50:
            run("train_dsec_detection.py", *cmd)
        validation = head_selection(head)
    run("evaluate_dsec_detection.py", "--checkpoint", head / "best.pt", "--feature-cache-dir", benchmark,
        "--labels-root", plan["labels"], "--dataset-root", plan["raw"], "--split-manifest", OFFICIAL,
        "--feature", feature, "--role", phase, "--protocol", "dsec-det", "--activity-manifest", plan["activity"],
        "--condition-label", job["label"], "--batch-size", "16", "--output", base / f"{phase}.json")
    result = verified_report(job, plan)
    history = base / "head_resumes.json"
    result.update(condition=condition, label=job["label"], feature=feature, phase=phase, seed=0,
                  validation_selection=validation, resumed_at_epoch_zero_based=read(history) if history.exists() else [],
                  backbone=job["backbone"], backbone_sha256=job["backbone_sha256"], cache=str(benchmark),
                  head=str(head / "best.pt"), head_sha256=sha256_file(head / "best.pt"),
                  report=str(base / f"{phase}.json"), report_sha256=sha256_file(base / f"{phase}.json"))
    save(receipt(root, condition, phase), result)
    save(a.worker_result, result)


def summary(a):
    rows = {}
    for key in LABELS:
        path = receipt(a.output, key, a.phase)
        if path.is_file():
            rows[key] = read(path)
    identities = {(row["samples_targets_sha256"], row["activity_assignments_sha256"],
                   row["activity_manifest_sha256"]) for row in rows.values()}
    if len(identities) > 1:
        raise ValueError(f"Detection {a.phase} samples/targets/activity assignments differ across conditions")
    save(a.output / f"summary_{a.phase}.json", dict(phase=a.phase, metric_unit="AP fraction [0,1]", results=rows))
    if len(rows) == len(LABELS):
        (a.output / f"_SUCCESS_{a.phase}").touch()


def plan_jobs(a, split, source_info, namespace, *, execute):
    from tools.placement_detection_protocol import check_backbone
    jobs = []
    for key in a.conditions:
        ckpt, feature = args_for(key, a)
        audited = check_backbone(ckpt, key, split)
        cache = a.scratch / namespace / key
        pending_roles = [r for r in roles(a.phase) if not (a.output / key / f"{r}.features.json").is_file()]
        frames = {r: sum(source_info[s]["frames"] for s in split[r]) for r in roles(a.phase)}
        allocated = {r: dict(benchmark=allocated_tensor_bytes(cache / "benchmark", split[r]),
                             rectified=allocated_tensor_bytes(cache / "rectified" / r, split[r])) for r in pending_roles}
        job = dict(name=key, label=LABELS[key], kind="detection", feature=feature, backbone=str(ckpt),
                   backbone_sha256=audited["sha256"], audit=audited, cache=str(cache), frames=frames,
                   pending_roles=pending_roles, allocated=allocated)
        identity = a.output / key / "backbone.json"
        if identity.is_file() and read(identity) != audited:
            raise ValueError(f"Different backbone recorded for {key}; choose a new output root")
        if execute and not identity.exists():
            save(identity, audited)
        if a.phase == "test":
            verify_saved(job, {"output": str(a.output)}, "val")
        if receipt(a.output, key, a.phase).exists():
            verify_saved(job, {"output": str(a.output)}, a.phase)
            print(f"[{key}] {LABELS[key]} | verified existing {a.phase}", flush=True)
            continue
        job["remaining_cache_bytes"] = remaining(job, pending_roles)
        print(f"[{key}] {LABELS[key]} | {a.phase} | scratch reserve {job['remaining_cache_bytes']/2**30:.1f} GiB", flush=True)
        jobs.append(job)
    return jobs


def prepare(a, split, *, lock_fd=None):
    execute = lock_fd is not None
    identity = {k: str(getattr(a, k)) for k in ("placement_root", "baseline_root", "temporal_root", "events",
                                                "raw", "labels", "scratch")}
    identity.update(split_sha256=sha256_file(OFFICIAL), train_manifest_sha256=sha256_file(TRAIN35))
    identity_path = a.output / "inputs.json"
    if identity_path.exists():
        existing = read(identity_path)
        namespace = existing.pop("namespace")
        if existing != identity or Path(namespace).name != namespace or not namespace.startswith("dsec-det-"):
            raise ValueError(f"Output belongs to different inputs: {a.output}")
    else:
        if a.output.exists() and any(p.name != ".lock" for p in a.output.iterdir()):
            raise ValueError(f"Choose a fresh Detection output root: {a.output}")
        namespace = "dsec-det-" + uuid4().hex
    if a.phase == "test":
        missing = [k for k in a.conditions if not receipt(a.output, k, "val").is_file()]
        if missing:
            raise ValueError(f"Finish val first for {', '.join(missing)} before scoring the fixed heads")
    info = inputs(a, split)
    previous_reference = None
    if a.phase == "test" and (a.output / "inputs_val.json").is_file():
        val_inputs = read(a.output / "inputs_val.json")
        previous_reference = val_inputs["activity_reference_sha256"]
        previous = val_inputs["source_info"]
        if any(info[s][k] != previous[s][k] for s in load_common("dsec")["splits"]["train"]
               for k in ("metadata", "timestamps")):
            raise ValueError("Activity reference train35 inputs changed since validation")
    manifest = a.activity_manifest or (a.output / f"activity_{a.phase}.json")
    if manifest.is_file():
        activity_hash, reference_hash = check_activity(manifest, split[a.phase], info)
    elif a.activity_manifest:
        raise FileNotFoundError(manifest)
    else:
        activity_hash, reference_hash = None, None
    source_path = a.output / f"inputs_{a.phase}.json"
    sources = dict(source_info=info, activity=str(manifest))
    if previous_reference and reference_hash and previous_reference != reference_hash:
        raise ValueError("Detection test activity thresholds differ from validation")
    if source_path.is_file() and read(source_path) != dict(sources, activity_manifest_sha256=activity_hash,
                                                           activity_reference_sha256=reference_hash):
        raise ValueError(f"Source labels, calibration, prepared events or activity path changed: {source_path}")
    if execute and not identity_path.exists():
        save(identity_path, dict(identity, namespace=namespace))
    jobs = plan_jobs(a, split, info, namespace, execute=execute)
    additional = sum(max(0, job["frames"][r] * WARP_BYTES - job["allocated"][r]["benchmark"])
                     for job in jobs for r in job["pending_roles"])
    print(f"Additional retained benchmark features for {a.phase}: {additional/2**30:.1f} GiB", flush=True)
    if not execute:
        print(f"Preflight passed. {len(jobs)} pending; activity {manifest} " +
              ("verified." if activity_hash else "will be fitted once to train35 at execution."))
        return
    if not jobs:
        summary(a)
        return
    available = idle_gpus(a.gpus)
    if not available:
        raise RuntimeError("No requested GPU is idle")
    if not activity_hash:
        temporary = manifest.with_name(manifest.name + "." + uuid4().hex + ".tmp")
        tool("prepare_dsec_activity_quantiles.py", "--event-cache-dir", a.events, "--train-manifest", TRAIN35,
             "--sequences", *split[a.phase], "--output", temporary, lock_fd=lock_fd, cpu=True)
        activity_hash, reference_hash = check_activity(temporary, split[a.phase], info)
        if previous_reference and previous_reference != reference_hash:
            raise ValueError("Detection test activity thresholds differ from validation")
        temporary.replace(manifest)
    save(source_path, dict(sources, activity_manifest_sha256=activity_hash, activity_reference_sha256=reference_hash))
    run = a.output / "logs" / (a.phase + "-" + uuid4().hex)
    path = run / "plan.json"
    plan = dict(output=str(a.output), phase=a.phase, split=split, source_info=info, gpus=available,
                events=str(a.events), raw=str(a.raw), labels=str(a.labels),
                activity=str(manifest), activity_hash=activity_hash, jobs=jobs)
    save(path, plan)
    a.scratch.mkdir(parents=True, exist_ok=True)
    def command(job, status, result):
        return [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-plan", str(path),
                "--worker-key", job["name"], "--worker-status", str(status), "--worker-result", str(result),
                "--lock-fd", str(lock_fd)]
    try:
        run_parallel(jobs, available, scratch=a.scratch, state_dir=run, worker_command=command, lock_fd=lock_fd,
                     on_complete=lambda _k, _r: summary(a))
    finally:
        summary(a)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="1,2")
    p.add_argument("--conditions", nargs="+", choices=LABELS, default=list(LABELS))
    p.add_argument("--phase", choices=("val", "test"), default="val")
    p.add_argument("--data", type=Path, default=Path("/home/iASL/Arata_repo/dataset"))
    for k in ("scratch", "activity-manifest"):
        p.add_argument("--" + k, type=Path)
    for k, name in (("output", "stage2_placement_detection_20261122"),
                    ("placement-root", "stage1_placement_full_20261118"),
                    ("baseline-root", "stage1_joined_baseline_full_20261107"),
                    ("temporal-root", "stage1_temporal_full_20261115")):
        p.add_argument("--" + k, type=Path, default=ROOT / "outputs" / name)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    for k in ("worker-plan", "worker-key", "worker-status", "worker-result", "lock-fd"):
        p.add_argument("--" + k, type=int if k == "lock-fd" else str if k == "worker-key" else Path,
                       help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker_plan:
        worker(a)
        return
    a.gpus = parse_gpus(a.gpus)
    if len(set(a.conditions)) != len(a.conditions):
        p.error("Specify each condition at most once")
    a.events, a.raw = a.data / "DSEC_cache/events/gep_rgb", a.data / "DSEC"
    a.labels = a.raw / "dsec_det_labels"
    a.scratch = a.scratch or a.data / "downstream_scratch"
    for k, v in vars(a).copy().items():
        if isinstance(v, Path):
            setattr(a, k, v.expanduser().resolve())
    for key in a.conditions:
        ckpt, feature = args_for(key, a)
        print(f"[{key}] {LABELS[key]} | {a.phase} | {feature} | GPU pool {','.join(a.gpus)} | {ckpt}")
    if not a.preflight and not a.execute:
        return
    from pycocotools.cocoeval import COCOeval  # Require the actual COCO evaluator before training.
    from event_state.detection.split import load_dsec_detection_split
    split = {r: list(v) for r, v in vars(load_dsec_detection_split(OFFICIAL)).items()}
    os.environ.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    if a.preflight:
        prepare(a, split)
        return
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(a, split, lock_fd=lock.fileno())
    print(f"Completed {a.phase}: {a.output / ('summary_' + a.phase + '.json')}")


if __name__ == "__main__":
    main()
