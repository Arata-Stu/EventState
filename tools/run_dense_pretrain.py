#!/usr/bin/env python3
"""Plan/run clean GEP pretraining for a published downstream depth protocol."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_state.dense.protocol import audit_pretraining, digest, load_protocol


def command(a):
    protocol = load_protocol(a.protocol)
    stats = json.loads(a.statistics.read_text())
    if stats.get("protocol_sha256") != digest(protocol):
        raise ValueError("Statistics were not computed for this published protocol")
    if stats.get("event_window_us") != a.window_us:
        raise ValueError("Statistics and pretraining accumulation windows differ")
    ds = protocol["dataset"]
    audit_pretraining(dict(dataset=dict(name=ds, train_sequences=protocol["splits"]["train"], val_sequences=[]),
                           training=dict(validation_enabled=False)), protocol,
                       statistics_sequences=stats.get("sequences"))
    if a.output_dir.exists():
        raise FileExistsError("Choose a fresh pretraining run; do not continue a leaked checkpoint")
    if a.window_us is not None and (ds != "dsec" or a.window_us <= 0):
        raise ValueError("Fixed raw event window is supported only for DSEC")
    cmd = [sys.executable, "train.py", f"dataset={'dsec_det_train41' if ds == 'dsec' else 'm3ed_half_dagr'}",
           "model=lstm", f"experiment={'h_distill_lstm_zloss' if a.condition == 'baseline' else 'activity_z_h_all'}",
           f"seed={a.seed}", f"dataset.root={a.data_root.resolve()}",
           f"dataset.train_sequences={json.dumps(protocol['splits']['train'], separators=(',', ':'))}",
           "dataset.val_sequences=null", "dataset.val_split=null", "training.validation_enabled=false",
           f"++dataset.dense_evaluation_protocol={a.protocol}",
           f"++dataset.normalization_provenance={_hydra_object(stats)}",
           f"dataset.representation.normalize_mean={json.dumps(stats['normalize_mean'])}",
           f"dataset.representation.normalize_std={json.dumps(stats['normalize_std'])}",
           f"dataset.event_cache_dir={a.event_cache_dir.resolve()}",
           f"teacher.cache_dir={a.teacher_cache_dir.resolve()}", "teacher.cache_features=true",
           f"teacher.checkpoint={a.teacher_checkpoint.resolve()}",
           f"++model.event_encoder.checkpoint={a.teacher_checkpoint.resolve()}",
           "dataset.augmentation.enabled=false", "training.event_dropout.enabled=false",
           f"++dataset.activity_mask={str(a.condition == 'alpha1').lower()}",
           "training.sampling.mode=mixed", f"training.batch_size={8 if ds == 'dsec' else 4}",
           f"training.sampling.random_batch_size={4 if ds == 'dsec' else 2}",
           f"training.sampling.stream_batch_size={4 if ds == 'dsec' else 2}",
           f"dataset.sequence_length={8 if ds == 'dsec' else 16}", "training.resume=null",
           f"training.max_steps={100 if a.stage == 'smoke' else 100000}",
           f"scheduler.warmup_steps={10 if a.stage == 'smoke' else 1000}",
           f"training.checkpoint_every={100 if a.stage == 'smoke' else 10000}",
           f"hydra.run.dir={a.output_dir.resolve()}"]
    if ds == "m3ed":
        cmd += [f"dataset.prepared_root={a.event_cache_dir.resolve()}"]
    if a.window_us is not None:
        cmd = [item for item in cmd if not item.startswith("dataset.event_cache_dir=")]
        cmd += ["dataset.event_cache_dir=null", f"++dataset.event_window_us={a.window_us}"]
    return cmd


def _hydra_object(stats):
    # Only the small reproducibility contract belongs in resolved checkpoint config.
    values = {k: stats[k] for k in ("sequences", "normalize_mean", "normalize_std", "protocol_sha256")}
    return "{" + ",".join(k + ":" + json.dumps(v, separators=(",", ":")) for k, v in values.items()) + "}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="DSEC raw root or M3ED prepared root")
    p.add_argument("--event-cache-dir", type=Path, required=True)
    p.add_argument("--teacher-cache-dir", type=Path, required=True)
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--condition", choices=("baseline", "alpha1"), default="baseline")
    p.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    p.add_argument("--window-us", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute", action="store_true", help="Actually launch; default only prints the command")
    a = p.parse_args()
    cmd = command(a)
    print(shlex.join(cmd), flush=True)
    if a.execute:
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
