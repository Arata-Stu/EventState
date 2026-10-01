#!/usr/bin/env python3
"""Four matched Hybrid pretraining runs. Standard library only; use on GPU host."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
TRAIN4 = {"car_urban_day_" + x for x in
          ("city_hall", "horse", "penno_big_loop", "penno_small_loop")}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--data-root', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--teacher-checkpoint', type=Path, default=Path(
        '/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    p.add_argument('--event-statistics', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-tests', action='store_true', help='Only after same-revision GPU-host preflight passes')
    return p.parse_args()


def build_plan(args):
    gpus = args.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3 or not all(g.isdigit() for g in gpus):
        raise ValueError('--gpus needs three distinct numeric IDs')
    if min(args.seed, args.num_workers) < 0:
        raise ValueError('seed and num-workers must be nonnegative')
    data = args.data_root.resolve()
    teacher = args.teacher_checkpoint.resolve()
    stats_path = (args.event_statistics or data / 'm3ed_cache/half_dagr/event_statistics_train4.json').resolve()
    for path in (data / 'DSEC', data / 'DSEC_cache/events/gep_rgb',
                 data / 'DSEC_cache/dinov3_vits16', data / 'm3ed_cache/half_dagr',
                 data / 'm3ed_cache/dinov3_vits16_640x352'):
        if not path.is_dir():
            raise ValueError(f'Missing data directory: {path}')
    if not teacher.is_file():
        raise ValueError(f'Missing teacher: {teacher}')
    stats = json.loads(stats_path.read_text())
    if set(stats.get('sequences', [])) != TRAIN4 or len(stats['sequences']) != 4:
        raise ValueError('Normalization must come from train4 only')
    mean, std = stats['normalize_mean'], stats['normalize_std']
    if (stats.get('representation') != 'gep_rgb' or len(mean) != 3 or len(std) != 3
            or not all(math.isfinite(x) for x in mean + std) or min(std) <= 0):
        raise ValueError('Invalid GEP normalization')
    output = (args.output_root or PROJECT / 'outputs' /
              f'hybrid_targets_{args.stage}_{datetime.datetime.now():%Y%m%d_%H%M%S}').resolve()
    if output.exists():
        raise ValueError(f'Output exists; choose a fresh directory: {output}')
    full = args.stage == 'full'
    jobs = []
    specs = [('dsec_z_only', 'dsec', 'z_distill_lstm', gpus[0]),
             ('dsec_z_h', 'dsec', 'h_distill_lstm_zloss', gpus[1]),
             ('m3ed_z_only', 'm3ed', 'z_distill_lstm', gpus[2]),
             ('m3ed_h_only', 'm3ed', 'h_distill_lstm', gpus[2])]
    for name, dataset, experiment, gpu in specs:
        is_dsec = dataset == 'dsec'
        branch_batch = 4 if is_dsec else 2
        cmd = [sys.executable, 'train.py',
               'dataset=' + ('dsec_det_train41' if is_dsec else 'm3ed_half_dagr'),
               'model=lstm', f'experiment={experiment}', f'seed={args.seed}',
               f'teacher.checkpoint={teacher}', 'teacher.cache_features=true',
               'dataset.activity_mask=false', 'dataset.augmentation.enabled=false',
               'training.event_dropout.enabled=false', 'training.sampling.mode=mixed',
               f'training.sampling.random_batch_size={branch_batch}',
               f'training.sampling.stream_batch_size={branch_batch}',
               'training.sampling.random_weight=1.0', 'training.sampling.stream_weight=1.0',
               'training.sampling.shuffle_sequences=true',
               f'training.batch_size={2 * branch_batch}',
               f'dataset.sequence_length={8 if is_dsec else 16}',
               'training.gradient_accumulation=1', 'training.precision=fp16',
               f'training.num_workers={args.num_workers}',
               f'training.max_steps={100000 if full else 100}',
               f'scheduler.warmup_steps={1000 if full else 10}',
               f'training.checkpoint_every={10000 if full else 100}',
               f'training.log_every={20 if full else 1}', 'training.validate_every=1000',
               'training.validation_enabled=' + str(not is_dsec).lower(),
               'training.resume=null', f'hydra.run.dir={output / name}']
        if is_dsec:
            cmd += [f'dataset.root={data / "DSEC"}',
                    f'dataset.event_cache_dir={data / "DSEC_cache/events/gep_rgb"}',
                    f'teacher.cache_dir={data / "DSEC_cache/dinov3_vits16"}']
        else:
            cmd += [f'dataset.root={data / "m3ed_cache/half_dagr"}',
                    f'dataset.prepared_root={data / "m3ed_cache/half_dagr"}',
                    f'teacher.cache_dir={data / "m3ed_cache/dinov3_vits16_640x352"}',
                    '+dataset.pretraining_protocol=m3ed_activity',
                    'dataset.representation.normalize_mean=' + json.dumps(mean, separators=(',', ':')),
                    'dataset.representation.normalize_std=' + json.dumps(std, separators=(',', ':'))]
        jobs.append(dict(name=name, gpu=gpu, command=cmd))
    return output, stats_path, jobs


def execute(args, output, stats_path, jobs):
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 20 * 2**30:
        raise ValueError('Less than 20 GiB free for pretraining outputs; free space first')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    shutil.copyfile(stats_path, output / 'event_statistics.json')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    (output / 'launch.json').write_text(json.dumps(
        dict(stage=args.stage, revision=revision, jobs=jobs), indent=2) + '\n')
    (output / 'source.patch').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=PROJECT, text=True))
    active = {}
    def stop(signum, frame):
        raise KeyboardInterrupt
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not args.skip_tests:
            test_cmd = [sys.executable, '-m', 'pytest', '-q',
                        'tests/test_hybrid_target_configs.py', 'tests/test_hybrid_target_launcher_stdlib.py',
                        'tests/test_stream_sampling.py',
                        'tests/test_training_runtime.py', 'tests/test_m3ed.py',
                        'tests/test_m3ed_activity.py', 'tests/test_activity_h_relaxation.py',
                        'tests/test_split_guard_stdlib.py']
            print(f'[targets] preflight: {output / "logs/preflight.log"}', flush=True)
            with (output / 'logs/preflight.log').open('w') as log:
                proc = subprocess.Popen(test_cmd, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active['preflight'] = (proc, log, 'preflight')
                code = proc.wait()
                del active['preflight']
                if code:
                    raise RuntimeError('Preflight failed; see logs/preflight.log. No training started.')
        pending = list(jobs)
        while pending or active:
            for job in pending[:]:
                if job['gpu'] in active:
                    continue
                log = (output / 'logs' / (job['name'] + '.log')).open('w')
                proc = subprocess.Popen(job['command'], cwd=PROJECT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu']),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job['gpu']] = (proc, log, job['name'])
                pending.remove(job)
                print(f"[targets] GPU {job['gpu']}: {job['name']}", flush=True)
            for gpu, (proc, log, name) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')
                print(f'[targets] complete: {name}', flush=True)
            time.sleep(0.5)
    finally:
        for proc, log, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, log, _ in active.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        for s, handler in old.items():
            signal.signal(s, handler)


def main():
    args = arguments()
    output, stats_path, jobs = build_plan(args)
    print(f'[targets] outputs: {output}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"# GPU {job['gpu']}: {job['name']} (same GPU jobs run sequentially)")
            print(f"CUDA_VISIBLE_DEVICES={job['gpu']} {shlex.join(job['command'])}")
        return
    execute(args, output, stats_path, jobs)
    print(f'[targets] all complete: {output}', flush=True)


if __name__ == '__main__':
    main()
