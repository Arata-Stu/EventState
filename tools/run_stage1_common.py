#!/usr/bin/env python3
"""Fresh common-backbone E0/E1/E2 runs and an E1 single-GPU vs 3-GPU comparison.

Standard-library planner. No training without --execute. Existing output is never
overwritten. Baselines occupy one GPU each; DDP jobs use all three exclusively.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from event_state.dense.common_protocol import load_common
from event_state.dense.protocol import digest
from tools.run_dense_pretrain import _hydra_object


def plan(a):
    gpus = a.gpus.split(',')
    if len(gpus) != 3 or len(set(gpus)) != 3 or not all(x.isdigit() for x in gpus):
        raise ValueError('Specify three distinct GPU IDs')
    if a.seed < 0 or a.num_workers < 0:
        raise ValueError('seed/workers must be nonnegative')
    if a.output_root.exists():
        raise FileExistsError('Use a fresh output root; do not overwrite prior runs')
    if not a.teacher_checkpoint.is_file():
        raise FileNotFoundError(a.teacher_checkpoint)
    jobs = []
    for ds in ('dsec', 'm3ed'):
        protocol = load_common(ds)
        stats_file = a.statistics_dir / f'{ds}_common.json'
        stats = json.loads(stats_file.read_text())
        if stats.get('protocol_sha256') != digest(protocol) or stats.get('sequences') != protocol['splits']['train']:
            raise ValueError(f'{ds}: recompute normalization for the common train pool')
        if stats.get('event_window_us') is not None:
            raise ValueError('This baseline uses RGB-interval GEP, not a fixed event window')
        mean, std = stats['normalize_mean'], stats['normalize_std']
        if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in mean + std) or min(std) <= 0:
            raise ValueError('Invalid normalization coefficients')
        events = a.data_root / ('DSEC_cache/events/gep_rgb' if ds == 'dsec' else 'm3ed_cache/half_dagr')
        teacher_cache = a.data_root / ('DSEC_cache/dinov3_vits16' if ds == 'dsec' else 'm3ed_cache/dinov3_vits16_640x352')
        root = a.data_root / 'DSEC' if ds == 'dsec' else events
        for directory in (root, events, teacher_cache):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        # Keep stream microbatch=1: M3ED has three independent train sequences.
        accumulation = 4 if ds == 'dsec' else 2
        clip = 8 if ds == 'dsec' else 16
        specs = [(e, 1) for e in ('E0', 'E1', 'E2')] + [('E1', 3)]
        for condition, world in specs:
            if a.phase == 'baseline' and world != 1 or a.phase == 'ddp' and world != 3:
                continue
            base_steps = 99 if a.stage == 'smoke' else 99999
            steps = base_steps // world
            warmup = (9 if a.stage == 'smoke' else 999) // world
            name = f'{ds}_{condition}_w{world}'
            experiment = {'E0': 'gep_baseline', 'E1': 'h_distill_lstm', 'E2': 'h_distill_lstm_zloss'}[condition]
            cmd = [sys.executable]
            if world > 1:
                cmd += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={world}']
            cmd += ['train.py', f'dataset={"dsec_det_train41" if ds == "dsec" else "m3ed_half_dagr"}',
                    f'model={"no_memory" if condition == "E0" else "lstm"}', f'experiment={experiment}',
                    f'seed={a.seed}', 'device=auto', f'dataset.root={root.resolve()}',
                    '++dataset.common_stage1=true',
                    'dataset.train_sequences=' + json.dumps(protocol['splits']['train'], separators=(',', ':')),
                    'dataset.val_sequences=null', 'dataset.val_split=null',
                    '++dataset.normalization_provenance=' + _hydra_object(stats),
                    'dataset.representation.normalize_mean=' + json.dumps(mean, separators=(',', ':')),
                    'dataset.representation.normalize_std=' + json.dumps(std, separators=(',', ':')),
                    f'dataset.event_cache_dir={events.resolve()}',
                    f'teacher.cache_dir={teacher_cache.resolve()}', 'teacher.cache_features=true',
                    f'teacher.checkpoint={a.teacher_checkpoint.resolve()}',
                    f'++model.event_encoder.checkpoint={a.teacher_checkpoint.resolve()}',
                    '++dataset.activity_mask=false', 'dataset.augmentation.enabled=false',
                    'training.event_dropout.enabled=false', 'training.validation_enabled=false',
                    'training.sampling.mode=mixed', 'training.sampling.random_batch_size=1',
                    'training.sampling.stream_batch_size=1', 'training.sampling.random_weight=1.0',
                    'training.sampling.stream_weight=1.0', 'training.sampling.shuffle_sequences=true',
                    'training.batch_size=2', f'training.gradient_accumulation={accumulation}',
                    f'++training.distributed_world_size={world}', f'dataset.sequence_length={clip}',
                    f'training.max_steps={steps}', f'scheduler.warmup_steps={warmup}',
                    'optimizer.lr=0.0001', 'training.precision=fp16',
                    f'training.num_workers={a.num_workers}', 'training.resume=null',
                    f'training.checkpoint_every={steps if a.stage == "smoke" else 9999 // world}',
                    f'training.log_every={1 if a.stage == "smoke" else 20}',
                    f'hydra.run.dir={(a.output_root / name).resolve()}',
                    'hydra.output_subdir=null', 'hydra/job_logging=disabled', 'hydra/hydra_logging=disabled']
            if ds == 'm3ed':
                cmd += [f'dataset.prepared_root={events.resolve()}']
            jobs.append(dict(name=name, world=world, command=cmd, steps=steps,
                effective_clips=2 * accumulation * world,
                planned_frames=steps * 2 * accumulation * world * clip,
                statistics=str(stats_file.resolve()), statistics_sha256=hashlib.sha256(stats_file.read_bytes()).hexdigest(),
                protocol=protocol))
    return sorted(jobs, key=lambda j: j['world'])


def execute(a, jobs):
    a.output_root.mkdir(parents=True)
    logs = a.output_root / 'logs'
    logs.mkdir()
    (a.output_root / 'plan.json').write_text(json.dumps(jobs, indent=2) + '\n')
    (a.output_root / 'source.patch').write_text(subprocess.check_output(['git','diff','HEAD'], cwd=PROJECT, text=True))
    # Include new, untracked implementation files too, not just git diff.
    snapshot = a.output_root / 'source'
    import shutil
    for folder in ('event_state', 'tools', 'configs', 'tests'):
        shutil.copytree(PROJECT / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT / 'train.py', snapshot / 'train.py')
    (a.output_root / 'revision.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT, text=True))
    for ds in ('dsec','m3ed'):
        shutil.copy2(a.statistics_dir / f'{ds}_common.json', a.output_root / f'{ds}_common.json')
    # Synthetic distributed correctness checks must pass before expensive data loading.
    if not a.skip_preflight:
        with (logs / 'preflight.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q',
                            'tests/test_ddp_training.py', 'tests/test_stage1_common_stdlib.py',
                            'tests/test_stream_sampling.py', 'tests/test_training_runtime.py',
                            'tests/test_stage1_common_configs.py'],
                           cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    pending, running = list(jobs), []
    gpu_ids = a.gpus.split(',')
    def stop(signum, frame):
        raise KeyboardInterrupt
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or running:
            used = {g for _, _, _, assigned in running for g in assigned}
            for job in pending[:]:
                free = [g for g in gpu_ids if g not in used]
                if len(free) < job['world']:
                    break
                assigned = free[:job['world']]
                log = (logs / f'{job["name"]}.log').open('w')
                proc = subprocess.Popen(job['command'], cwd=PROJECT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(assigned), OMP_NUM_THREADS='1'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running.append((proc, log, job, assigned))
                pending.remove(job)
                used.update(assigned)
                print(f'[stage1] GPUs {assigned}: {job["name"]}', flush=True)
            for item in running[:]:
                proc, log, job, _ = item
                if proc.poll() is None:
                    continue
                log.close()
                running.remove(item)
                ckpt = a.output_root / job['name'] / 'checkpoints' / f'step_{job["steps"]:08d}.pt'
                if proc.returncode or not ckpt.is_file() or not ckpt.stat().st_size:
                    raise RuntimeError(f'{job["name"]} failed or missing final checkpoint; inspect log')
                print(f'[stage1] complete: {job["name"]}', flush=True)
            time.sleep(.5)
    finally:
        for proc, _, _, _ in running:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, log, _, _ in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        for s, handler in old.items():
            signal.signal(s, handler)
    (a.output_root / '_SUCCESS').write_text('All jobs exited successfully and final checkpoints exist.\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke','full'), default='smoke')
    p.add_argument('--phase', choices=('all','baseline','ddp'), default='all')
    p.add_argument('--data-root', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--teacher-checkpoint', type=Path, default=Path('/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    p.add_argument('--statistics-dir', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--skip-preflight', action='store_true', help='Only after same-code server preflight passes')
    a = p.parse_args()
    jobs = plan(a)
    for job in jobs:
        print(f'{job["name"]}: clips/update={job["effective_clips"]}, steps={job["steps"]}, frames={job["planned_frames"]}')
        print(shlex.join(job['command']))
    if a.execute:
        execute(a, jobs)


if __name__ == '__main__':
    main()
