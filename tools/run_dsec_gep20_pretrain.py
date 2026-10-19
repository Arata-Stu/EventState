#!/usr/bin/env python3
"""Two matched Hybrid GEP runs with exact 20ms causal input and cached RGB teacher."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def plan(a, output):
    jobs = []
    for name, experiment, gpu in zip(('baseline', 'alpha1'),
            ('h_distill_lstm_zloss', 'activity_z_h_all'), a.gpus.split(',')):
        cmd = [sys.executable, 'train.py', 'dataset=dsec_det_train41', 'model=lstm',
            f'experiment={experiment}', f'seed={a.seed}', f'dataset.root={a.data_root / "DSEC"}',
            'dataset.event_cache_dir=null', '++dataset.event_window_us=20000',
            'dataset.event_window_fraction=1.0', 'dataset.augmentation.enabled=false',
            '++dataset.activity_mask=' + str(name == 'alpha1').lower(),
            f'teacher.cache_dir={a.data_root / "DSEC_cache/dinov3_vits16"}',
            f'teacher.checkpoint={a.teacher_checkpoint}', 'teacher.cache_features=true',
            'training.event_dropout.enabled=false', 'training.sampling.mode=mixed',
            'training.sampling.random_batch_size=4', 'training.sampling.stream_batch_size=4',
            'training.sampling.random_weight=1.0', 'training.sampling.stream_weight=1.0',
            'training.sampling.shuffle_sequences=true', 'training.batch_size=8',
            'dataset.sequence_length=8', 'training.gradient_accumulation=1',
            'training.precision=fp16', f'training.num_workers={a.num_workers}',
            'training.validation_enabled=false', 'training.resume=null',
            f'training.max_steps={100000 if a.stage == "full" else 100}',
            f'scheduler.warmup_steps={1000 if a.stage == "full" else 10}',
            f'training.checkpoint_every={10000 if a.stage == "full" else 100}',
            f'training.log_every={20 if a.stage == "full" else 1}',
            f'hydra.run.dir={output / name}']
        jobs.append(dict(name=name, gpu=gpu, command=cmd))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['smoke','full'], default='smoke')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--data-root', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--teacher-checkpoint', type=Path, default=Path('/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    p.add_argument('--output-root', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    gpus = a.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus) or min(a.seed,a.num_workers) < 0:
        p.error('Specify two distinct GPU IDs and nonnegative seed/workers')
    a.data_root = a.data_root.resolve(); a.teacher_checkpoint = a.teacher_checkpoint.resolve()
    output = (a.output_root or PROJECT/'outputs'/f'dsec_gep20_{a.stage}_{datetime.datetime.now():%Y%m%d_%H%M%S}').resolve()
    jobs = plan(a, output)
    if a.dry_run:
        for job in jobs:
            print(f"CUDA_VISIBLE_DEVICES={job['gpu']} {shlex.join(job['command'])}")
        return
    for path in (a.data_root/'DSEC', a.data_root/'DSEC_cache/dinov3_vits16', a.teacher_checkpoint):
        if not path.exists(): raise FileNotFoundError(path)
    if output.exists(): raise ValueError('Choose a new output root; existing runs are preserved')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_gep_fixed_window.py'], cwd=PROJECT, check=True)
    output.mkdir(parents=True); (output/'logs').mkdir()
    (output/'plan.json').write_text(json.dumps(dict(jobs=jobs, window_us=20000,
        normalization='unchanged DSEC train41 coefficients', teacher='cached at original RGB timestamps'),indent=2)+'\n')
    processes, logs = [], []
    try:
        for job in jobs:
            log = (output/'logs'/f"{job['name']}.log").open('w'); logs.append(log)
            proc = subprocess.Popen(job['command'], cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu']),
                                    stdout=log, stderr=subprocess.STDOUT)
            processes.append((job['name'],proc))
            print(f"[gep20] GPU {job['gpu']}: {job['name']}",flush=True)
        pending = list(processes)
        while pending:
            for name, proc in pending[:]:
                code = proc.poll()
                if code is None: continue
                if code: raise RuntimeError(f'{name} failed ({code}); inspect {output}/logs/{name}.log')
                print(f'[gep20] complete: {name}',flush=True); pending.remove((name,proc))
            if pending: time.sleep(1)
    finally:
        for _,proc in processes:
            if proc.poll() is None: proc.terminate()
        for _,proc in processes:
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for log in logs: log.close()
    print(f'[gep20] all complete: {output}',flush=True)

if __name__ == '__main__': main()
