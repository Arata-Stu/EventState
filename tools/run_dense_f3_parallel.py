#!/usr/bin/env python3
"""Resume F3 depth jobs across independent GPUs; no DDP or batch-size change."""
import argparse
import hashlib
import json
import fcntl
import os
from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_root', type=Path)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--cache-root', type=Path, help='Omit to locate existing cache by saved head metadata hash')
    a = p.parse_args()
    root = a.run_root.resolve()
    if not (root / 'pretrain').is_dir():
        raise ValueError('Expected existing F3 run root')
    if a.cache_root is None:
        config = json.loads((root / 'heads/dsec_alpha1/h/config.json').read_text())
        candidates = [root / 'features/dsec_alpha1/train_continuous/metadata.json']
        scratch = Path(os.environ.get('DATA_ROOT', '/home/iASL/Arata_repo/dataset')) / 'downstream_scratch'
        candidates.extend(scratch.glob('*/dsec_alpha1/train_continuous/metadata.json'))
        matches = {p.resolve().parents[2] for p in candidates if p.is_file() and
                   hashlib.sha256(p.read_bytes()).hexdigest() == config['train_cache_metadata_sha256']}
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely locate original cache: {matches}; specify --cache-root')
        a.cache_root = matches.pop()
    print(f'[parallel] cache root: {a.cache_root.resolve()}', flush=True)
    gpus = a.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError('Use distinct numeric GPU IDs')
    # Refuse old sequential runner or independent head/cache workers on this run.
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    for line in processes.splitlines():
        if any(s in line for s in ('run_dense_f3_pipeline.sh', 'train_dense_probe.py', 'cache_dense_features.py', 'evaluate_dense_probe.py')) and (str(root) in line or root.name in line):
            raise RuntimeError(f'Stop existing pipeline and children first: {line.strip()}')
    lock = (root / '.parallel_depth.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Prioritize unfinished DSEC plus both M3ED conditions; completed baseline audited last.
    pending = [('dsec', 'alpha1'), ('m3ed', 'baseline'), ('m3ed', 'alpha1'), ('dsec', 'baseline')]
    running = {}
    env = dict(os.environ, CACHE_ROOT=str(a.cache_root.resolve()), RESUME='1', SKIP_MODEL_COMPARISON='1')
    try:
        while pending or running:
            for gpu in gpus:
                if pending and gpu not in running:
                    ds, condition = pending.pop(0)
                    print(f'[parallel] GPU {gpu}: {ds}_{condition}', flush=True)
                    proc = subprocess.Popen(['bash', 'tools/run_dense_f3_pipeline.sh', 'depth', str(root)],
                        cwd=PROJECT, env=dict(env, DATASETS=ds, CONDITIONS=condition, CUDA_VISIBLE_DEVICES=gpu),
                        start_new_session=True)
                    running[gpu] = proc
            for gpu, proc in list(running.items()):
                code = proc.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'GPU {gpu} worker failed: {code}; inspect per-step logs')
                    del running[gpu]
            time.sleep(1)
        # All writers finished. Validate/reuse all artifacts, then compute model comparisons serially.
        subprocess.run(['bash', 'tools/run_dense_f3_pipeline.sh', 'depth', str(root)], cwd=PROJECT,
            env=dict(env, DATASETS='dsec m3ed', CONDITIONS='baseline alpha1',
                     SKIP_MODEL_COMPARISON='0', CUDA_VISIBLE_DEVICES=gpus[0]), check=True)
    finally:
        import signal
        for proc in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in running.values():
            proc.wait()
        lock.close()
    print(f'[parallel] complete: {root}', flush=True)


if __name__ == '__main__':
    main()
