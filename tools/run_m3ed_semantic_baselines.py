"""Run the three single-GPU Stage 1 backbones with matched semantic probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-target-root', type=Path)
    args = parser.parse_args()
    data = Path('/home/iASL/Arata_repo/dataset')
    stage1 = Path('outputs/stage1_joined_baseline_full_20261107')
    out = Path('outputs/stage2_m3ed_semantic_linear_20261109')
    cache = data / 'downstream_scratch' / out.name
    jobs = []
    for gpu, (condition, label, feature) in enumerate([
        ('E0', 'Encoder（z蒸留）', 'z'),
        ('E1', 'Encoder＋LSTM（h蒸留）', 'h'),
        ('E2', 'Encoder＋LSTM（z+h蒸留）', 'h'),
    ]):
        checkpoint = stage1 / f'm3ed_{condition}_w1/checkpoints/step_00099999.pt'
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise SystemExit(f'Missing checkpoint: {checkpoint}')
        cmd = [sys.executable, '-u', 'tools/run_common_m3ed_semantic.py',
               '--checkpoint', str(checkpoint), '--features', feature,
               '--statistics', 'outputs/stage1_common_20261105/statistics/m3ed_common.json',
               '--prepared-root', str(data / 'm3ed_cache/half_dagr'),
               '--target-root', str(data / 'm3ed_cache/m3ed_downstream'),
               '--cache', str(cache / condition), '--output', str(out / condition)]
        if args.extra_target_root:
            cmd.extend(['--extra-target-root', str(args.extra_target_root)])
        jobs.append((gpu, f'{condition} {label}', cmd, out / condition / feature / 'continuous_groups.json'))
    running = []
    try:
        for gpu, label, cmd, report in jobs:
            log = out / 'logs' / f'gpu{gpu}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'GPU {gpu} | {label} | {log}', flush=True)
            with log.open('a') as handle:
                proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                        stdout=handle, stderr=subprocess.STDOUT)
            running.append((label, proc, report, log))
        failures = []
        for label, proc, report, log in running:
            if proc.wait():
                failures.append(str(log))
            else:
                metrics = json.loads(report.read_text())['all']
                print(f"{label} | val mIoU={metrics['mIoU']:.2%} | {metrics['valid_pixels']:,} pixels")
        if failures:
            raise SystemExit('Failed; see ' + ', '.join(failures))
    finally:
        for _, proc, _, _ in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _, _ in running:
            proc.wait()


if __name__ == '__main__':
    main()
