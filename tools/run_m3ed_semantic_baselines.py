"""Run matched M3ED semantic probes on up to three GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-target-root', type=Path)
    parser.add_argument('--include-ddp', action='store_true',
                        help='Also probe the old and joined E1 DDP checkpoints')
    args = parser.parse_args()
    data = Path('/home/iASL/Arata_repo/dataset')
    stage1 = Path('outputs/stage1_joined_baseline_full_20261107')
    out = Path('outputs/stage2_m3ed_semantic_linear_20261109')
    cache = data / 'downstream_scratch' / out.name
    jobs = []
    specs = []
    for condition, label, feature in [
        ('E0', 'Encoder（z蒸留）', 'z'),
        ('E1', 'Encoder＋LSTM（h蒸留）', 'h'),
        ('E2', 'Encoder＋LSTM（z+h蒸留）', 'h'),
    ]:
        checkpoint = stage1 / f'm3ed_{condition}_w1/checkpoints/step_00099999.pt'
        specs.append((condition, f'{condition} {label}・単GPU', feature, checkpoint))
    if args.include_ddp:
        for key, variant, root in [
            ('E1_old_ddp', '旧型', 'stage1_rvt_ddp_full_20261105'),
            ('E1_new_ddp', '新型RVT', 'stage1_joined_ddp_full_new_20261106'),
        ]:
            checkpoint = Path('outputs') / root / 'm3ed_E1_w3/checkpoints/step_00033333.pt'
            specs.append((key, f'E1 Encoder＋LSTM（h蒸留）・{variant}DDP', 'h', checkpoint))
    for condition, label, feature, checkpoint in specs:
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
        report = out / condition / feature / 'continuous_groups.json'
        if (out / condition / '_SUCCESS').is_file() and report.is_file():
            metrics = json.loads(report.read_text())['all']
            print(f"{label} | 完了済み | val mIoU={metrics['mIoU']:.2%}")
            continue
        jobs.append((label, cmd, report))
    running = {}
    failures = []
    try:
        while jobs or running:
            for gpu in range(3):
                if gpu in running:
                    label, proc, report, log = running[gpu]
                    if proc.poll() is None:
                        continue
                    del running[gpu]
                    if proc.returncode:
                        failures.append(str(log))
                    else:
                        metrics = json.loads(report.read_text())['all']
                        print(f"{label} | val mIoU={metrics['mIoU']:.2%} | {metrics['valid_pixels']:,} pixels", flush=True)
                if jobs and not failures:
                    label, cmd, report = jobs.pop(0)
                    log = out / 'logs' / f'gpu{gpu}.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    print(f'GPU {gpu} | {label} | {log}', flush=True)
                    with log.open('a') as handle:
                        proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                                stdout=handle, stderr=subprocess.STDOUT)
                    running[gpu] = (label, proc, report, log)
            if failures and not running:
                break
            if running:
                time.sleep(1)
        if failures:
            raise SystemExit('Failed; see ' + ', '.join(failures))
    finally:
        for _, proc, _, _ in running.values():
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _, _ in running.values():
            proc.wait()


if __name__ == '__main__':
    main()
