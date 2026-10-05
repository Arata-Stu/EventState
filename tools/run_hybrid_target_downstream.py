#!/usr/bin/env python3
"""Sequential frozen validation probes with owned-cache cleanup and restart markers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
VERSION = 1


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_cleanup(cache, cache_root, identity):
    """Delete only the exact model directory owned by this run, never a parent."""
    if not cache.exists():
        return
    if cache.is_symlink() or cache.parent != cache_root or cache.resolve() != cache:
        raise ValueError(f'Unsafe cleanup path: {cache}')
    marker = cache / '.owner.json'
    if not marker.is_file() or json.loads(marker.read_text()) != identity:
        raise ValueError(f'Cache ownership mismatch; preserving: {cache}')
    if any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError(f'Symlink inside cache; preserving: {cache}')
    shutil.rmtree(cache)


def verify_results(results):
    for head, metric, feature, key, role in results:
        for filename in ('best.pt', 'last.pt'):
            path = head / filename
            if not path.is_file() or not zipfile.is_zipfile(path):
                raise ValueError(f'Missing/invalid head checkpoint: {path}')
            with zipfile.ZipFile(path) as z:
                if not any(n.endswith('/data.pkl') for n in z.namelist()):
                    raise ValueError(f'Missing checkpoint metadata: {path}')
                if z.testzip() is not None:
                    raise ValueError(f'Corrupt checkpoint ZIP: {path}')
        d = json.loads(metric.read_text())
        value = d.get(key)
        if (d.get('feature') != feature or d.get('role') != role or
            not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f'Invalid validation metrics: {metric}')
        if key == 'mIoU' and d.get('evaluated_pixels', 0) <= 0:
            raise ValueError(f'No evaluated pixels: {metric}')
        if key == 'mAP' and d.get('protocol') != 'dsec-det':
            raise ValueError(f'Wrong detection protocol: {metric}')


def model_plan(name, checkpoint, data, teacher, cache, output):
    dsec = name.startswith('dsec')
    features = ['z'] if name.endswith('z_only') else ['z', 'h']
    probes = ['z'] if len(features) == 1 else ['z', 'h', 'concat']
    stages, results = [], []
    def cmd(script, *args):
        return [sys.executable, f'tools/{script}.py', *map(str, args)]
    def stage(label, command, head=None):
        stages.append((label, command, head))
    if dsec:
        for task in ('semantic', 'detection'):
            source = cache / task
            feature_cache = source if task == 'semantic' else cache / 'dsec_det'
            labels = data / ('DSEC/task_labels/semantic' if task == 'semantic' else 'DSEC/dsec_det_labels')
            for role in ('train', 'val'):
                args = ['--checkpoint', checkpoint, '--event-cache-dir', data / 'DSEC_cache/events/gep_rgb',
                        '--output-dir', source, '--role', role, '--features', *features,
                        '--state-policy', 'continuous', '--teacher-checkpoint', teacher, '--device', 'cuda']
                if task == 'semantic':
                    args += ['--labels-root', labels]
                stage(f'{task}_cache_{role}', cmd(f'cache_dsec_{task}_features', *args))
                if task == 'detection':
                    stage(f'{task}_warp_{role}', cmd('prepare_dsec_detection_benchmark_features',
                        '--input-dir', source, '--output-dir', feature_cache, '--dataset-root', data / 'DSEC',
                        '--role', role, '--features', *features, '--device', 'cuda'))
            for feature in probes:
                head = output / task / feature / 'seed_0'
                batch = '8' if task == 'semantic' else '16'
                args = ['--feature-cache-dir', feature_cache, '--labels-root', labels,
                        '--feature', feature, '--batch-size', batch, '--num-workers', '4',
                        '--device', 'cuda', '--precision', 'fp16']
                specific = (['--protocol', 'development', '--head-type', 'linear', '--loss', 'ce']
                            if task == 'semantic' else ['--protocol', 'dsec-det', '--dataset-root', data / 'DSEC'])
                stage(f'{task}_train_{feature}', cmd(f'train_dsec_{task}', *args, *specific,
                      '--output-dir', head, '--epochs', '50', '--seed', '0'), head)
                metric = head / 'validation_metrics.json'
                evaluate_args = [] if task == 'semantic' else ['--protocol', 'dsec-det', '--dataset-root', data / 'DSEC']
                stage(f'{task}_eval_{feature}', cmd(f'evaluate_dsec_{task}', *args, *evaluate_args,
                      '--checkpoint', head / 'best.pt', '--role', 'val', '--output', metric))
                results.append((head, metric, feature, 'mIoU' if task == 'semantic' else 'mAP', 'val'))
    else:
        for role in ('train', 'validation'):
            stage(f'cache_{role}', cmd('cache_m3ed_semantic_features', '--checkpoint', checkpoint,
                '--prepared-root', data / 'm3ed_cache/half_dagr',
                '--teacher-cache-dir', data / 'm3ed_cache/dinov3_vits16_640x352',
                '--teacher-checkpoint', teacher, '--target-cache-dir', data / 'm3ed_cache/m3ed_downstream',
                '--output-dir', cache / 'semantic', '--role', role, '--features', *features, '--device', 'cuda'))
        for feature in probes:
            head = output / 'semantic' / feature / 'seed_0'
            stage(f'train_{feature}', cmd('train_m3ed_semantic', '--feature-cache-dir', cache / 'semantic',
                '--target-cache-dir', data / 'm3ed_cache/m3ed_downstream', '--output-dir', head,
                '--feature', feature, '--head-type', 'linear', '--loss', 'ce', '--epochs', '50',
                '--validate-every', '5', '--batch-size', '8', '--num-workers', '4', '--precision', 'fp16',
                '--seed', '0', '--device', 'cuda'), head)
            results.append((head, head / 'validation_metrics.json', feature, 'mIoU', 'validation'))
    return stages, results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrain-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True, help='NEW dedicated scratch directory; owned model caches are deleted on success')
    p.add_argument('--data-root', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--teacher-checkpoint', type=Path, default=Path('/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    p.add_argument('--dsec-h-checkpoint', type=Path, help='Optional existing matched Hybrid h-only; adds both tasks')
    p.add_argument('--m3ed-zh-checkpoint', type=Path, help='Optional existing Hybrid full-region z+h')
    p.add_argument('--only', nargs='+', choices=['dsec_z_only', 'dsec_z_h', 'm3ed_z_only',
                   'm3ed_h_only', 'dsec_h_only', 'm3ed_z_h'],
                   help='Run only selected models in a NEW output/cache root')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.gpu < 0:
        p.error('--gpu must be nonnegative')
    data, teacher, root, output, scratch = [x.resolve() for x in
        (a.data_root, a.teacher_checkpoint, a.pretrain_root, a.output_root, a.cache_root)]
    if (scratch == output or scratch.is_relative_to(output) or output.is_relative_to(scratch)
        or root.is_relative_to(scratch) or data == scratch or data.is_relative_to(scratch)):
        p.error('Cache/output/input roots must not overlap unsafely')
    models = [(n, root / n / 'checkpoints/step_00100000.pt') for n in
              ('dsec_z_only', 'dsec_z_h', 'm3ed_z_only', 'm3ed_h_only')]
    for name, checkpoint in [('dsec_h_only', a.dsec_h_checkpoint), ('m3ed_z_h', a.m3ed_zh_checkpoint)]:
        if checkpoint:
            models.append((name, checkpoint.resolve()))
    if a.only:
        missing = set(a.only) - {name for name, _ in models}
        if missing:
            p.error(f'Missing checkpoint option for selected models: {sorted(missing)}')
        models = [(name, checkpoint) for name, checkpoint in models if name in a.only]
    if a.dry_run:
        for name, checkpoint in models:
            stages, _ = model_plan(name, checkpoint, data, teacher, scratch / name, output / name)
            for label, command, _ in stages:
                print(f'# {name}: {label}\nCUDA_VISIBLE_DEVICES={a.gpu} {shlex.join(command)}')
            print(f'# Verify results, then delete OWNED cache only: {scratch / name}')
        return
    for path in [teacher, *(c for _, c in models)]:
        if path.is_relative_to(scratch):
            raise ValueError(f'Input checkpoint cannot be inside scratch root: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
    required_dirs = []
    if any(name.startswith('dsec') for name, _ in models):
        required_dirs += ['DSEC_cache/events/gep_rgb', 'DSEC/task_labels/semantic', 'DSEC/dsec_det_labels']
    if any(name.startswith('m3ed') for name, _ in models):
        required_dirs += ['m3ed_cache/half_dagr', 'm3ed_cache/dinov3_vits16_640x352', 'm3ed_cache/m3ed_downstream']
    for rel in required_dirs:
        if not (data / rel).is_dir():
            raise FileNotFoundError(data / rel)
    identity = dict(version=VERSION, output=str(output), scratch=str(scratch), data=str(data),
                    teacher_sha256=digest(teacher), models={n: dict(path=str(c), sha256=digest(c)) for n, c in models})
    if a.resume:
        if json.loads((output / 'run.json').read_text()) != identity:
            raise ValueError('Resume identity differs; preserve original inputs and paths')
        if json.loads((scratch / '.owner.json').read_text()) != identity:
            raise ValueError('Scratch root ownership mismatch')
    else:
        if output.exists() or scratch.exists():
            raise ValueError('Use fresh output/cache roots, or --resume for this same run')
        output.mkdir(parents=True)
        scratch.mkdir(parents=True)
        atomic_json(output / 'run.json', identity)
        atomic_json(scratch / '.owner.json', identity)
    for name, checkpoint in models:
        out, cache = output / name, scratch / name
        out.mkdir(exist_ok=True)
        stages, results = model_plan(name, checkpoint, data, teacher, cache, out)
        done = out / 'complete.json'
        if done.exists():
            verify_results(results)
            safe_cleanup(cache, scratch, identity)
            print(f'[downstream] already complete: {name}', flush=True)
            continue
        if cache.exists():
            if cache.is_symlink() or json.loads((cache / '.owner.json').read_text()) != identity:
                raise ValueError('Unowned model cache')
        else:
            cache.mkdir()
            atomic_json(cache / '.owner.json', identity)
        for label, command, head in stages:
            marker = out / (label + '.done.json')
            if marker.exists():
                saved = json.loads(marker.read_text())['command']
                if '--resume' in saved:
                    saved = saved[:saved.index('--resume')]
                if saved != command:
                    raise ValueError(f'Stage command changed: {marker}')
                continue
            if head is not None and (head / 'last.pt').is_file():
                command = [*command, '--resume', str(head / 'last.pt')]
            print(f'[downstream] {name}: {label}', flush=True)
            with (out / (label + '.log')).open('a') as log:
                log.write(shlex.join(command) + '\n'); log.flush()
                subprocess.run(command, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            atomic_json(marker, dict(command=command))
        verify_results(results)
        atomic_json(done, dict(metrics=[str(m) for _, m, *_ in results]))
        safe_cleanup(cache, scratch, identity)
        atomic_json(out / 'cache_deleted.json', dict(path=str(cache)))
        print(f'[downstream] complete and cache removed: {name}', flush=True)
    print(f'[downstream] all complete: {output}', flush=True)


if __name__ == '__main__':
    main()
