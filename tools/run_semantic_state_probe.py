#!/usr/bin/env python3
"""Validation-only fixed-head state/activity probes, one owned cache at a time."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from run_hybrid_target_downstream import atomic_json, digest, safe_cleanup, PROJECT


def default_models():
    pre = Path('outputs/hybrid_targets_full_20261001_231617')
    down = Path('outputs/hybrid_target_downstream_20261004')
    return [
        dict(name='dsec_h_only', dataset='dsec', checkpoint='outputs/dsec_hybrid_aug/hybrid_noaug_cache/checkpoints/step_00100000.pt',
             heads='outputs/hybrid_h_only_downstream_20261006/dsec_h_only/semantic'),
        dict(name='dsec_z_h', dataset='dsec', checkpoint=str(pre/'dsec_z_h/checkpoints/step_00100000.pt'), heads=str(down/'dsec_z_h/semantic')),
        dict(name='m3ed_h_only', dataset='m3ed', checkpoint=str(pre/'m3ed_h_only/checkpoints/step_00100000.pt'), heads=str(down/'m3ed_h_only/semantic')),
        dict(name='m3ed_z_h', dataset='m3ed', checkpoint='outputs/m3ed_activity_hybrid_full_20260929_131030/baseline_e2/checkpoints/step_00100000.pt',
             heads='outputs/m3ed_semantic_activity_random_hybrid_20260930/hybrid/baseline_e2'),
    ]


def commands(model, policy, cache, output, data, teacher, threshold):
    ds = model['dataset']
    labels = data / ('DSEC/task_labels/semantic' if ds == 'dsec' else 'm3ed_cache/m3ed_downstream')
    common = ['--checkpoint', model['checkpoint'], '--output-dir', cache,
              '--features', 'z', 'h', '--state-policy', policy, '--include-activity',
              '--teacher-checkpoint', teacher, '--device', 'cuda']
    if ds == 'dsec':
        extra = ['--event-cache-dir', data/'DSEC_cache/events/gep_rgb', '--labels-root', labels, '--role', 'val']
    else:
        extra = ['--prepared-root', data/'m3ed_cache/half_dagr', '--teacher-cache-dir', data/'m3ed_cache/dinov3_vits16_640x352',
                 '--target-cache-dir', labels, '--role', 'validation']
    def cmd(script, args):
        return [sys.executable, str(PROJECT/'tools'/script), *map(str, args)]
    stages = [('cache', cmd(f'cache_{ds}_semantic_features.py', common + extra))]
    for feature in ('z', 'h', 'concat'):
        stages.append((feature, cmd('evaluate_semantic_state_probe.py', [
            '--dataset', ds, '--checkpoint', Path(model['heads'])/feature/'seed_0/best.pt',
            '--feature-cache-dir', cache, '--labels-root', labels, '--feature', feature,
            '--state-policy', policy, '--low-activity-threshold', threshold,
            '--output', output/f'{feature}.json'])))
    return stages


def summarize(output, models):
    rows = []
    for m in models:
        for feature in ('z', 'h', 'concat'):
            left, right = [json.loads((output/m['name']/p/f'{feature}.json').read_text())
                           for p in ('continuous', 'frame')]
            for key in ('samples_sha256', 'activity_sha256', 'backbone_sha256', 'head_sha256', 'low_activity_threshold'):
                if left[key] != right[key]:
                    raise ValueError(f"Paired comparison mismatch: {m['name']} {feature} {key}")
            if feature == 'z' and left['z_features_sha256'] != right['z_features_sha256']:
                raise ValueError(f"z negative control changed for {m['name']}; inspect before interpretation")
            for group, metrics in left['metrics'].items():
                c, r = metrics['mIoU'], right['metrics'][group]['mIoU']
                if metrics['class_support'] != right['metrics'][group]['class_support']:
                    raise ValueError('Paired class support differs')
                rows.append(dict(model=m['name'], feature=feature, group=group,
                    continuous_mIoU=c, frame_reset_mIoU=r,
                    delta_points=None if c is None or r is None else (c-r)*100,
                    pixels=metrics['evaluated_pixels']))
    atomic_json(output/'summary.json', rows)
    lines = ['model\tfeature\tgroup\tcontinuous_mIoU\tframe_reset_mIoU\tdelta_points\tpixels']
    lines += ['\t'.join(str(row[k]) for k in row) for row in rows]
    (output/'summary.tsv').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--models-json', type=Path, help='Optional list of {name,dataset,checkpoint,heads}; head root contains z/h/concat/seed_0/best.pt')
    p.add_argument('--only', nargs='+')
    p.add_argument('--data-root', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--teacher-checkpoint', type=Path, default=Path('/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    p.add_argument('--low-activity-threshold', type=float, default=0.1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 0 <= a.low_activity_threshold <= 1 or a.gpu < 0:
        p.error('Invalid threshold/GPU')
    models = json.loads(a.models_json.read_text()) if a.models_json else default_models()
    if not models or len({m['name'] for m in models}) != len(models):
        p.error('Models must be nonempty and uniquely named')
    for m in models:
        if not m['name'] or Path(m['name']).name != m['name'] or m['name'] in {'.', '..'} or m['dataset'] not in {'dsec','m3ed'}:
            p.error('Invalid model name/dataset')
    if a.only:
        if set(a.only) - {m['name'] for m in models}:
            p.error('Unknown --only model')
        models = [m for m in models if m['name'] in a.only]
    output, scratch, data, teacher = [v.resolve() for v in (a.output_root,a.cache_root,a.data_root,a.teacher_checkpoint)]
    def overlaps(x, y):
        return x == y or x.is_relative_to(y) or y.is_relative_to(x)
    if overlaps(output, scratch) or (data == scratch or data.is_relative_to(scratch)) or overlaps(PROJECT, scratch):
        p.error('Use scratch outside repository, data root, and output (a dedicated subdirectory of data is allowed)')
    # A scratch subdirectory of data is safe; inputs below are checked individually.
    inputs = [teacher]
    for m in models:
        m['checkpoint'] = str(Path(m['checkpoint']).resolve())
        m['heads'] = str(Path(m['heads']).resolve())
        inputs += [Path(m['checkpoint']), *(Path(m['heads'])/f/'seed_0/best.pt' for f in ('z','h','concat'))]
    if any(v.is_relative_to(scratch) for v in inputs):
        p.error('Input inside scratch')
    if a.dry_run:
        for m in models:
            for policy in ('continuous','frame'):
                for _, cmd in commands(m, policy, scratch/f"{m['name']}_{policy}", output/m['name']/policy, data, teacher, a.low_activity_threshold):
                    print(shlex.join(cmd))
        return
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = dict(version=1, models=models, output=str(output), scratch=str(scratch), data=str(data),
                    threshold=a.low_activity_threshold, inputs={str(v):digest(v) for v in inputs},
                    code={str(v):digest(v) for v in [Path(__file__),
                        PROJECT/'tools/evaluate_semantic_state_probe.py',
                        PROJECT/'tools/cache_dsec_semantic_features.py',
                        PROJECT/'tools/cache_m3ed_semantic_features.py',
                        PROJECT/'event_state/segmentation/state_probe.py',
                        PROJECT/'event_state/segmentation/m3ed.py']})
    if a.resume:
        if json.loads((output/'run.json').read_text()) != identity:
            raise ValueError('Resume identity changed')
    else:
        if output.exists() or scratch.exists():
            raise ValueError('Use new output/cache roots or --resume')
        output.mkdir(parents=True); scratch.mkdir(parents=True)
        atomic_json(output/'run.json', identity)
    for m in models:
        for policy in ('continuous','frame'):
            dest = output/m['name']/policy; dest.mkdir(parents=True, exist_ok=True)
            cache = scratch/f"{m['name']}_{policy}"
            if (dest/'complete.json').exists():
                if any(not (dest/f'{f}.json').is_file() for f in ('z','h','concat')):
                    raise ValueError('Completed result missing')
                safe_cleanup(cache, scratch, identity)
                continue
            if cache.exists():
                if cache.is_symlink() or json.loads((cache/'.owner.json').read_text()) != identity:
                    raise ValueError('Unowned cache')
            else:
                cache.mkdir(); atomic_json(cache/'.owner.json', identity)
            for label, cmd in commands(m, policy, cache, dest, data, teacher, a.low_activity_threshold):
                print(f"[state-probe] {m['name']} {policy}: {label}", flush=True)
                with (dest/f'{label}.log').open('a') as log:
                    subprocess.run(cmd, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            atomic_json(dest/'complete.json', dict(complete=True))
            safe_cleanup(cache, scratch, identity)
    summarize(output, models)
    print(f'[state-probe] complete: {output}/summary.tsv', flush=True)

if __name__ == '__main__':
    main()
