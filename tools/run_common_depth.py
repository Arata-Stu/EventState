#!/usr/bin/env python3
"""Train VKITTI DPT probes on DSEC then M3ED with frozen common Stage 1 weights."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    ('E0', 'E0 Encoder（z蒸留）', 'stage1_joined_baseline_full_20261107', 'E0_w1', 99999),
    ('E1', 'E1 Encoder＋LSTM（h蒸留）', 'stage1_joined_baseline_full_20261107', 'E1_w1', 99999),
    ('E2', 'E2 Encoder＋LSTM（z+h蒸留）', 'stage1_joined_baseline_full_20261107', 'E2_w1', 99999),
    ('E1_old_ddp', 'E1 Encoder＋LSTM（h蒸留）・旧型DDP', 'stage1_rvt_ddp_full_20261105', 'E1_w3', 33333),
    ('E1_new_ddp', 'E1 Encoder＋LSTM（h蒸留）・新型DDP', 'stage1_joined_ddp_full_new_20261106', 'E1_w3', 33333),
]


def run(tool, *args):
    subprocess.run([sys.executable, '-u', str(ROOT/'tools'/tool), *map(str, args)], check=True)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def checkpoint(ds, condition):
    _, _, folder, model, step = condition
    return ROOT/'outputs'/folder/f'{ds}_{model}'/'checkpoints'/f'step_{step:08d}.pt'


def timestamps(out):
    spec = read(ROOT/'tools/manifests/dense/dsec_submission_csv.json')
    dest = out/'official_csv'
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            archive = Path(tmp)/'timestamps.zip'
            urllib.request.urlretrieve(spec['source'], archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != spec['zip_sha256']:
                raise ValueError('Official timestamp archive checksum mismatch')
            csv = Path(tmp)/'csv'
            csv.mkdir()
            with zipfile.ZipFile(archive) as z:
                for seq, digest in spec['csv_sha256'].items():
                    files = [k for k in z.namelist() if Path(k).name == seq+'.csv']
                    if len(files) != 1:
                        raise ValueError(f'Missing or duplicate official CSV: {seq}')
                    content = z.read(files[0])
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError(f'Invalid official CSV: {seq}')
                    (csv/(seq+'.csv')).write_bytes(content)
            csv.rename(dest)
    return dest


def verify_decoder(weights):
    from event_state.dense.dav2 import MetricDPTSmall
    import torch
    decoder = MetricDPTSmall()
    decoder.initialize_vkitti(weights)
    with torch.no_grad():
        probe = decoder(torch.zeros(1, 1536, 4, 6), (64, 96))
    if probe.shape != (1, 64, 96) or not torch.isfinite(probe).all() or (probe <= 0).any():
        raise ValueError('Invalid VKITTI decoder output on rectangular patch grid')



def pipeline(a):
    from event_state.dense.protocol import sha256_file
    conditions = CONDITIONS[:3] if a.baselines_only else CONDITIONS
    verify_decoder(a.weights)
    out, data = a.output, a.data
    identity = dict(weights=sha256_file(a.weights), data=str(data.resolve()), conditions={})
    for ds in ('dsec','m3ed'):
        for c in conditions:
            identity['conditions'][ds+'_'+c[0]] = sha256_file(checkpoint(ds,c))
    settings = out/'run.json'
    if settings.exists() and read(settings) != identity:
        raise ValueError('Different depth run at this output path; use --output')
    save(settings, identity)
    if not a.local_only:
        export_args = ['--run', out, '--dsec', data/'DSEC', '--timestamps', timestamps(out),
                       '--output', out/'submission']
        run('export_dsec_depth_submission.py', *export_args, '--preflight')
    indexes = {}
    for ds in ('dsec','m3ed'):
        for role in ('train','val'):
            target = data/'DSEC/train_disparity'
            if ds == 'm3ed':
                target = a.m3ed_val_targets if role == 'val' else a.m3ed_targets
            path = out/'targets'/f'{ds}_{role}.json'
            if not path.exists():
                run('prepare_dense_targets.py', '--protocol', ds+'_f3', '--role', role,
                    '--task', 'depth', '--target-root', target, '--output', path,
                    *(['--calibration-root', data/'DSEC/train_calibration'] if ds=='dsec' else []))
            indexes[ds,role] = path

    summary = {}
    for ds in ('dsec','m3ed'):
        stats = ROOT/'outputs/stage1_common_20261105/statistics'/f'{ds}_common.json'
        events = data/('DSEC_cache/events/gep_rgb' if ds=='dsec' else 'm3ed_cache/half_dagr')
        for c in conditions:
            name = ds+'_'+c[0]
            ckpt = checkpoint(ds,c)
            head = out/'heads'/name/'stages'
            report = head/'continuous.json'
            print(f'[{ds}] {c[1]} | {head}', flush=True)
            if not report.exists():
                roles = ['val'] if (head/'_SUCCESS').is_file() else ['train','val']
                grid = 30*40 if ds=='dsec' else 22*40
                needed = sum(len(read(indexes[ds,r])['records']) for r in roles)*grid*1536*2
                if shutil.disk_usage(a.scratch).free < needed*1.1:
                    raise OSError(f'{name}: need approx {needed*1.1/2**30:.1f} GiB temporary space at {a.scratch}')
                with tempfile.TemporaryDirectory(dir=a.scratch, prefix=name+'_') as tmp:
                    cache = Path(tmp)
                    for role in roles:
                        run('cache_dense_features.py','--four-level','--checkpoint',ckpt,
                            '--target-index',indexes[ds,role],'--statistics-provenance',stats,
                            '--event-cache-dir',events,'--output-dir',cache/role)
                    if 'train' in roles:
                        # A partial training run is retained. Its cache metadata contains
                        # absolute temp paths, so a fresh cache needs a fresh head directory.
                        if head.exists():
                            head.rename(head.with_name('partial_'+os.urandom(4).hex()))
                        run('train_dense_probe.py','--feature','stages','--train-cache',cache/'train',
                            '--val-cache',cache/'val','--output-dir',head,'--dav2-weights',a.weights)
                    run('evaluate_dense_probe.py','--checkpoint',head/'best.pt','--cache',cache/'val','--output',report)
            summary[name] = dict(condition=c[1], depth=read(report)['metrics']['all'])
            if ds=='dsec':
                score = head/'disparity_val.json'
                if not score.exists():
                    run('evaluate_dsec_disparity_window.py','--run',out,'--name',name,'--feature','stages',
                        '--backbone',ckpt,'--dsec',data/'DSEC','--window-ms','rgb-interval','--output',score)
                summary[name]['mae_px'] = read(score)['mean_sequence_mae_px']
            save(out/'summary.json', summary)
        if ds=='dsec' and not a.local_only and not (out/'submission/submission.zip').exists():
            winner = min(conditions,key=lambda c: summary['dsec_'+c[0]]['mae_px'])
            if (out/'submission').exists():
                (out/'submission').rename(out/('partial_submission_'+os.urandom(4).hex()))
            run('export_dsec_depth_submission.py',*export_args,'--name','dsec_'+winner[0],
                '--backbone',checkpoint(ds,winner),'--feature','stages')
    print(f"Completed: {out/'summary.json'}", flush=True)


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--weights', type=Path, default=Path('weights/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth'))
    p.add_argument('--output', type=Path, default=Path('outputs/stage2_common_depth_vkitti_20261110'))
    p.add_argument('--scratch', type=Path)
    p.add_argument('--m3ed-targets', type=Path, help='Override the default M3ED target root for both splits')
    p.add_argument('--m3ed-val-targets', type=Path, help='Separate prepared depth targets for validation')
    p.add_argument('--local-only', action='store_true', help='Evaluate both public validation sets without preparing a hidden-test ZIP')
    p.add_argument('--baselines-only', action='store_true')
    a = p.parse_args()
    a.m3ed_val_targets = a.m3ed_val_targets or a.m3ed_targets or a.data/'m3ed_cache/f3_val_addition_20261027_183108/targets'
    a.m3ed_targets = a.m3ed_targets or a.data/'m3ed_cache/m3ed_downstream'
    a.scratch = a.scratch or a.data/'downstream_scratch'
    a.output.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    with (a.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0,str(ROOT))
        pipeline(a)


if __name__ == '__main__':
    main()
