#!/usr/bin/env python3
"""Train VKITTI DPT probes on DSEC then M3ED with frozen common Stage 1 weights."""
import argparse
import fcntl
import hashlib
import json
import math
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
_LOCK_FD = None


def run(tool, *args):
    # Preserve the parent's run lock even if it is killed before its GPU child.
    subprocess.run([sys.executable, '-u', str(ROOT/'tools'/tool), *map(str, args)], check=True,
                   pass_fds=(() if _LOCK_FD is None else (_LOCK_FD,)))


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def checkpoint(ds, condition):
    _, _, folder, model, step = condition
    return ROOT/'outputs'/folder/f'{ds}_{model}'/'checkpoints'/f'step_{step:08d}.pt'


def checked_activity_report(path, *, dataset, label, activity_index, head, backbone):
    """Validate an existing score before reusing it or keeping it as a legacy score."""
    from event_state.dense.depth_frame_metrics import F3_FRAME_PROTOCOL
    from event_state.dense.protocol import sha256_file

    payload = read(path)
    recorded = payload.get('activity_evaluation', {})
    if (recorded.get('activity_manifest') != activity_index.identity
            or recorded.get('condition_label') != label):
        raise ValueError(f'{path}: different activity evaluation is preserved')
    expected = dict(task='depth', role='val', head_sha256=sha256_file(head/'best.pt'),
                    backbone_sha256=sha256_file(backbone))
    if (payload.get('protocol', {}).get('dataset') != dataset
            or any(payload.get(key) != value for key, value in expected.items())):
        raise ValueError(f'{path}: stale head/backbone or evaluation protocol')
    frame_protocol = payload.get('depth_frame_metric_protocol')
    if frame_protocol is not None and frame_protocol != F3_FRAME_PROTOCOL:
        raise ValueError(f'{path}: different depth frame metric protocol is preserved')
    groups = recorded.get('groups', {})
    complete = (frame_protocol == F3_FRAME_PROTOCOL
                and 'all' in payload.get('metrics_frame_mean_f3', {})
                and 'all' in groups
                and all('metrics_frame_mean_f3' in group for group in groups.values()))
    return payload, complete


def activity_report_path(head, *, dataset, label, activity_index, backbone):
    """Add frame metrics beside an older pooled-only score without overwriting it."""
    original = head/'activity_continuous.json'
    fallback = head/'activity_continuous_frame_mean.json'
    selected = original
    if original.exists():
        _, complete = checked_activity_report(original, dataset=dataset, label=label,
            activity_index=activity_index, head=head, backbone=backbone)
        if not complete:
            selected = fallback
    elif fallback.exists():
        selected = fallback
    if selected.exists():
        _, complete = checked_activity_report(selected, dataset=dataset, label=label,
            activity_index=activity_index, head=head, backbone=backbone)
        if not complete:
            raise ValueError(f'{selected}: incomplete depth frame metrics are preserved')
    return selected


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




def _conditions(a):
    return CONDITIONS[:3] if a.baselines_only else CONDITIONS


def _activity_indexes(a):
    paths = {ds: getattr(a, ds + '_activity_manifest', None) for ds in ('dsec', 'm3ed')}
    if getattr(a, 'activity_only', False) and not any(paths.values()):
        raise ValueError('--activity-only requires at least one dataset activity manifest')
    from event_state.activity_reporting import load_activity_options
    return {ds: load_activity_options(path, 'Depth activity preflight', dataset=ds)
            for ds, path in paths.items() if path is not None}


def checked_report(path, *, dataset, head, backbone, disparity=False):
    """Return a complete report, or None for interrupted/incomplete JSON.

    Present provenance must match. Older reports with fewer provenance fields
    remain readable; current production reports supply all these fields.
    """
    from event_state.dense.protocol import digest, load_protocol, sha256_file
    try:
        payload = read(path)
        score = payload['mean_sequence_mae_px'] if disparity else payload['metrics']['all']['mae']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    source = payload.get('model', {}) if disparity else payload
    expected = dict(head_sha256=lambda: sha256_file(head/'best.pt'),
        backbone_sha256=lambda: sha256_file(backbone), feature=lambda: 'stages',
        task=lambda: 'depth', role=lambda: 'val', state_policy=lambda: 'continuous',
        protocol_sha256=lambda: digest(load_protocol(dataset+'_f3')))
    if not isinstance(source, dict):
        return None
    for key, value in expected.items():
        if key in source and source[key] != value():
            raise ValueError(f'{path}: stale head/backbone or evaluation protocol ({key})')
    if 'protocol' in payload:
        if not isinstance(payload['protocol'], dict):
            return None
        if payload['protocol'].get('dataset') != dataset:
            raise ValueError(f'{path}: stale head/backbone or evaluation protocol (dataset)')
    if disparity:
        if 'role' in payload and payload['role'] != 'public_val':
            raise ValueError(f'{path}: not a public validation disparity score')
        if 'window_ms' in payload and payload['window_ms'] is not None:
            raise ValueError(f'{path}: different disparity accumulation window')
    return payload


def build_jobs(a, indexes, activity_indexes):
    """Describe each condition without creating files or starting GPU work."""
    activity_only = bool(getattr(a, 'activity_only', False))
    datasets = tuple(ds for ds in ('dsec', 'm3ed') if not activity_only or ds in activity_indexes)
    jobs = []
    for ds in datasets:
        for condition in _conditions(a):
            name, label = ds+'_'+condition[0], condition[1]
            ckpt = checkpoint(ds, condition)
            head = a.output/'heads'/name/'stages'
            report = head/'continuous.json'
            activity_index = activity_indexes.get(ds)
            activity_report = head/'activity_continuous.json'
            complete_head = (head/'best.pt').is_file() and (head/'_SUCCESS').is_file()
            if activity_only and not complete_head:
                raise ValueError(f'{head}: --activity-only requires a completed existing head')
            if activity_index is not None and complete_head:
                activity_report = activity_report_path(head, dataset=ds, label=label,
                    activity_index=activity_index, backbone=ckpt)
            invalid_reports = []
            main_ok = False
            if report.exists():
                main_ok = complete_head and checked_report(report, dataset=ds, head=head, backbone=ckpt) is not None
                if not main_ok and not activity_only:
                    invalid_reports.append(str(report))
            need_main = not activity_only and not main_ok
            need_activity = activity_index is not None and (not complete_head or not activity_report.exists())
            roles = (['val'] if complete_head or activity_only or not need_main else ['train', 'val']) if (
                need_main or need_activity) else []
            role_bytes = {role: len(read(indexes[ds, role])['records'])*(
                30*40 if ds == 'dsec' else 22*40)*1536*2 for role in roles}
            score = head/'disparity_val.json'
            score_ok = False
            if ds == 'dsec' and not activity_only and score.exists():
                score_ok = complete_head and checked_report(score, dataset=ds, head=head,
                    backbone=ckpt, disparity=True) is not None
                if not score_ok:
                    invalid_reports.append(str(score))
            jobs.append(dict(name=name, label=label, dataset=ds, backbone=str(ckpt),
                head=str(head), report=str(report), activity_report=str(activity_report),
                activity_manifest=str(getattr(a, ds+'_activity_manifest')) if activity_index is not None else None,
                indexes={role: str(indexes[ds, role]) for role in roles},
                statistics=str(ROOT/'outputs/stage1_common_20261105/statistics'/f'{ds}_common.json'),
                events=str(a.data/('DSEC_cache/events/gep_rgb' if ds=='dsec' else 'm3ed_cache/half_dagr')),
                roles=roles, role_bytes=role_bytes, remaining_cache_bytes=sum(role_bytes.values()),
                need_main=need_main, need_activity=need_activity,
                need_disparity=ds=='dsec' and not activity_only and not score_ok,
                invalid_reports=invalid_reports))
    return jobs


def _preserve_partial(path):
    destination = path.with_name(path.name+'.partial_'+os.urandom(4).hex())
    path.rename(destination)
    print(f'Preserved incomplete artifact: {destination}', flush=True)


def run_condition(a, job, status_callback=None):
    """Run one independent condition; never write a shared summary or ZIP."""
    from tools.common_depth_parallel import reservation_bytes
    ds, label, name = job['dataset'], job['label'], job['name']
    head, ckpt = Path(job['head']), Path(job['backbone'])
    report, activity_report = Path(job['report']), Path(job['activity_report'])
    remaining = job['remaining_cache_bytes']
    cache = None

    def phase(value):
        status = dict(name=name, condition=label, phase=value, remaining_cache_bytes=remaining,
                      cache_dir=str(cache) if cache else None)
        print(f'[{name}] {label} | {value}', flush=True)
        if status_callback is not None:
            status_callback(status)

    phase('checking existing artifacts')
    for invalid in job['invalid_reports']:
        path = Path(invalid)
        if path.exists():
            _preserve_partial(path)
    if job['roles']:
        if shutil.disk_usage(a.scratch).free < reservation_bytes(remaining):
            raise OSError(f'{name}: need approx {reservation_bytes(remaining)/2**30:.1f} GiB '
                          f'temporary space at {a.scratch}')
        cache = Path(tempfile.mkdtemp(dir=a.scratch, prefix=name+'_'))
        try:
            for role in job['roles']:
                phase('extracting '+role+' features')
                run('cache_dense_features.py', '--four-level', '--checkpoint', ckpt,
                    '--target-index', Path(job['indexes'][role]), '--statistics-provenance', Path(job['statistics']),
                    '--event-cache-dir', Path(job['events']), '--output-dir', cache/role)
                # Space now exists on disk. Release only this fully extracted
                # role's *unallocated* reservation, not a partially written one.
                remaining -= job['role_bytes'][role]
                phase(role+' features ready')
            if 'train' in job['roles']:
                if head.exists():
                    previous = head.with_name('partial_'+os.urandom(4).hex())
                    head.rename(previous)
                    print(f'[{name}] {label} | incomplete head preserved at {previous}; '
                          'training a fresh head with the original budget', flush=True)
                phase('training frozen-backbone DPT head')
                run('train_dense_probe.py', '--feature', 'stages', '--train-cache', cache/'train',
                    '--val-cache', cache/'val', '--output-dir', head, '--dav2-weights', a.weights)
            if job['need_main']:
                phase('evaluating depth')
                run('evaluate_dense_probe.py', '--checkpoint', head/'best.pt', '--cache', cache/'val',
                    '--output', report)
            if job['need_activity']:
                phase('evaluating activity quintiles')
                run('evaluate_dense_probe.py', '--checkpoint', head/'best.pt', '--cache', cache/'val',
                    '--output', activity_report, '--activity-manifest', Path(job['activity_manifest']),
                    '--condition-label', label)
        except BaseException:
            phase('interrupted or failed; cache preserved')
            raise
        else:
            shutil.rmtree(cache)
            cache = None
            phase('temporary features removed')
    overall_path = report if report.exists() else activity_report
    overall = checked_report(overall_path, dataset=ds, head=head, backbone=ckpt)
    if overall is None:
        raise ValueError(f'{overall_path}: incomplete depth evaluation')
    result = dict(condition=label, depth=overall['metrics']['all'])
    if job['activity_manifest'] is not None:
        from event_state.activity_reporting import load_activity_options
        activity_index = load_activity_options(Path(job['activity_manifest']), 'Depth activity preflight', dataset=ds)
        saved, complete = checked_activity_report(activity_report, dataset=ds, label=label,
            activity_index=activity_index, head=head, backbone=ckpt)
        if not complete:
            raise ValueError(f'{activity_report}: evaluation did not provide depth frame metrics')
        result.update(activity_report=str(activity_report),
            activity_groups=saved['activity_evaluation']['groups'],
            overall_frame_mean_f3=saved['metrics_frame_mean_f3']['all'])
    if ds == 'dsec' and not getattr(a, 'activity_only', False):
        score = head/'disparity_val.json'
        if job['need_disparity']:
            phase('evaluating disparity validation (four sequences)')
            run('evaluate_dsec_disparity_window.py', '--run', a.output, '--name', name,
                '--feature', 'stages', '--backbone', ckpt, '--dsec', a.data/'DSEC',
                '--window-ms', 'rgb-interval', '--output', score)
        saved = checked_report(score, dataset=ds, head=head, backbone=ckpt, disparity=True)
        if saved is None:
            raise ValueError(f'{score}: incomplete disparity validation')
        result['mae_px'] = saved['mean_sequence_mae_px']
    phase('completed')
    return result


def _existing_timestamps(out):
    """Read-only counterpart of timestamps for preflight."""
    spec = read(ROOT/'tools/manifests/dense/dsec_submission_csv.json')
    dest = out/'official_csv'
    for seq, expected in spec['csv_sha256'].items():
        path = dest/(seq+'.csv')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{path}: missing/invalid official CSV; prepare it with the normal runner first')
    return dest


def _prepare(a):
    from event_state.dense.protocol import sha256_file
    preflight = bool(getattr(a, 'preflight', False))
    activity_only = bool(getattr(a, 'activity_only', False))
    activity_indexes = _activity_indexes(a)
    out, data = a.output, a.data
    if not activity_only:
        identity = dict(weights=sha256_file(a.weights), data=str(data.resolve()), conditions={})
        for ds in ('dsec', 'm3ed'):
            for condition in _conditions(a):
                identity['conditions'][ds+'_'+condition[0]] = sha256_file(checkpoint(ds, condition))
        settings = out/'run.json'
        if settings.exists() and read(settings) != identity:
            raise ValueError('Different depth run at this output path; use --output')
        if not preflight:
            verify_decoder(a.weights)
            save(settings, identity)
    export_args = None
    if not a.local_only and not activity_only:
        csv = _existing_timestamps(out) if preflight else timestamps(out)
        export_args = ['--run', out, '--dsec', data/'DSEC', '--timestamps', csv,
                       '--output', out/'submission']
        if not preflight:
            run('export_dsec_depth_submission.py', *export_args, '--preflight')
    indexes = {}
    datasets = tuple(ds for ds in ('dsec', 'm3ed') if not activity_only or ds in activity_indexes)
    for ds in datasets:
        for role in (('val',) if activity_only else ('train', 'val')):
            target = data/'DSEC/train_disparity'
            if ds == 'm3ed':
                target = a.m3ed_val_targets if role == 'val' else a.m3ed_targets
            path = out/'targets'/f'{ds}_{role}.json'
            if not path.exists():
                if preflight:
                    raise FileNotFoundError(f'{path}: target index required for read-only --preflight; '
                                            'prepare it with the normal runner first')
                run('prepare_dense_targets.py', '--protocol', ds+'_f3', '--role', role,
                    '--task', 'depth', '--target-root', target, '--output', path,
                    *(['--calibration-root', data/'DSEC/train_calibration'] if ds=='dsec' else []))
            indexes[ds, role] = path
    return build_jobs(a, indexes, activity_indexes), export_args


def _preflight(a, jobs):
    from tools.common_depth_parallel import admission_bytes, parse_gpus, reservation_bytes
    # File checks deliberately avoid torch.load/model construction/GPU inference.
    for job in jobs:
        if job['roles']:
            for key in ('backbone', 'statistics'):
                if not Path(job[key]).is_file():
                    raise FileNotFoundError(f'{job["name"]}: missing {key}: {job[key]}')
            if not Path(job['events']).is_dir():
                raise FileNotFoundError(f'{job["name"]}: missing event cache: {job["events"]}')
            if job['need_activity'] and not Path(job['activity_manifest']).is_file():
                raise FileNotFoundError(job['activity_manifest'])
    scratch = a.scratch
    while not scratch.exists() and scratch != scratch.parent:
        scratch = scratch.parent
    free = shutil.disk_usage(scratch).free
    gpus = parse_gpus(a.gpus) if getattr(a, 'gpus', None) else ('current CUDA device',)
    print(f'Read-only preflight: GPU(s) {", ".join(gpus)}; scratch free {free/2**30:.1f} GiB at {a.scratch}', flush=True)
    for job in jobs:
        actions = (['extract '+','.join(job['roles'])] if job['roles'] else [])
        if 'train' in job['roles']:
            actions.append('train head')
        for key, text in (('need_main', 'depth score'), ('need_activity', 'activity quintiles'),
                          ('need_disparity', 'disparity val')):
            if job[key]:
                actions.append(text)
        print(f'[{job["name"]}] {job["label"]} | {"; ".join(actions) or "reuse completed reports"} | '
              f'{reservation_bytes(job["remaining_cache_bytes"])/2**30:.1f} GiB temporary reservation', flush=True)
    pending = sorted(jobs, key=lambda job: job['remaining_cache_bytes'])
    selected = []
    for job in pending:
        if len(selected) < len(gpus) and admission_bytes(job, selected) <= free:
            selected.append(job)
    print(f'Initially feasible parallel conditions: {len(selected)}/{len(gpus)} '
          '(scratch reservation estimate; GPU availability is not probed)', flush=True)
    print('Preflight completed: no files created, no model loading or inference.', flush=True)


def _export(a, summary, export_args):
    if export_args is None or (a.output/'submission/submission.zip').exists():
        return
    winner = min(_conditions(a), key=lambda condition: summary['dsec_'+condition[0]]['mae_px'])
    if (a.output/'submission').exists():
        (a.output/'submission').rename(a.output/('partial_submission_'+os.urandom(4).hex()))
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    if getattr(a, 'gpus', None):
        from tools.common_depth_parallel import parse_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = parse_gpus(a.gpus)[0]
    try:
        run('export_dsec_depth_submission.py', *export_args, '--name', 'dsec_'+winner[0],
            '--backbone', checkpoint('dsec', winner), '--feature', 'stages')
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous


def _worker_command(a, plan_path, lock_fd):
    def command(job, status_path, result_path):
        return [sys.executable, '-u', str(ROOT/'tools/run_common_depth.py'),
            '--worker-plan', str(plan_path), '--worker-name', job['name'],
            '--worker-status', str(status_path), '--worker-result', str(result_path),
            '--parent-lock-fd', str(lock_fd)]
    return command


def pipeline(a, lock_fd=None):
    jobs, export_args = _prepare(a)
    if getattr(a, 'preflight', False):
        _preflight(a, jobs)
        return
    activity_only = bool(getattr(a, 'activity_only', False))
    summary_path = a.output/('activity_summary.json' if activity_only else 'summary.json')
    summary = read(summary_path) if activity_only and summary_path.exists() else {}
    if getattr(a, 'gpus', None):
        from tools.common_depth_parallel import parse_gpus, run_parallel
        if lock_fd is None:
            raise ValueError('Parallel workers require the parent run.lock')
        state_dir = Path(tempfile.mkdtemp(dir=a.output, prefix='parallel_'))
        plan_path = state_dir/'plan.json'
        values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        save(plan_path, dict(arguments=values, jobs=jobs))
        results = run_parallel(jobs, parse_gpus(a.gpus), scratch=a.scratch, state_dir=state_dir,
            worker_command=_worker_command(a, plan_path, lock_fd), lock_fd=lock_fd)
        for job in jobs:
            summary[job['name']] = results[job['name']]
        save(summary_path, summary)
        _export(a, summary, export_args)
    else:
        for i, job in enumerate(jobs):
            summary[job['name']] = run_condition(a, job)
            save(summary_path, summary)
            # Preserve the original serial DSEC -> submission -> M3ED order.
            if job['dataset'] == 'dsec' and (i+1 == len(jobs) or jobs[i+1]['dataset'] != 'dsec'):
                _export(a, summary, export_args)
    print(f'Completed: {summary_path}', flush=True)


def _worker(a):
    global _LOCK_FD
    if a.parent_lock_fd is None:
        raise ValueError('Internal workers require the inherited parent run.lock')
    _LOCK_FD = a.parent_lock_fd
    plan = read(a.worker_plan)
    values = plan['arguments']
    for key in ('data', 'weights', 'output', 'scratch', 'm3ed_targets', 'm3ed_val_targets',
                'dsec_activity_manifest', 'm3ed_activity_manifest'):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    args = argparse.Namespace(**values)
    held, expected = os.fstat(_LOCK_FD), (args.output/'run.lock').stat()
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Worker inherited an unrelated lock file descriptor')
    matches = [job for job in plan['jobs'] if job['name'] == a.worker_name]
    if len(matches) != 1:
        raise ValueError('Worker condition absent or duplicated in plan')
    try:
        result = run_condition(args, matches[0], status_callback=lambda status: save(a.worker_status, status))
        save(a.worker_result, result)
    except BaseException as exc:
        previous = read(a.worker_status) if a.worker_status.exists() else dict(
            remaining_cache_bytes=matches[0]['remaining_cache_bytes'])
        save(a.worker_status, dict(previous, phase='failed', error=str(exc)))
        raise


def find_active_run_processes(output, proc_root=Path('/proc')):
    """Find old Linux GPU subprocesses that did not inherit the run lock.

    Old serial runners did not pass run.lock to their children. Inspect only
    known Depth tools and their explicit run/head/index paths; never signal or
    terminate any process discovered here.
    """
    if not proc_root.is_dir():
        return []
    tools = {'cache_dense_features.py', 'train_dense_probe.py', 'evaluate_dense_probe.py',
             'evaluate_dsec_disparity_window.py', 'export_dsec_depth_submission.py'}
    options = {'--run', '--target-index', '--checkpoint', '--output-dir', '--output'}
    output = output.resolve()
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            command = [part.decode(errors='replace') for part in (entry/'cmdline').read_bytes().split(b'\0') if part]
            tool = next((Path(part).name for part in command if Path(part).name in tools), None)
            if tool is None:
                continue
            cwd = (entry/'cwd').resolve()
            for i, option in enumerate(command[:-1]):
                if option not in options:
                    continue
                path = Path(command[i+1])
                path = (path if path.is_absolute() else cwd/path).resolve()
                if path == output or output in path.parents:
                    matches.append(dict(pid=int(entry.name), tool=tool))
                    break
        except (OSError, ValueError):
            # Other users' processes may be inaccessible on a shared server.
            continue
    return matches


def main():
    global _LOCK_FD
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    p.add_argument('--weights', type=Path, default=Path('weights/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth'))
    p.add_argument('--output', type=Path, default=Path('outputs/stage2_common_depth_vkitti_20261110'))
    p.add_argument('--scratch', type=Path)
    p.add_argument('--m3ed-targets', type=Path, help='Override the default M3ED target root for both splits')
    p.add_argument('--m3ed-val-targets', type=Path, help='Separate prepared depth targets for validation')
    p.add_argument('--local-only', action='store_true', help='Evaluate both public validation sets without preparing a hidden-test ZIP')
    p.add_argument('--baselines-only', action='store_true')
    p.add_argument('--dsec-activity-manifest', type=Path, help='Shared train-fitted DSEC event-rate quintiles')
    p.add_argument('--m3ed-activity-manifest', type=Path, help='Shared train-fitted M3ED event-rate quintiles')
    p.add_argument('--activity-only', action='store_true', help='Score completed heads; never train or export')
    p.add_argument('--gpus', help='Independent condition workers on up to three GPUs, e.g. 0,1,2')
    p.add_argument('--preflight', action='store_true', help='Read-only input checks and job plan; no model loading or inference')
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-name', help=argparse.SUPPRESS)
    p.add_argument('--worker-status', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--parent-lock-fd', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker_plan is not None:
        _worker(a)
        return
    if a.gpus is not None:
        from tools.common_depth_parallel import parse_gpus
        try:
            parse_gpus(a.gpus)
        except ValueError as exc:
            p.error(str(exc))
    a.m3ed_val_targets = a.m3ed_val_targets or a.m3ed_targets or a.data/'m3ed_cache/f3_val_addition_20261027_183108/targets'
    a.m3ed_targets = a.m3ed_targets or a.data/'m3ed_cache/m3ed_downstream'
    a.scratch = a.scratch or a.data/'downstream_scratch'
    lock_path = a.output/'run.lock'
    if a.preflight and not lock_path.exists():
        active = find_active_run_processes(a.output)
        if active:
            p.error('Depth subprocesses still use this run; stop the previous run before preflight')
        pipeline(a)
        return
    if not a.preflight:
        a.output.mkdir(parents=True, exist_ok=True)
        a.scratch.mkdir(parents=True, exist_ok=True)
    with lock_path.open('r' if a.preflight else 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error(f'{lock_path}: another Depth runner is still active; stop it before switching GPU mode')
        active = find_active_run_processes(a.output)
        if active:
            p.error('Depth subprocesses still use this run: '+', '.join(
                f'{item["tool"]} (PID {item["pid"]})' for item in active)+
                '; wait for or stop the previous run before switching GPU mode')
        _LOCK_FD = lock.fileno()
        pipeline(a, lock_fd=lock.fileno())


if __name__ == '__main__':
    main()
