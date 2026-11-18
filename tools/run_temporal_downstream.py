#!/usr/bin/env python3
"""Run the three remaining ConvLSTM probes on idle GPUs with unchanged recipes.

Read-only preflight is the default. A fresh run owns its output and scratch
namespace; interrupted artifacts are retained and are never silently retrained.
"""
from __future__ import annotations

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
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from event_state.activity_quantiles import ActivityIndex
from event_state.dense.protocol import digest, load_protocol
from tools.common_depth_parallel import (
    MAX_PARALLEL_DEPTH_JOBS, initially_feasible, parse_gpus, reservation_bytes, run_parallel,
)
from tools.temporal_downstream_semantic import build_semantic_job, run_semantic_job

LABEL = 'E1 Encoder＋LSTM（h蒸留）・ConvLSTM'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'Missing or empty input: {path}')
    return str(path.resolve())


def idle_gpus(requested, *, run=subprocess.run):
    """Use physical IDs and exclude devices with an existing compute process."""
    inventory = run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True)
    processes = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    devices = {}
    for line in inventory.stdout.splitlines():
        if line.strip():
            index, uuid = (part.strip() for part in line.split(','))
            devices[index] = uuid
    unknown = set(requested) - devices.keys()
    if unknown:
        raise ValueError(f'Unknown physical GPU IDs: {sorted(unknown)}')
    busy = set()
    for line in processes.stdout.splitlines():
        if line.strip():
            uuid, pid = (part.strip() for part in line.split(','))
            if not pid.isdecimal() or uuid not in devices.values():
                raise ValueError(f'Cannot identify GPU compute process safely: {line}')
            busy.add(uuid)
    return tuple(gpu for gpu in requested if devices[gpu] not in busy)


def _statistics(a, dataset):
    path = a.statistics / f'{dataset}_common.json'
    value = read(file(path))
    common = read(a.project_root / f'tools/manifests/{dataset}_stage1_common_v1.json')
    threshold = value.get('event_rate_p10_hz')
    if (set(value.get('sequences', [])) != set(common['splits']['train'])
            or value.get('event_window_us') is not None
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError(f'{path}: expected original common train/RGB-interval statistics')
    return path


def build_depth_job(a, dataset):
    name = f'{dataset}_E1_convlstm'
    backbone = a.stage1_root / name.replace('_E1_convlstm', '_E1_convlstm_w1') / 'checkpoints/step_00099999.pt'
    statistics = _statistics(a, dataset)
    events = a.data / ('DSEC_cache/events/gep_rgb' if dataset == 'dsec' else 'm3ed_cache/half_dagr')
    activity = a.activity_root / f'{dataset}.json'
    index = ActivityIndex.load(file(activity))
    if index.identity['source_contract']['dataset'] != dataset:
        raise ValueError(f'{activity}: activity dataset differs')
    inputs = [file(backbone), file(statistics), file(activity), file(a.weights)]
    inputs.append(file(a.project_root / f'tools/manifests/dense/{dataset}_f3.json'))
    sources, indexes, role_bytes, expected_hashes, stat_inputs = {}, {}, {}, {}, []
    protocol = load_protocol(dataset + '_f3')
    for role in ('train', 'val'):
        source = a.depth_run / 'targets' / f'{dataset}_{role}.json'
        payload = read(file(source))
        if (payload.get('protocol') != protocol or payload.get('protocol_sha256') != digest(protocol)
                or payload.get('dataset') != dataset
                or payload.get('role') != role or payload.get('task') != 'depth'
                or payload.get('labeled') is not True
                or {r['sequence'] for r in payload['records']} != set(protocol['splits'][role])):
            raise ValueError(f'{source}: different Depth protocol/split')
        inputs.append(file(source))
        for path, expected in payload['source_identity'].items():
            path = file(path)
            inputs.append(path)
            expected_hashes[path] = expected
        for record in payload['records']:
            path = file(record['target_path'])
            inputs.append(path)
            if 'target_sha256' in record:
                expected_hashes[path] = record['target_sha256']
        for sequence in protocol['splits'][role]:
            inputs.append(file(events / sequence / 'metadata.json'))
            # Also fail before training if a validation timestamp is not covered.
            if role == 'val':
                first = index.lookup(sequence, index.timestamps(sequence)[0])
                for record in (r for r in payload['records'] if r['sequence'] == sequence):
                    # The existing extractor legally skips only the first RGB
                    # timestamp, which has no preceding event interval.
                    if record['timestamp'] != first['previous_timestamp']:
                        index.lookup(sequence, record['timestamp'])
        sources[role] = str(source)
        indexes[role] = str(a.output / 'targets' / f'{dataset}_{role}.json')
        role_bytes[role] = len(payload['records']) * (30 * 40 if dataset == 'dsec' else 22 * 40) * 1536 * 2
    if dataset == 'dsec':
        for sequence in protocol['splits']['val']:
            for filename in ('events.h5', 'rectify_map.h5'):
                stat_inputs.append(file(a.data / 'DSEC/train_events' / sequence / 'events/left' / filename))
            inputs.append(file(a.data / 'DSEC/train_images' / sequence / 'images/timestamps.txt'))
    head = a.output / 'heads' / name / 'stages'
    return dict(name=name, kind='depth', dataset=dataset, label=LABEL,
                backbone=str(backbone), head=str(head), report=str(head / 'continuous.json'),
                activity_report=str(head / 'activity_continuous.json'),
                activity_manifest=str(activity), indexes=indexes, index_sources=sources,
                statistics=str(statistics), events=str(events), role_bytes=role_bytes,
                remaining_cache_bytes=sum(role_bytes.values()), invalid_reports=[],
                roles=['train', 'val'], need_main=True, need_activity=True,
                need_disparity=dataset == 'dsec', inputs=sorted(set(inputs)),
                expected_sha256=expected_hashes, stat_inputs=stat_inputs)


def stamp(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def build_plan(a):
    if a.output.exists():
        raise FileExistsError(f'Output exists; preserving all artifacts: {a.output}. Choose a fresh --output.')
    jobs = [build_semantic_job(a), build_depth_job(a, 'm3ed'), build_depth_job(a, 'dsec')]
    paths = {path for job in jobs for path in job['inputs']}
    print(f'Read-only input verification: {len(paths)} files (including original GT provenance).', flush=True)
    identity = {path: sha256(path) for path in sorted(paths)}
    for job in jobs:
        for path, expected in job.get('expected_sha256', {}).items():
            if identity[path] != expected:
                raise ValueError(f'{path}: original target index source hash differs; preserving inputs')
    stamps = {path: stamp(path) for job in jobs for path in job.get('stat_inputs', [])}
    return dict(format_version=1, arguments={key: str(value) if isinstance(value, Path) else value
                for key, value in vars(a).items()}, jobs=jobs, inputs_sha256=identity, inputs_stat=stamps,
                recipe='Existing E1 single-GPU probes; seed0; unchanged splits, GT, losses and budgets')


def _existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def validate_backbone_config(config, dataset):
    """Verify the actual checkpoint recipe rather than infer it from its path."""
    temporal = config.get('model', {}).get('temporal', {})
    loss = config.get('loss', {})
    expected = dict(type='convlstm', num_layers=1, input_dim=384, hidden_dim=384, output_dim=384)
    if (config.get('dataset', {}).get('name') != dataset
            or any(temporal.get(key) != value for key, value in expected.items())
            or config.get('seed') != 0 or config.get('training', {}).get('max_steps') != 99999
            or loss.get('h_distill', {}).get('enabled') is not True
            or loss.get('h_distill', {}).get('cosine_weight') != 1
            or loss.get('h_distill', {}).get('mse_weight') != 1
            or loss.get('z_objective', {}).get('type') != 'none'
            or loss.get('z_objective', {}).get('weight') != 0):
        raise ValueError('Expected final-stage ConvLSTM E1 h-only distillation, seed0/full99999 recipe')


def check_backbone(path, dataset):
    # This is worker-only: read-only preflight remains independent of Torch.
    from event_state.training import load_checkpoint_config_metadata
    validate_backbone_config(load_checkpoint_config_metadata(path).config, dataset)


def preflight(a, plan, gpus):
    free = shutil.disk_usage(_existing_parent(a.scratch)).free
    print(f'Read-only preflight | idle physical GPU(s): {", ".join(gpus) or "none"} | '
          f'scratch free {free / 2**30:.1f} GiB', flush=True)
    for job in plan['jobs']:
        task = 'DSEC Semantic: development → refit → test/Q1–Q5' if job['kind'] == 'semantic' else (
            f'{job["dataset"].upper()} Depth: train/val features → DPT head → all/Q1–Q5')
        print(f'[{job["name"]}] {job["label"]} | {task} | '
              f'{reservation_bytes(job["remaining_cache_bytes"]) / 2**30:.1f} GiB reservation', flush=True)
    selected = initially_feasible(plan['jobs'], len(gpus), free)
    print(f'Initially feasible parallel jobs: {len(selected)}/{len(plan["jobs"])}; '
          f'Depth limit {MAX_PARALLEL_DEPTH_JOBS}; host slots checked at launch; '
          'additional jobs wait for a slot or scratch.', flush=True)
    if not gpus:
        raise RuntimeError('No requested GPU is idle. No training or output creation performed.')
    if not selected:
        raise OSError('Insufficient scratch for any pending job; existing files are preserved.')


def _worker(a):
    plan = read(a.worker_plan)
    held = os.fstat(a.parent_lock_fd)
    expected_lock = (Path(plan['arguments']['output']) / 'run.lock').stat()
    if (held.st_dev, held.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
        raise ValueError('Internal worker did not inherit this run lock')
    job = next(job for job in plan['jobs'] if job['name'] == a.worker_name)
    for path in job['inputs']:
        if sha256(path) != plan['inputs_sha256'][path]:
            raise ValueError(f'Input changed after preflight: {path}')
    for path in job.get('stat_inputs', []):
        if stamp(path) != plan['inputs_stat'][path]:
            raise ValueError(f'Raw disparity-evaluation input changed: {path}')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(parse_gpus(gpu)) != 1 or idle_gpus((gpu,)) != (gpu,):
        raise RuntimeError(f'Assigned GPU {gpu} has become busy; worker did not start inference.')
    args = SimpleNamespace(**plan['arguments'])
    for key in ('project_root', 'data', 'stage1_root', 'output', 'scratch', 'depth_run',
                'statistics', 'weights', 'activity_root', 'baseline_semantic_config'):
        setattr(args, key, Path(getattr(args, key)))
    args.parent_lock_fd = a.parent_lock_fd
    args.activity_only = False
    check_backbone(job['backbone'], job['dataset'])

    def status(value):
        save(a.worker_status, value)

    if job['kind'] == 'semantic':
        result = run_semantic_job(args, job, status_callback=status)
    else:
        from tools import run_common_depth
        run_common_depth._LOCK_FD = a.parent_lock_fd
        result = run_common_depth.run_condition(args, job, status_callback=status)
    save(a.worker_result, result)


def execute(a, plan, gpus):
    # Recheck immediately before creating a fresh output namespace.
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    with (a.output / 'run.lock').open('x') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (a.output / 'targets').mkdir()
        for job in plan['jobs']:
            for role, source in job.get('index_sources', {}).items():
                shutil.copyfile(source, job['indexes'][role])
        plan_path = a.output / 'plan.json'
        save(plan_path, plan)
        state = a.output / 'logs'
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'

        def command(job, status_path, result_path):
            return [sys.executable, '-u', str(PROJECT / 'tools/run_temporal_downstream.py'),
                    '--worker-plan', str(plan_path), '--worker-name', job['name'],
                    '--worker-status', str(status_path), '--worker-result', str(result_path),
                    '--parent-lock-fd', str(lock.fileno())]

        results = run_parallel(plan['jobs'], gpus, scratch=a.scratch, state_dir=state,
                               worker_command=command, lock_fd=lock.fileno())
        save(a.output / 'summary.json', results)
        (a.output / '_SUCCESS').write_text('Three remaining ConvLSTM probes and activity evaluations completed.\n')
    print(f'Completed: {a.output / "summary.json"}', flush=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1,2', help='Physical GPU IDs to consider; busy GPUs are excluded')
    parser.add_argument('--data', type=Path, default=Path('/home/iASL/Arata_repo/dataset'))
    parser.add_argument('--stage1-root', type=Path, default=Path('outputs/stage1_temporal_full_20261115'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stage2_temporal_remaining_20261117'))
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--depth-run', type=Path, default=Path('outputs/stage2_common_depth_vkitti_20261110'))
    parser.add_argument('--statistics', type=Path, default=Path('outputs/stage1_common_20261105/statistics'))
    parser.add_argument('--weights', type=Path, default=Path('weights/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth'))
    parser.add_argument('--activity-root', type=Path, default=Path('outputs/semantic_depth_activity_20261110/manifests'))
    parser.add_argument('--baseline-semantic-config', type=Path,
                        default=Path('outputs/stage2_dsec_semantic_linear_20261109/E1/final/config.json'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='Read-only checks; no Torch/model loading or files created')
    mode.add_argument('--execute', action='store_true', help='Run after checks; only a fresh output is accepted')
    parser.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-name', help=argparse.SUPPRESS)
    parser.add_argument('--worker-status', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--parent-lock-fd', type=int, help=argparse.SUPPRESS)
    a = parser.parse_args(argv)
    a.project_root = PROJECT
    a.seed = 0
    if a.scratch is None:
        a.scratch = a.data / 'downstream_scratch' / a.output.name
    for key, value in vars(a).copy().items():
        if isinstance(value, Path):
            setattr(a, key, (PROJECT / value).resolve() if not value.is_absolute() else value.resolve())
    return a


def main(argv=None):
    a = arguments(argv)
    if a.worker_plan:
        if a.parent_lock_fd is None or a.parent_lock_fd < 0:
            raise ValueError('Internal worker requires the inherited run lock')
        os.fstat(a.parent_lock_fd)
        _worker(a)
        return
    os.chdir(PROJECT)
    requested = parse_gpus(a.gpus)
    gpus = idle_gpus(requested)
    plan = build_plan(a)
    preflight(a, plan, gpus)
    if a.execute:
        # Availability can change during source checks; keep only still-idle GPUs.
        gpus = idle_gpus(gpus)
        if not gpus:
            raise RuntimeError('GPUs became busy during preflight; nothing was started.')
        execute(a, plan, gpus)


if __name__ == '__main__':
    main()
