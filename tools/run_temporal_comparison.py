#!/usr/bin/env python3
"""Plan/run final-stage Patchwise LSTM versus 3x3 ConvLSTM pretraining.

Reuse the common E1 single-GPU recipe and its protected DSEC/M3ED train pools.
By default only the two ConvLSTM jobs are new; completed Patchwise E1 checkpoints
can be reused. No training or output creation occurs without --execute.
"""
import argparse
import copy
from pathlib import Path
import os
import shlex
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tools.run_stage1_common import (
    DEFAULT_PREFLIGHT_TESTS,
    execute,
    plan as common_plan,
)


ARCHITECTURES = ('patchwise', 'convlstm')


def parse_architectures(value):
    """Return a nonempty, ordered selection without aliases or duplicates."""
    if not isinstance(value, str):
        raise ValueError('architectures must be a comma-separated string')
    values = tuple(item.strip() for item in value.split(','))
    if (not values or any(item not in ARCHITECTURES for item in values)
            or len(set(values)) != len(values)):
        raise ValueError('Choose unique architectures from patchwise,convlstm')
    return values


def plan(a):
    architectures = parse_architectures(a.architectures)
    common_args = copy.copy(a)
    common_args.phase = 'baseline'
    common_args.four_stage_lstm = False
    common_args.lstm_from_stage = None
    baselines = {job['name']: job for job in common_plan(common_args)}
    jobs = []
    for dataset in ('dsec', 'm3ed'):
        baseline = baselines[f'{dataset}_E1_w1']
        for architecture in architectures:
            job = copy.deepcopy(baseline)
            name = f'{dataset}_E1_{architecture}_w1'
            model = 'lstm' if architecture == 'patchwise' else 'convlstm'
            command = []
            for argument in job['command']:
                if argument == 'model=lstm':
                    argument = f'model={model}'
                elif argument.startswith('hydra.run.dir='):
                    argument = f'hydra.run.dir={(a.output_root / name).resolve()}'
                command.append(argument)
            command.append(f'experiment.name=temporal_compare_e1_{architecture}_h_distill')
            temporal_label = 'Patchwise LSTM' if architecture == 'patchwise' else 'ConvLSTM'
            job.update(name=name, label=f'E1 Encoder＋LSTM（h蒸留）・{temporal_label}',
                       architecture=architecture, condition='E1', command=command)
            jobs.append(job)
    return jobs


def main():
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architectures', default='convlstm',
                        help='Comma-separated patchwise,convlstm; default: convlstm only')
    parser.add_argument('--stage', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--data-root', type=Path,
                        default=Path('/home/iASL/Arata_repo/dataset'))
    parser.add_argument('--teacher-checkpoint', type=Path,
                        default=Path('/home/iASL/Arata_repo/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'))
    parser.add_argument('--statistics-dir', type=Path, required=True,
                        help='Existing common-train dsec_common.json and m3ed_common.json')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Fresh root; existing output is never overwritten')
    parser.add_argument('--gpus', default='0,1', help='One to three distinct physical GPU IDs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Only after same-code server preflight passes')
    args = parser.parse_args()
    args.preflight_tests = DEFAULT_PREFLIGHT_TESTS + (
        'tests/test_convlstm.py', 'tests/test_four_stage_depth.py',
        'tests/test_temporal_comparison_stdlib.py',
        'tests/test_temporal_comparison_configs.py',
    )
    jobs = plan(args)
    for job in jobs:
        print(f'{job["name"]} {job["label"]}: clips/update={job["effective_clips"]}, '
              f'steps={job["steps"]}, frames={job["planned_frames"]}')
        print(shlex.join(job['command']))
    if args.execute:
        execute(args, jobs)


if __name__ == '__main__':
    main()
