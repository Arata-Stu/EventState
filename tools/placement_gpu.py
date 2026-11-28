"""Idle GPU admission shared by Stage 1 and the extended placement Semantic suite."""
import os
from pathlib import Path

from tools.run_temporal_downstream import idle_gpus


PROJECT = Path(__file__).resolve().parents[1]


def semantic_worker_gpus(proc_root=Path('/proc')):
    """Keep the physical GPU assigned to a live Semantic worker during its CPU-only phases."""
    reserved = set()
    if proc_root.is_dir():
        launcher = os.fsencode(PROJECT / 'tools/run_placement_semantic.py')
        for process in proc_root.iterdir():
            if not process.name.isdecimal():
                continue
            try:
                argv = (process / 'cmdline').read_bytes().split(b'\0')
                if launcher not in argv or b'--worker-result' not in argv:
                    continue
                env = (process / 'environ').read_bytes().split(b'\0')
            except OSError:
                continue
            for value in env:
                if value.startswith(b'CUDA_VISIBLE_DEVICES='):
                    reserved.update(os.fsdecode(value.split(b'=', 1)[1]).split(','))
    return reserved


def idle_placement_gpus(requested):
    free = idle_gpus(requested)
    if not free:
        return free
    reserved = semantic_worker_gpus()
    return tuple(gpu for gpu in free if gpu not in reserved)
