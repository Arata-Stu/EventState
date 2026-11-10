"""Standard-library scheduling for independent common Depth conditions."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def parse_gpus(value):
    """Return up to three distinct physical GPU IDs, preserving their order."""
    values = tuple(part.strip() for part in value.split(','))
    if not values or any(not part or not part.isascii() or not part.isdecimal() for part in values):
        raise ValueError('--gpus requires comma-separated numeric GPU IDs, e.g. 0,1,2')
    values = tuple(str(int(part)) for part in values)
    if len(values) > 3 or len(set(values)) != len(values):
        raise ValueError('--gpus requires one to three distinct GPU IDs')
    return values


def reservation_bytes(cache_bytes):
    return (int(cache_bytes) * 11 + 9) // 10


def admission_bytes(job, active_statuses):
    """Free space must cover this job and the unallocated caches of active jobs."""
    return reservation_bytes(job['remaining_cache_bytes']) + sum(
        reservation_bytes(status['remaining_cache_bytes']) for status in active_statuses)


def stop_process_group(proc, timeout=10):
    """Stop a worker session, including a GPU subprocess left behind by the worker."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # The worker can exit before its child. Always signal the original group again.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _read_status(path, fallback):
    if not path.exists():
        return fallback
    status = json.loads(path.read_text())
    remaining = status.get('remaining_cache_bytes')
    if not isinstance(remaining, int) or remaining < 0:
        raise ValueError(f'{path}: invalid scratch reservation')
    if remaining > fallback['remaining_cache_bytes']:
        raise ValueError(f'{path}: scratch reservation increased unexpectedly')
    return status


def run_parallel(jobs, gpus, *, scratch, state_dir, worker_command, lock_fd=None,
                 poll_interval=2.0, popen_factory=None, disk_free=None, sleep=None):
    """Run condition workers with exclusive GPUs and aggregate scratch reservations.

    Workers atomically write status and result JSON files. A cache reservation may
    decrease only after that entire cache role has been extracted. Interrupted
    worker-owned caches and head/report artifacts are deliberately left in place.
    """
    popen_factory = popen_factory or subprocess.Popen
    disk_free = disk_free or (lambda: shutil.disk_usage(scratch).free)
    sleep = sleep or time.sleep
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    if len({job['name'] for job in pending}) != len(pending):
        raise ValueError('Duplicate condition jobs')
    active = {}
    available = list(gpus)
    results = {}
    last_waiting = None
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        while pending or active:
            for name, entry in list(active.items()):
                status = _read_status(entry['status_path'], entry['status'])
                if status.get('phase') != entry['status'].get('phase'):
                    print(f"[{name}] GPU {entry['gpu']} | {entry['job']['label']} | "
                          f"{status.get('phase')} | log: {entry['log_path']}", flush=True)
                entry['status'] = status
                code = entry['proc'].poll()
                if code is None:
                    continue
                entry['proc'].wait()
                if code != 0:
                    # A dead worker can still have a live CUDA subprocess.
                    stop_process_group(entry['proc'])
                    raise RuntimeError(f'{name}: worker exited with status {code}; '
                                       f"see {entry['log_path']}")
                if not entry['result_path'].is_file():
                    stop_process_group(entry['proc'])
                    raise RuntimeError(f'{name}: worker finished without a result; '
                                       f"see {entry['log_path']}")
                results[name] = json.loads(entry['result_path'].read_text())
                entry['log'].close()
                available.append(entry['gpu'])
                del active[name]
                print(f"[{name}] GPU {entry['gpu']} | {entry['job']['label']} | completed | "
                      f"log: {entry['log_path']}", flush=True)

            while pending and available:
                free = disk_free()
                statuses = [entry['status'] for entry in active.values()]
                # A disparity-only or smaller M3ED job can fit while a larger
                # DSEC extraction waits; preserve condition order among fits.
                fitting = next((job for job in pending
                                if admission_bytes(job, statuses) <= free), None)
                if fitting is None:
                    minimum = min(admission_bytes(job, statuses) for job in pending)
                    waiting = (tuple(job['name'] for job in pending),
                               tuple(entry['status']['remaining_cache_bytes'] for entry in active.values()))
                    if waiting != last_waiting:
                        print(f'Scratch wait: {len(available)} GPU(s) idle; need '
                              f'{minimum / 2**30:.1f} GiB free including outstanding reservations, '
                              f'{free / 2**30:.1f} GiB available at {scratch}', flush=True)
                        last_waiting = waiting
                    if not active:
                        raise OSError(f'No pending condition fits scratch: need at least '
                                      f'{minimum / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB '
                                      f'at {scratch}; retained partial caches are not removed automatically')
                    break
                pending.remove(fitting)
                gpu = available.pop(0)
                name = fitting['name']
                status_path = state_dir / f'{name}.status.json'
                result_path = state_dir / f'{name}.result.json'
                log_path = state_dir / f'{name}.log'
                # Each scheduler invocation uses a fresh state_dir. Never accept
                # results or reservations belonging to an earlier invocation.
                if status_path.exists() or result_path.exists():
                    raise ValueError(f'{state_dir}: worker state already exists')
                log = log_path.open('w')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                command = worker_command(fitting, status_path, result_path)
                print(f'[{name}] GPU {gpu} | {fitting["label"]} | starting; '
                      f'{reservation_bytes(fitting["remaining_cache_bytes"]) / 2**30:.1f} GiB '
                      f'reserved | log: {log_path}', flush=True)
                try:
                    proc = popen_factory(command, stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True,
                        pass_fds=(() if lock_fd is None else (lock_fd,)))
                except BaseException:
                    log.close()
                    raise
                active[name] = dict(proc=proc, gpu=gpu, job=fitting, status_path=status_path,
                    result_path=result_path, log_path=log_path, log=log,
                    status=dict(phase='starting', remaining_cache_bytes=fitting['remaining_cache_bytes']))
                last_waiting = None
            if active:
                sleep(poll_interval)
    finally:
        # A second Ctrl+C must not leave a CUDA child or the inherited run lock
        # alive. Restore the caller's handlers after all groups are reaped.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for entry in active.values():
            try:
                os.killpg(entry['proc'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for entry in active.values():
            try:
                stop_process_group(entry['proc'])
            finally:
                entry['log'].close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return results
