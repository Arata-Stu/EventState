"""Host-local Depth worker slots shared by this user's managed launchers."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import time


MAX_PARALLEL_DEPTH_JOBS = 2
SLOT_DIR = Path('/tmp') / f'eventstate-depth-{os.getuid()}'
SLOT_FD_ENV = 'EVENTSTATE_DEPTH_SLOT_FD'


def inherited_depth_fds():
    return (int(os.environ[SLOT_FD_ENV]),) if SLOT_FD_ENV in os.environ else ()


def try_depth_slot():
    """Return an exclusively locked file or None. Keep its inode across runs."""
    SLOT_DIR.mkdir(mode=0o700, exist_ok=True)
    info = SLOT_DIR.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f'Expected a private Depth slot directory owned by this user: {SLOT_DIR}')
    for index in range(MAX_PARALLEL_DEPTH_JOBS):
        slot = (SLOT_DIR / f'{index}.lock').open('a')
        try:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            slot.close()
        except BaseException:
            slot.close()
            raise
        else:
            return slot
    return None


@contextmanager
def hold_depth_slot():
    """Wait for a slot for a serial job and make its descriptor available to children."""
    slot = try_depth_slot()
    if slot is None:
        print(f'Waiting for a host Depth slot ({MAX_PARALLEL_DEPTH_JOBS} in use)', flush=True)
        while slot is None:
            time.sleep(1)
            slot = try_depth_slot()
    with slot:
        previous = os.environ.get(SLOT_FD_ENV)
        os.environ[SLOT_FD_ENV] = str(slot.fileno())
        try:
            yield slot
        finally:
            if previous is None:
                del os.environ[SLOT_FD_ENV]
            else:
                os.environ[SLOT_FD_ENV] = previous
