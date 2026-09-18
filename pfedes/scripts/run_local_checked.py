#!/usr/bin/env python
"""Run ONE local test tree under a memory watchdog on the 8 GB WSL host.

Usage: conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/...
The lock prevents this project's test invocations from overlapping. No CUDA virtual-memory cap.
"""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('MemAvailable unavailable')


def group_rss_mib(pgid):
    total = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            for line in (proc / 'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total / 1024


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    lock = open('/tmp/pfedes-local-test.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another local test is running; wait for it to finish.')
    free = available_mib()
    reserve, ceiling = 1536, min(3000, free - 2048)
    if ceiling < 1536:
        raise SystemExit(f'Refusing local test: only {free:.0f} MiB available.')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONUNBUFFERED='1')
    print(f'Local memory guard: tree RSS <= {ceiling:.0f} MiB; keep {reserve} MiB available.', flush=True)
    child = subprocess.Popen(sys.argv[1:], env=env, start_new_session=True)
    peak = 0
    try:
        while child.poll() is None:
            rss, free = group_rss_mib(child.pid), available_mib()
            peak = max(peak, rss)
            if rss > ceiling or free < reserve:
                print(f'MEMORY GUARD STOP: tree RSS {rss:.0f} MiB, available {free:.0f} MiB', flush=True)
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 99
            time.sleep(0.5)
        print(f'Peak test-tree RSS: {peak:.0f} MiB; exit {child.returncode}', flush=True)
        return child.returncode
    finally:
        # Terminate only this invocation's process group, including orphaned test workers.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
