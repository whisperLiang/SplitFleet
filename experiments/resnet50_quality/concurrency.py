"""Native failure detection and cleanup for the current paired experiment."""

from __future__ import annotations

import os
import re
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path


def native_failure_detail(log, *, limit=1500):
    """Keep the crashing thread before discarding large native failure logs."""
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    fatal = list(re.finditer(r'Fatal Python error:[^\n]*', log))
    if fatal:
        context = log[fatal[-1].end():]
        current = re.search(r'Current thread[^\n]*\n', context)
        frames = []
        if current:
            for line in context[current.end():].splitlines():
                if 'File "' not in line:
                    break
                frames.append(line[line.index('File "'):])
                if len(frames) == 8:
                    break
        return '\n'.join([fatal[-1].group(), *frames])[:limit]
    # A long extension-module inventory hides the exception or signal above it.
    return log.split('Extension modules:', 1)[0][-limit:]


def native_crash_report(*, output, scratch=None):
    """Read owned native crash evidence before failed-output cleanup."""
    paths = [Path(output) / 'stdout.log']
    if scratch is not None:
        logs = Path(scratch) / 'ray/session_latest/logs'
        paths.extend(sorted(logs.glob('worker-*.err')))
    crashes = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            prefix = stream.read(131072)
            stream.seek(max(len(prefix), stream.seek(0, os.SEEK_END) - 131072))
            data = (prefix + stream.read()).decode('utf-8', errors='replace')
        fatal = re.search(r'^Fatal Python error: [^\n]+', data, re.MULTILINE)
        signal = re.search(r'^\*\*\* (SIG(?:SEGV|ILL|ABRT|BUS|FPE)) received[^\n]*', data, re.MULTILINE)
        if fatal or signal:
            marker = (signal or fatal).start()
            crashes.append({'file': path.name,
                            'reason': native_failure_detail(data) if fatal else signal.group(),
                            'signal': signal.group(1) if signal else None,
                            'native_context': data[max(0, marker - 700):marker].strip()})
    if not crashes:
        return None
    # Worker stderr contains the original crash even if Ray's log forwarding
    # deduplicates it or the enclosing Flower error only says "incomplete".
    return {'kind': 'native_crash', 'reason': crashes[0]['reason'], 'processes': crashes[:16]}


def memory_available_mb():
    values = {row.split(':')[0]: int(row.split()[1]) / 1024
              for row in Path('/proc/meminfo').read_text().splitlines()
              if row.startswith(('MemAvailable:', 'MemTotal:'))}
    available = values['MemAvailable']
    for root in (Path('/sys/fs/cgroup'),):
        maximum, current = root / 'memory.max', root / 'memory.current'
        if maximum.exists() and maximum.read_text().strip() != 'max':
            available = min(available, (int(maximum.read_text()) - int(current.read_text())) / 2**20)
    return max(0.0, available)


def discard_output(path, *, root):
    """Delete only a known child output after its processes have stopped."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Cleanup must target a child of this experiment directory')
    if not path.exists():
        return

    def writable(function, name, _error):
        file = Path(name)
        file.chmod(file.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        function(name)

    if path.is_dir():
        # Read-only source snapshots are owned by this run, including their dirs.
        for directory, _, _ in os.walk(path):
            p = Path(directory)
            p.chmod(p.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        shutil.rmtree(path, onerror=writable)
    else:
        path.unlink()


def run_bounded(child, *, deadline, floor_mb=4096, on_poll=None, failure_check=None):
    """Watch the owned child; terminate its native process tree on admission loss."""
    import psutil

    reason, owned = None, {}
    owner = psutil.Process(child.pid)
    while child.poll() is None:
        try:
            for process in owner.children(recursive=True):
                owned[(process.pid, process.create_time())] = process
        except psutil.NoSuchProcess:
            pass
        failure = failure_check() if failure_check is not None else None
        if failure:
            reason = failure
        elif memory_available_mb() < floor_mb:
            reason = 'host memory reserve exhausted'
        elif time.monotonic() >= deadline:
            reason = 'native prerequisite timeout'
        if reason:
            try:
                descendants = list(owned.values())
                # psutil retains birth times, so a recycled PID is never signaled.
                for process in reversed(descendants):
                    if process.is_running():
                        process.send_signal(signal.SIGTERM)
                owner.terminate()
                # Popen owns this waitpid. Letting psutil reap its child can
                # make Popen subsequently report exit 0 for a terminated run.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    owner.kill()
                    child.wait()
                _, alive = psutil.wait_procs(descendants, timeout=5)
                for process in alive:
                    if process.is_running():
                        process.kill()
            except psutil.NoSuchProcess:
                pass
            child.wait()
            break
        if on_poll is not None:
            on_poll()
        time.sleep(1)
    # A crashed native interpreter cannot run Ray's normal shutdown. Discard
    # any remaining descendants observed while this owned controller was alive.
    alive = []
    for process in owned.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.terminate()
                alive.append(process)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    for process in alive:
        try:
            if process.is_running():
                process.kill()
        except psutil.NoSuchProcess:
            pass
    return child.wait(), reason
