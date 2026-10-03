"""Process accounting for the children the harness spawns — verifier runs
and coding agent calls — so their CPU time is what they actually burned.

CPU is read at reaping: `os.wait4` hands back the child's user+system time
together with its exit status, and that figure already includes every
descendant the child itself waited for. Per-pid, never RUSAGE_CHILDREN
deltas — those are process-wide and would mix the concurrent trials and
operators one engine runs. Two things the kernel's own accounting misses
are covered here:

- a process group killed at a timeout: its still-running descendants die
  before anyone waits for them, so `Reaper.kill_group` samples their CPU
  from `ps` first and adds it to what the leader reports;
- a shell that `exec`s its last command, which on macOS drops the CPU of
  the children the shell had already waited for — nothing can recover
  that after the fact, so the starter verifiers call their scorer plainly
  (`tests/test_verifier_scripts.py` keeps it that way).

Platforms without `os.wait4` (Windows) fall back to a plain wait and
report `cpu_s=None`.
"""

from __future__ import annotations

import os
import subprocess
import time

from hillclimb.harness.oscompat import IS_WINDOWS, kill_group


def parse_ps_time(text: str) -> float:
    """Seconds from a `ps -o time` field: `mm:ss.ss` (macOS), `hh:mm:ss` or
    `dd-hh:mm:ss` (Linux). Anything else reads as zero."""
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return 0.0
    seconds = 0.0
    for part in text.split(":"):
        try:
            seconds = seconds * 60 + float(part)
        except ValueError:
            return 0.0
    return days * 86400 + seconds


def descendant_cpu_s(pid: int) -> float:
    """CPU seconds of every live descendant of `pid`, read from `ps` — what
    a process-group kill is about to throw away. Zero when `ps` is missing
    or says nothing (Windows: no `ps`, so the orphans' CPU goes uncounted)."""
    if IS_WINDOWS:
        return 0.0
    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,time="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    children: dict[int, list[int]] = {}
    cpu: dict[int, float] = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            child, parent = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        children.setdefault(parent, []).append(child)
        cpu[child] = parse_ps_time(fields[2])
    total = 0.0
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        total += cpu.get(child, 0.0)
        stack.extend(children.get(child, []))
    return total


class Reaper:
    """Waits on one `subprocess.Popen` child through `os.wait4`, so its CPU
    time arrives with its exit status. Use `poll`/`wait` instead of the
    Popen's own — those reap through `waitpid` and lose the usage — and
    `kill_group` instead of a bare killpg. `proc.returncode` is set by hand
    once reaped, so subprocess never re-waits a recycled pid."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        # user+system seconds of the child and what it waited for, plus the
        # descendants sampled before a group kill; None until reaped, or
        # where the platform cannot say
        self.cpu_s: float | None = None

    def poll(self) -> int | None:
        """The exit code once the child has been reaped, else None."""
        proc = self.proc
        if proc.returncode is not None:
            return proc.returncode
        if not hasattr(os, "wait4"):
            return proc.poll()
        try:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:  # reaped elsewhere — exit status and cpu lost
            proc.wait()
            return proc.returncode
        if pid == 0:
            return None
        proc.returncode = os.waitstatus_to_exitcode(status)
        self.cpu_s = (self.cpu_s or 0.0) + usage.ru_utime + usage.ru_stime
        return proc.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        """Poll until the child exits or `timeout` seconds pass; None on
        timeout, like `poll`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def kill_group(self) -> int | None:
        """SIGKILL the child's whole process group and reap the child. The
        CPU of descendants still alive at the kill is sampled first — the
        kill orphans them, so nothing else would count it. (A descendant
        that exits between the sample and the kill is reaped by the leader
        and counted twice; that window is a few milliseconds.)"""
        proc = self.proc
        if proc.returncode is not None:
            return proc.returncode
        orphans = descendant_cpu_s(proc.pid)
        kill_group(proc.pid)
        code = self.wait()
        if orphans > 0.0:
            self.cpu_s = (self.cpu_s or 0.0) + orphans
        return code
