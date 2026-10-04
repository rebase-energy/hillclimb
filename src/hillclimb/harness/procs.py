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

import json
import os
import subprocess
import threading
import time
from pathlib import Path

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


def _proc_table() -> list[tuple[int, int, float]]:
    """(pid, ppid, CPU seconds) of every process, from Linux's /proc: tick
    resolution, where procps `ps -o time` rounds down to whole seconds (a
    descendant's 0.4 s would read as 0). Empty where there is no /proc."""
    proc = Path("/proc")
    if not (proc / "self" / "stat").exists():
        return []
    try:
        ticks = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return []
    table = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue  # gone since the listing
        # "pid (comm) state ppid ... utime stime ...": comm may hold spaces
        fields = stat[stat.rfind(")") + 2 :].split()
        try:
            table.append((int(entry.name), int(fields[1]), (int(fields[11]) + int(fields[12])) / ticks))
        except (IndexError, ValueError):
            continue
    return table


def _ps_table() -> list[tuple[int, int, float]]:
    """(pid, ppid, CPU seconds) of every process, from `ps` (macOS, BSD)."""
    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,time="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    table = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            table.append((int(fields[0]), int(fields[1]), parse_ps_time(fields[2])))
        except ValueError:
            continue
    return table


def descendant_cpu_s(pid: int) -> float:
    """CPU seconds of every live descendant of `pid`, read from `ps` — what
    a process-group kill is about to throw away. Zero when `ps` is missing
    or says nothing (Windows: no `ps`, so the orphans' CPU goes uncounted)."""
    if IS_WINDOWS:
        return 0.0
    children: dict[int, list[int]] = {}
    cpu: dict[int, float] = {}
    for child, parent, seconds in _proc_table() or _ps_table():
        children.setdefault(parent, []).append(child)
        cpu[child] = seconds
    total = 0.0
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        total += cpu.get(child, 0.0)
        stack.extend(children.get(child, []))
    return total


# --- the ledger of children ---------------------------------------------
#
# Every coding agent and verifier an engine starts leads a process group of
# its own (`start_new_session`), so an engine killed with SIGKILL takes none
# of them with it: they run on, and a coding agent goes on billing. Each
# child is written to the search's ledger while it lives, so whoever finds
# the engine dead (`resume`, `stop`, `kill`) can stop what it left behind
# (`orphans.stop_orphaned_children`).

CHILDREN_FILE = "children.json"  # in the search dir, while children live
_ledger: Path | None = None
_ledger_lock = threading.Lock()
_live: dict[int, dict] = {}


def track_children(path: Path | None) -> None:
    """Keep the ledger of this process's children at `path` (None: stop, and
    remove the file). An engine calls it once its search dir exists."""
    global _ledger
    with _ledger_lock:
        if _ledger is not None and path is None:
            _ledger.unlink(missing_ok=True)
        _ledger = Path(path) if path is not None else None
        _live.clear()
        _write_ledger()


def _program(proc: subprocess.Popen) -> str:
    """What the child runs, past a sandbox wrapper (`sandbox-exec -p PROFILE
    cmd …`, `bwrap … -- cmd …`), for the messages that name it."""
    args = [str(a) for a in (proc.args if isinstance(proc.args, (list, tuple)) else str(proc.args).split())]
    if args and Path(args[0]).name == "sandbox-exec" and len(args) > 3:
        args = args[3:]
    elif args and Path(args[0]).name == "bwrap" and "--" in args:
        args = args[args.index("--") + 1 :]
    return Path(args[0]).name if args else ""


def _write_ledger() -> None:
    if _ledger is None:
        return
    try:
        if _live:
            staging = _ledger.with_suffix(".tmp")
            staging.write_text(json.dumps(list(_live.values())))
            os.replace(staging, _ledger)
        else:
            _ledger.unlink(missing_ok=True)
    except OSError:
        pass  # a ledger that cannot be written never costs the search


def _track(proc: subprocess.Popen) -> None:
    with _ledger_lock:
        if _ledger is not None:
            _live[proc.pid] = {"pid": proc.pid, "program": _program(proc), "started": time.time()}
            _write_ledger()


def _untrack(pid: int) -> None:
    with _ledger_lock:
        if _live.pop(pid, None) is not None:
            _write_ledger()


class Reaper:
    """Waits on one `subprocess.Popen` child through `os.wait4`, so its CPU
    time arrives with its exit status. Use `poll`/`wait` instead of the
    Popen's own — those reap through `waitpid` and lose the usage — and
    `kill_group` instead of a bare killpg. `proc.returncode` is set by hand
    once reaped, so subprocess never re-waits a recycled pid."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        _track(proc)  # on the search's ledger until it is reaped
        # user+system seconds of the child and what it waited for, plus the
        # descendants sampled before a group kill; None until reaped, or
        # where the platform cannot say
        self.cpu_s: float | None = None

    def poll(self) -> int | None:
        """The exit code once the child has been reaped, else None."""
        proc = self.proc
        if proc.returncode is not None:
            _untrack(proc.pid)
            return proc.returncode
        if not hasattr(os, "wait4"):
            code = proc.poll()
            if code is not None:
                _untrack(proc.pid)
            return code
        try:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:  # reaped elsewhere — exit status and cpu lost
            proc.wait()
            _untrack(proc.pid)
            return proc.returncode
        if pid == 0:
            return None
        proc.returncode = os.waitstatus_to_exitcode(status)
        self.cpu_s = (self.cpu_s or 0.0) + usage.ru_utime + usage.ru_stime
        _untrack(proc.pid)
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
