"""Engines whose hillclimb dir is gone.

`hillclimb stop|kill` route through the search's control/ queue, which lives
under the hillclimb dir — delete that dir and the engines (and their coding agents
and verifiers) keep running with nowhere to receive commands. The launcher
pins `HILLCLIMB_DIR` in every child engine's environment, so a live engine
pointing at a path that no longer exists is an orphan; each was started in
its own session (`start_new_session=True`), so signalling its process group
takes the coding agents and verifier children down with it.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from hillclimb.harness.oscompat import IS_WINDOWS, KILL_SIGNAL, pid_alive, signal_pid

# the launcher's argv: `<python> -m hillclimb.cli run|resume ...` (anchored,
# so a shell whose command line merely mentions it is not an engine)
ENGINE_RE = re.compile(r"^\S*python\S*\s+-m\s+hillclimb\.cli\s+(?:run|resume)\b")
_DIR_RE = re.compile(r"HILLCLIMB_DIR=(\S+)")


def is_engine(command: str) -> bool:
    return ENGINE_RE.search(command) is not None


@dataclass(frozen=True)
class Engine:
    pid: int
    pgid: int
    hillclimb_dir: Path | None  # None when the env could not be read


def _ps(args: list[str]) -> str:
    if IS_WINDOWS:
        return _ps_windows(args)
    try:
        return subprocess.run(["ps", *args], capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def _ps_windows(args: list[str]) -> str:
    """The `ps` listings this module reads, synthesized from psutil (Windows
    has no ps). The engine leads its own process group, so pgid = pid."""
    import psutil

    if args[:1] == ["-Ewwo"]:  # one pid's environment
        try:
            env = psutil.Process(int(args[-1])).environ()
        except (psutil.Error, OSError, ValueError):
            return ""
        return " ".join(f"{k}={v}" for k, v in env.items() if k == "HILLCLIMB_DIR")
    fields = args[1].rstrip("=").split("=,")
    rows = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "cpu_percent", "memory_info", "create_time"]):
        info = proc.info
        cmdline = info.get("cmdline") or []
        command = " ".join([Path(cmdline[0]).name, *cmdline[1:]]) if cmdline else "?"
        memory = info.get("memory_info")
        elapsed = int(time.time() - (info.get("create_time") or time.time()))
        values = {
            "pid": info["pid"],
            "ppid": info.get("ppid") or 0,
            "pgid": info["pid"],
            "%cpu": f"{info.get('cpu_percent') or 0.0:.1f}",
            "rss": (memory.rss // 1024) if memory else 0,
            "etime": f"{elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}",
            "command": command,
        }
        rows.append(" ".join(str(values[name]) for name in fields))
    return "\n".join(rows)


def _environ_dir(pid: int) -> Path | None:
    if IS_WINDOWS:
        text = _ps(["-Ewwo", "command=", "-p", str(pid)])
    elif sys.platform == "linux":
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
        except OSError:
            return None
        text = raw.replace("\0", " ")
    else:  # macOS/BSD: `ps -E` appends the environment to the command line
        text = _ps(["-Ewwo", "command=", "-p", str(pid)])
    match = _DIR_RE.search(text)
    return Path(match.group(1)) if match else None


def live_engines(listing: str | None = None) -> list[Engine]:
    """Every `hillclimb.cli run` engine on the machine. `listing` overrides
    the `ps -Ao pid=,pgid=,command=` output (tests)."""
    if listing is None:
        listing = _ps(["-Ao", "pid=,pgid=,command="])
    engines = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not is_engine(parts[2]):
            continue
        pid, pgid = int(parts[0]), int(parts[1])
        engines.append(Engine(pid=pid, pgid=pgid, hillclimb_dir=_environ_dir(pid)))
    return engines


def orphan_engines(engines: list[Engine] | None = None) -> list[Engine]:
    """Engines whose HILLCLIMB_DIR no longer exists."""
    if engines is None:
        engines = live_engines()
    return [e for e in engines if e.hillclimb_dir is not None and not e.hillclimb_dir.exists()]


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def engines_for(hillclimb_dir: Path, engines: list[Engine] | None = None) -> list[Engine]:
    """Engines pinned (via HILLCLIMB_DIR) to exactly this hillclimb dir —
    `hillclimb reset` takes only these down, never another folder's."""
    if engines is None:
        engines = live_engines()
    return [e for e in engines if e.hillclimb_dir is not None and _same_dir(e.hillclimb_dir, hillclimb_dir)]


def _descendants(root: int, listing: str | None = None) -> list[int]:
    """All live descendants of `root`, from `ps -Ao pid=,ppid=` (or
    `listing`). Taken before any signal: once the parent dies the children are
    reparented to launchd/init and the tree is lost."""
    if listing is None:
        listing = _ps(["-Ao", "pid=,ppid="])
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    pgid: int
    cpu: float
    rss_mb: float
    elapsed: str
    command: str


def process_table(listing: str | None = None) -> dict[int, Proc]:
    """Every process on the machine, from `ps -Ao pid=,ppid=,pgid=,%cpu=,rss=,etime=,command=`."""
    if listing is None:
        listing = _ps(["-Ao", "pid=,ppid=,pgid=,%cpu=,rss=,etime=,command="])
    table = {}
    for line in listing.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        try:
            proc = Proc(
                pid=int(parts[0]), ppid=int(parts[1]), pgid=int(parts[2]),
                cpu=float(parts[3].replace(",", ".")), rss_mb=int(parts[4]) / 1024,
                elapsed=parts[5], command=parts[6],
            )
        except ValueError:
            continue
        table[proc.pid] = proc
    return table


def engine_trees(table: dict[int, Proc] | None = None) -> list[tuple[Engine, list[Proc]]]:
    """Each live engine with its descendants (coding agents, verifiers, their
    children) in tree order, for `hillclimb ps`."""
    if table is None:
        table = process_table()
    listing = "\n".join(f"{p.pid} {p.ppid}" for p in table.values())
    trees = []
    for proc in sorted(table.values(), key=lambda p: p.pid):
        if not is_engine(proc.command):
            continue
        engine = Engine(pid=proc.pid, pgid=proc.pgid, hillclimb_dir=_environ_dir(proc.pid))
        kids = [table[pid] for pid in _descendants(proc.pid, listing) if pid in table]
        trees.append((engine, kids))
    return trees


def _alive(pid: int) -> bool:
    return pid_alive(pid)


def _signal_tree(engine: Engine, pids: list[int], sig: int) -> None:
    if IS_WINDOWS:  # taskkill /T takes the tree; survivors that left it go one by one
        for pid in (engine.pid, *pids):
            signal_pid(pid, sig)
        return
    groups = {engine.pgid}
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for pgid in groups:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
    for pid in pids:  # anything that changed group on its own
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def kill_engines(engines: list[Engine], grace_s: float = 5.0) -> list[Engine]:
    """SIGTERM each engine with its whole descendant tree (coding agents, verifiers,
    their children), then SIGKILL whatever survives `grace_s`. Returns the
    engines whose tree needed SIGKILL."""
    trees = {engine.pid: _descendants(engine.pid) for engine in engines}
    for engine in engines:
        _signal_tree(engine, trees[engine.pid], signal.SIGTERM)

    def survivors(engine: Engine) -> list[int]:
        return [pid for pid in (engine.pid, *trees[engine.pid]) if _alive(pid)]

    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(survivors(e) for e in engines):
        time.sleep(0.2)
    forced = [e for e in engines if survivors(e)]
    for engine in forced:
        _signal_tree(engine, trees[engine.pid], KILL_SIGNAL)
    return forced
