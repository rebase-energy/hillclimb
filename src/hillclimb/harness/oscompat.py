"""The one place hillclimb's POSIX assumptions meet native Windows.

Everything the harness does with processes, locks, venvs and links goes
through here, so the POSIX path stays exactly what it was and Windows gets
the nearest equivalent:

- file locks: `fcntl.flock` / `msvcrt.locking` on byte 0 (both die with the
  handle, so a crashed holder frees its lock);
- process groups: `start_new_session` / `CREATE_NEW_PROCESS_GROUP`, and a
  group kill is `killpg` / `taskkill /T /F` (Windows has no SIGTERM a
  detached console process can catch — an engine killed there is left
  `crashed` and resumes like one);
- liveness: `os.kill(pid, 0)` on POSIX — on Windows that call would
  TerminateProcess the pid, so it is psutil there;
- venv interpreters live in `bin/python` / `Scripts/python.exe`;
- directory links are symlinks, falling back to junctions on Windows (a
  symlink needs admin or developer mode there, a junction does not);
- a problem fetched on Windows carries verifier.py instead of verifier.sh
  (`problem.windows_edition`), run by this interpreter; a `.sh` verifier
  that has no `.py` edition runs through Git for Windows' bash
  (`HILLCLIMB_BASH` points at another).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


# ---- file locks --------------------------------------------------------------


def lock_fd(fd: int, *, blocking: bool = True) -> bool:
    """Exclusive lock on an open file descriptor; False when `blocking` is
    off and someone else holds it. Released by `unlock_fd` or closing."""
    if IS_WINDOWS:
        import msvcrt
        import time

        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if not blocking:
                    return False
                time.sleep(0.1)
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except OSError:
        if blocking:
            raise
        return False
    return True


def lock_file(handle, *, blocking: bool = True) -> bool:
    return lock_fd(handle.fileno(), blocking=blocking)


# ---- processes ---------------------------------------------------------------


def new_group_kwargs(*, detached: bool = False) -> dict:
    """Popen kwargs that put the child in its own process group (so a group
    kill takes its descendants and a terminal Ctrl-C does not reach it).
    `detached` also hides it from the launching console on Windows, so an
    engine outlives the terminal that started it."""
    if IS_WINDOWS:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if detached:
            flags |= subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


def _taskkill(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def kill_group(pid: int) -> None:
    """Hard-kill the process group `pid` leads (its whole tree on Windows).
    A group that is already gone is not an error."""
    if IS_WINDOWS:
        _taskkill(pid)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def signal_group(pid: int, sig: int) -> None:
    """Send `sig` to the process group led by `pid`. Windows has no signal a
    detached group can catch, so every signal there is a tree kill."""
    if IS_WINDOWS:
        _taskkill(pid)
        return
    os.killpg(pid, sig)


def signal_pid(pid: int, sig: int) -> None:
    """`os.kill` on POSIX; a tree kill on Windows (where os.kill with a plain
    signal number is TerminateProcess anyway)."""
    if IS_WINDOWS:
        _taskkill(pid)
        return
    os.kill(pid, sig)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if IS_WINDOWS:
        import psutil

        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


# signal a hard kill is reported under (SIGKILL does not exist on Windows)
KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def replace_file(source: Path | str, target: Path | str) -> None:
    """`os.replace`, retried briefly on Windows, where it fails while another
    process (a `watch` reading status.json) has the target open."""
    if not IS_WINDOWS:
        os.replace(source, target)
        return
    import time

    for attempt in range(50):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


# ---- venvs, links, scripts ---------------------------------------------------


def venv_python(venv_dir: Path) -> Path:
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_dir_of(python: Path) -> Path:
    """The venv a `venv_python()` path belongs to."""
    return python.parents[1]


def link_dir(link: Path, target: Path) -> None:
    """`link` -> `target` (a directory): a symlink, or a junction on Windows
    when symlinks need privileges this user lacks."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if not IS_WINDOWS:
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def is_link(path: Path) -> bool:
    return path.is_symlink() or (IS_WINDOWS and path.is_junction())


class BashNotFound(RuntimeError):
    pass


BASH_HINT = (
    "hillclimb verifiers are bash scripts: install Git for Windows "
    "(https://git-scm.com/downloads/win — Claude Code on Windows needs it too), "
    "or point HILLCLIMB_BASH at a bash.exe"
)


def find_bash() -> str:
    """Git for Windows' bash.exe — never System32's, which is WSL's and would
    run the verifier in another OS."""
    for var in ("HILLCLIMB_BASH", "CLAUDE_CODE_GIT_BASH_PATH"):
        value = os.environ.get(var)
        if value and Path(value).is_file():
            return value
    candidates = []
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent  # <Git>/cmd/git.exe or <Git>/bin/git.exe
        candidates += [root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += [Path(base) / "Git" / "bin" / "bash.exe", Path(base) / "Programs" / "Git" / "bin" / "bash.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    raise BashNotFound(BASH_HINT)


def runnable(cmd: list[str]) -> list[str]:
    """`cmd` as the OS can start it: unchanged on POSIX (a script's shebang
    decides); on Windows a `.sh` first token runs through bash, a `.py` one
    through this interpreter (a verifier.py needs only the stdlib), and a bare
    program name resolved through PATHEXT there (npm installs `codex` and
    `pi` as `.cmd` shims, which CreateProcess never finds by bare name)."""
    if not IS_WINDOWS or not cmd:
        return cmd
    if cmd[0].lower().endswith(".sh"):
        return [find_bash(), Path(cmd[0]).as_posix(), *cmd[1:]]
    if cmd[0].lower().endswith(".py"):  # no shebangs: the engine's interpreter runs it
        return [sys.executable, *cmd]
    if not any(sep in cmd[0] for sep in "/\\"):
        found = shutil.which(cmd[0])
        if found:
            return [found, *cmd[1:]]
    return cmd


def env_path(path: Path | str) -> str:
    """A path for a child's environment. Forward slashes on Windows: bash
    runs `"$HILLCLIMB_PYTHON"` as a PATH lookup unless it contains a `/`,
    and Python and Windows accept both separators."""
    return Path(path).as_posix() if IS_WINDOWS else str(path)


# ---- text mode ---------------------------------------------------------------


def ensure_utf8_mode(argv: list[str]) -> None:
    """On Windows, re-run this command in Python's UTF-8 mode and exit with
    its status. Every text file hillclimb reads or writes (problem docs,
    prompts, yaml, journals) is UTF-8, and pre-3.15 Windows Pythons default
    to the ANSI code page. PYTHONUTF8 rides along to every child (engines,
    verifiers, agents' Python)."""
    if not IS_WINDOWS or sys.flags.utf8_mode:
        return
    env = dict(os.environ, PYTHONUTF8="1")
    # the child owns the console: a Ctrl-C reaches it directly, the parent
    # only waits for its exit status
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", "-m", "hillclimb.cli", *argv], env=env))
