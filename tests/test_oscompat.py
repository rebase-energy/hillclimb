"""harness/oscompat.py: the POSIX path unchanged, the Windows one real.

Runs everywhere; the quickstart workflow also runs it on windows-latest,
where the junction, bash and taskkill branches are the ones exercised."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hillclimb.harness import oscompat
from hillclimb.harness.oscompat import (
    IS_WINDOWS,
    is_link,
    kill_group,
    link_dir,
    lock_fd,
    new_group_kwargs,
    pid_alive,
    replace_file,
    runnable,
    venv_python,
)


def test_lock_is_exclusive_across_handles(tmp_path):
    path = tmp_path / "slot"
    first = os.open(path, os.O_CREAT | os.O_RDWR)
    second = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        assert lock_fd(first, blocking=False)
        assert not lock_fd(second, blocking=False)
        os.close(first)
        first = None
        assert lock_fd(second, blocking=False)  # closing freed it
    finally:
        for fd in (first, second):
            if fd is not None:
                os.close(fd)


def test_machine_slots_cap(tmp_path):
    from hillclimb.harness.slots import MachineSlots

    slots = MachineSlots(tmp_path, limit=1)
    held = slots.try_acquire()
    assert held is not None
    assert slots.try_acquire() is None
    held.release()
    assert slots.try_acquire() is not None


def test_venv_python_layout(tmp_path):
    expected = ("Scripts", "python.exe") if IS_WINDOWS else ("bin", "python")
    assert venv_python(tmp_path).parts[-2:] == expected


def test_link_dir_and_rmtree_leave_the_target(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    if IS_WINDOWS:  # force the junction branch: symlinks need privileges users lack
        def refuse(*_args, **_kwargs):
            raise OSError("symlinks need privileges")

        monkeypatch.setattr(Path, "symlink_to", refuse)
    holder = tmp_path / "holder"
    holder.mkdir()
    link = holder / "data"
    link_dir(link, target)
    assert is_link(link)
    assert (link / "keep.txt").read_text() == "x"
    shutil.rmtree(holder)
    assert (target / "keep.txt").exists()


def test_runnable(tmp_path):
    script = tmp_path / "verifier.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    if not IS_WINDOWS:
        assert runnable([str(script), "--holdout"]) == [str(script), "--holdout"]
        return
    argv = runnable([str(script), "--holdout"])
    assert argv[0].lower().endswith("bash.exe")
    assert "system32" not in argv[0].lower()
    assert argv[1:] == [script.as_posix(), "--holdout"]
    assert runnable(["python", "-V"])[0].lower().endswith(".exe")


def test_bash_verifier_reads_forward_slash_env(tmp_path):
    """The whole verifier contract on this OS: bash runs the script and
    `"$HILLCLIMB_PYTHON"` starts the interpreter from the env path."""
    if IS_WINDOWS:
        try:
            oscompat.find_bash()
        except oscompat.BashNotFound:
            pytest.skip("no Git Bash on this machine")
    elif shutil.which("bash") is None:
        pytest.skip("no bash")
    script = tmp_path / "verifier.sh"
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n"$HILLCLIMB_PYTHON" -c "print(42)" > "$HILLCLIMB_RESULT"\n')
    script.chmod(0o755)
    result = tmp_path / "out.txt"
    env = dict(os.environ, HILLCLIMB_PYTHON=oscompat.env_path(sys.executable), HILLCLIMB_RESULT=oscompat.env_path(result))
    subprocess.run(runnable([str(script)]), env=env, check=True, cwd=tmp_path)
    assert result.read_text().strip() == "42"


def test_pid_alive_and_kill_group_takes_the_tree(tmp_path):
    assert pid_alive(os.getpid())
    assert not pid_alive(None)
    marker = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time, pathlib;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], **new_group_kwargs())
    deadline = time.monotonic() + 30
    while not marker.exists() or not marker.read_text():
        assert time.monotonic() < deadline, "child never started its grandchild"
        time.sleep(0.1)
    grandchild = int(marker.read_text())
    assert pid_alive(proc.pid) and pid_alive(grandchild)
    kill_group(proc.pid)
    proc.wait(timeout=30)
    deadline = time.monotonic() + 10
    while pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not pid_alive(grandchild)
    assert not pid_alive(proc.pid)
    kill_group(proc.pid)  # already gone: not an error


def test_replace_file(tmp_path):
    source, target = tmp_path / "a.tmp", tmp_path / "a.json"
    target.write_text("old")
    source.write_text("new")
    replace_file(source, target)
    assert target.read_text() == "new" and not source.exists()


def test_cli_imports_and_runs_utf8():
    out = subprocess.run(
        [sys.executable, "-m", "hillclimb.cli", "--help"],
        capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, HILLCLIMB_NO_INTRO="1"),
    )
    assert out.returncode == 0, out.stderr
    assert "init" in out.stdout
