from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

VAL_SCORE_RE = re.compile(r"^val_score:\s*([-+0-9.eE]+)\s*$")


class ExecResult(BaseModel):
    returncode: int | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    stdout_path: str = ""
    stderr_path: str = ""
    val_score: float | None = None
    submission_ok: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.val_score is not None
            and self.submission_ok
        )


class Executor(Protocol):
    def execute(
        self,
        script: Path,
        workspace: Path,
        timeout_s: int,
        verifier: Path | None = None,
    ) -> ExecResult: ...


def parse_val_score(stdout_text: str) -> float | None:
    """Last `val_score: <float>` line in stdout wins."""
    for line in reversed(stdout_text.splitlines()):
        match = VAL_SCORE_RE.match(line.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


class LocalExecutor:
    """Runs solution scripts as subprocesses of a dedicated runtime venv.
    Kills the whole process group on timeout so stray workers don't linger."""

    def __init__(self, python: Path):
        # absolute() not resolve(): a venv python must be invoked via its
        # symlink path or the interpreter escapes the venv's site-packages
        self.python = python.absolute()

    def execute(
        self,
        script: Path,
        workspace: Path,
        timeout_s: int,
        verifier: Path | None = None,
    ) -> ExecResult:
        script = script.absolute()
        workspace = workspace.absolute()
        verifier = verifier.absolute() if verifier is not None else None
        stdout_path = workspace / "exec_stdout.log"
        stderr_path = workspace / "exec_stderr.log"
        start = time.monotonic()
        timed_out = False
        returncode: int | None = None
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            proc = subprocess.Popen(
                [str(self.python), str(script)],
                cwd=workspace,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            returncode = proc.returncode
            if not timed_out and proc.returncode == 0 and verifier is not None:
                remaining = max(1, int(timeout_s - (time.monotonic() - start)))
                verifier_proc = subprocess.Popen(
                    [str(self.python), str(verifier)],
                    cwd=workspace,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                try:
                    verifier_proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(os.getpgid(verifier_proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    verifier_proc.wait()
                returncode = verifier_proc.returncode
        duration = time.monotonic() - start
        stdout_text = stdout_path.read_text()
        return ExecResult(
            returncode=returncode,
            duration_s=duration,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_score=None if timed_out else parse_val_score(stdout_text),
            submission_ok=(workspace / "submission.csv").exists() and not timed_out,
        )
