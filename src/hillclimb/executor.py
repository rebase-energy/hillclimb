from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Protocol

from pydantic import BaseModel

VAL_SCORE_RE = re.compile(r"^val_score:\s*([-+0-9.eE]+)\s*$")

# Secrets must never reach agent-authored code. Deny-list (not allow-list):
# solution subprocesses legitimately need PATH/HOME/venv/locale/thread-pool
# vars that no allow-list would enumerate reliably.
SECRET_ENV_EXACT = frozenset({
    "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN",
    "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_ACCESS_TOKEN",
})
SECRET_ENV_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_SECRET_KEY", "_PASSWORD")


def scrubbed_env(**extra: str) -> dict[str, str]:
    """Parent env minus credentials, for running agent-authored code."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in SECRET_ENV_EXACT and not k.upper().endswith(SECRET_ENV_SUFFIXES)
    }
    env.update(extra)
    return env


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
        seed: int | None = None,
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


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_logged(
    cmd: list[str],
    workspace: Path,
    timeout_s: int,
    out: IO,
    err: IO,
    env: dict[str, str] | None = None,
    abort: "threading.Event | None" = None,
) -> tuple[int | None, bool]:
    """Run cmd in its own process group with logs redirected; kill the whole
    group on timeout or abort so stray workers don't linger. Returns
    (returncode, timed_out) — an abort reports as timed_out."""
    proc = subprocess.Popen(
        cmd,
        cwd=workspace,
        stdout=out,
        stderr=err,
        env=env,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            proc.wait(timeout=1.0)
            return proc.returncode, False
        except subprocess.TimeoutExpired:
            if abort is not None and abort.is_set():
                _kill_group(proc)
                return proc.returncode, True
            if time.monotonic() >= deadline:
                _kill_group(proc)
                return proc.returncode, True


class LocalExecutor:
    """Runs solution scripts as subprocesses of a dedicated runtime venv,
    with credentials scrubbed from their environment."""

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
        seed: int | None = None,
    ) -> ExecResult:
        script = script.absolute()
        workspace = workspace.absolute()
        verifier = verifier.absolute() if verifier is not None else None
        stdout_path = workspace / "exec_stdout.log"
        stderr_path = workspace / "exec_stderr.log"
        # stale-result guard, mirroring the emflow executor: a leftover
        # eval_result.json must never masquerade as this execution's report
        (workspace / "eval_result.json").unlink(missing_ok=True)
        env = scrubbed_env(**({"HILLCLIMB_TRIAL_SEED": str(seed)} if seed is not None else {}))
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                [str(self.python), str(script)], workspace, timeout_s, out, err, env
            )
            if not timed_out and returncode == 0 and verifier is not None:
                # trust boundary: on verifier problems only the verifier may
                # produce eval_result.json — discard anything the (agent-
                # authored) solution wrote, exactly as its val_score line is
                # overridden by the verifier's
                (workspace / "eval_result.json").unlink(missing_ok=True)
                remaining = max(1, int(timeout_s - (time.monotonic() - start)))
                returncode, timed_out = run_logged(
                    [str(self.python), str(verifier)], workspace, remaining, out, err, env
                )
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
