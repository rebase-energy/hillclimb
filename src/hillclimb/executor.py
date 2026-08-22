"""Verifier-driven execution: the one way hillclimb scores a candidate.

Every problem — hand-written, emflow, MLE-bench — is defined by a **verifier
command**. It is the only process the engine starts. It drives `solution.py`
itself (run it, import it, shell out to it), and reports the score by writing
`$HILLCLIMB_RESULT`:

    exit 0                 the candidate is valid
    $HILLCLIMB_RESULT      `{"score": <float>, ...}` — or a bare number;
                           other numeric keys are journaled as trial
                           `metrics` (feature dimensions for quality-
                           diversity policies — never scores)

The result file is both the score carrier and the completion proof: the engine
deletes it before every run, so a stale file can never masquerade as this
run's result, and a candidate that exits 0 without writing one is a contract
violation rather than a silent zero.

Verifier environment:

    $HILLCLIMB_PYTHON      the managed runtime venv's interpreter (bare
                           `python` resolves via PATH inside the scrubbed
                           env: wrong interpreter)
    $HILLCLIMB_SOLUTION    the solution path for this run (trial-dir aware)
    $HILLCLIMB_RESULT      where to write the score
    $HILLCLIMB_SPLIT       `validation` or `holdout`
    $HILLCLIMB_TRIAL_SEED  set when the engine runs repeated trials

The command runs with cwd = the candidate candidate_dir, where `./problem/` and
`./data/` symlinks always exist (candidate candidate dirs and trial dirs by
construction; the hidden holdout dir recreates them here).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Protocol

from pydantic import BaseModel

RESULT_FILE = "eval_result.json"

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
    # journal field names predate the verifier contract and are schema v2 on
    # disk (hillclimb-go reads them): val_score is the score the verifier
    # reported, submission_ok is "the verifier wrote a usable result file".
    val_score: float | None = None
    submission_ok: bool = False
    # extra numeric keys the verifier wrote next to `score` (see
    # result_metrics); opaque to the engine, consumed by policies
    metrics: dict[str, float] = {}

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.val_score is not None
            and self.submission_ok
        )


class HoldoutScorer(Protocol):
    def score(self, candidate_dir: Path) -> tuple[float | None, str | None]: ...


class Executor(Protocol):
    def execute(
        self,
        script: Path,
        candidate_dir: Path,
        timeout_s: int,
        seed: int | None = None,
    ) -> ExecResult: ...


def read_result(path: Path) -> tuple[float | None, dict | None]:
    """(score, payload) from a verifier's result file.

    Accepts a JSON object carrying a numeric `score` (the full form, which
    may also carry a `report` breakdown) or a bare number, so the simplest
    possible verifier is `echo 12.3 > "$HILLCLIMB_RESULT"`. NaN is not a
    score: an evaluation that silently found nothing to grade must read as a
    contract violation, not as the worst possible result.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return None, None
    if not text:
        return None, None
    try:
        payload = json.loads(text)
    except ValueError:
        try:
            value = float(text)
        except ValueError:
            return None, None
        return (None if value != value else value), None
    if isinstance(payload, dict):
        score = payload.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score != score:
            return None, payload
        return float(score), payload
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or payload != payload:
        return None, None
    return float(payload), None


RESERVED_RESULT_KEYS = frozenset({"score", "report"})


def result_metrics(payload: dict | None) -> dict[str, float]:
    """Numeric keys of a result object other than the reserved ones — the
    verifier's auxiliary measurements (e.g. runtime_s, n_params, code_len)
    that a quality-diversity policy can bin on. Non-numeric, bool and NaN
    values are dropped rather than rejected: metrics are advisory."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        if key in RESERVED_RESULT_KEYS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == value:
            out[str(key)] = float(value)
    return out


def verifier_env(
    python: Path,
    solution: Path,
    result: Path,
    split: str,
    seed: int | None = None,
) -> dict[str, str]:
    """The `$HILLCLIMB_*` contract a verifier command reads."""
    env = {
        "HILLCLIMB_PYTHON": str(python),
        "HILLCLIMB_SOLUTION": str(solution),
        "HILLCLIMB_RESULT": str(result),
        "HILLCLIMB_SPLIT": split,
    }
    if seed is not None:
        env["HILLCLIMB_TRIAL_SEED"] = str(seed)
    return env


def render_argv(argv: list[str], python: Path, solution: Path, result: Path) -> list[str]:
    """Provider-supplied commands carry placeholders (a shell verifier reads
    the env instead). Substituted per token so paths with spaces survive."""
    return [
        token.replace("{python}", str(python))
        .replace("{solution}", str(solution))
        .replace("{result}", str(result))
        for token in argv
    ]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_logged(
    cmd: list[str],
    candidate_dir: Path,
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
        cwd=candidate_dir,
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


class CommandExecutor:
    """Executor-protocol impl: runs the problem's verifier command on the
    validation split, in the candidate (or trial) candidate_dir."""

    def __init__(self, python: Path, argv: list[str], env_extra: dict[str, str] | None = None):
        # absolute() not resolve(): a venv python must be invoked via its
        # symlink path or the interpreter escapes the venv's site-packages
        self.python = python.absolute()
        self.argv = list(argv)
        self.env_extra = dict(env_extra or {})

    def execute(
        self,
        script: Path,
        candidate_dir: Path,
        timeout_s: int,
        seed: int | None = None,
    ) -> ExecResult:
        script = script.absolute()
        candidate_dir = candidate_dir.absolute()
        stdout_path = candidate_dir / "exec_stdout.log"
        stderr_path = candidate_dir / "exec_stderr.log"
        result_path = candidate_dir / RESULT_FILE
        result_path.unlink(missing_ok=True)  # staleness must never fake success
        # agent-authored code runs inside the verifier process: credentials are
        # scrubbed at run time (not snapshotted at construction) so a change to
        # the orchestrator's environment can never leak into a later run
        env = scrubbed_env(**self.env_extra)
        env.update(verifier_env(self.python, script, result_path, "validation", seed))
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                render_argv(self.argv, self.python, script, result_path),
                candidate_dir, timeout_s, out, err, env,
            )
        duration = time.monotonic() - start
        score, payload = (None, None) if timed_out else read_result(result_path)
        return ExecResult(
            returncode=returncode,
            duration_s=duration,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_score=score,
            # completion proof: the verifier writes it only after scoring
            submission_ok=score is not None,
            metrics=result_metrics(payload),
        )


class CommandHoldoutScorer:
    """Hidden holdout scoring: runs the problem's holdout command against a
    copy of the candidate's solution in a dir agents never see, with the full
    environment (credentials flow — private holdout data may be gated)."""

    def __init__(
        self,
        python: Path,
        argv: list[str],
        problem_dir: Path,
        data_dir: Path,
        work_root: Path,
        timeout_s: int,
    ):
        self.python = python.absolute()
        self.argv = list(argv)
        self.problem_dir = problem_dir
        self.data_dir = data_dir
        self.work_root = work_root
        self.timeout_s = timeout_s

    def score(self, candidate_dir: Path) -> tuple[float | None, str | None]:
        candidate_dir = candidate_dir.absolute()
        solution = candidate_dir / "solution.py"
        if not solution.exists():
            return None, "solution.py missing at holdout time"
        eval_dir = self.work_root.absolute() / candidate_dir.name
        eval_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(solution, eval_dir / "solution.py")
        # ensemble candidates import candidate_N modules from their candidate_dir
        for extra in candidate_dir.glob("candidate_*.py"):
            shutil.copy(extra, eval_dir / extra.name)
        for name, target in (("problem", self.problem_dir), ("data", self.data_dir)):
            link = eval_dir / name
            if not link.exists():
                link.symlink_to(Path(target).resolve(), target_is_directory=True)
        result_path = eval_dir / RESULT_FILE
        result_path.unlink(missing_ok=True)
        env = dict(os.environ)  # full env: credentials flow
        env.update(
            verifier_env(self.python, eval_dir / "solution.py", result_path, "holdout")
        )
        stdout_path = eval_dir / "exec_stdout.log"
        stderr_path = eval_dir / "exec_stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                render_argv(self.argv, self.python, eval_dir / "solution.py", result_path),
                eval_dir, self.timeout_s, out, err, env,
            )
        if timed_out:
            return None, "holdout evaluation timed out"
        if returncode != 0:
            tail = stderr_path.read_text(errors="replace")[-300:].strip()
            return None, f"holdout evaluation failed: {tail or f'exit {returncode}'}"
        score, _ = read_result(result_path)
        if score is None:
            return None, "holdout evaluation produced no score"
        return score, None
