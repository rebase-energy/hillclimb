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
    $HILLCLIMB_REPLICATE_SEED  set when the engine runs repeated replicates
                           (also exported under the old name HILLCLIMB_TRIAL_SEED)
    $HILLCLIMB_PARAMS      the trial's params.json when the candidate declares
                           tunable parameters (`spaces.params()` follows it)
    $HILLCLIMB_ENGINE_PYTHON  the engine's interpreter (hillclimb importable) — run it
                           with `env -u PYTHONPATH`: the verifier's PYTHONPATH is the
                           runtime venv's `hillclimb.spaces` shim, which shadows the package
    $HILLCLIMB_DIR         the hillclimb dir of the search (set by api.build_executor)

The command runs with cwd = the candidate candidate_dir, where `./problem/` and
`./data/` symlinks always exist (candidate candidate dirs and trial dirs by
construction; the hidden holdout dir recreates them here).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, NamedTuple, Protocol

from pydantic import BaseModel

from hillclimb.harness.procs import Reaper
from hillclimb.harness.oscompat import env_path, link_dir, new_group_kwargs, runnable

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


# One thread per solution process. Every verifier and agent experiment runs a
# numpy/scipy/torch workload that would otherwise fan out across all cores;
# with N of them in flight that is N x cores of demand, the machine stalls,
# and timing-based metrics measure the contention. Parent values win, so a
# problem that really wants multithreaded solutions can export its own.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def single_threaded(env: dict[str, str]) -> dict[str, str]:
    """`env` with SINGLE_THREAD_ENV filled in where unset."""
    for key, value in SINGLE_THREAD_ENV.items():
        env.setdefault(key, value)
    return env


def prepend_pythonpath(env: dict[str, str], path: str | None) -> dict[str, str]:
    """`env` with `path` prepended to PYTHONPATH (no-op when path is None).
    Carries the interface shim (`runtime.ensure_interface_shim`) into verifier
    runs so `from hillclimb import spaces` resolves inside the runtime venvs."""
    if path:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{path}{os.pathsep}{existing}" if existing else path
    return env


def scrubbed_env(**extra: str) -> dict[str, str]:
    """Parent env minus credentials, single-threaded, for running
    agent-authored code."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in SECRET_ENV_EXACT and not k.upper().endswith(SECRET_ENV_SUFFIXES)
    }
    env.update(extra)
    return single_threaded(env)


class ExecResult(BaseModel):
    returncode: int | None = None
    duration_s: float = 0.0
    # CPU seconds (user+system) the verifier process actually burned — the
    # cost signal duration_s only approximates (wall-clock counts I/O waits).
    # None where the platform can't report it (no os.wait4).
    cpu_s: float | None = None
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
    # per-instance breakdown of `score` from the reserved `instances` key
    # (see result_instances); empty when the verifier does not emit one
    instance_scores: dict[str, float] = {}

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.val_score is not None
            and self.submission_ok
        )


PARAMS_FILE = "params.json"


def trial_params_doc(candidate_dir: Path, trial: object) -> dict | None:
    """The params.json document a trial runs with: the candidate's declared
    space (its root params.json) with the trial's `value` per entry. None
    when the candidate declares nothing (the runtime helper then uses the
    solution's own defaults) or the declaration is unreadable."""
    root = candidate_dir / PARAMS_FILE
    if not root.exists():
        return None
    try:
        raw = json.loads(root.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
        return None
    from hillclimb.spaces import with_values

    return with_values(raw, getattr(trial, "params", None) or {})


class HoldoutScorer(Protocol):
    def score(
        self, candidate_dir: Path, trial: object = None
    ) -> tuple[float | None, str | None, float | None]:
        """Score the candidate's immutable code with `trial`'s params (a
        `candidate.Trial`; None = the defaults). Returns (score, error, cpu_s)."""
        ...


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


RESERVED_RESULT_KEYS = frozenset({"score", "report", "instances"})


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


def result_instances(payload: dict | None) -> dict[str, float]:
    """Per-instance scores from a result object's reserved `instances` key:
    the breakdown of `score` over the problem's sub-instances (zones, folds,
    test cases), in the same metric and direction as `score`. Keys must stay
    stable across a search — engines compare candidates per key. Parsing is
    advisory like result_metrics: a malformed value degrades to empty."""
    if not isinstance(payload, dict):
        return {}
    instances = payload.get("instances")
    if not isinstance(instances, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in instances.items():
        if isinstance(value, bool):
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
    params: Path | None = None,
) -> dict[str, str]:
    """The `$HILLCLIMB_*` contract a verifier command reads."""
    env = {
        "HILLCLIMB_PYTHON": env_path(python),
        "HILLCLIMB_SOLUTION": env_path(solution),
        "HILLCLIMB_RESULT": env_path(result),
        "HILLCLIMB_SPLIT": split,
        # the engine's own interpreter, where hillclimb itself is importable
        # (the runtime venv above only carries the interface shim): what a
        # meta-problem's verifier starts inner searches with
        "HILLCLIMB_ENGINE_PYTHON": sys.executable,
    }
    if seed is not None:
        env["HILLCLIMB_REPLICATE_SEED"] = str(seed)
        env["HILLCLIMB_TRIAL_SEED"] = str(seed)  # pre-rename spelling, still read by harvested skills
    if params is not None:
        env["HILLCLIMB_PARAMS"] = env_path(params)  # the trial's params.json (spaces.params() follows it)
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


class RunResult(NamedTuple):
    returncode: int | None
    timed_out: bool  # an abort reports as timed_out
    # ru_utime + ru_stime of the direct child; None where unavailable
    cpu_s: float | None


def run_logged(
    cmd: list[str],
    candidate_dir: Path,
    timeout_s: int,
    out: IO,
    err: IO,
    env: dict[str, str] | None = None,
    abort: "threading.Event | None" = None,
) -> RunResult:
    """Run cmd in its own process group with logs redirected; kill the whole
    group on timeout or abort so stray workers don't linger.

    The child is reaped through `procs.Reaper`, so its CPU time (user+system,
    with everything it waited for and — at a kill — the descendants the kill
    orphans) rides along in the result; cpu_s is None where the platform
    cannot say.
    """
    proc = subprocess.Popen(
        runnable(cmd),
        cwd=candidate_dir,
        stdout=out,
        stderr=err,
        env=env,
        **new_group_kwargs(),
    )
    reaper = Reaper(proc)
    deadline = time.monotonic() + timeout_s
    while True:
        if reaper.wait(timeout=0.2) is not None:
            return RunResult(proc.returncode, False, reaper.cpu_s)
        if (abort is not None and abort.is_set()) or time.monotonic() >= deadline:
            reaper.kill_group()
            return RunResult(proc.returncode, True, reaper.cpu_s)


class CommandExecutor:
    """Executor-protocol impl: runs the problem's verifier command on the
    validation split, in the candidate (or trial) candidate_dir."""

    def __init__(
        self,
        python: Path,
        argv: list[str],
        env_extra: dict[str, str] | None = None,
        pythonpath: str | None = None,
    ):
        # absolute() not resolve(): a venv python must be invoked via its
        # symlink path or the interpreter escapes the venv's site-packages
        self.python = python.absolute()
        self.argv = list(argv)
        self.env_extra = dict(env_extra or {})
        # deliberately not part of env_extra: that is the problem's
        # "validation runs only" env, while the interface shim is engine
        # infrastructure applied symmetrically here and on the holdout scorer
        self.pythonpath = pythonpath

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
        params_path = candidate_dir / PARAMS_FILE
        env.update(verifier_env(
            self.python, script, result_path, "validation", seed,
            params=params_path if params_path.exists() else None,
        ))
        prepend_pythonpath(env, self.pythonpath)
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out, cpu_s = run_logged(
                render_argv(self.argv, self.python, script, result_path),
                candidate_dir, timeout_s, out, err, env,
            )
        duration = time.monotonic() - start
        score, payload = (None, None) if timed_out else read_result(result_path)
        return ExecResult(
            returncode=returncode,
            duration_s=duration,
            cpu_s=cpu_s,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_score=score,
            # completion proof: the verifier writes it only after scoring
            submission_ok=score is not None,
            metrics=result_metrics(payload),
            instance_scores=result_instances(payload),
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
        pythonpath: str | None = None,
    ):
        self.python = python.absolute()
        self.argv = list(argv)
        self.problem_dir = problem_dir
        self.data_dir = data_dir
        self.work_root = work_root
        self.timeout_s = timeout_s
        self.pythonpath = pythonpath

    def score(
        self, candidate_dir: Path, trial: object = None
    ) -> tuple[float | None, str | None, float | None]:
        """Returns (score, error, cpu_s). CPU seconds are reported on every
        exit — a failed or timed-out holdout run burned them all the same.
        The holdout process sees exactly the candidate's scripts plus the
        trial's params.json — never a trial dir's logs or outputs."""
        candidate_dir = candidate_dir.absolute()
        solution = candidate_dir / "solution.py"
        if not solution.exists():
            return None, "solution.py missing at holdout time", None
        index = getattr(trial, "index", 0) or 0
        eval_dir = self.work_root.absolute() / candidate_dir.name / f"t{index}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(solution, eval_dir / "solution.py")
        # ensemble candidates import candidate_N modules from their candidate_dir
        for extra in candidate_dir.glob("candidate_*.py"):
            shutil.copy(extra, eval_dir / extra.name)
        params_path = eval_dir / PARAMS_FILE
        params_path.unlink(missing_ok=True)
        params_doc = trial_params_doc(candidate_dir, trial)
        if params_doc is not None:
            params_path.write_text(json.dumps(params_doc, indent=2, sort_keys=True) + "\n")
        for name, target in (("problem", self.problem_dir), ("data", self.data_dir)):
            link = eval_dir / name
            if not link.exists():
                link_dir(link, Path(target).resolve())
        result_path = eval_dir / RESULT_FILE
        result_path.unlink(missing_ok=True)
        env = dict(os.environ)  # full env: credentials flow
        env.update(verifier_env(
            self.python, eval_dir / "solution.py", result_path, "holdout",
            params=params_path if params_doc is not None else None,
        ))
        prepend_pythonpath(env, self.pythonpath)
        stdout_path = eval_dir / "exec_stdout.log"
        stderr_path = eval_dir / "exec_stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out, cpu_s = run_logged(
                render_argv(self.argv, self.python, eval_dir / "solution.py", result_path),
                eval_dir, self.timeout_s, out, err, env,
            )
        if timed_out:
            return None, "holdout evaluation timed out", cpu_s
        if returncode != 0:
            tail = stderr_path.read_text(errors="replace")[-300:].strip()
            return None, f"holdout evaluation failed: {tail or f'exit {returncode}'}", cpu_s
        score, _ = read_result(result_path)
        if score is None:
            return None, "holdout evaluation produced no score", cpu_s
        return score, None, cpu_s
