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
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, NamedTuple, Protocol

from pydantic import BaseModel

from hillclimb.harness.procs import Reaper
from hillclimb.harness.oscompat import env_path, link_dir, new_group_kwargs, runnable

if TYPE_CHECKING:
    from hillclimb.harness.sandbox import SandboxPolicy

RESULT_FILE = "eval_result.json"

# Secrets must never reach coding-agent-authored code. Deny-list (not allow-list):
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


# A fixed CPU allotment per solution process (`concurrency.solution_cpus`,
# default 1). Every verifier and coding agent experiment runs a
# numpy/scipy/torch workload that would otherwise fan out across all cores;
# with N of them in flight that is N x cores of demand, the machine stalls,
# and timing-based metrics measure the contention. The math libraries' thread
# pools are capped at the allotment; parent values win, so a problem that
# really wants other settings can export its own.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
# The allotment itself, which nothing caps for us: a solution that starts its
# own processes (multiprocessing, joblib, concurrent.futures) sizes them from
# this, never from os.cpu_count() — the contract prompt says so, and the
# journal's cpu_s / duration_s shows who did not (`Replicate.oversubscribed`).
CPUS_ENV = "HILLCLIMB_CPUS"


def single_threaded(env: dict[str, str], cpus: int = 1) -> dict[str, str]:
    """`env` capped at `cpus` cores: the math libraries' thread variables
    filled in where unset, and `$HILLCLIMB_CPUS` (the harness's, always set)."""
    for key in SINGLE_THREAD_ENV:
        env.setdefault(key, str(cpus))
    env[CPUS_ENV] = str(cpus)
    return env


def prepend_pythonpath(env: dict[str, str], path: str | None) -> dict[str, str]:
    """`env` with `path` prepended to PYTHONPATH (no-op when path is None).
    Carries the interface shim (`runtime.ensure_interface_shim`) into verifier
    runs so `from hillclimb import spaces` resolves inside the runtime venvs."""
    if path:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{path}{os.pathsep}{existing}" if existing else path
    return env


def scrubbed_env(cpus: int = 1, **extra: str) -> dict[str, str]:
    """Parent env minus credentials, capped at `cpus` cores, for running
    coding-agent-authored code."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in SECRET_ENV_EXACT and not k.upper().endswith(SECRET_ENV_SUFFIXES)
    }
    env.update(extra)
    return single_threaded(env, cpus)


class ExecResult(BaseModel):
    returncode: int | None = None
    duration_s: float = 0.0
    # CPU seconds (user+system) the verifier process actually burned — the
    # cost signal duration_s only approximates (wall-clock counts I/O waits).
    # None where the platform can't report it (no os.wait4).
    cpu_s: float | None = None
    # the CPU allotment the run was given ($HILLCLIMB_CPUS); None from an
    # executor that sets none
    cpus: int | None = None
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
    # why the run has no score when it exited 0 (see result_problem)
    result_error: str | None = None

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
    contract violation, not as the worst possible result. Nor is infinity: it
    would win (or lose) every comparison and cannot be journaled as JSON, so
    a broken verifier must not crown a candidate with it.
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
        return (value if math.isfinite(value) else None), None
    if isinstance(payload, dict):
        score = payload.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return None, payload
        return float(score), payload
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or not math.isfinite(payload):
        return None, None
    return float(payload), None


def result_problem(path: Path) -> str:
    """Why a result file holds no score, in words a verifier's author can act
    on — `read_result` only says that it does not."""
    try:
        text = path.read_text().strip()
    except OSError:
        return "no score written to $HILLCLIMB_RESULT"
    if not text:
        return "$HILLCLIMB_RESULT is empty"
    try:
        payload = json.loads(text)
    except ValueError:
        return f"$HILLCLIMB_RESULT is neither JSON nor a number: {text[:60]!r}"
    if isinstance(payload, dict):
        if "score" not in payload:
            return f"the result has no `score` key (keys: {', '.join(map(str, list(payload)[:6])) or 'none'})"
        payload = payload["score"]
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return f"the score is not a number: {payload!r}"[:120]
    if payload != payload:
        return "the score is NaN"
    if not math.isfinite(payload):
        return f"the score is {payload}: a score must be a finite number"
    return "no score"


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
        if isinstance(value, (int, float)) and math.isfinite(value):
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
        if isinstance(value, (int, float)) and math.isfinite(value):
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
        token.replace("{engine_python}", sys.executable)
        .replace("{python}", str(python))
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
    sandbox: "SandboxPolicy | None" = None,
    end_group: bool = False,
) -> RunResult:
    """Run cmd in its own process group with logs redirected; kill the whole
    group on timeout or abort so stray workers don't linger. With a `sandbox`
    policy the command runs inside the OS sandbox (`harness/sandbox.py`).

    The child is reaped through `procs.Reaper`, so its CPU time (user+system,
    with everything it waited for and — at a kill — the descendants the kill
    orphans) rides along in the result; cpu_s is None where the platform
    cannot say.
    """
    if sandbox is not None:
        from hillclimb.harness.sandbox import launch

        started = launch(list(cmd), sandbox)
        cmd = started.argv
        if started.env:
            env = {**(os.environ if env is None else env), **started.env}
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
            if end_group:
                # what it left running must not outlive it (a solution's
                # straggler could rewrite its output while the scorer reads)
                reaper.kill_group()
            return RunResult(proc.returncode, False, reaper.cpu_s)
        if (abort is not None and abort.is_set()) or time.monotonic() >= deadline:
            reaper.kill_group()
            return RunResult(proc.returncode, True, reaper.cpu_s)


def run_then_score(
    run_argv: list[str],
    score_argv: list[str],
    run_dir: Path,
    timeout_s: float,
    out: IO,
    err: IO,
    *,
    run_env: dict[str, str],
    score_env: dict[str, str],
    result_path: Path,
    run_sandbox: "SandboxPolicy | None",
    score_sandbox: "SandboxPolicy | None",
    time_limit_s: float | None = None,
    abort: "threading.Event | None" = None,
) -> RunResult:
    """A two-step problem's run: the solution, then the scorer, each a
    process in a sandbox of its own (the solution's cannot read the
    problem's private paths; the scorer's can). Whatever the solution leaves
    behind — a result file, a process still running — is gone before the
    scorer starts. The scorer gets the time the solution left over."""
    started = time.monotonic()
    limit = timeout_s if time_limit_s is None else min(timeout_s, time_limit_s)
    ran = run_logged(run_argv, run_dir, limit, out, err, run_env, abort=abort, sandbox=run_sandbox, end_group=True)
    if ran.timed_out or ran.returncode != 0:
        if ran.timed_out and not (abort is not None and abort.is_set()) and limit < timeout_s:
            err.write(f"\nsolution.py did not finish within the time limit of {limit:g}s\n")
            err.flush()
            # its own limit, not the engine's clock: a failed attempt, not a cut-off one
            return RunResult(124, False, ran.cpu_s)
        return ran
    result_path.unlink(missing_ok=True)  # only the scorer reports a score
    left = max(1.0, timeout_s - (time.monotonic() - started))
    scored = run_logged(score_argv, run_dir, left, out, err, score_env, abort=abort, sandbox=score_sandbox)
    cpu = None if ran.cpu_s is None and scored.cpu_s is None else (ran.cpu_s or 0.0) + (scored.cpu_s or 0.0)
    return RunResult(scored.returncode, scored.timed_out, cpu)


def confined(policy: "SandboxPolicy | None", run_dir: Path) -> "SandboxPolicy | None":
    """`policy` for one run in `run_dir`: writable are that dir and the
    candidate it belongs to (a solution may write beside itself)."""
    if policy is None:
        return None
    from hillclimb.harness.sandbox import candidate_root

    return policy.writable(run_dir, candidate_root(run_dir))


class CommandExecutor:
    """Executor-protocol impl: runs the problem's verifier command on the
    validation split, in the candidate (or trial) candidate_dir."""

    def __init__(
        self,
        python: Path,
        argv: list[str],
        env_extra: dict[str, str] | None = None,
        pythonpath: str | None = None,
        sandbox: "SandboxPolicy | None" = None,
        score_argv: list[str] | None = None,
        private: tuple[str, ...] = (),
        holdout_inputs: tuple[str, ...] = (),
        time_limit_s: float | None = None,
        cpus: int = 1,
    ):
        # what every run is confined to; each run adds its candidate's dir
        self.sandbox = sandbox
        # cores each run may use (`concurrency.solution_cpus`)
        self.cpus = cpus
        # a two-step problem: `argv` runs the solution, then `score_argv`
        # scores it. Only the scorer may read `private`; a validation run's
        # output reaches the agents, so neither step reads `holdout_inputs`
        self.score_argv = list(score_argv) if score_argv else None
        self.private = tuple(private)
        self.holdout_inputs = tuple(holdout_inputs)
        self.time_limit_s = time_limit_s
        # absolute() not resolve(): a venv python must be invoked via its
        # symlink path or the interpreter escapes the venv's site-packages
        self.python = python.absolute()
        self.argv = list(argv)
        self.env_extra = dict(env_extra or {})
        # deliberately not part of env_extra: that is the problem's
        # "validation runs only" env, while the interface shim is engine
        # infrastructure applied symmetrically here and on the holdout scorer
        self.pythonpath = pythonpath
        # the engine's stop signal, wired in by the Harness (as it is for
        # agents): a stop or a hard deadline kills a running verifier at once
        # instead of waiting for it to finish
        self.abort: "threading.Event | None" = None

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
        # coding-agent-authored code runs inside the verifier process: credentials are
        # scrubbed at run time (not snapshotted at construction) so a change to
        # the orchestrator's environment can never leak into a later run
        env = scrubbed_env(self.cpus, **self.env_extra)
        params_path = candidate_dir / PARAMS_FILE
        env.update(verifier_env(
            self.python, script, result_path, "validation", seed,
            params=params_path if params_path.exists() else None,
        ))
        prepend_pythonpath(env, self.pythonpath)
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            policy = confined(self.sandbox, candidate_dir)
            if self.score_argv:
                returncode, timed_out, cpu_s = run_then_score(
                    render_argv(self.argv, self.python, script, result_path),
                    render_argv(self.score_argv, self.python, script, result_path),
                    candidate_dir, timeout_s, out, err,
                    run_env=env, score_env=env, result_path=result_path,
                    run_sandbox=policy.unreadable(*self.private, *self.holdout_inputs) if policy is not None else None,
                    score_sandbox=policy.unreadable(*self.holdout_inputs) if policy is not None else None,
                    time_limit_s=self.time_limit_s, abort=self.abort,
                )
            else:
                returncode, timed_out, cpu_s = run_logged(
                    render_argv(self.argv, self.python, script, result_path),
                    candidate_dir, timeout_s, out, err, env,
                    abort=self.abort,
                    sandbox=policy,
                )
        duration = time.monotonic() - start
        score, payload = (None, None) if timed_out else read_result(result_path)
        return ExecResult(
            returncode=returncode,
            duration_s=duration,
            cpu_s=cpu_s,
            cpus=self.cpus,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_score=score,
            # completion proof: the verifier writes it only after scoring
            submission_ok=score is not None,
            metrics=result_metrics(payload),
            instance_scores=result_instances(payload),
            result_error=None if timed_out or score is not None else result_problem(result_path),
        )


class CommandHoldoutScorer:
    """Hidden holdout scoring: runs the problem's holdout command against a
    copy of the candidate's solution in a dir coding agents never see, with the full
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
        sandbox: "SandboxPolicy | None" = None,
        score_argv: list[str] | None = None,
        private: tuple[str, ...] = (),
        holdout_inputs: tuple[str, ...] = (),  # readable here: this is the holdout split
        time_limit_s: float | None = None,
        cpus: int = 1,
    ):
        self.sandbox = sandbox
        self.cpus = cpus  # the same allotment as validation runs (CommandExecutor)
        self.python = python.absolute()
        self.argv = list(argv)
        # a two-step problem (see CommandExecutor); its scorer is told `--holdout`
        self.score_argv = list(score_argv) + ["--holdout"] if score_argv else None
        self.private = tuple(private)
        self.time_limit_s = time_limit_s
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
        env = single_threaded(dict(os.environ), self.cpus)  # full env: credentials flow
        env.update(verifier_env(
            self.python, eval_dir / "solution.py", result_path, "holdout",
            params=params_path if params_doc is not None else None,
        ))
        prepend_pythonpath(env, self.pythonpath)
        stdout_path = eval_dir / "exec_stdout.log"
        stderr_path = eval_dir / "exec_stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            policy = self.sandbox.writable(eval_dir) if self.sandbox is not None else None
            if self.score_argv:
                # the solution gets the scrubbed environment; only the scorer
                # gets credentials (gated holdout data is the scorer's to fetch)
                run_env = scrubbed_env(self.cpus)
                run_env.update({
                    key: value for key, value in env.items() if key.startswith("HILLCLIMB_") or key == "PYTHONPATH"
                })
                returncode, timed_out, cpu_s = run_then_score(
                    render_argv(self.argv, self.python, eval_dir / "solution.py", result_path),
                    render_argv(self.score_argv, self.python, eval_dir / "solution.py", result_path),
                    eval_dir, self.timeout_s, out, err,
                    run_env=run_env, score_env=env, result_path=result_path,
                    run_sandbox=policy.unreadable(*self.private) if policy is not None else None,
                    score_sandbox=policy,
                    time_limit_s=self.time_limit_s,
                )
            else:
                returncode, timed_out, cpu_s = run_logged(
                    render_argv(self.argv, self.python, eval_dir / "solution.py", result_path),
                    eval_dir, self.timeout_s, out, err, env,
                    sandbox=policy,
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
