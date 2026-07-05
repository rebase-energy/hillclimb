"""Evaluator-driven execution for emflow problems.

The candidate's solution.py is a Predictor module — it is never executed
directly; eval_runner.py (running in the emflow runtime venv) loads it,
fits it on the official training view, and scores it on the requested split.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from hillclimb.executor import ExecResult, parse_val_score, run_logged, scrubbed_env

EVAL_RUNNER = Path(__file__).parent / "eval_runner.py"
RESULT_JSON = "eval_result.json"


class EmflowLocalExecutor:
    """Executor-protocol impl: runs eval_runner.py against the candidate's
    solution.py on the validation split. The `verifier` param of the protocol
    is ignored — scoring is the evaluator's job."""

    def __init__(self, python: Path, problem_name: str, allow_network: bool = False):
        self.python = python.absolute()
        self.problem_name = problem_name
        # cache pre-warmed at resolve time; offline keeps agent-side evals
        # hermetic (and no ambient HF credentials exist either way)
        self.env = scrubbed_env(**({} if allow_network else {"HF_HUB_OFFLINE": "1"}))

    def execute(
        self,
        script: Path,
        workspace: Path,
        timeout_s: int,
        verifier: Path | None = None,
    ) -> ExecResult:
        script = script.absolute()
        workspace = workspace.absolute()
        stdout_path = workspace / "exec_stdout.log"
        stderr_path = workspace / "exec_stderr.log"
        result_json = workspace / RESULT_JSON
        result_json.unlink(missing_ok=True)
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                [
                    str(self.python), str(EVAL_RUNNER), str(script),
                    "--problem", self.problem_name,
                    "--split", "validation",
                    "--result-json", str(result_json),
                ],
                workspace, timeout_s, out, err, self.env,
            )
        duration = time.monotonic() - start
        return ExecResult(
            returncode=returncode,
            duration_s=duration,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_score=None if timed_out else parse_val_score(stdout_path.read_text()),
            # completion proof: the evaluator writes it only after scoring
            submission_ok=result_json.exists() and not timed_out,
        )


class EmflowHoldoutScorer:
    """Hidden holdout scoring: re-evaluates the candidate's solution.py on the
    holdout split in a dir agents never see, with credentials intact (the
    private holdout data may live in a gated rb:// repo)."""

    def __init__(self, python: Path, problem_name: str, work_root: Path, timeout_s: int):
        self.python = python.absolute()
        self.problem_name = problem_name
        self.work_root = work_root
        self.timeout_s = timeout_s

    def score(self, workspace: Path) -> tuple[float | None, str | None]:
        solution = workspace / "solution.py"
        if not solution.exists():
            return None, "solution.py missing at holdout time"
        eval_dir = self.work_root / workspace.name
        eval_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(solution, eval_dir / "solution.py")
        # ensemble candidates import candidate_N modules from their workspace
        for extra in workspace.glob("candidate_*.py"):
            shutil.copy(extra, eval_dir / extra.name)
        stdout_path = eval_dir / "exec_stdout.log"
        stderr_path = eval_dir / "exec_stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                [
                    str(self.python), str(EVAL_RUNNER), str(eval_dir / "solution.py"),
                    "--problem", self.problem_name,
                    "--split", "holdout",
                    "--result-json", str(eval_dir / RESULT_JSON),
                ],
                eval_dir, self.timeout_s, out, err, env=None,  # full env: token flows
            )
        if timed_out:
            return None, "holdout evaluation timed out"
        if returncode != 0:
            tail = stderr_path.read_text(errors="replace")[-300:].strip()
            return None, f"holdout evaluation failed: {tail or f'exit {returncode}'}"
        score = parse_val_score(stdout_path.read_text())
        if score is None:
            return None, "holdout evaluation produced no score"
        return score, None
