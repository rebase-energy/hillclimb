"""Command-driven execution for `kind: evaluator` problems.

The problem folder supplies its own evaluation command — the only process
that runs; it drives solution.py itself (import it, exec it, shell out),
prints the final `val_score:` line, and writes `eval_result.json` (the
completion proof and evaluator-report carrier). Generalizes the emflow
executor pair (`integrations/emflow/executor.py`), whose eval command just
happens to be `eval_runner.py --problem <name>`.

Command strings are shlex-split at construction; two placeholders are
substituted per token at run time:

- `{python}`   — the managed runtime venv's python (bare `python` would
                 resolve via PATH inside the scrubbed env: wrong interpreter)
- `{solution}` — the solution path for this run (trial-dir aware)

Relative paths in the command resolve against the run cwd, where
`./problem/` and `./data/` symlinks always exist (candidate workspaces and
trial dirs by construction; the hidden holdout dir recreates them here).
"""

from __future__ import annotations

import shlex
import shutil
import time
from pathlib import Path

from hillclimb.executor import ExecResult, parse_val_score, run_logged, scrubbed_env

RESULT_JSON = "eval_result.json"


def _render_command(tokens: list[str], python: Path, solution: Path) -> list[str]:
    return [
        token.replace("{python}", str(python)).replace("{solution}", str(solution))
        for token in tokens
    ]


class CommandExecutor:
    """Executor-protocol impl: runs the problem's eval command in the
    candidate/trial workspace. The `verifier` param of the protocol is
    ignored — scoring is the eval command's job."""

    def __init__(self, python: Path, command: str):
        self.python = python.absolute()
        self.tokens = shlex.split(command)
        # agent-authored code runs inside the evaluator process: credentials
        # are scrubbed exactly as for direct solution execution (the problem's
        # allow_network flag is prompt-side policy, not an env switch here)
        self.env = scrubbed_env()

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
        stdout_path = workspace / "exec_stdout.log"
        stderr_path = workspace / "exec_stderr.log"
        result_json = workspace / RESULT_JSON
        result_json.unlink(missing_ok=True)  # staleness must never fake success
        env = dict(self.env)
        if seed is not None:
            env["HILLCLIMB_TRIAL_SEED"] = str(seed)
        start = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                _render_command(self.tokens, self.python, script),
                workspace, timeout_s, out, err, env,
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


class CommandHoldoutScorer:
    """Hidden holdout scoring: runs the problem's holdout command against a
    copy of the candidate's solution in a dir agents never see, with the full
    environment (credentials flow). Unlike the emflow scorer — whose runner
    finds the problem in a registry — a generic command needs `./problem/`
    and `./data/`, so they are symlinked into the hidden dir."""

    def __init__(
        self,
        python: Path,
        command: str,
        problem_dir: Path,
        data_dir: Path,
        work_root: Path,
        timeout_s: int,
    ):
        self.python = python.absolute()
        self.tokens = shlex.split(command)
        self.problem_dir = problem_dir
        self.data_dir = data_dir
        self.work_root = work_root
        self.timeout_s = timeout_s

    def score(self, workspace: Path) -> tuple[float | None, str | None]:
        workspace = workspace.absolute()
        solution = workspace / "solution.py"
        if not solution.exists():
            return None, "solution.py missing at holdout time"
        eval_dir = self.work_root.absolute() / workspace.name
        eval_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(solution, eval_dir / "solution.py")
        # ensemble candidates import candidate_N modules from their workspace
        for extra in workspace.glob("candidate_*.py"):
            shutil.copy(extra, eval_dir / extra.name)
        for name, target in (("problem", self.problem_dir), ("data", self.data_dir)):
            link = eval_dir / name
            if not link.exists():
                link.symlink_to(Path(target).resolve(), target_is_directory=True)
        stdout_path = eval_dir / "exec_stdout.log"
        stderr_path = eval_dir / "exec_stderr.log"
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            returncode, timed_out = run_logged(
                _render_command(self.tokens, self.python, eval_dir / "solution.py"),
                eval_dir, self.timeout_s, out, err, env=None,  # full env: credentials flow
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
