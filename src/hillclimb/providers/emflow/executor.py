"""One official emflow Verifier run for a finished search.

Ordinary evaluation goes through the generic verifier command the provider
puts on the ProblemSpec (`executor.CommandExecutor`); this is the separate,
once-per-search leaderboard submission.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.harness.executor import RESULT_FILE, read_result, run_logged

EVAL_RUNNER = Path(__file__).parent / "eval_runner.py"


def official_verify(
    python: Path,
    problem_name: str,
    candidate_dir: Path,
    out_dir: Path,
    name: str,
    n_trials: int,
    timeout_s: int,
    log=print,
) -> float | None:
    """One official Verifier run (scorecard + emflow leaderboard row) on a
    candidate candidate_dir's solution.py, with n_trials recorded for selection
    honesty. Returns the holdout score, or None on failure."""
    import json

    candidate_dir = candidate_dir.absolute()
    out_dir = out_dir.absolute()
    solution = candidate_dir / "solution.py"
    if not solution.exists():
        log("official verify skipped: solution.py missing")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(solution, out_dir / "solution.py")
    for extra in candidate_dir.glob("candidate_*.py"):
        shutil.copy(extra, out_dir / extra.name)
    stdout_path = out_dir / "verify_stdout.log"
    stderr_path = out_dir / "verify_stderr.log"
    # cpu_s is discarded: this once-per-search submission produces no Trial
    # record to carry it, so its CPU stays outside the cost accounting.
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        returncode, timed_out, _cpu_s = run_logged(
            [
                str(python.absolute()), str(EVAL_RUNNER), str(out_dir / "solution.py"),
                "--problem", problem_name,
                "--split", "holdout",
                "--verify",
                "--name", name,
                "--metadata-json", json.dumps({"n_trials": n_trials}),
                "--result-json", str(out_dir / RESULT_FILE),
            ],
            out_dir, timeout_s, out, err, env=None,  # full env: token flows
        )
    if timed_out or returncode != 0:
        tail = stderr_path.read_text(errors="replace")[-300:].strip()
        log(f"official verify failed: {'timeout' if timed_out else tail}")
        return None
    stdout_text = stdout_path.read_text()
    # surface the scorecard block in the orchestrator log
    if "=" * 20 in stdout_text:
        log(stdout_text[stdout_text.index("=" * 20):].strip())
    return read_result(out_dir / RESULT_FILE)[0]
