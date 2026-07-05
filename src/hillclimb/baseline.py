from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.candidate import Candidate, utcnow
from hillclimb.problem import ProblemSpec


def write_baseline(
    problem: ProblemSpec,
    search_dir: Path,
    executor=None,
    holdout_scorer=None,
    timeout_s: int = 1800,
) -> Candidate:
    """t=0 safety net / scored floor, by problem kind.

    csv: copy of sample_submission.csv — valid by construction so the search
    always has *something* gradeable in best/, never selected (no trials).

    emflow: the benchmark's reference model (get_model()) evaluated for real,
    so agent drafts must beat it to become best."""
    if problem.kind == "emflow":
        from hillclimb.integrations.emflow.provider import write_emflow_baseline

        return write_emflow_baseline(problem, search_dir, executor, holdout_scorer, timeout_s)

    workspace = search_dir / "candidates" / "c000"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(problem.sample_submission, workspace / "submission.csv")
    shutil.copy(problem.sample_submission, search_dir / "best" / "submission.csv")
    return Candidate(
        candidate_id="c000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
        summary="baseline: copy of sample_submission.csv",
        finished_at=utcnow(),
    )
