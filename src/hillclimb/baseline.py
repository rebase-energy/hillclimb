from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.candidate import Candidate, utcnow
from hillclimb.problem import ProblemSpec


def write_baseline(problem: ProblemSpec, search_dir: Path) -> Candidate:
    """t=0 safety net: the sample submission is valid by construction, so the
    search always has *something* gradeable in best/ even if every agent
    candidate fails. Never selected by greedy search (it carries no trials,
    hence no val_score)."""
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
