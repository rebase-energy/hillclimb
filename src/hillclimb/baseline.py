from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.node import Node, utcnow
from hillclimb.task import TaskSpec


def write_baseline(task: TaskSpec, run_dir: Path) -> Node:
    """t=0 safety net: the sample submission is valid by construction, so the
    run always has *something* gradeable in best/ even if every agent node
    fails. Never selected by greedy search (it carries no val_score)."""
    workspace = run_dir / "nodes" / "n000"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(task.sample_submission, workspace / "submission.csv")
    shutil.copy(task.sample_submission, run_dir / "best" / "submission.csv")
    return Node(
        node_id="n000",
        operator="baseline",
        status="ok",
        workspace=str(workspace),
        summary="baseline: copy of sample_submission.csv",
        finished_at=utcnow(),
    )
