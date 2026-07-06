from __future__ import annotations

import shutil
from pathlib import Path

from hillclimb.run import SEARCHES_DIRNAME


def create_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_search_dir(run_dir: Path, search_id: str) -> Path:
    search_dir = run_dir / SEARCHES_DIRNAME / search_id
    (search_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (search_dir / "best").mkdir(parents=True, exist_ok=True)
    return search_dir


def create_trial_dir(workspace: Path, index: int) -> Path:
    """Per-trial working directory under a candidate workspace (n_trials > 1):
    same data/problem symlinks, own copies of the solution and ensemble inputs
    so parallel trials can't collide on artifacts."""
    trial_dir = workspace / "trials" / f"t{index}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    for link_name in ("data", "problem"):
        source = workspace / link_name
        link = trial_dir / link_name
        if source.exists() and not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
    for script in ["solution.py", *(p.name for p in workspace.glob("candidate_*.py"))]:
        source = workspace / script
        if source.exists():
            shutil.copy(source, trial_dir / script)
    return trial_dir


def create_candidate_workspace(
    search_dir: Path,
    candidate_id: str,
    data_dir: Path,
    problem_dir: Path,
    parent_solution: Path | None = None,
) -> Path:
    """Per-candidate working directory.

    `problem` symlinks to the problem definition folder (verifier, sample, docs).
    `data` symlinks to the runtime data view. For simple verifier-only problems
    these may point at the same directory.
    """
    workspace = search_dir / "candidates" / candidate_id
    workspace.mkdir(parents=True, exist_ok=True)
    link = workspace / "data"
    if not link.exists():
        link.symlink_to(data_dir.resolve(), target_is_directory=True)
    problem_link = workspace / "problem"
    if not problem_link.exists():
        problem_link.symlink_to(problem_dir.resolve(), target_is_directory=True)
    if parent_solution and parent_solution.exists():
        shutil.copy(parent_solution, workspace / "solution.py")
    return workspace
