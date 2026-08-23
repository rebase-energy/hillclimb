from __future__ import annotations

import os
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


def allocate_search_dir(run_dir: Path, problem_id: str) -> Path:
    """A fresh search dir for a search on `problem_id` inside `run_dir`.

    The problem is an attribute of the search, not its name: the first
    search on a problem in a run is `<problem-id>` (so refs from before
    suffixes stay valid), the next are `<problem-id>-2`, `-3`, ... The claim
    is an atomic mkdir, so engines started in parallel for one run (a demo,
    a suite with a problem listed twice) never share a dir."""
    root = run_dir / SEARCHES_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        search_id = problem_id if index == 1 else f"{problem_id}-{index}"
        try:
            os.mkdir(root / search_id)
        except FileExistsError:
            index += 1
            continue
        return create_search_dir(run_dir, search_id)


def create_trial_dir(candidate_dir: Path, index: int) -> Path:
    """Per-trial working directory under a candidate dir (n_trials > 1):
    same data/problem symlinks, own copies of the solution and ensemble inputs
    so parallel trials can't collide on artifacts."""
    trial_dir = candidate_dir / "trials" / f"t{index}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    for link_name in ("data", "problem"):
        source = candidate_dir / link_name
        link = trial_dir / link_name
        if source.exists() and not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
    for script in ["solution.py", *(p.name for p in candidate_dir.glob("candidate_*.py"))]:
        source = candidate_dir / script
        if source.exists():
            shutil.copy(source, trial_dir / script)
    return trial_dir


def create_candidate_dir(
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
    candidate_dir = search_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    link = candidate_dir / "data"
    if not link.exists():
        link.symlink_to(data_dir.resolve(), target_is_directory=True)
    problem_link = candidate_dir / "problem"
    if not problem_link.exists():
        problem_link.symlink_to(problem_dir.resolve(), target_is_directory=True)
    if parent_solution and parent_solution.exists():
        shutil.copy(parent_solution, candidate_dir / "solution.py")
    return candidate_dir
