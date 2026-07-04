from __future__ import annotations

import shutil
from pathlib import Path


def create_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "nodes").mkdir(parents=True, exist_ok=True)
    (run_dir / "best").mkdir(parents=True, exist_ok=True)
    return run_dir


def create_node_workspace(
    run_dir: Path,
    node_id: str,
    data_dir: Path,
    problem_dir: Path,
    parent_solution: Path | None = None,
) -> Path:
    """Per-node working directory.

    `problem` symlinks to the problem definition folder (verifier, sample, docs).
    `data` symlinks to the runtime data view. For simple verifier-only problems
    these may point at the same directory.
    """
    workspace = run_dir / "nodes" / node_id
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
