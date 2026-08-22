"""The bundled demo problem, shipped as package data so `hillclimb demo`
works from a bare `pip install hillclimb` with no repo checkout."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

DEMO_PROBLEM_ID = "circle-packing"


def demo_problem_resource():
    """Traversable for the bundled problem folder."""
    return resources.files(__package__) / DEMO_PROBLEM_ID


def install_demo_problem(problems_dir: Path) -> tuple[Path, bool]:
    """Copy the bundled problem into `problems_dir` unless it is already
    there. Returns (problem_dir, created). An existing folder is never
    touched — the user may have edited it."""
    target = problems_dir / DEMO_PROBLEM_ID
    if (target / "problem.yaml").exists():
        return target, False
    problems_dir.mkdir(parents=True, exist_ok=True)
    with resources.as_file(demo_problem_resource()) as source:
        shutil.copytree(source, target, dirs_exist_ok=True)
    # wheels keep mode bits, but a copy through some installers does not
    (target / "verifier.sh").chmod(0o755)
    return target, True
