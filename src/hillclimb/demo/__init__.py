"""Bundled problems shipped as package data.

Keeping these inside the wheel makes `hillclimb fetch` and `hillclimb demo`
work from a bare `pip install hillclimb` with no repository checkout.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

DEMO_PROBLEM_ID = "circle-packing"
BUNDLED_PROBLEM_IDS = (DEMO_PROBLEM_ID, "knapsack", "heilbronn-convex-13")


def demo_problem_resource(problem_id: str = DEMO_PROBLEM_ID):
    """Traversable for one of the bundled problem folders."""
    if problem_id not in BUNDLED_PROBLEM_IDS:
        available = ", ".join(BUNDLED_PROBLEM_IDS)
        raise ValueError(f"no bundled problem {problem_id!r} (available: {available})")
    return resources.files(__package__) / problem_id


def install_demo_problem(
    problems_dir: Path,
    problem_id: str = DEMO_PROBLEM_ID,
) -> tuple[Path, bool]:
    """Copy a bundled problem into `problems_dir` unless it is already
    there. Returns (problem_dir, created). An existing folder is never
    touched — the user may have edited it."""
    # Validate before inspecting the destination: callers should never be able
    # to turn an arbitrary id into a filesystem copy target through this API.
    source_resource = demo_problem_resource(problem_id)
    target = problems_dir / problem_id
    if (target / "problem.yaml").exists():
        return target, False
    problems_dir.mkdir(parents=True, exist_ok=True)
    with resources.as_file(source_resource) as source:
        shutil.copytree(source, target, dirs_exist_ok=True)
    # wheels keep mode bits, but a copy through some installers does not
    (target / "verifier.sh").chmod(0o755)
    return target, True
