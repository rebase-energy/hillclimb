"""Runtime-venv requirements, shipped as package data so they survive
installation into site-packages (no repo-root assumptions)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

# generic verifier for self-reported problems (see run_solution.py)
RUN_SOLUTION = Path(__file__).parent / "run_solution.py"

KINDS = ("csv", "emflow")


def requirements_resource(kind: str = "csv"):
    """Traversable for the kind's requirements file (use resources.as_file
    to get a real path for subprocesses). The evaluator kind runs in the csv
    runtime unless the problem ships its own requirements file."""
    if kind == "evaluator":
        kind = "csv"
    if kind not in KINDS:
        raise ValueError(f"Unknown runtime kind {kind!r}; expected one of {KINDS}")
    return resources.files(__package__) / f"requirements-{kind}.txt"


def _parse_requirements(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def runtime_packages(kind: str = "csv", requirements_file: Path | None = None) -> list[str]:
    """Package names for the prompt's 'Available packages' line: the kind's
    packaged requirement set, or the problem's own requirements file when it
    pins one (per-problem venv)."""
    if requirements_file is not None:
        return _parse_requirements(Path(requirements_file).read_text())
    pkgs = _parse_requirements(requirements_resource(kind).read_text())
    if kind == "emflow":
        pkgs.append("emflow")  # installed separately from the configured source
    return pkgs
