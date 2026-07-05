"""Runtime-venv requirements, shipped as package data so they survive
installation into site-packages (no repo-root assumptions)."""

from __future__ import annotations

from importlib import resources

KINDS = ("csv", "emflow")


def requirements_resource(kind: str = "csv"):
    """Traversable for the kind's requirements file (use resources.as_file
    to get a real path for subprocesses)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown runtime kind {kind!r}; expected one of {KINDS}")
    return resources.files(__package__) / f"requirements-{kind}.txt"


def runtime_packages(kind: str = "csv") -> list[str]:
    """Package names listed for the kind (comment lines stripped), for the
    prompt's 'Available packages' line."""
    text = requirements_resource(kind).read_text()
    pkgs = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if kind == "emflow":
        pkgs.append("emflow")  # installed separately from the configured source
    return pkgs
