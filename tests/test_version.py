"""One version: `hillclimb.__version__` is what `--version` prints and a
search records, and `pyproject.toml` declares no version of its own —
hatchling reads that line at build time (`tests/test_catalog.py` holds the
built wheel's name to it)."""

import tomllib
from pathlib import Path

import hillclimb

REPO = Path(__file__).resolve().parents[1]


def test_pyproject_reads_the_version_from_the_package():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert "version" not in pyproject["project"] and pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/hillclimb/__init__.py"
    parts = hillclimb.__version__.split(".")
    assert len(parts) == 3 and all(part.isdigit() for part in parts)
