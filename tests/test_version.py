"""The one version, in two places: `pyproject.toml` (what pip installs) and
`hillclimb.__version__` (what `--version` prints and a search records)."""

import tomllib
from pathlib import Path

import hillclimb


def test_the_package_version_is_the_one_pyproject_declares():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert hillclimb.__version__ == pyproject["project"]["version"]
