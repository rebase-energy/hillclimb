"""Ship the catalog in the wheel.

The repository's `problems/` and `climbers/` folders are hillclimb's catalog:
examples the CLI copies out (`hillclimb problem get`, `hillclimb climber get`),
never code the engine imports. They live beside `src/`, so hatch's `packages`
rule does not pick them up; this hook adds them to the wheel as package data
under `hillclimb/_catalog/`, file by file, so `__pycache__` and anything not
in `catalog.PROBLEM_IDS` (the generators, meta-heilbronn, a developer's own
problems) stay out. An editable install gets nothing: `catalog.root()` reads
the checkout then. `tests/test_catalog.py` holds the file list to the catalog.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

try:  # the hook runs under hatchling; tests import the enumeration without it
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # pragma: no cover
    BuildHookInterface = object  # type: ignore[assignment,misc]

CATALOG_MODULE = Path("src") / "hillclimb" / "catalog.py"
SKIP_DIRS = {"__pycache__"}


def _catalog_module(root: Path):
    """`hillclimb.catalog` loaded by path: the build env has hatchling only."""
    spec = importlib.util.spec_from_file_location("_hillclimb_catalog", root / CATALOG_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        if any(part in SKIP_DIRS or part.startswith(".") for part in relative.parts) or path.suffix == ".pyc":
            continue
        yield path, relative


def catalog_files(root: Path) -> dict[str, str]:
    """{absolute source: path inside the wheel} for every catalog file."""
    catalog = _catalog_module(root)
    bundled = f"hillclimb/{catalog.BUNDLED_DIRNAME}"
    included: dict[str, str] = {}
    for problem_id in catalog.PROBLEM_IDS:
        folder = root / catalog.PROBLEMS_DIRNAME / problem_id
        if not folder.is_dir():
            raise FileNotFoundError(f"catalog problem {problem_id!r} is not in {folder.parent}")
        for path, relative in _files(folder):
            included[str(path)] = f"{bundled}/{catalog.PROBLEMS_DIRNAME}/{problem_id}/{relative.as_posix()}"
    climbers = root / catalog.CLIMBERS_DIRNAME
    folders = sorted(p for p in climbers.iterdir() if (p / catalog.CLIMBER_ENTRY).is_file()) if climbers.is_dir() else []
    for folder in folders:
        for path, relative in _files(folder):
            included[str(path)] = f"{bundled}/{catalog.CLIMBERS_DIRNAME}/{folder.name}/{relative.as_posix()}"
    return included


class CatalogHook(BuildHookInterface):  # type: ignore[misc]
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        build_data.setdefault("force_include", {}).update(catalog_files(Path(self.root)))
