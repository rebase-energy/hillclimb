"""The one seam the tests reach the catalog climbers through.

hillclimb ships no climber as engine code: `greedy`, `openevolve` and `gepa`
are catalog folders (`<repo>/climbers/<name>/`), read in place from a
checkout. A test that needs "the" climber pins the catalog file on its
config (`pin`), one that needs the classes takes them from `module(name)`,
one that needs a ref writes `class_ref(name, "Greedy")` — never a registry
name. `isinstance` holds only across classes from ONE load chain
(`module()` / `climber()` of the same name); across a snapshot or a copy,
compare `type(x).__name__`.
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path
from types import ModuleType

from hillclimb import catalog
from hillclimb.climber import Climber
from hillclimb.modules.spec import ClimberSpec

ROOT: Path = catalog.root()  # this checkout
GREEDY: Path = catalog.climber_path("greedy") / catalog.CLIMBER_ENTRY
OPENEVOLVE: Path = catalog.climber_path("openevolve") / catalog.CLIMBER_ENTRY
GEPA: Path = catalog.climber_path("gepa") / catalog.CLIMBER_ENTRY
META_BASELINE: Path = ROOT / "problems" / "meta-heilbronn" / "greedy.py"


def path(name: str) -> Path:
    return catalog.climber_path(name) / catalog.CLIMBER_ENTRY


def class_ref(name: str, cls: str) -> str:
    """How a block names a class of a catalog climber: `<file>:<Class>`."""
    return f"{path(name)}:{cls}"


def block(name: str) -> dict:
    """The block the catalog climber's file builds, for `==` assertions."""
    return ClimberSpec.model_validate(str(path(name))).block()


def pinned(name: str = "greedy", **overrides):
    """A `Config(**overrides)` with the catalog climber pinned: what a test
    that builds its own config instead of taking the fixture needs."""
    from hillclimb.config import Config

    config = Config(**overrides)
    pin(config, name)
    return config


def pin(config, name: str = "greedy") -> None:
    """Make the catalog climber the config's — what `conftest.config` does,
    so a test that names no climber runs greedy as before."""
    config.climber = ClimberSpec.model_validate(str(path(name)))


@functools.lru_cache
def climber(name: str) -> Climber:
    return catalog.climber(name)


def module(name: str) -> ModuleType:
    """The climber's classes, from the catalog file (one module per process)."""
    return catalog.module(name)


def greedy_classes() -> tuple[type, type]:
    """(Greedy, Best)."""
    greedy = module("greedy")
    return greedy.Greedy, greedy.Best


def openevolve_classes() -> tuple[type, type]:
    """(Greedy, MapElites) of the openevolve climber."""
    openevolve = module("openevolve")
    return openevolve.Greedy, openevolve.MapElites


def gepa_module(part: str = "loop") -> ModuleType:
    """One module of the gepa climber (`loop`, `operator`, `proposer`,
    `config`, `evaluator`, `driver`), from the climber's own scope — the
    package its `loop.py` and siblings are imported as (never the one
    `composed_block` imports `policy.py` into)."""
    gepa = climber("gepa")
    gepa.brain  # noqa: B018 — imports the scope
    return importlib.import_module(f"{gepa.scope.package}.{part}")


def gepa_loop(config, *, driver=None, log=print):
    """What `build_gepa_loop(config)` did when gepa was engine code: the loop
    from the block's params, gated as the engine gates it."""
    GepaLoop = gepa_module("loop").GepaLoop
    return GepaLoop(config.climber.params or {}, driver=driver, log=log, parallelism=config.concurrency.parallel_agents)
