"""The catalog: the problems and climbers hillclimb ships as EXAMPLES, never
as engine code.

The engine defines neither a problem nor a climber — those are the user's.
What it ships is a catalog to copy from: the repository's own `problems/` and
`climbers/` folders, bundled into the wheel as package data
(`hillclimb/_catalog/`, by `hatch_build.py`) and found here. `hillclimb
problem get <id>` and `hillclimb climber get <name>` copy an entry into the
user's hillclimb dir; nothing in the package imports from it.

`root()` is the repository when hillclimb runs from a checkout (an editable
install: `problems/` and `climbers/` sit beside `src/`), the bundled copy
otherwise — so a fresh `pip install hillclimb` and a developer's checkout read
the same files, and there is no second copy to keep in sync.

This module keeps stdlib-only imports at the top: the build hook loads it by
path with nothing but hatchling installed, and `modules/refs.py` (stdlib
only) imports it when a recorded name must find its catalog file.
"""

from __future__ import annotations

import shutil
import sys
from importlib import resources
from pathlib import Path

BUNDLED_DIRNAME = "_catalog"  # where hatch_build.py puts the catalog inside the wheel
PROBLEMS_DIRNAME = "problems"
CLIMBERS_DIRNAME = "climbers"
CLIMBER_ENTRY = "policy.py"  # the file that makes a catalog folder a climber
CLIMBER_REQUIREMENTS = "requirements.txt"  # what a climber imports beyond hillclimb, if anything

DEMO_PROBLEM_ID = "circle-packing"

# The example problems, in ladder order: construction problems in one shape —
# a small CSV of numbers, an exact verifier, no data, no holdout, no noise —
# each family from the instance a ten-minute run climbs to the one an hour
# does not saturate. Every family is stamped by a generator in problems/
# (`make_<family>.py`). circle-packing has a lean runtime so a first run
# starts in seconds.
STARTER_PROBLEM_IDS = (
    DEMO_PROBLEM_ID, "circle-packing-32",
    "heilbronn-11", "heilbronn-14", "heilbronn-17", "heilbronn-convex-13",
    "labs-40", "labs-60",
    "tammes-30", "tammes-50",
    "thomson-50", "thomson-100",
    "autocorr-1", "autocorr-3", "erdos-overlap",
    "kissing-11",
    "golomb-20", "golomb-27",
    "tsp-200",
    "mknap-100-5", "mknap-250-10",
)
# shipped beyond the example set: a data + holdout example, and the terrain
# the scripted `toy` agent walks (the Python SDK's examples run on it). What is
# NOT listed never leaves the repository: meta-heilbronn, bin-packing, the
# generators, a user's own problems in a checkout.
PROBLEM_IDS = (*STARTER_PROBLEM_IDS, "knapsack", "fitness-landscape")

# names a search recorded before 0.9, when the bundled climbers were engine
# code the registry named: (kind, name) -> (catalog folder, file, class). Only a
# RECORD (a snapshot, a run folder) resolves them; a new config is told to
# fetch the climber instead. `RECORDED_MODULES` are the same classes by the
# module paths 0.6–0.8 wrote (after `_moved.modernize`), `RECORDED_PRESETS` the
# blocks the preset names stood for, for v2/v3 records and 0.5 manifests.
RECORDED = {
    ("operator_policy", "greedy"): ("greedy", "policy.py", "Greedy"),
    ("selector_policy", "best"): ("greedy", "policy.py", "Best"),
    ("operator_policy", "openevolve"): ("openevolve", "policy.py", "Greedy"),
    ("selector_policy", "map-elites"): ("openevolve", "policy.py", "MapElites"),
    ("loop", "gepa"): ("gepa", "loop.py", "GepaLoop"),
}
RECORDED_MODULES = {
    "hillclimb.climbers.greedy.policy:Greedy": ("greedy", "policy.py", "Greedy"),
    "hillclimb.climbers.greedy.policy:Best": ("greedy", "policy.py", "Best"),
    "hillclimb.climbers.openevolve.policy:Greedy": ("openevolve", "policy.py", "Greedy"),
    "hillclimb.climbers.openevolve.policy:MapElites": ("openevolve", "policy.py", "MapElites"),
    "hillclimb.climbers.gepa.loop:GepaLoop": ("gepa", "loop.py", "GepaLoop"),
    "hillclimb.climbers.gepa.operator:GepaReflect": ("gepa", "operator.py", "GepaReflect"),
}
RECORDED_PRESETS = {
    "greedy": {"operator_policy": "greedy"},
    "openevolve": {"name": "openevolve", "operator_policy": "openevolve", "selector_policy": "map-elites"},
    "gepa": {"loop": "gepa"},
}


class CatalogUnavailable(RuntimeError):
    """The catalog's files cannot be read as paths (an install that keeps
    the package in an archive)."""


def root() -> Path:
    """The folder holding `problems/` and `climbers/`: the bundled copy in an
    installed wheel, else this checkout."""
    bundled = resources.files(__package__) / BUNDLED_DIRNAME
    if bundled.is_dir():
        if not isinstance(bundled, Path):
            raise CatalogUnavailable(
                f"hillclimb's catalog is not on disk ({bundled!r}); install hillclimb as files, not an archive"
            )
        return bundled
    return Path(__file__).resolve().parents[2]


def problems_dir() -> Path:
    return root() / PROBLEMS_DIRNAME


def climbers_dir() -> Path:
    return root() / CLIMBERS_DIRNAME


# --- problems ---


def problem_ids() -> tuple[str, ...]:
    return PROBLEM_IDS


def problem_path(problem_id: str) -> Path:
    """The catalog folder of one problem."""
    if problem_id not in PROBLEM_IDS:
        raise ValueError(f"no catalog problem {problem_id!r} (available: {', '.join(PROBLEM_IDS)})")
    return problems_dir() / problem_id


def install_problem(
    problems_dir_: Path,
    problem_id: str = DEMO_PROBLEM_ID,
    *,
    windows: bool | None = None,
) -> tuple[Path, bool]:
    """Copy a catalog problem into `problems_dir_` unless it is already
    there. Returns (problem_dir, created). An existing folder is never
    touched — the user may have edited it.

    The verifier is written for the machine that gets it: verifier.sh on
    macOS/Linux, verifier.py on Windows (the problem's own, else the shared
    `scaffold.WINDOWS_VERIFIER`), so a Windows user never needs bash.
    `windows` defaults to this machine."""
    import yaml

    from hillclimb.scaffold import windows_verifier_path

    if windows is None:
        windows = sys.platform == "win32"
    # Validate before inspecting the destination: callers should never be able
    # to turn an arbitrary id into a filesystem copy target through this API.
    source = problem_path(problem_id)
    target = problems_dir_ / problem_id
    if (target / "problem.yaml").exists():
        return target, False
    problems_dir_.mkdir(parents=True, exist_ok=True)
    skip = "verifier.sh" if windows else "verifier.py"
    # a two-step problem (`score:` in problem.yaml) has no verifier to pick
    two_step = "score" in (yaml.safe_load((source / "problem.yaml").read_text()) or {})
    shutil.copytree(
        source, target, dirs_exist_ok=True,
        ignore=lambda directory, names: (
            ({skip} & set(names) if Path(directory) == source else set()) | ({"__pycache__"} & set(names))
        ),
    )
    if two_step:
        return target, True
    if windows:
        if not (target / "verifier.py").exists():
            (target / "verifier.py").write_bytes(windows_verifier_path().read_bytes())
    else:
        # wheels keep mode bits, but a copy through some installers does not
        (target / "verifier.sh").chmod(0o755)
    return target, True


# --- climbers ---


def climber_names() -> tuple[str, ...]:
    """Every catalog climber: a folder under `climbers/` holding `policy.py`."""
    folder = climbers_dir()
    if not folder.is_dir():
        return ()
    return tuple(sorted(p.name for p in folder.iterdir() if (p / CLIMBER_ENTRY).is_file()))


def climber_path(name: str) -> Path:
    """The catalog folder of one climber."""
    if name not in climber_names():
        raise ValueError(f"no catalog climber {name!r} (available: {', '.join(climber_names()) or 'none'})")
    return climbers_dir() / name


def recorded_file(entry: tuple[str, str, str]) -> tuple[Path, str]:
    """(file, class) a recorded name stands for today, from a `RECORDED*` value."""
    folder, file, cls = entry
    return climber_path(folder) / file, cls


def install_climber(climbers_dir_: Path, name: str, *, as_name: str | None = None) -> tuple[Path, bool]:
    """Copy a catalog climber into `climbers_dir_/<as_name or name>/` unless
    that folder exists (never overwritten: the user may have edited it).
    Returns (folder, created). The folder is the catalog's as it is —
    `policy.py` IS the climber — plus, when the catalog folder brought no
    `prompts/` of its own, a `prompts/` holding the templates its operators
    render and a README of what fills them: the built-in templates stay one
    source in the package, and the copy is the climber as it runs."""
    source = climber_path(name)
    target = climbers_dir_ / (as_name or name)
    if target.exists():
        return target, False
    climbers_dir_.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    if as_name and as_name != name:  # the copy is called what it was fetched as: its `Climber(name=…)`
        entry = target / CLIMBER_ENTRY
        entry.write_text(entry.read_text(encoding="utf-8").replace(f"name={name!r},", f"name={as_name!r},", 1), encoding="utf-8")
    if not (target / "prompts").is_dir():
        from hillclimb.climber import load_climber, write_prompts_folder

        write_prompts_folder(load_climber(str(target / CLIMBER_ENTRY)), target / "prompts", as_name or name)
    return target, True


def climber(name: str):
    """A catalog climber as a `Climber`, read in place (the Python SDK's
    `hc.catalog.climber("greedy")`); its own classes are `.module.Greedy`,
    `.module.Best`. A search run from it snapshots the file like any other."""
    from hillclimb.climber import load_climber

    return load_climber(str(climber_path(name) / CLIMBER_ENTRY))


def module(name: str):
    """The module a catalog climber's file is: `hc.catalog.module("greedy").Greedy`
    to subclass or compose with. One module per process (`FileScope` names
    the package by place and bytes), so classes compare by identity."""
    return climber(name).module
