from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from hillclimb.harness.oscompat import link_dir
from hillclimb.harness.run import SEARCHES_DIRNAME


def create_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_search_dir(run_dir: Path, search_id: str) -> Path:
    search_dir = run_dir / SEARCHES_DIRNAME / search_id
    (search_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (search_dir / "best").mkdir(parents=True, exist_ok=True)
    return search_dir


def allocate_search_dir(run_dir: Path, problem_id: str) -> Path:
    """A fresh search dir for a search on `problem_id` inside `run_dir`.

    The problem is an attribute of the search, not its name: the first
    search on a problem in a run is `<problem-id>` (so refs from before
    suffixes stay valid), the next are `<problem-id>-2`, `-3`, ... The claim
    is an atomic mkdir, so engines started in parallel for one run (`--parallel-searches`,
    a suite with a problem listed twice) never share a dir."""
    root = run_dir / SEARCHES_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        search_id = problem_id if index == 1 else f"{problem_id}-{index}"
        try:
            os.mkdir(root / search_id)
        except FileExistsError:
            index += 1
            continue
        return create_search_dir(run_dir, search_id)


PARAMS_FILE = "params.json"
_SCRIPT_GLOBS = ("solution.py", "candidate_*.py")


def _link_and_copy(source_dir: Path, target_dir: Path, extra: tuple[str, ...] = ()) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for link_name in ("data", "problem"):
        source = source_dir / link_name
        link = target_dir / link_name
        if source.exists() and not link.exists():
            link_dir(link, source.resolve())
    names = ["solution.py", *(p.name for p in source_dir.glob("candidate_*.py")), *extra]
    for name in names:
        source = source_dir / name
        if source.exists():
            shutil.copy(source, target_dir / name)


def trial_dir(candidate_dir: Path, index: int) -> Path:
    return candidate_dir / "trials" / f"t{index}"


def replicate_dir(trial_dir_: Path, index: int) -> Path:
    return trial_dir_ / "replicates" / f"r{index}"


def create_trial_dir(candidate_dir: Path, index: int, params_doc: dict | None = None) -> Path:
    """Per-trial working directory (one parameter set) under a candidate dir:
    same data/problem symlinks, own copies of the solution and ensemble
    inputs. `params_doc` — the candidate's declared parameter space with this
    trial's `value` per entry — is written as params.json when given; a
    candidate without a declaration gets no file (the runtime helper then
    falls back to the solution's own defaults)."""
    tdir = trial_dir(candidate_dir, index)
    _link_and_copy(candidate_dir, tdir)
    if params_doc is not None:
        (tdir / PARAMS_FILE).write_text(json.dumps(params_doc, indent=2, sort_keys=True) + "\n")
    return tdir


def create_replicate_dir(trial_dir_: Path, index: int) -> Path:
    """Per-replicate working directory (one seeded execution) under a trial
    dir — the executor's cwd — with its own copies of the scripts and
    params.json so parallel replicates can't collide on artifacts."""
    rdir = replicate_dir(trial_dir_, index)
    _link_and_copy(trial_dir_, rdir, extra=(PARAMS_FILE,))
    return rdir


def hoist_replicate(candidate_dir: Path, source_dir: Path, names) -> None:
    """Surface one replicate's outputs at the candidate-dir root so
    selection, pruning, summit and engine-specific consumers all see the
    same declared artifact set. eval_result.json and the exec logs are
    evaluator infrastructure, not shippable artifacts, but are hoisted for
    report reading and debugging."""
    for name in [*names, "eval_result.json", "exec_stdout.log", "exec_stderr.log"]:
        if (source_dir / name).exists():
            shutil.copy(source_dir / name, candidate_dir / name)


def create_candidate_dir(
    search_dir: Path,
    candidate_id: str,
    data_dir: Path,
    problem_dir: Path,
    parent_solution: Path | None = None,
    unit_tests_dir: Path | None = None,
) -> Path:
    """Per-candidate working directory.

    `problem` symlinks to the problem definition folder (verifier, sample, docs).
    `data` symlinks to the runtime data view. For simple verifier-only problems
    these may point at the same directory.
    """
    candidate_dir = search_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    link = candidate_dir / "data"
    if not link.exists():
        link_dir(link, data_dir.resolve())
    problem_link = candidate_dir / "problem"
    if not problem_link.exists():
        link_dir(problem_link, problem_dir.resolve())
    if parent_solution and parent_solution.exists():
        shutil.copy(parent_solution, candidate_dir / "solution.py")
    if unit_tests_dir is not None:
        from hillclimb.harness.unit_tests import copy_visible_root

        copy_visible_root(unit_tests_dir, candidate_dir)
    return candidate_dir
