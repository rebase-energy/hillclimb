from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from hillclimb.config import Config
from hillclimb.holdout import HoldoutOverride


class ProblemSpec(BaseModel):
    """Portable verifier-first problem definition.

    A problem is a directory with `problem.yaml`, `description.md`, a baseline
    `sample_submission.csv`, and a verifier script. The generated solution
    writes `submission.csv`; the orchestrator runs the verifier and uses the
    final `val_score:` line as the validation score.
    """

    problem_id: str
    problem_dir: Path
    data_dir: Path
    description: str
    metric_name: str
    lower_is_better: bool
    sample_submission: Path
    verifier: Path | None = None
    time_budget_s: int
    holdout: HoldoutOverride | None = None
    allow_network: bool = False


class SuiteSpec(BaseModel):
    suite_id: str
    suite_path: Path
    problems: list[str]


@dataclass(frozen=True)
class ResolvedTarget:
    kind: Literal["problem", "suite"]
    problem: ProblemSpec | None = None
    suite: SuiteSpec | None = None


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _candidate_paths(target: str, config: Config) -> list[Path]:
    raw = Path(target)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(config.paths.problems_dir / raw)
        if raw.suffix != ".yaml":
            candidates.append(config.paths.problems_dir / f"{target}.yaml")
    return candidates


def resolve_problem_yaml(target: str | Path, config: Config) -> Path:
    """Resolve a problem target to its `problem.yaml`.

    Accepted forms:
    - `problems/circle-packing`
    - `problems/circle-packing/problem.yaml`
    - `circle-packing` under `config.paths.problems_dir`
    """
    for candidate in _candidate_paths(str(target), config):
        if candidate.is_dir() and (candidate / "problem.yaml").exists():
            return (candidate / "problem.yaml").resolve()
        if candidate.is_file() and candidate.name == "problem.yaml":
            return candidate.resolve()
    raise FileNotFoundError(f"No problem found for target {target!r}")


def load_problem(target: str | Path, config: Config) -> ProblemSpec:
    problem_yaml = resolve_problem_yaml(target, config)
    problem_dir = problem_yaml.parent
    meta = _read_yaml(problem_yaml)

    problem_id = meta.get("problem_id") or problem_dir.name
    description_path = problem_dir / meta.get("description", "description.md")
    sample_path = problem_dir / meta.get("sample_submission", "sample_submission.csv")
    verifier_value = meta.get("verifier", "verify.py")
    verifier_path = (problem_dir / verifier_value).resolve() if verifier_value else None
    if verifier_path is not None and not verifier_path.exists():
        raise FileNotFoundError(f"Verifier not found: {verifier_path}")
    if not description_path.exists():
        raise FileNotFoundError(f"Description not found: {description_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_path}")

    if meta.get("data_dir"):
        data_dir = (problem_dir / meta["data_dir"]).resolve()
    elif (problem_dir / "data").exists():
        data_dir = (problem_dir / "data").resolve()
    else:
        data_dir = problem_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Problem data directory not found: {data_dir}")

    return ProblemSpec(
        problem_id=problem_id,
        problem_dir=problem_dir.resolve(),
        data_dir=data_dir,
        description=description_path.read_text(),
        metric_name=meta["metric"],
        lower_is_better=bool(meta["lower_is_better"]),
        sample_submission=sample_path.resolve(),
        verifier=verifier_path,
        time_budget_s=meta.get("time_budget_s", config.budget.total_s),
        holdout=HoldoutOverride.model_validate(meta["holdout"]) if meta.get("holdout") else None,
        allow_network=bool(meta.get("allow_network", False)),
    )


def resolve_suite_yaml(target: str | Path, config: Config) -> Path:
    for candidate in _candidate_paths(str(target), config):
        if candidate.is_file() and candidate.suffix in {".yaml", ".yml"}:
            meta = _read_yaml(candidate)
            if "problems" in meta:
                return candidate.resolve()
    raise FileNotFoundError(f"No suite found for target {target!r}")


def load_suite(target: str | Path, config: Config) -> SuiteSpec:
    suite_yaml = resolve_suite_yaml(target, config)
    meta = _read_yaml(suite_yaml)
    problems = meta.get("problems")
    if not isinstance(problems, list) or not problems:
        raise ValueError(f"{suite_yaml} must define a non-empty `problems` list")
    if not all(isinstance(problem, str) and problem.strip() for problem in problems):
        raise ValueError(f"{suite_yaml} `problems` entries must be non-empty strings")
    return SuiteSpec(
        suite_id=meta.get("suite_id") or suite_yaml.stem,
        suite_path=suite_yaml,
        problems=problems,
    )


def suite_problem_targets(suite: SuiteSpec, config: Config) -> list[str]:
    """Resolve suite entries relative to the suite file when possible."""
    targets = []
    for problem in suite.problems:
        raw = Path(problem)
        if raw.is_absolute() or raw.exists():
            targets.append(str(raw))
            continue
        sibling = suite.suite_path.parent / raw
        if sibling.exists():
            targets.append(str(sibling))
            continue
        configured = config.paths.problems_dir / raw
        if configured.exists():
            targets.append(str(configured))
            continue
        targets.append(problem)
    return targets


def resolve_target(target: str | Path, config: Config) -> ResolvedTarget:
    """Resolve a `hillclimb run <target>` argument as a problem or suite."""
    try:
        suite = load_suite(target, config)
    except FileNotFoundError:
        suite = None
    if suite is not None:
        return ResolvedTarget(kind="suite", suite=suite)
    return ResolvedTarget(kind="problem", problem=load_problem(target, config))
