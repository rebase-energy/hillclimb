from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

from hillclimb.config import Config
from hillclimb.holdout import HoldoutOverride


class ProblemSpec(BaseModel):
    """Portable problem definition.

    kind="csv" (default): a directory with `problem.yaml`, `description.md`,
    a baseline `sample_submission.csv`, and a verifier script. The generated
    solution writes `submission.csv`; the orchestrator runs the verifier and
    uses the final `val_score:` line as the validation score.

    kind="emflow": an emflow registry problem (`emflow://<name>`); the
    generated solution is a Predictor module driven by the emflow evaluator,
    and holdout scoring is a second evaluator invocation (`holdout_mode=
    "evaluator"`) rather than a hillclimb-built data view.

    kind="evaluator": a directory whose problem.yaml supplies its own eval
    command (`eval:`) — the only process that runs; it drives solution.py
    itself, prints the final `val_score:` line, and writes `eval_result.json`
    (the completion proof and report carrier). Optional `holdout_eval:` runs
    the same way in a hidden dir with the full environment.
    """

    kind: Literal["csv", "emflow", "evaluator"] = "csv"
    problem_id: str
    problem_dir: Path
    data_dir: Path
    description: str
    metric_name: str
    lower_is_better: bool
    sample_submission: Path | None = None  # required for csv (loader enforces)
    verifier: Path | None = None
    time_budget_s: int
    holdout: HoldoutOverride | None = None
    holdout_mode: Literal["data-view", "evaluator"] = "data-view"
    allow_network: bool = False
    emflow_problem: str | None = None   # registry name, e.g. "gefcom2014:solar"
    emflow_baseline: str | None = None  # module exposing get_model(), if any
    emflow_quantiles: list[float] | None = None  # probabilistic problems only
    # evaluator kind: raw command strings (shlex-split at execution;
    # `{python}`/`{solution}` placeholders substituted per token)
    eval_command: str | None = None
    holdout_command: str | None = None
    contract: str | None = None  # problem-authored solution-contract prompt section
    requirements_file: Path | None = None  # per-problem venv requirements
    baseline_solution: Path | None = None  # floor solution scored as c000
    # MLE-bench competition id; set only by the mlebench provider. Marks the
    # search for one official grade-sample run on the selected candidate
    # after the search finishes (never during — selection integrity).
    mlebench_comp_id: str | None = None


class SuiteEntry(BaseModel):
    """One search in a run-spec file. Bare-string entries are shorthand for
    `{target: <string>}`; dict entries carry per-search parameter overrides,
    so a committed spec fully describes a run (git-versionable parameters).
    CLI flags passed alongside the spec override these values."""

    target: str
    name: str | None = None
    model: str | None = None
    backend: str | None = None
    budget: str | None = None  # "2h" / "30m" / seconds — parsed by the CLI
    parallel_agents: int | None = None
    n_trials: int | None = None
    seed_from: str | None = None  # incumbent solution.py, relative to the spec file


class SuiteSpec(BaseModel):
    suite_id: str
    suite_path: Path
    problems: list[SuiteEntry]

    @field_validator("problems", mode="before")
    @classmethod
    def _coerce_entries(cls, value):
        if isinstance(value, list):
            return [{"target": v} if isinstance(v, str) else v for v in value]
        return value


@dataclass(frozen=True)
class ResolvedTarget:
    kind: Literal["problem", "suite"]
    problem: ProblemSpec | None = None
    suite: SuiteSpec | None = None


# Target schemes served by optional problem providers (lazy imports so the
# core has no hard dependency on them).
PROVIDER_SCHEMES = ("emflow", "mlebench")


def _split_scheme(target: str | Path) -> tuple[str, str] | None:
    """(scheme, rest) when the target uses a provider scheme, else None."""
    scheme, sep, rest = str(target).partition("://")
    if sep and scheme in PROVIDER_SCHEMES:
        return scheme, rest
    return None


def _emflow_provider():
    try:
        from hillclimb.integrations.emflow import provider
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "emflow:// targets need the emflow extra: "
            "pip install 'hillclimb[emflow]'"
        ) from exc
    return provider


def _provider_calls(scheme: str):
    """(load_problem, resolve_target) pair for a provider scheme."""
    if scheme == "emflow":
        provider = _emflow_provider()
        return provider.load_emflow_problem, provider.resolve_emflow_target
    from hillclimb.integrations.mlebench import provider

    return provider.load_mlebench_problem, provider.resolve_mlebench_target


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


def _optional_file(problem_dir: Path, meta: dict, key: str, default: str | None = None) -> Path | None:
    """Resolve an optional problem-relative file. An explicitly named file
    must exist; a missing default candidate is simply absent."""
    value = meta.get(key, default)
    if not value:
        return None
    path = (problem_dir / value).resolve()
    if not path.exists():
        if key not in meta:
            return None
        raise FileNotFoundError(f"{key} file not found: {path}")
    return path


def _load_evaluator_problem(meta: dict, problem_yaml: Path, problem_dir: Path, common: dict) -> ProblemSpec:
    """kind: evaluator — the problem supplies its own eval (and optional
    holdout) command; no sample_submission, no verifier, no data-view holdout."""
    if meta.get("holdout"):
        raise ValueError(
            f"{problem_yaml}: `holdout:` (data-view override) does not apply to "
            "kind: evaluator — supply a `holdout_eval:` command instead"
        )
    eval_command = str(meta.get("eval") or "").strip()
    if not eval_command or not shlex.split(eval_command):
        raise ValueError(f"{problem_yaml}: kind: evaluator requires a non-empty `eval:` command")
    holdout_command = str(meta.get("holdout_eval") or "").strip() or None
    if holdout_command:
        shlex.split(holdout_command)  # a malformed command fails at load, not mid-search
    contract_path = _optional_file(problem_dir, meta, "contract", default="contract.md")
    return ProblemSpec(
        kind="evaluator",
        **common,
        eval_command=eval_command,
        holdout_command=holdout_command,
        contract=contract_path.read_text() if contract_path else None,
        requirements_file=_optional_file(problem_dir, meta, "requirements"),
        baseline_solution=_optional_file(problem_dir, meta, "baseline"),
        holdout_mode="evaluator",
    )


def load_problem(target: str | Path, config: Config) -> ProblemSpec:
    scheme = _split_scheme(target)
    if scheme is not None:
        load, _ = _provider_calls(scheme[0])
        return load(scheme[1], config)
    problem_yaml = resolve_problem_yaml(target, config)
    problem_dir = problem_yaml.parent
    meta = _read_yaml(problem_yaml)

    kind = meta.get("kind", "csv")
    problem_id = meta.get("problem_id") or problem_dir.name
    description_path = problem_dir / meta.get("description", "description.md")
    if not description_path.exists():
        raise FileNotFoundError(f"Description not found: {description_path}")

    if meta.get("data_dir"):
        data_dir = (problem_dir / meta["data_dir"]).resolve()
    elif (problem_dir / "data").exists():
        data_dir = (problem_dir / "data").resolve()
    else:
        data_dir = problem_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Problem data directory not found: {data_dir}")

    common = dict(
        problem_id=problem_id,
        problem_dir=problem_dir.resolve(),
        data_dir=data_dir,
        description=description_path.read_text(),
        metric_name=meta["metric"],
        lower_is_better=bool(meta["lower_is_better"]),
        time_budget_s=meta.get("time_budget_s", config.budget.total_s),
        allow_network=bool(meta.get("allow_network", False)),
    )
    if kind == "evaluator":
        return _load_evaluator_problem(meta, problem_yaml, problem_dir, common)
    if kind != "csv":
        raise ValueError(f"{problem_yaml}: unknown problem kind {kind!r}")

    sample_path = problem_dir / meta.get("sample_submission", "sample_submission.csv")
    verifier_value = meta.get("verifier", "verify.py")
    verifier_path = (problem_dir / verifier_value).resolve() if verifier_value else None
    if verifier_path is not None and not verifier_path.exists():
        raise FileNotFoundError(f"Verifier not found: {verifier_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_path}")

    return ProblemSpec(
        **common,
        sample_submission=sample_path.resolve(),
        verifier=verifier_path,
        holdout=HoldoutOverride.model_validate(meta["holdout"]) if meta.get("holdout") else None,
    )


def resolve_suite_yaml(target: str | Path, config: Config) -> Path:
    for candidate in _candidate_paths(str(target), config):
        if candidate.is_file() and candidate.suffix in {".yaml", ".yml"}:
            meta = _read_yaml(candidate)
            if "problems" in meta or "target" in meta:
                return candidate.resolve()
    raise FileNotFoundError(f"No suite found for target {target!r}")


def _parse_entry(raw, suite_yaml: Path) -> SuiteEntry:
    if isinstance(raw, str) and raw.strip():
        return SuiteEntry(target=raw)
    if isinstance(raw, dict):
        return SuiteEntry.model_validate(raw)
    raise ValueError(
        f"{suite_yaml} `problems` entries must be target strings or "
        f"{{target: ..., model: ..., budget: ...}} mappings (got {raw!r})"
    )


def load_suite(target: str | Path, config: Config) -> SuiteSpec:
    """Load a run spec: a `problems:` list (strings or per-entry parameter
    dicts), or the single-search form with a top-level `target:` plus the
    same parameter keys."""
    suite_yaml = resolve_suite_yaml(target, config)
    meta = _read_yaml(suite_yaml)
    if "problems" not in meta:  # single-target spec
        entry_keys = SuiteEntry.model_fields.keys()
        entry = SuiteEntry.model_validate({k: v for k, v in meta.items() if k in entry_keys})
        return SuiteSpec(
            suite_id=meta.get("suite_id") or suite_yaml.stem,
            suite_path=suite_yaml,
            problems=[entry],
        )
    problems = meta.get("problems")
    if not isinstance(problems, list) or not problems:
        raise ValueError(f"{suite_yaml} must define a non-empty `problems` list")
    return SuiteSpec(
        suite_id=meta.get("suite_id") or suite_yaml.stem,
        suite_path=suite_yaml,
        problems=[_parse_entry(raw, suite_yaml) for raw in problems],
    )


def suite_problem_targets(suite: SuiteSpec, config: Config) -> list[str]:
    """Resolve suite entries relative to the suite file when possible."""
    targets = []
    for entry in suite.problems:
        problem = entry.target
        if _split_scheme(problem) is not None:
            targets.append(problem)  # provider targets resolve by name, not path
            continue
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
    scheme = _split_scheme(target)
    if scheme is not None:
        _, resolve = _provider_calls(scheme[0])
        return resolve(scheme[1], config)
    try:
        suite = load_suite(target, config)
    except FileNotFoundError:
        suite = None
    if suite is not None:
        return ResolvedTarget(kind="suite", suite=suite)
    return ResolvedTarget(kind="problem", problem=load_problem(target, config))
