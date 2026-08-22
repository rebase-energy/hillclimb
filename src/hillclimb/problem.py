from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from hillclimb.config import Config
from hillclimb.direction import legacy_direction_key


class ProblemSpec(BaseModel):
    """Portable problem definition.

    A problem is defined by its **verifier command**: the only process the
    engine starts, which drives `solution.py` and writes the score to
    `$HILLCLIMB_RESULT` (see `executor.py` for the contract). A directory
    problem supplies it as `verifier.sh`; providers (`emflow://`,
    `mlebench://`) supply their own argv for the same contract.
    """

    problem_id: str
    problem_dir: Path
    data_dir: Path
    description: str
    metric_name: str
    higher_is_better: bool
    time_budget_s: int
    allow_network: bool = False

    # --- the verifier contract ---
    # argv of the validation command; `{python}`/`{solution}`/`{result}`
    # tokens are substituted per run (shell verifiers read the env instead)
    verifier_cmd: list[str]
    # same contract against the hidden split; None = this problem has no
    # holdout and selection climbs on validation alone
    holdout_cmd: list[str] | None = None
    verifier_env: dict[str, str] = Field(default_factory=dict)  # extra env, validation runs only
    verifier_display: str = "./problem/verifier.sh"  # prompt-facing form of the command
    # True: the verifier computes the score (and any eval_result.json report)
    # independently of the agent. False: the number is the solution's own
    # claim (MLE-bench), so reports are stored labelled self-reported.
    report_trusted: bool = True
    holdout_needs_credentials: bool = False  # fail fast when the hidden split is gated
    runtime: Literal["csv", "emflow"] = "csv"  # which shared runtime venv to build
    requirements_file: Path | None = None  # per-problem venv requirements

    # --- prompt assembly ---
    contract_template: str = "contract_verifier"  # prompts/<name>.md
    contract: str | None = None  # problem-authored solution-contract section

    # --- t=0 floor ---
    baseline_text: str | None = None  # solution.py source scored as c000
    # declared floor (`baseline: 0.5` in problem.yaml): c000 carries this
    # score without running anything — for problems whose trivial solution is
    # obvious (one big circle) but not worth shipping as code
    baseline_score: float | None = None
    baseline_summary: str = "baseline"
    # {name in the candidate_dir: source file} copied into c000 and best/ when the
    # problem has no scored baseline, so a search that never lands a working
    # candidate still ships something gradeable (a sample submission)
    baseline_files: dict[str, Path] = Field(default_factory=dict)

    # --- provider extras ---
    emflow_problem: str | None = None   # registry name, e.g. "gefcom2014:solar"
    emflow_quantiles: list[float] | None = None  # probabilistic problems only
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


def _verifier_argv(problem_yaml: Path, problem_dir: Path, meta: dict) -> list[str]:
    """The problem's verifier, as argv. Absolute: a relative program path is
    resolved against the ENGINE's cwd, not the candidate dir the command runs in
    (subprocess does not search cwd for the executable)."""
    name = str(meta.get("verifier", "verifier.sh"))
    path = (problem_dir / name).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{problem_yaml}: verifier not found: {path} — a problem is defined by its "
            "verifier (see `hillclimb init` for a scaffold)"
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{problem_yaml}: verifier is not executable: chmod +x {path}")
    return [str(path)]


def load_problem(target: str | Path, config: Config) -> ProblemSpec:
    scheme = _split_scheme(target)
    if scheme is not None:
        load, _ = _provider_calls(scheme[0])
        return load(scheme[1], config)
    problem_yaml = resolve_problem_yaml(target, config)
    problem_dir = problem_yaml.parent
    meta = _read_yaml(problem_yaml)

    if "kind" in meta:
        raise ValueError(
            f"{problem_yaml}: `kind:` is gone — every problem is now defined by a "
            "verifier command (`verifier: verifier.sh`, the default)"
        )
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
        higher_is_better=bool(legacy_direction_key(meta)["higher_is_better"]),
        time_budget_s=meta.get("time_budget_s", config.budget.total_s),
        allow_network=bool(meta.get("allow_network", False)),
    )
    verifier_cmd = _verifier_argv(problem_yaml, problem_dir, meta)
    contract_path = _optional_file(problem_dir, meta, "contract", default="contract.md")
    baseline_raw = meta.get("baseline")
    baseline_score = float(baseline_raw) if isinstance(baseline_raw, (int, float)) and not isinstance(baseline_raw, bool) else None
    baseline_path = None if baseline_score is not None else _optional_file(problem_dir, meta, "baseline")
    if baseline_score is not None:
        baseline_summary = f"baseline: {baseline_score:g} (declared)"
    elif baseline_path:
        baseline_summary = f"baseline: {baseline_path.name}"
    else:
        baseline_summary = "baseline"
    return ProblemSpec(
        **common,
        verifier_cmd=verifier_cmd,
        holdout_cmd=(verifier_cmd + ["--holdout"]) if meta.get("holdout") else None,
        verifier_display=f"./problem/{Path(verifier_cmd[0]).name}",
        contract=contract_path.read_text() if contract_path else None,
        requirements_file=_optional_file(problem_dir, meta, "requirements"),
        baseline_text=baseline_path.read_text() if baseline_path else None,
        baseline_score=baseline_score,
        baseline_summary=baseline_summary,
        baseline_files={
            dest: (problem_dir / src).resolve()
            for dest, src in (meta.get("baseline_files") or {}).items()
        },
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
