from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from hillclimb.config import Config
from hillclimb.harness.direction import legacy_direction_key


SOLUTION_KINDS = ("program", "climber")
# the harness-owned contract prompt per solution kind (prompts/<name>.md)
CONTRACT_TEMPLATES = {"program": "contract_verifier", "climber": "contract_climber"}


class UnitTestSpec(BaseModel):
    """Optional, framework-neutral correctness gate for a problem.

    ``root`` is the tree snapshotted at run start.  The command is argv, not
    a shell string, and may reference ``{python}``, ``{solution}``, and
    ``{tests}``.  ``sha256`` is filled only for a frozen run bundle.
    """

    root: Path
    command: list[str]
    sha256: str | None = None

    @field_validator("command")
    @classmethod
    def _valid_command(cls, value: list[str]) -> list[str]:
        if not value or not all(isinstance(token, str) and token for token in value):
            raise ValueError("unit_tests.command must be a non-empty argv list")
        allowed = {"python", "solution", "tests"}
        import string

        for token in value:
            for _literal, field, _format, _conversion in string.Formatter().parse(token):
                if field is not None and field not in allowed:
                    raise ValueError(
                        f"unknown unit-test command token {{{field}}}; "
                        "allowed: {python}, {solution}, {tests}"
                    )
        return value


class ProblemSpec(BaseModel):
    """Portable problem definition.

    A problem is defined by its **verifier command**: the scoring process,
    which drives `solution.py` and writes the score to `$HILLCLIMB_RESULT`
    (see `executor.py` for the contract). A directory problem supplies it as
    `verifier.sh`; providers (`emflow://`, `mlebench://`) supply their own argv
    for the same contract. An optional unit-test command is a second,
    framework-neutral correctness gate.
    """

    problem_id: str
    problem_dir: Path
    data_dir: Path
    description: str
    metric_name: str
    higher_is_better: bool
    time_budget_s: int
    # may solution.py reach the internet when the verifier runs it? Only
    # the contract prompt says so; the verifier's network is not jailed.
    # (The operator AGENTS' internet is the user's `allow_internet_for_agents`.)
    allow_internet_during_solution: bool = False
    # What a solution.py IS. `program`: a script or module the verifier
    # drives (every ordinary problem). `climber`: a one-file hillclimb climber
    # — the problem is a META-problem whose verifier runs inner searches with
    # the candidate as their climber (`hillclimb meta evaluate`), and a search
    # on it runs its climber in the improver role (`SearchMeta.role`).
    solution_kind: Literal["program", "climber"] = "program"
    # Named score references drawn as horizontal lines by `hillclimb chart`.
    # A mapping keeps problem.yaml compact and preserves legend order.
    chart_baselines: dict[str, float] = Field(default_factory=dict)

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
    unit_tests: UnitTestSpec | None = None

    # --- prompt assembly ---
    contract_template: str = "contract_verifier"  # prompts/<name>.md
    contract: str | None = None  # problem-authored solution-contract section
    # optional machine-checkable I/O declaration (`interface.py`, spaces.py
    # vocabulary): the path is what verifiers/agents re-load to run checks,
    # the text is its rendered describe() for prompts — objects themselves
    # never ride on the spec (it stays serializable)
    interface_path: Path | None = None
    interface_text: str | None = None

    # --- surface view (`hillclimb surface`) ---
    # optional terrain module (`landscape.py`, picked up by default like
    # `contract.md`): exposes `elevation(x, y)` and `grid(n)`. The verifier
    # journals each candidate's position by writing the `surface_metrics`
    # keys as extra numeric keys next to `score` (they land in
    # Trial.metrics), which is what lets the surface view drop candidates
    # onto the terrain. No landscape module = no surface view.
    landscape_path: Path | None = None
    surface_metrics: list[str] = Field(default_factory=lambda: ["x", "y"])

    # --- similarity view (`hillclimb similarity`) ---
    # optional behavioral fingerprint module (`fingerprint.py`, picked up by
    # default): exposes `fingerprint(candidate_dir) -> Sequence[float] | None`,
    # a vector that captures what a candidate's output *is* (invariant to
    # whatever the problem considers equivalent — point order, symmetry).
    # Without it the view falls back to the flattened submission file.
    fingerprint_path: Path | None = None

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

    # Files produced by a valid candidate that travel with solution.py through
    # trial hoisting, best/ selection, pruning, resume, and summit.  CSV is the
    # historical default; JSON-native benchmark providers override it.
    output_artifacts: list[str] = Field(default_factory=lambda: ["submission.csv"])

    @field_validator("output_artifacts")
    @classmethod
    def _safe_output_artifacts(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for name in value:
            path = Path(name)
            if not name or path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
                raise ValueError(f"output artifact must be a file name, got {name!r}")
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    # --- provider extras ---
    # Generic provider identity. ``provider_target`` is the resumable source
    # target (content-pinned where the provider supports revisions), while
    # ``problem_key_override`` groups revisions of the same benchmark problem.
    provider_target: str | None = None
    provider_revision: str | None = None
    problem_key_override: str | None = None
    emflow_problem: str | None = None   # registry name, e.g. "gefcom2014:solar"
    emflow_quantiles: list[float] | None = None  # probabilistic problems only
    # MLE-bench competition id; set only by the mlebench provider. Marks the
    # search for one official grade-sample run on the selected candidate
    # after the search finishes (never during — selection integrity).
    mlebench_comp_id: str | None = None

    @property
    def target(self) -> str | None:
        """Provider target string when the problem comes from one."""
        if self.provider_target:
            return self.provider_target
        if self.emflow_problem:
            return f"emflow://{self.emflow_problem}"
        if self.mlebench_comp_id:
            return f"mlebench://{self.mlebench_comp_id}"
        return None

    @property
    def problem_key(self) -> str:
        """Canonical identity of the problem across runs (SearchMeta.problem_key)."""
        if self.problem_key_override:
            return self.problem_key_override
        from hillclimb.harness.run import problem_key_for

        return problem_key_for(self.target or "", self.problem_id)


class SuiteEntry(BaseModel):
    """One search in a run-spec file. Bare-string entries are shorthand for
    `{target: <string>}`; dict entries carry per-search parameter overrides,
    so a committed spec fully describes a run (git-versionable parameters).
    CLI flags passed alongside the spec override these values."""

    target: str
    name: str | None = None
    model: str | None = None
    agent: str | None = None
    budget: str | None = None  # "2h" / "30m" / seconds — parsed by the CLI
    # the climber this search runs: the `climber:` block (`modules/spec.py`),
    # or a preset's name / one .py file as shorthand. File refs in it are
    # relative to the spec file. None = the spec's own `climber:`, else the folder's
    climber: str | dict | None = None
    parallel_agents: int | None = None
    n_replicates: int | None = None
    seed_from: str | None = None  # incumbent solution.py, relative to the spec file
    set: list[str] = Field(default_factory=list)  # `key=value` config overrides (`--set`)

    @model_validator(mode="before")
    @classmethod
    def _legacy_keys(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            for old, new in (("backend", "agent"), ("parallel_operators", "parallel_agents"), ("n_trials", "n_replicates")):
                if old in data:
                    data.setdefault(new, data.pop(old))
        return data


class SuiteSpec(BaseModel):
    suite_id: str
    suite_path: Path
    problems: list[SuiteEntry]
    climber: str | dict | None = None  # the default for entries that name none

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
LEGACY_PROVIDER_SCHEMES = ("emflow", "mlebench")


def _split_scheme(target: str | Path) -> tuple[str, str] | None:
    """(scheme, rest) when the target uses a provider scheme, else None."""
    scheme, sep, rest = str(target).partition("://")
    from hillclimb.benchmark_providers import has_benchmark_provider

    if sep and (scheme in LEGACY_PROVIDER_SCHEMES or has_benchmark_provider(scheme)):
        return scheme, rest
    return None


def _emflow_provider():
    try:
        from hillclimb.providers.emflow import provider
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "emflow:// targets need the emflow extra: "
            "pip install 'hillclimb[emflow]'"
        ) from exc
    return provider


def _provider_calls(scheme: str):
    """(load_problem, resolve_target) pair for a provider scheme."""
    from hillclimb.benchmark_providers import get_benchmark_provider, has_benchmark_provider

    if has_benchmark_provider(scheme):
        provider = get_benchmark_provider(scheme)
        return provider.load_problem, provider.resolve_target
    if scheme == "emflow":
        provider = _emflow_provider()
        return provider.load_emflow_problem, provider.resolve_emflow_target
    from hillclimb.providers.mlebench import provider

    return provider.load_mlebench_problem, provider.resolve_mlebench_target


def provider_chart_baselines(target: str) -> dict[str, float]:
    """Chart reference lines for a provider target, resolved lazily without
    materializing any data. Empty when the target is not a provider's or the
    provider publishes no reference scores."""
    scheme = _split_scheme(target)
    if scheme is None:
        return {}
    from hillclimb.benchmark_providers import get_benchmark_provider, has_benchmark_provider

    if has_benchmark_provider(scheme[0]):
        provider = get_benchmark_provider(scheme[0])
        chart_baselines = getattr(provider, "chart_baselines", None)
        return chart_baselines(scheme[1]) if chart_baselines is not None else {}
    if scheme[0] == "emflow":
        return _emflow_provider().chart_baselines_for(scheme[1])
    return {}


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


def load_interface_fields(problem_dir: Path, meta: dict) -> tuple[Path | None, str | None]:
    """(interface_path, rendered describe text) for a problem dir's optional
    `interface.py` (same optional-default semantics as `contract.md`).
    Imported eagerly so an unloadable interface fails at problem load, with
    the file named — not mid-search. Reused by providers that synthesize an
    interface into their materialized problem dirs."""
    interface_path = _optional_file(problem_dir, meta, "interface", default="interface.py")
    if interface_path is None:
        return None, None
    from hillclimb import spaces

    try:
        module = spaces.load_interface(interface_path)
    except spaces.InterfaceError as exc:
        raise ValueError(f"invalid interface file {interface_path}: {exc}") from exc
    return interface_path, spaces.describe_interface(module)


def windows_edition(path: Path, windows: bool | None = None) -> Path:
    """On Windows, a shell verifier's `.py` sibling when the problem ships one
    (`hillclimb problem get` writes verifier.py there instead of verifier.sh),
    so no bash is needed; otherwise `path` itself."""
    if windows is None:
        windows = sys.platform == "win32"
    if windows and path.suffix == ".sh":
        sibling = path.with_suffix(".py")
        if sibling.exists():
            return sibling
    return path


def _verifier_argv(problem_yaml: Path, problem_dir: Path, meta: dict) -> list[str]:
    """The problem's verifier, as argv. Absolute: a relative program path is
    resolved against the ENGINE's cwd, not the candidate dir the command runs in
    (subprocess does not search cwd for the executable)."""
    name = str(meta.get("verifier", "verifier.sh"))
    path = windows_edition((problem_dir / name).resolve())
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

    solution_kind = meta.get("solution_kind", "program")
    if solution_kind not in SOLUTION_KINDS:
        raise ValueError(
            f"{problem_yaml}: solution_kind must be one of {', '.join(SOLUTION_KINDS)}, not {solution_kind!r}"
        )
    common = dict(
        problem_id=problem_id,
        problem_dir=problem_dir.resolve(),
        data_dir=data_dir,
        description=description_path.read_text(),
        metric_name=meta["metric"],
        higher_is_better=bool(legacy_direction_key(meta)["higher_is_better"]),
        time_budget_s=meta.get("time_budget_s", config.budget.total_s),
        # `allow_network` is the pre-0.6 spelling
        allow_internet_during_solution=bool(
            meta.get("allow_internet_during_solution", meta.get("allow_network", False))
        ),
        solution_kind=solution_kind,
        # a climber is prompted for as a climber, not as a script
        contract_template=CONTRACT_TEMPLATES[solution_kind],
    )
    verifier_cmd = _verifier_argv(problem_yaml, problem_dir, meta)
    contract_path = _optional_file(problem_dir, meta, "contract", default="contract.md")
    interface_path, interface_text = load_interface_fields(problem_dir, meta)
    baseline_raw = meta.get("baseline")
    baseline_score = float(baseline_raw) if isinstance(baseline_raw, (int, float)) and not isinstance(baseline_raw, bool) else None
    chart_baselines = dict(meta.get("chart_baselines") or {})
    if baseline_score is not None:
        # A declared numeric floor is both a real c000 search candidate and an
        # obvious chart reference. Keep it first in the legend and authoritative
        # if an older config redundantly declared `chart_baselines.baseline`.
        chart_baselines = {
            "baseline": baseline_score,
            **{label: value for label, value in chart_baselines.items() if label != "baseline"},
        }
    baseline_path = None if baseline_score is not None else _optional_file(problem_dir, meta, "baseline")
    unit_tests = None
    if meta.get("unit_tests") is not None:
        raw_tests = meta["unit_tests"]
        if not isinstance(raw_tests, dict):
            raise ValueError(f"{problem_yaml}: unit_tests must be a mapping")
        root_value = raw_tests.get("root")
        command = raw_tests.get("command")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError(f"{problem_yaml}: unit_tests.root must be a relative directory")
        test_root = (problem_dir / root_value).resolve()
        try:
            test_root.relative_to(problem_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"{problem_yaml}: unit_tests.root must stay inside the problem dir") from exc
        if not test_root.is_dir():
            raise FileNotFoundError(f"unit test directory not found: {test_root}")
        unit_tests = UnitTestSpec(root=test_root, command=command)
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
        interface_path=interface_path,
        interface_text=interface_text,
        landscape_path=_optional_file(problem_dir, meta, "landscape", default="landscape.py"),
        surface_metrics=list(meta.get("surface_metrics") or ["x", "y"]),
        fingerprint_path=_optional_file(problem_dir, meta, "fingerprint", default="fingerprint.py"),
        requirements_file=_optional_file(problem_dir, meta, "requirements"),
        unit_tests=unit_tests,
        baseline_text=baseline_path.read_text() if baseline_path else None,
        baseline_score=baseline_score,
        baseline_summary=baseline_summary,
        chart_baselines=chart_baselines,
        baseline_files={
            dest: (problem_dir / src).resolve()
            for dest, src in (meta.get("baseline_files") or {}).items()
        },
        output_artifacts=(
            list(meta["output_artifacts"])
            if "output_artifacts" in meta
            else ["submission.csv"]
        ),
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


def _climber_block(value, suite_yaml: Path, where: str) -> dict:
    """A `climber:` value of a run spec as the full block, its file refs
    resolved from the spec's own folder (like `seed_from`)."""
    from hillclimb.modules.spec import ClimberSpec, block_error

    try:
        return ClimberSpec.model_validate(value).anchored(suite_yaml.parent).block()
    except ValueError as exc:
        raise ValueError(f"{suite_yaml}: {where}climber: {block_error(exc)}") from exc


def load_suite(target: str | Path, config: Config) -> SuiteSpec:
    """Load a run spec: a `problems:` list (strings or per-entry parameter
    dicts), or the single-search form with a top-level `target:` plus the
    same parameter keys. A top-level `climber:` beside `problems:` is the
    default for entries that name none; every entry's climber comes back as
    the full block."""
    suite_yaml = resolve_suite_yaml(target, config)
    meta = _read_yaml(suite_yaml)
    if "problems" not in meta:  # single-target spec
        entry_keys = SuiteEntry.model_fields.keys()
        entries = [SuiteEntry.model_validate({k: v for k, v in meta.items() if k in entry_keys})]
        default = None
    else:
        problems = meta.get("problems")
        if not isinstance(problems, list) or not problems:
            raise ValueError(f"{suite_yaml} must define a non-empty `problems` list")
        entries = [_parse_entry(raw, suite_yaml) for raw in problems]
        default = meta.get("climber")
    if default is not None:
        default = _climber_block(default, suite_yaml, "")
    for index, entry in enumerate(entries, 1):
        if entry.climber is not None:
            entry.climber = _climber_block(entry.climber, suite_yaml, f"problems[{index}].")
        elif default is not None:
            entry.climber = dict(default)
    return SuiteSpec(
        suite_id=meta.get("suite_id") or suite_yaml.stem,
        suite_path=suite_yaml,
        problems=entries,
        climber=default,
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
