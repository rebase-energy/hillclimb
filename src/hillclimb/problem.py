from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from hillclimb.config import Config, EvaluationConfig, ReportConfig
from hillclimb.harness.direction import legacy_direction_key


SOLUTION_KINDS = ("program", "climber")
# the harness-owned contract prompt per solution kind (prompts/<name>.md)
CONTRACT_TEMPLATES = {"program": "contract_verifier", "climber": "contract_climber"}


class ProblemError(Exception):
    """A problem folder the user has to fix (a missing file, a bad key in
    problem.yaml). The CLI prints it as one line, not a traceback; the
    subclasses keep the built-in types, so callers that catch those still do."""


class ProblemValueError(ProblemError, ValueError):
    pass


class ProblemFileNotFound(ProblemError, FileNotFoundError):
    pass


class ProblemPermissionError(ProblemError, PermissionError):
    pass


# Every key load_problem (or a view) reads from problem.yaml, plus the ones
# hillclimb.Problem writes. Anything else is most likely a typo, and is
# warned about once: a misspelt `higher_is_beter` must not pass silently.
KNOWN_PROBLEM_KEYS = frozenset({
    "problem_id", "metric", "higher_is_better", "lower_is_better", "description", "contract",  # legacy-key
    "verifier", "score", "run", "private", "holdout", "holdout_inputs", "time_limit_s",
    "baseline", "baseline_files", "chart_baselines", "output_artifacts",
    "requirements", "unit_tests", "data_dir", "allow_internet_during_solution", "allow_network",
    "evaluation", "report",
    "solution_kind", "interface", "landscape", "surface_metrics", "fingerprint", "plot",
    "written_by", "score_function", "kind",
})
_WARNED_KEYS: set[tuple[str, str]] = set()
# Keys a problem used to carry, ignored now, with where the setting went.
RETIRED_PROBLEM_KEYS = {
    "time_budget_s": "the budget is the run's: pass --budget, or set budget.total_s in runs/config.yaml",
}


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
    # may solution.py reach the internet when the verifier runs it? Only
    # the contract prompt says so; the verifier's network is not jailed.
    # (The operator AGENTS' internet is the user's `allow_internet_for_agents`.)
    allow_internet_during_solution: bool = False
    # What a solution.py IS. `program`: a script or module the verifier
    # drives (every ordinary problem). `climber`: a one-file hillclimb climber
    # — the problem is a META-problem whose verifier runs inner searches with
    # the candidate as their climber (`hillclimb grade`), and a search
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
    # Two steps instead of one verifier (`score:` in problem.yaml): the
    # engine runs `verifier_cmd` (the solution) and then this scorer, each in
    # a sandbox of its own, so the scorer may read what the solution may not
    # (`private_paths`) and nothing the solution leaves behind runs as the
    # scorer. None = one verifier that runs the solution itself.
    score_cmd: list[str] | None = None
    # files and folders only the scorer reads: hidden labels, holdout
    # instances, a seed. Unreadable to coding agents and to solutions.
    private_paths: list[Path] = Field(default_factory=list)
    # the hidden split's inputs (instances the solution must solve there):
    # read by the solution and the scorer only when they run on the holdout
    # split; never by coding agents, never during validation runs, whose
    # output reaches the agents
    holdout_inputs: list[Path] = Field(default_factory=list)
    # two-step problems: the engine stops a solution that runs longer
    solution_time_limit_s: float | None = None
    verifier_env: dict[str, str] = Field(default_factory=dict)  # extra env, validation runs only
    verifier_display: str = "./problem/verifier.sh"  # prompt-facing form of the command
    # True: the verifier computes the score (and any eval_result.json report)
    # independently of the coding agent. False: the number is the solution's own
    # claim (MLE-bench), so reports are stored labelled self-reported.
    report_trusted: bool = True
    holdout_needs_credentials: bool = False  # fail fast when the hidden split is gated
    runtime: Literal["csv", "emflow"] = "csv"  # which shared runtime venv to build
    requirements_file: Path | None = None  # per-problem venv requirements
    unit_tests: UnitTestSpec | None = None
    # How noisy its score is (replicates, the noise band) and what its report
    # shows: the problem's to say. None = the defaults. Only the keys it sets
    # are applied (`apply_problem_settings`).
    evaluation: EvaluationConfig | None = None
    report: ReportConfig | None = None

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

    # --- solution plot (`hillclimb plot`, `summit --plot`) ---
    # optional module (`plot.py`, picked up by default): exposes
    # `plot(solution_dir, ax) -> str | None`, drawing a solution's output
    # files (submission.csv, …) on matplotlib Axes and returning a caption.
    # It runs in the problem's runtime venv like the verifier
    # (runtime/plot_solution.py; matplotlib is added there on first use).
    plot_path: Path | None = None

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
    # or one .py file / a climber folder as shorthand. File refs in it are
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
        raise ProblemValueError(f"{path} must contain a YAML mapping")
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
    raise ProblemFileNotFound(f"No problem {str(target)!r} in {config.paths.problems_dir}. {_how_to_get(str(target))}")


def _how_to_get(target: str) -> str:
    """The command that would give the user the problem they named."""
    from hillclimb.catalog import PROBLEM_IDS

    name = Path(target).name
    if name in PROBLEM_IDS:
        return f"It is a catalog problem: fetch it with `hillclimb problem get {name}`."
    return f"See `hillclimb problem list`, or start your own with `hillclimb problem new {name}`."


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
        raise ProblemFileNotFound(f"{key} file not found: {path}")
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
        raise ProblemValueError(f"invalid interface file {interface_path}: {exc}") from exc
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
        raise ProblemFileNotFound(
            f"{problem_yaml}: verifier not found: {path} — a problem needs a verifier, or "
            "`score:` (a scorer) in problem.yaml; `hillclimb problem new <id>` writes one that runs"
        )
    if not os.access(path, os.X_OK):
        raise ProblemPermissionError(f"{problem_yaml}: verifier is not executable: chmod +x {path}")
    return [str(path)]


def _scorer_argv(problem_yaml: Path, problem_dir: Path, raw) -> list[str]:
    """`score:` as argv: a script (`verify.py`, run with the runtime's
    python) or an argv whose tokens may name files of the problem. Paths are
    made absolute: the scorer runs in the candidate's directory, and its
    script must not be found relative to anything the solution can write."""
    tokens = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else None
    if not tokens or not all(isinstance(token, str) and token for token in tokens):
        raise ProblemValueError(f"{problem_yaml}: score must be a script (score: verify.py) or an argv list")
    if len(tokens) == 1 and tokens[0].endswith(".py"):
        tokens = ["{python}", tokens[0]]
    argv = []
    for token in tokens:
        candidate = problem_dir / token
        argv.append(str(candidate.resolve()) if "{" not in token and candidate.is_file() else token)
    named = [token for token in tokens if token.endswith((".py", ".sh")) and "{" not in token]
    for token in named:
        if not (problem_dir / token).is_file():
            raise ProblemFileNotFound(f"{problem_yaml}: scorer not found: {problem_dir / token}")
    return argv


def _private_paths(problem_yaml: Path, problem_dir: Path, raw, key: str = "private") -> list[Path]:
    """`private:` (or `holdout_inputs:`) as absolute paths: relative ones are the problem's own."""
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
        raise ProblemValueError(f"{problem_yaml}: {key} must be a path or a list of paths")
    paths = []
    for item in items:
        path = Path(item).expanduser()
        path = (path if path.is_absolute() else problem_dir / path).resolve()
        if not path.exists():
            raise ProblemFileNotFound(f"{problem_yaml}: {key} path not found: {path}")
        paths.append(path)
    return paths


def _settings_block(problem_yaml: Path, meta: dict, key: str, model):
    """`evaluation:` / `report:` of a problem.yaml, as the config model it
    fills (its unset keys stay the defaults)."""
    raw = meta.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProblemValueError(f"{problem_yaml}: `{key}:` must be a mapping")
    unknown = sorted(set(raw) - set(model.model_fields))
    if unknown:
        raise ProblemValueError(
            f"{problem_yaml}: unknown `{key}.{unknown[0]}` (it takes {', '.join(model.model_fields)})"
        )
    try:
        return model.model_validate(raw)
    except ValueError as exc:
        raise ProblemValueError(f"{problem_yaml}: `{key}:` {exc}") from exc


def apply_problem_settings(config: Config, problem: ProblemSpec) -> None:
    """The problem's `evaluation:` and `report:` onto the config a search
    runs with: each key the problem.yaml sets replaces the default."""
    for name in ("evaluation", "report"):
        block = getattr(problem, name, None)
        if block is None:
            continue
        target = getattr(config, name)
        for field in block.model_fields_set:
            setattr(target, field, getattr(block, field))


def _check_meta(problem_yaml: Path, meta: dict) -> None:
    """The keys every problem needs, and the one that decides which way the
    search climbs, which must be a real YAML boolean: `bool("false")` is
    True, so a quoted "false" would silently climb the wrong way."""
    import difflib

    for key in sorted(set(meta) - KNOWN_PROBLEM_KEYS):
        if (str(problem_yaml), key) not in _WARNED_KEYS:
            _WARNED_KEYS.add((str(problem_yaml), key))
            if key in RETIRED_PROBLEM_KEYS:
                print(f"warning: {problem_yaml}: `{key}:` is ignored ({RETIRED_PROBLEM_KEYS[key]})", file=sys.stderr)
                continue
            close = difflib.get_close_matches(key, KNOWN_PROBLEM_KEYS, n=1)
            hint = f"did you mean `{close[0]}:`?" if close else "a typo?"
            print(f"warning: {problem_yaml}: unknown key `{key}:` is ignored ({hint})", file=sys.stderr)
    if not meta.get("metric"):
        raise ProblemValueError(f"{problem_yaml}: `metric:` is required (a name for the score, e.g. `metric: accuracy`)")
    direction = legacy_direction_key(meta).get("higher_is_better")
    if direction is None:
        raise ProblemValueError(f"{problem_yaml}: `higher_is_better:` is required (true or false)")
    if not isinstance(direction, bool):
        key = "higher_is_better" if "higher_is_better" in meta else "lower_is_better"  # legacy-key
        raise ProblemValueError(f"{problem_yaml}: `{key}:` must be true or false, unquoted (not {direction!r})")


def _sdk_config() -> Config:
    from hillclimb.api import sdk_config  # makes a fresh folder a hillclimb dir

    return sdk_config()


def load_problem(target: str | Path, config: Config) -> ProblemSpec:
    scheme = _split_scheme(target)
    if scheme is not None:
        load, _ = _provider_calls(scheme[0])
        return load(scheme[1], config)
    problem_yaml = resolve_problem_yaml(target, config)
    problem_dir = problem_yaml.parent
    meta = _read_yaml(problem_yaml)

    _check_meta(problem_yaml, meta)
    if "kind" in meta:
        raise ProblemValueError(
            f"{problem_yaml}: `kind:` is gone — every problem is now defined by a "
            "verifier command (`verifier: verifier.sh`, the default)"
        )
    problem_id = meta.get("problem_id") or problem_dir.name
    description_path = problem_dir / meta.get("description", "description.md")
    if not description_path.exists():
        raise ProblemFileNotFound(f"Description not found: {description_path}")

    if meta.get("data_dir"):
        data_dir = (problem_dir / meta["data_dir"]).resolve()
    elif (problem_dir / "data").exists():
        data_dir = (problem_dir / "data").resolve()
    else:
        data_dir = problem_dir.resolve()
    if not data_dir.exists():
        raise ProblemFileNotFound(f"Problem data directory not found: {data_dir}")

    solution_kind = meta.get("solution_kind", "program")
    if solution_kind not in SOLUTION_KINDS:
        raise ProblemValueError(
            f"{problem_yaml}: solution_kind must be one of {', '.join(SOLUTION_KINDS)}, not {solution_kind!r}"
        )
    common = dict(
        problem_id=problem_id,
        problem_dir=problem_dir.resolve(),
        data_dir=data_dir,
        description=description_path.read_text(),
        metric_name=meta["metric"],
        higher_is_better=bool(legacy_direction_key(meta)["higher_is_better"]),
        # `allow_network` is the pre-0.6 spelling
        allow_internet_during_solution=bool(
            meta.get("allow_internet_during_solution", meta.get("allow_network", False))
        ),
        solution_kind=solution_kind,
        # a climber is prompted for as a climber, not as a script
        contract_template=CONTRACT_TEMPLATES[solution_kind],
    )
    score_cmd = _scorer_argv(problem_yaml, problem_dir, meta["score"]) if meta.get("score") else None
    private_paths = _private_paths(problem_yaml, problem_dir, meta.get("private"))
    holdout_inputs = _private_paths(problem_yaml, problem_dir, meta.get("holdout_inputs"), "holdout_inputs")
    if (private_paths or holdout_inputs or meta.get("run")) and score_cmd is None:
        raise ProblemValueError(
            f"{problem_yaml}: private data needs a scorer of its own (`score: verify.py`): a "
            "verifier.sh runs the solution inside its own sandbox, so anything it can read, the "
            "solution can read too"
        )
    if holdout_inputs and not meta.get("holdout"):
        raise ProblemValueError(f"{problem_yaml}: holdout_inputs need `holdout: true`")
    # two steps: the engine runs the solution, or the problem's own runner
    # for it (`run:`, which imports or drives solution.py and runs in the
    # solution's sandbox), then the scorer
    if score_cmd and meta.get("run"):
        verifier_cmd = _scorer_argv(problem_yaml, problem_dir, meta["run"])
    elif score_cmd:
        verifier_cmd = ["{python}", "{solution}"]
    else:
        verifier_cmd = _verifier_argv(problem_yaml, problem_dir, meta)
    time_limit = meta.get("time_limit_s")
    if time_limit is not None and (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or time_limit <= 0):
        raise ProblemValueError(f"{problem_yaml}: time_limit_s must be a positive number of seconds")
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
            raise ProblemValueError(f"{problem_yaml}: unit_tests must be a mapping")
        root_value = raw_tests.get("root")
        command = raw_tests.get("command")
        if not isinstance(root_value, str) or not root_value:
            raise ProblemValueError(f"{problem_yaml}: unit_tests.root must be a relative directory")
        test_root = (problem_dir / root_value).resolve()
        try:
            test_root.relative_to(problem_dir.resolve())
        except ValueError as exc:
            raise ProblemValueError(f"{problem_yaml}: unit_tests.root must stay inside the problem dir") from exc
        if not test_root.is_dir():
            raise ProblemFileNotFound(f"unit test directory not found: {test_root}")
        unit_tests = UnitTestSpec(root=test_root, command=command)
    evaluation = _settings_block(problem_yaml, meta, "evaluation", EvaluationConfig)
    report = _settings_block(problem_yaml, meta, "report", ReportConfig)
    if baseline_score is not None:
        baseline_summary = f"baseline: {baseline_score:g} (declared)"
    elif baseline_path:
        baseline_summary = f"baseline: {baseline_path.name}"
    else:
        baseline_summary = "baseline"
    return ProblemSpec(
        **common,
        verifier_cmd=verifier_cmd,
        # a two-step problem's scorer is told the split; the solution step is the same
        holdout_cmd=(
            (list(verifier_cmd) if score_cmd else verifier_cmd + ["--holdout"]) if meta.get("holdout") else None
        ),
        score_cmd=score_cmd,
        private_paths=private_paths,
        holdout_inputs=holdout_inputs,
        solution_time_limit_s=float(time_limit) if score_cmd and time_limit is not None else None,
        verifier_display=(
            f"./problem/{Path(score_cmd[-1]).name}" if score_cmd else f"./problem/{Path(verifier_cmd[0]).name}"
        ),
        contract=contract_path.read_text() if contract_path else None,
        interface_path=interface_path,
        interface_text=interface_text,
        landscape_path=_optional_file(problem_dir, meta, "landscape", default="landscape.py"),
        surface_metrics=list(meta.get("surface_metrics") or ["x", "y"]),
        fingerprint_path=_optional_file(problem_dir, meta, "fingerprint", default="fingerprint.py"),
        plot_path=_optional_file(problem_dir, meta, "plot", default="plot.py"),
        requirements_file=_optional_file(problem_dir, meta, "requirements"),
        unit_tests=unit_tests,
        evaluation=evaluation,
        report=report,
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


# --- the Python side: a problem as the thing you hand a climber -----------------

# A Python-defined problem has two steps, which the engine runs as two
# processes, each in a sandbox of its own: the solution, and then the scorer
# (`score:` in its problem.yaml). The scorer runs with the engine's
# interpreter, because the scoring function lives in the user's own code, and
# it may read the problem's private paths, which the solution may not.
_DEFINED_SCORE_ARGV = ["{engine_python}", "verify.py"]

_DEFINED_VERIFY_PY = '''\
"""Scorer written by hillclimb.Problem. hillclimb has already run the
solution, in a sandbox of its own (and stopped it at the problem's time limit,
when it has one); this calls `{name}` from {source}
on the directory it ran in, and writes what it returns to $HILLCLIMB_RESULT.
A number is the score; a mapping must hold "score" and may add other numbers
(journaled as the candidate's metrics)."""

import importlib.util
import json
import math
import numbers
import os
import sys
import traceback
from pathlib import Path

SOURCE = {source!r}
NAME = {name!r}


def load_score():
    # the verifier's PYTHONPATH carries a shim `hillclimb` package (only
    # `spaces`, for solutions); the scoring file wants the real one
    sys.path[:] = [
        entry for entry in sys.path
        if not (Path(entry, "hillclimb", "spaces.py").is_file() and not Path(entry, "hillclimb", "api.py").is_file())
    ]
    for loaded in [name for name in sys.modules if name == "hillclimb" or name.startswith("hillclimb.")]:
        del sys.modules[loaded]
    spec = importlib.util.spec_from_file_location("hillclimb_problem_score", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, NAME)


def number(value):
    """A real number as a float (numpy scalars included), else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def main() -> int:
    try:
        value = load_score()(Path.cwd())
    except Exception:  # noqa: BLE001 - the scorer's own failure is a buggy candidate, with the trace
        traceback.print_exc()
        return 1
    payload = dict(value) if isinstance(value, dict) else {{"score": value}}
    score = number(payload.get("score"))
    if score is None or not math.isfinite(score):
        print(f"score() must return a finite number or a mapping with one, got {{value!r}}", file=sys.stderr)
        return 1
    # other numbers become metrics; anything else is dropped
    payload = {{key: number(item) if number(item) is not None else item for key, item in payload.items()}}
    payload = {{key: item for key, item in payload.items() if isinstance(item, (float, int, str))}}
    payload["score"] = score
    Path(os.environ["HILLCLIMB_RESULT"]).write_text(json.dumps(payload))
    print(f"score: {{score}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class Problem:
    """A problem, as the thing you hand a climber.

    One that exists already — in the folder's `problems/`, or bundled with
    hillclimb, or at a path, or a provider's (`emflow://…`):

        problem = Problem("fitness-landscape")

    Or a new one, from a scoring function of your own. The solution a coding
    agent writes is a script; hillclimb runs it, then calls your function in
    the directory it ran in, and the number it returns is the score:

        def score(run_dir: Path) -> float:
            return float(Path(run_dir, "answer.txt").read_text())

        problem = Problem(
            "largest-number",
            score=score,
            higher_is_better=True,
            description="Write the largest number you can to answer.txt.",
            output="answer.txt",
        )

    hillclimb runs the solution and then `score` as two processes, each in a
    sandbox of its own. `time_limit_s` stops a solution that runs longer (a
    failed attempt, which the climber repairs like any other). `private`
    names files or folders only `score` may read, such as hidden labels:
    coding agents and solutions cannot open them. `score` may
    also return a mapping with a `score` and other numbers, which the search
    journals as the candidate's metrics. It must be a function at
    the top level of a .py file: the verifier imports that file again (keep
    the code that starts a search under `if __name__ == "__main__":`). The
    problem's folder is written under the hillclimb dir's `problems/` the
    first time a climber uses it (`save` writes it by hand), so it is a
    problem like any other afterwards: `hillclimb verify`, `hillclimb run`,
    `hillclimb watch` all know it.
    """

    def __init__(
        self,
        name: str | Path,
        *,
        score: Callable[[Path], float | Mapping[str, float]] | None = None,
        higher_is_better: bool = True,
        metric: str = "score",
        description: str = "",
        baseline: float | None = None,
        output: str = "submission.csv",
        files: Mapping[str, str | Path] | None = None,
        requirements: Sequence[str] | None = None,
        time_limit_s: float | None = None,
        time_budget_s: int | None = None,
        allow_internet_during_solution: bool = False,
        chart_baselines: Mapping[str, float] | None = None,
        private: Sequence[str | Path] | None = None,
    ):
        self.name = str(name)
        self.score = score
        self.higher_is_better = bool(higher_is_better)
        self.metric = metric
        self._description = description
        self.baseline = baseline
        self.output = output
        self.files = dict(files or {})
        self.requirements = list(requirements) if requirements is not None else None
        self.time_limit_s = float(time_limit_s) if time_limit_s is not None else None
        if time_budget_s is not None:
            raise TypeError(
                "Problem(time_budget_s=...) is gone: the budget is the run's "
                "(hc.run(problem, budget='30m'), or budget.total_s in runs/config.yaml)"
            )
        self.allow_internet_during_solution = bool(allow_internet_during_solution)
        self.chart_baselines = dict(chart_baselines or {})
        # what only `score` reads (hidden labels, a held-back set): absolute,
        # since the scorer and the paths live in the user's code, not the problem
        self.private = [Path(p).expanduser().resolve() for p in (private or ())]
        if self.defined:
            if _split_scheme(self.name) is not None or "/" in self.name or self.name in ("", ".", ".."):
                raise ValueError(f"a problem defined in Python needs a plain name, not {self.name!r}")
            if not description:
                raise ValueError(f"Problem({self.name!r}): give the coding agent a description of what to write")
            if self.time_limit_s is not None and not self.time_limit_s > 0:
                raise ValueError(f"Problem({self.name!r}): time_limit_s must be positive, not {time_limit_s!r}")
            _score_source(score)  # fails now, with the fix, not at the first verifier run
            missing = [str(p) for p in self.private if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Problem({self.name!r}): private path not found: {', '.join(missing)}")

    @property
    def defined(self) -> bool:
        """Defined here, from a scoring function (else it exists already)."""
        return self.score is not None

    @property
    def id(self) -> str:
        scheme = _split_scheme(self.name)
        if scheme is not None:
            return scheme[1].replace(":", "-").replace("/", "-")
        return Path(self.name).name.removesuffix(".yaml") if "/" in self.name else self.name

    def __repr__(self) -> str:
        return f"Problem({self.name!r}{', defined here' if self.defined else ''})"

    # --- where it is ---

    def save(self, problems_dir: Path | str | None = None) -> Path:
        """Write a defined problem's folder (and rewrite it when the
        definition changed). Returns the folder. `problems_dir` defaults to
        the hillclimb dir's."""
        if not self.defined:
            raise ValueError(f"Problem({self.name!r}) exists already; only a problem defined in Python is saved")
        root = Path(problems_dir) if problems_dir is not None else _sdk_config().paths.problems_dir
        folder = root / self.name
        marker = folder / "problem.yaml"
        if marker.exists() and not (_read_yaml(marker).get("written_by") == "hillclimb.Problem"):
            raise FileExistsError(
                f"{folder} is a problem of its own, not one hillclimb.Problem wrote: "
                "give this one another name"
            )
        folder.mkdir(parents=True, exist_ok=True)
        source, function = _score_source(self.score)
        meta = {
            "problem_id": self.name,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "description": "description.md",
            "allow_internet_during_solution": self.allow_internet_during_solution,
            "output_artifacts": [self.output],
            "written_by": "hillclimb.Problem",
            "score_function": f"{source}:{function}",
            "score": list(_DEFINED_SCORE_ARGV),
        }
        if self.private:
            meta["private"] = [str(p) for p in self.private]
        if self.time_limit_s is not None:
            meta["time_limit_s"] = self.time_limit_s
        if self.baseline is not None:
            meta["baseline"] = self.baseline
        if self.chart_baselines:
            meta["chart_baselines"] = dict(self.chart_baselines)
        if self.requirements is not None:
            meta["requirements"] = "requirements.txt"
            (folder / "requirements.txt").write_text("".join(f"{line}\n" for line in self.requirements))
        (folder / "problem.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        (folder / "description.md").write_text(self._description.rstrip("\n") + "\n" + self._submission_note())
        for dest, content in self.files.items():
            if Path(dest).name != dest:
                raise ValueError(f"Problem({self.name!r}): files are named by a bare file name, not {dest!r}")
            if isinstance(content, Path) or (isinstance(content, str) and "\n" not in content and Path(content).is_file()):
                (folder / dest).write_bytes(Path(content).read_bytes())
            else:
                (folder / dest).write_text(str(content))
        (folder / "verify.py").write_text(_DEFINED_VERIFY_PY.format(source=str(source), name=function))
        # written by an earlier hillclimb: a verifier that ran the solution itself
        for stale in ("verifier.sh", "verifier.py"):
            (folder / stale).unlink(missing_ok=True)
        return folder

    def _submission_note(self) -> str:
        limit = (
            f" It must finish within {self.time_limit_s:g} seconds; a run that does not is a failed attempt."
            if self.time_limit_s is not None else ""
        )
        return (
            "\n## Submission format\n\n"
            f"Your `solution.py` runs in the working directory and must write `{self.output}` there.{limit} "
            "`problem/verify.py` reads what it wrote and reports the score "
            f"(`{self.metric}`, {'higher' if self.higher_is_better else 'lower'} is better).\n"
        )

    def resolve(self, config: Config) -> str:
        """The target `load_problem` takes for this problem, making sure it
        exists: a defined problem is saved, a bundled one that the folder
        lacks is copied in."""
        if self.defined:
            return str(self.save(config.paths.problems_dir))
        if _split_scheme(self.name) is None:
            try:
                resolve_problem_yaml(self.name, config)
            except FileNotFoundError:
                from hillclimb.catalog import PROBLEM_IDS, install_problem

                if self.name in PROBLEM_IDS:
                    install_problem(config.paths.problems_dir, self.name)
        return self.name

    def spec(self, config: Config | None = None) -> ProblemSpec:
        """What the engine loads for it."""
        config = config if config is not None else _sdk_config()
        return load_problem(self.resolve(config), config)

    @property
    def description(self) -> str:
        return self._description if self.defined else self.spec().description


def _score_source(score: Callable) -> tuple[Path, str]:
    """Where a scoring function lives, as (file, name), so the verifier can
    import it in its own process."""
    import inspect

    if not callable(score):
        raise TypeError(f"score must be a function taking the directory the solution ran in, not {score!r}")
    name = getattr(score, "__name__", "")
    qualname = getattr(score, "__qualname__", name)
    if not name or name == "<lambda>" or "<locals>" in qualname or qualname != name:
        raise ValueError(
            "score must be a plain function at the top level of a .py file (not a lambda, a method, "
            "or a function defined inside another), so the verifier can import it"
        )
    try:
        file = inspect.getsourcefile(score)
    except TypeError:
        file = None
    if not file or not Path(file).is_file():
        raise ValueError(
            f"score {name}() must live in a .py file the verifier can import, not in a REPL or a notebook cell"
        )
    return Path(file).resolve(), name
