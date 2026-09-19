from __future__ import annotations

import os

from pathlib import Path

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hillclimb.project import (
    find_hillclimb_dir,
    MARKER_FILE,
    user_config_path,
    HillclimbDirNotFound,
)


class BudgetConfig(BaseModel):
    total_s: int = 7200
    agent_timeout_s: int = 1800
    exec_timeout_s: int = 1800
    stop_margin_s: int = 300
    # what happens to operators still in flight when `total_s` runs out:
    # `graceful` lets them finish and commit (the search overruns by up to one
    # operator; the TUI shows by how much), `hard` aborts them at the deadline
    # and journals them abandoned
    deadline: Literal["graceful", "hard"] = "graceful"
    # hard agent-spend ceiling; the search parks (resumable) when cumulative
    # backend cost reaches it. 0 = no ceiling.
    max_cost_usd: float = 0.0
    # The budget is the USER's, in every dimension — a climber sees what is
    # left (BudgetView) and never sets it. Both end the search like the clock
    # does (no new work; what is in flight still lands). 0 = no limit.
    #   max_evaluations  verifier trials the climber caused: one per scored
    #                    attempt plus one per tune trial; the baseline and the
    #                    seed are the harness's own floor and do not count
    #   max_tokens       tokens its agent calls consumed, all kinds summed
    max_evaluations: int = 0
    max_tokens: int = 0


def default_machine_max_operators() -> int:
    """min(8, cores - 2): each operator is an API-bound agent plus, at worst,
    one single-threaded solution process, so this keeps a laptop responsive
    however many searches are launched."""
    return max(1, min(8, (os.cpu_count() or 4) - 2))


# pre-rename config keys, mapped on load AND by apply_overrides (which walks
# model_fields and would otherwise reject `--set search.n_replicates=3` or an old
# experiment spec). "agent" was overloaded (the engine noun is operator);
# "trial" moved up a level when trials gained parameters (a trial is one
# parameter set, a replicate one seeded execution of it).
LEGACY_SEARCH_KEYS = {
    "parallel_agents": "parallel_operators",
    "machine_max_agents": "machine_max_operators",
    "n_trials": "n_replicates",
    "trial_mode": "replicate_mode",
}
LEGACY_SETTINGS = {f"search.{old}": f"search.{new}" for old, new in LEGACY_SEARCH_KEYS.items()}


class SearchConfig(BaseModel):
    """Policy knobs for one Search (the `search:` config block), not the
    Search entity itself — that lives in run.py as SearchMeta."""

    num_drafts: int = 3
    max_debug_depth: int = 3
    parallel_operators: int = 1  # >1 enables the worker pool; 1 = serial (default)

    @model_validator(mode="before")
    @classmethod
    def _legacy_parallel_agents(cls, data):
        # pre-rename keys; "agent" is overloaded, the engine noun is operator
        if isinstance(data, dict):
            data = dict(data)
            for old, new in LEGACY_SEARCH_KEYS.items():
                if old in data:
                    data.setdefault(new, data.pop(old))
        return data

    def effective_machine_max_operators(self) -> int:
        if self.machine_max_operators is None:
            return default_machine_max_operators()
        return self.machine_max_operators
    # seeded executions per trial (one parameter set); the trial's score is
    # their MEDIAN. Replicate variance is noise, never something to climb.
    n_replicates: int = 1
    # how replicates run. "parallel" is right for seed variance (and 3x
    # faster); "serial" is REQUIRED for anything that measures time — runs
    # sharing a machine contend, and the contention is the measurement.
    replicate_mode: Literal["parallel", "serial"] = "parallel"
    # Noise guard. A candidate is only better than the incumbent when it beats
    # it by more than the band, so the search cannot climb measurement noise.
    #   min_improvement: absolute floor, in metric units
    #   noise_k: multiples of the observed noise floor (the median per-trial
    #            replicate spread); needs n_replicates > 1 to have anything to measure
    # Both default to 0 = off, which is the strict comparison.
    min_improvement: float = 0.0
    noise_k: float = 0.0
    # Machine-wide cap on concurrent operators across every search on this
    # machine (flock slots in ~/.cache/hillclimb/agent-slots/). Operators
    # beyond it wait (`waiting-slot` in watch). 0 = off; None = default_machine_max_operators().
    machine_max_operators: int | None = None
    policy: str = "greedy"  # search policy (policies registry)
    policy_params: dict = Field(default_factory=dict)  # opaque; validated by the policy factory
    # which parameter set a `tune` action tries next on a candidate that
    # declares params.json (tuners registry: random | optuna). WHEN to tune is
    # the policy's call (greedy: policy_params.tune_budget etc.)
    tuner: str = "random"
    tuner_params: dict = Field(default_factory=dict)  # opaque; validated by the tuner factory


class OperatorsConfig(BaseModel):
    """Operator-scaffold knobs (the `operators:` config block). Both gate
    ONLY prompt injection, so A/B arms differ in what the agent is told, not
    in what the engine records."""

    # draft: instruct the agent to web-search current SOTA methods for the
    # problem class before writing code (counters training-data staleness)
    draft_retrieval: bool = True
    # improve: instruct the agent to run a component ablation of the parent
    # solution and target only the highest-impact component
    improve_ablation: bool = True
    # all operators: contract clause advertising the read-only
    # `hillclimb knowledge query` memory lookup (needs learning enabled)
    knowledge_tool: bool = True


class RouteConfig(BaseModel):
    """Per-operator backend/model override (the `routing:` config block).
    None fields inherit the global `backend`/`model`/`backend_auth` scalars."""

    model_config = ConfigDict(extra="forbid")

    backend: str | None = None
    model: str | None = None
    backend_auth: str | None = None
    sampling: dict[str, int | float] | None = None
    # model POOL: when set (2+ entries), a UCB1 bandit picks the model per
    # call, rewarded by whether the candidate improved on its parent
    # (bandit.py). Takes precedence over `model` in the same layer; a
    # single-entry pool behaves like `model`.
    models: list[str] | None = None

    @field_validator("sampling")
    @classmethod
    def _finite_sampling(cls, value: dict[str, int | float] | None):
        if value is None:
            return None
        import math

        bad = [name for name, number in value.items() if not math.isfinite(number)]
        if bad:
            raise ValueError(f"sampling values must be finite: {', '.join(bad)}")
        return value


class HoldoutConfig(BaseModel):
    """Selection hygiene for problems whose verifier ships a hidden split
    (`holdout: true`). `enabled: false` turns the split off search-wide."""

    enabled: bool = True
    climb_on: str = "val"  # seam only; 'holdout' climbing is a future experiment
    selection: str = "rank-blend"  # rank-blend | holdout | val
    # holdout hygiene: only candidates whose val score ranks top-k get a
    # holdout evaluation (0 = score every passing candidate). Non-top-k candidates
    # climb on val but cannot win rank-blend selection.
    top_k: int = 5
    # when the hidden split is scored: `inline` as each candidate lands
    # (`watch` shows it live), `after` once the search has finished. A
    # climber can only ever see val, either way.
    timing: Literal["inline", "after"] = "inline"


class EnsembleConfig(BaseModel):
    enabled: bool = True
    reserve_fraction: float = 0.2  # final slice of budget reserved for ensembling
    top_k: int = 3
    max_attempts: int = 2


class PathsConfig(BaseModel):
    # Relative runs_dir/problems_dir resolve at load time against the folder
    # holding the hillclimb dir; everything hillclimb writes stays inside the
    # hillclimb/ folder by default.
    runs_dir: Path = Path("hillclimb/runs")
    problems_dir: Path = Path("hillclimb/problems")
    # operator prompt overrides: `<prompts_dir>/<template>.md` shadows the
    # package template of the same name (prompts/render.py); the effective
    # set is hashed into SearchMeta.templates_sha256
    prompts_dir: Path = Path("hillclimb/prompts")
    # None = shared machine venv under ~/.cache/hillclimb/venvs/, keyed by a
    # hash of the requirements (+ emflow source). Set explicitly to pin.
    runtime_python: Path | None = None
    emflow_runtime_python: Path | None = None
    mlebench_python: Path = Path("../mle-bench/.venv/bin/python")
    mlebench_data_dir: Path | None = None
    kaggle_bin: Path = Path("../mle-bench/.venv/bin/kaggle")


class StoreConfig(BaseModel):
    """Where run/search/candidate records are indexed for the cross-run
    views (see store.py). `files` reads the hillclimb folder directly;
    `sqlite` keeps one database file the engine feeds live and
    `hillclimb store sync` rebuilds from the folder."""

    backend: str = "files"  # files | sqlite
    sqlite_path: Path = Path("hillclimb/store.sqlite")


class LearningConfig(BaseModel):
    """Cross-search learning: distill a knowledge card from every finished
    search and inject prior experience into draft prompts."""

    enabled: bool = True
    # default: <hillclimb dir>/knowledge (git-versionable); explicit
    # path overrides; None + no hillclimb dir = learning off
    dir: Path | None = None
    max_cards: int = 3  # cards rendered into the prompt
    # live sharing: republish this search's card after every executed
    # candidate and read siblings' cards (runs/<run-id>/knowledge/), so
    # concurrent searches in one run learn from each other mid-flight
    live: bool = True
    # opt-in policy bias: start the draft complexity schedule one step up
    # when past winners were never 'minimal'
    complexity_prior: bool = False
    # semantic layer: one cheap agent pass after each finished search
    # distills typed claims into the card (claims.py). Routed via
    # `routing: distill:` (default model: haiku).
    claims: bool = True
    claims_timeout_s: int = 300
    # inject graph-ranked claims into operator prompts when
    # knowledge/graph.json exists (its own flag so A/B stays possible)
    graph_retrieval: bool = True
    # credit assignment: injected claims share the search's outcome reward;
    # track records adjust retrieval confidence and retire failing claims
    # (credit.py)
    credit: bool = True
    # consolidated playbooks (knowledge/playbooks/<concept>.md) replace the
    # raw claims block in draft prompts when one matches the problem's
    # concepts; credit flows to the playbook's source claims
    playbooks: bool = True
    # skill library: harvest scored winners into knowledge/skills/ and hand
    # the best match to the first draft as reference_solution.py
    skills: bool = True


class ReportConfig(BaseModel):
    """Trial evaluation reports: per-zone/horizon/quantile breakdowns from
    emflow validation evals."""

    # gates ONLY prompt injection — computation and journal storage always
    # run, so A/B arms record identical data and differ only in what the
    # improve operator sees
    enabled: bool = True


class EmflowConfig(BaseModel):
    """Optional emflow problem-provider settings (hillclimb[emflow] extra)."""

    # pip requirement installed into the emflow runtime venv; supports a
    # leading "-e " for editable local checkouts (e.g. "-e ../emflow")
    source: str = "emflow @ git+https://github.com/rebase-energy/emflow.git"


class EinsteinArenaConfig(BaseModel):
    """Public, read-only Einstein Arena problem-provider settings."""

    base_url: str = "https://einsteinarena.com"
    request_timeout_s: float = Field(default=30.0, gt=0, le=300)


class SimilarityConfig(BaseModel):
    """Similarity scores `hillclimb similarity scores` computes, name -> params.
    A name is a registry entry (solution-card, api-calls, code-tokens), a
    `.py` file (relative to the folder holding the hillclimb dir), or
    `module:Class` — see similarity_scores/__init__.py."""

    scores: dict[str, dict] = Field(default_factory=lambda: {
        "solution-card": {},
        "api-calls": {},
    })


class PiConfig(BaseModel):
    """pi backend settings shared by routed pi instances."""

    models_file: Path | None = None


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _load_dotenv(path: Path) -> None:
    """`KEY=VALUE` lines into the environment, never overriding the shell.
    Provider keys (OPENROUTER_API_KEY) live here; the value stays in the
    environment and never enters the Config object, so it cannot be
    journaled."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


BACKEND_AUTHS = ("subscription", "api-key", "openrouter")


class Config(BaseModel):
    backend: str = "claude-code"
    backend_auth: str = "subscription"  # one of BACKEND_AUTHS
    model: str = "sonnet"
    # per-operator routing; keys: draft | debug | improve | ensemble |
    # distill | default. Missing keys (or an absent block) fall back to the
    # scalars above (except distill's model, which defaults to haiku).
    routing: dict[str, RouteConfig] = Field(default_factory=dict)
    budget: BudgetConfig = BudgetConfig()
    search: SearchConfig = SearchConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    paths: PathsConfig = PathsConfig()
    store: StoreConfig = StoreConfig()
    emflow: EmflowConfig = EmflowConfig()
    einsteinarena: EinsteinArenaConfig = EinsteinArenaConfig()
    pi: PiConfig = PiConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    learning: LearningConfig = LearningConfig()
    report: ReportConfig = ReportConfig()
    operators: OperatorsConfig = OperatorsConfig()
    # Resolved at load time; None for embedders that construct Config()
    # directly and set absolute paths themselves (e.g. the hosted container).
    hillclimb_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_backend_auth(self):
        """Reject auth/sampling combinations a backend would silently ignore.

        `openrouter` is implemented by codex and pi. Sampling is implemented
        only by pi's explicitly loaded provider-payload extension.
        """
        default_route = self.routing.get("default")

        def effective_backend(name: str, route: RouteConfig) -> str:
            if route.backend:
                return route.backend
            if name != "default" and default_route and default_route.backend:
                return default_route.backend
            return self.backend

        def effective_auth(name: str, route: RouteConfig) -> str:
            if route.backend_auth:
                return route.backend_auth
            if name != "default" and default_route and default_route.backend_auth:
                return default_route.backend_auth
            return self.backend_auth

        layers = [("backend_auth", self.backend, self.backend_auth)]
        for name, route in self.routing.items():
            layers.append(
                (
                    f"routing.{name}",
                    effective_backend(name, route),
                    effective_auth(name, route),
                )
            )
        for where, backend, auth in layers:
            if auth not in BACKEND_AUTHS:
                raise ValueError(
                    f"{where}: unknown backend_auth {auth!r} "
                    f"(one of {', '.join(BACKEND_AUTHS)})"
                )
            if auth == "openrouter" and backend not in {"codex", "pi"}:
                raise ValueError(
                    f"{where}: backend_auth: openrouter needs backend: codex or pi, "
                    f"not {backend!r}"
                )
        for name, route in self.routing.items():
            backend = effective_backend(name, route)
            sampling = route.sampling
            if sampling is None and default_route is not None:
                sampling = default_route.sampling
            if sampling and backend != "pi":
                raise ValueError(
                    f"routing.{name}: sampling needs backend: pi, not {backend!r}"
                )
        return self

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        require_dir: bool = True,
        **overrides,
    ) -> Config:
        """Resolve configuration. Precedence (highest wins): keyword
        overrides > the hillclimb dir's `config.yaml` > user
        `~/.config/hillclimb/config.yaml` > built-in defaults.

        An explicit `path` reads only that file (no discovery, no user
        config) — the escape hatch for tests and embedders. Otherwise the
        hillclimb dir is found by upward search; with `require_dir` (the
        default) a missing one raises HillclimbDirNotFound."""
        if path is not None:
            config = cls.model_validate(_read_yaml(path))
            config.hillclimb_dir = None
            if (path.parent / ".env").exists():
                _load_dotenv(path.parent / ".env")
            if config.pi.models_file is not None:
                config.pi.models_file = path.parent / config.pi.models_file.expanduser()
        else:
            found = find_hillclimb_dir()
            if found is None and require_dir:
                raise HillclimbDirNotFound(Path.cwd())
            data = _read_yaml(user_config_path())
            if found is not None:
                data = _deep_merge(data, _read_yaml(found / MARKER_FILE))
            config = cls.model_validate(data)
            config.hillclimb_dir = found
            if found is not None:
                # a repo keeps .env at its root, a standalone hillclimb dir
                # beside config.yaml
                for env_file in (found / ".env", found.parent / ".env"):
                    if env_file.exists():
                        _load_dotenv(env_file)
                        break
        for key, value in overrides.items():
            if value is None:
                continue
            if "." in key:
                section, field = key.split(".", 1)
                setattr(getattr(config, section), field, value)
            else:
                setattr(config, key, value)
        # Keyword overrides are applied after loading so they need the same
        # validation pass as file values (including routed sampling rules).
        hillclimb_dir = config.hillclimb_dir
        config = cls.model_validate(config.model_dump())
        config.hillclimb_dir = hillclimb_dir
        config._resolve_paths()
        return config

    def apply_overrides(self, overrides: dict[str, object]) -> None:
        """Set dotted config paths (`search.policy`, `learning.enabled`,
        `search.policy_params.population_size`, top-level `model`) with
        pydantic validation at each level — the one way an experiment arm
        or `hillclimb run --set` changes a setting. Unknown paths raise
        KeyError naming the offending key."""
        working = self.model_copy(deep=True)
        for key, value in overrides.items():
            key = LEGACY_SETTINGS.get(key, key)
            parts = key.split(".")
            target: object = working
            for part in parts[:-1]:
                if isinstance(target, dict):
                    target = target.setdefault(part, {})
                elif isinstance(target, BaseModel) and part in type(target).model_fields:
                    if part == "sampling" and getattr(target, part) is None:
                        setattr(target, part, {})
                    target = getattr(target, part)
                else:
                    raise KeyError(f"unknown config setting {key!r}")
            leaf = parts[-1]
            if isinstance(target, dict):
                target[leaf] = value
            elif isinstance(target, BaseModel) and leaf in type(target).model_fields:
                annotation = type(target).model_fields[leaf].annotation
                setattr(target, leaf, _coerce(value, annotation))
            else:
                raise KeyError(f"unknown config setting {key!r}")

        # Dotted traversal may have built raw dictionaries inside typed
        # sections (notably routing.<op>.sampling). Re-validate once so
        # experiment/CLI overrides have exactly the same guarantees as YAML.
        hillclimb_dir = self.hillclimb_dir
        validated = type(self).model_validate(working.model_dump(warnings=False))
        for field in type(self).model_fields:
            if field != "hillclimb_dir":
                setattr(self, field, getattr(validated, field))
        self.hillclimb_dir = hillclimb_dir
        self._resolve_paths()


    def _resolve_paths(self) -> None:
        """Anchor relative runs_dir/problems_dir at the folder holding the
        hillclimb dir, so commands work from any subdirectory. Without one
        (explicit-path loads, embedders) relative paths keep their CWD
        meaning."""
        if self.hillclimb_dir is None:
            return
        for section, name in (
            (self.paths, "runs_dir"),
            (self.paths, "problems_dir"),
            (self.paths, "prompts_dir"),
            (self.store, "sqlite_path"),
            (self.pi, "models_file"),
        ):
            value: Path = getattr(section, name)
            if value is not None and not value.is_absolute():
                setattr(section, name, self.hillclimb_dir.parent / value)


def _coerce(value: object, annotation: object) -> object:
    """Parse a string override (`--set search.n_replicates=3`) to the field's
    declared type; non-strings (from YAML) pass through."""
    if not isinstance(value, str):
        return value
    from typing import get_args

    candidates = [annotation, *get_args(annotation)]
    for candidate in candidates:
        if candidate is bool:
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "on", "1"):
                return True
            if lowered in ("false", "no", "off", "0"):
                return False
        elif candidate is int:
            try:
                return int(value)
            except ValueError:
                continue
        elif candidate is float:
            try:
                return float(value)
            except ValueError:
                continue
        elif candidate is Path:
            return Path(value)
    return value


def parse_set_overrides(pairs: list[str]) -> dict[str, object]:
    """`key=value` strings (the `--set` flag) → override mapping; a value that
    parses as YAML/JSON (numbers, lists, `{a: 1}`) is taken as such."""
    out: dict[str, object] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects key=value, got {pair!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        out[key.strip()] = raw if isinstance(value, str) or value is None else value
    return out
