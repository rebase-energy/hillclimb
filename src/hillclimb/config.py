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


# Every pre-0.4 spelling of a setting -> where it lives now. Applied to a
# config file on load (`Config._from_legacy_blocks`) and to dotted overrides
# (`apply_overrides`, so `--set search.policy=openevolve` and an old
# experiment spec keep working). Prefix match: `search.policy_params.k` maps
# to `climber.params.k`. The method knobs moved into the `climber:` block
# (they are the climber's params now), the rest of `search:` split by what it
# is about.
LEGACY_SETTINGS = {
    "search.policy_params": "climber.params",
    "search.policy": "climber.ref",
    "search.tuner_params": "climber.tuner_params",
    "search.tuner": "climber.tuner",
    "search.num_drafts": "climber.params.num_drafts",
    "search.max_debug_depth": "climber.params.max_debug_depth",
    "ensemble.enabled": "climber.params.ensemble",
    "ensemble.reserve_fraction": "climber.params.ensemble_reserve_fraction",
    "ensemble.top_k": "climber.params.ensemble_top_k",
    "ensemble.max_attempts": "climber.params.ensemble_max_attempts",
    "operators.draft_retrieval": "climber.operators.draft.retrieval",
    "operators.improve_ablation": "climber.operators.improve.ablation",
    "operators.knowledge_tool": "learning.tool",
    "search.parallel_operators": "concurrency.parallel_operators",
    "search.parallel_agents": "concurrency.parallel_operators",
    "search.machine_max_operators": "concurrency.machine_max_operators",
    "search.machine_max_agents": "concurrency.machine_max_operators",
    "search.n_replicates": "evaluation.n_replicates",
    "search.n_trials": "evaluation.n_replicates",
    "search.replicate_mode": "evaluation.replicate_mode",
    "search.trial_mode": "evaluation.replicate_mode",
    "search.noise_k": "evaluation.noise_k",
    "search.min_improvement": "evaluation.min_improvement",
}
# settings that have no new home, and what to do instead
REMOVED_SETTINGS = {
    "paths.prompts_dir": (
        "prompts belong to a climber now: `hillclimb climber new mine --from greedy`, "
        "edit mine/prompts/, then run with `--climber mine`"
    ),
}


def current_setting(key: str) -> str:
    """A dotted setting in today's spelling (longest legacy prefix wins)."""
    for old in sorted(LEGACY_SETTINGS, key=len, reverse=True):
        if key == old or key.startswith(old + "."):
            return LEGACY_SETTINGS[old] + key[len(old):]
    for old, advice in REMOVED_SETTINGS.items():
        if key == old or key.startswith(old + "."):
            raise KeyError(f"{old} is gone: {advice}")
    return key


class ClimberConfig(BaseModel):
    """Which climber drives the search, and what the USER lays over it (the
    `climber:` block; `climber: greedy` is shorthand for `{ref: greedy}`).

    `ref` is a bundled name (greedy | openevolve | gepa), a directory holding
    climber.yaml, or one .py file — relative paths resolve from the folder
    holding the hillclimb dir. The rest are the manifest keys that are always
    the user's to change without copying the climber: its `params`, its
    operators' params, its tuner, its memory."""

    model_config = ConfigDict(extra="forbid")

    ref: str = "greedy"
    params: dict = Field(default_factory=dict)  # laid over the manifest's params
    # per-operator params laid over the manifest's: {draft: {retrieval: false}}
    operators: dict[str, dict] = Field(default_factory=dict)
    tuner: str | None = None  # None = the manifest's (random | optuna)
    tuner_params: dict = Field(default_factory=dict)
    memory: Literal["knowledge-graph", "none"] | None = None  # None = the manifest's

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data):
        return {"ref": data} if isinstance(data, str) else data


class EvaluationConfig(BaseModel):
    """How a candidate is measured (the `evaluation:` block) — the harness's,
    never a climber's."""

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


class ConcurrencyConfig(BaseModel):
    """How much runs at once (the `concurrency:` block)."""

    parallel_operators: int = 1  # attempts in flight per search; 1 = serial (default)
    # Machine-wide cap on concurrent operators across every search on this
    # machine (flock slots in ~/.cache/hillclimb/agent-slots/). Operators
    # beyond it wait (`waiting-slot` in watch). 0 = off; None = default_machine_max_operators().
    machine_max_operators: int | None = None

    def effective_machine_max_operators(self) -> int:
        if self.machine_max_operators is None:
            return default_machine_max_operators()
        return self.machine_max_operators


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


class PathsConfig(BaseModel):
    # Relative runs_dir/problems_dir resolve at load time against the folder
    # holding the hillclimb dir; everything hillclimb writes stays inside the
    # hillclimb/ folder by default.
    runs_dir: Path = Path("hillclimb/runs")
    problems_dir: Path = Path("hillclimb/problems")
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
    # advertise the read-only `hillclimb knowledge query` lookup to every
    # agent (a clause of the contract; needs `enabled`)
    tool: bool = True
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
    `module:Class` — see modules/similarity/__init__.py."""

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
    climber: ClimberConfig = ClimberConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    paths: PathsConfig = PathsConfig()
    store: StoreConfig = StoreConfig()
    emflow: EmflowConfig = EmflowConfig()
    einsteinarena: EinsteinArenaConfig = EinsteinArenaConfig()
    pi: PiConfig = PiConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    learning: LearningConfig = LearningConfig()
    report: ReportConfig = ReportConfig()
    # Resolved at load time; None for embedders that construct Config()
    # directly and set absolute paths themselves (e.g. the hosted container).
    hillclimb_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_blocks(cls, data):
        """A config file written for 0.3 (`search:`, `ensemble:`,
        `operators:` blocks) still loads: every setting is moved to where it
        lives now (LEGACY_SETTINGS). A setting with no new home raises with
        what to do instead."""
        if not isinstance(data, dict) or not any(k in data for k in ("search", "ensemble", "operators")) and not (
            isinstance(data.get("paths"), dict) and "prompts_dir" in data["paths"]
        ):
            return data
        data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in data.items()}
        moved: dict[str, object] = {}
        for block in ("search", "ensemble", "operators"):
            for name, value in (data.pop(block, None) or {}).items():
                moved[current_setting(f"{block}.{name}")] = value
        if isinstance(data.get("paths"), dict) and "prompts_dir" in data["paths"]:
            current_setting("paths.prompts_dir")  # raises with the advice
        for key, value in moved.items():
            target = data
            *parents, leaf = key.split(".")
            for part in parents:
                node = target.get(part)
                if isinstance(node, str) and part == "climber":
                    node = {"ref": node}
                target[part] = node = dict(node or {})
                target = node
            if isinstance(value, dict) and isinstance(target.get(leaf), dict):
                target[leaf] = {**value, **target[leaf]}  # an explicit new-style value wins
            else:
                target.setdefault(leaf, value)
        return data

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
            *parents, leaf = current_setting(key).split(".")
            target: object = config
            for part in parents:
                target = target[part] if isinstance(target, dict) else getattr(target, part)
            if isinstance(target, dict):
                target[leaf] = value
            else:
                setattr(target, leaf, value)
        # Keyword overrides are applied after loading so they need the same
        # validation pass as file values (including routed sampling rules).
        hillclimb_dir = config.hillclimb_dir
        config = cls.model_validate(config.model_dump())
        config.hillclimb_dir = hillclimb_dir
        config._resolve_paths()
        return config

    def apply_overrides(self, overrides: dict[str, object]) -> None:
        """Set dotted config paths (`climber.ref`, `learning.enabled`,
        `climber.params.population_size`, top-level `model`) with
        pydantic validation at each level — the one way an experiment arm
        or `hillclimb run --set` changes a setting. Unknown paths raise
        KeyError naming the offending key."""
        working = self.model_copy(deep=True)
        for key, value in overrides.items():
            key = current_setting(key)
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
