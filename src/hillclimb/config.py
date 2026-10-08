from __future__ import annotations

import os

from pathlib import Path

from typing import Any, Literal

import yaml
from pydantic import PrivateAttr, BaseModel, ConfigDict, Field, field_validator, model_validator

from hillclimb.project import (
    find_hillclimb_dir,
    MARKER_FILE,
    user_config_path,
    user_env_path,
    HillclimbDirNotFound,
)
from hillclimb.modules.spec import ClimberSpec


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
    # hard coding-agent-spend ceiling; the search parks (resumable) when cumulative
    # coding agent cost reaches it. 0 = no ceiling.
    max_cost_usd: float = 0.0
    # The budget is the USER's, in every dimension — a climber sees what is
    # left (BudgetView) and never sets it. Both end the search like the clock
    # does (no new work; what is in flight still lands). 0 = no limit.
    #   max_evaluations  verifier trials the climber caused: one per scored
    #                    attempt plus one per tune trial; the baseline and the
    #                    seed are the harness's own floor and do not count
    #   max_tokens       tokens its coding agent calls consumed, all kinds summed
    max_evaluations: int = 0
    max_tokens: int = 0


def default_machine_max_agents() -> int:
    """min(8, cores - 2): each operator is an API-bound coding agent plus, at worst,
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
    "search.parallel_agents": "concurrency.parallel_agents",
    "search.parallel_operators": "concurrency.parallel_agents",
    "concurrency.parallel_operators": "concurrency.parallel_agents",
    "search.machine_max_agents": "concurrency.machine_max_agents",
    "search.machine_max_operators": "concurrency.machine_max_agents",
    "concurrency.machine_max_operators": "concurrency.machine_max_agents",
    # the coding agent used to be called the backend
    "backend": "agent",
    "backend_auth": "agent_auth",
    "search.n_replicates": "evaluation.n_replicates",
    "search.n_trials": "evaluation.n_replicates",
    # how many of a trial's replicates run at once was a mode (parallel |
    # serial) until 0.7.2; it is a count now (`LEGACY_VALUES` maps the words)
    "search.replicate_mode": "concurrency.parallel_replicates",
    "search.trial_mode": "concurrency.parallel_replicates",
    "evaluation.replicate_mode": "concurrency.parallel_replicates",
    "search.noise_k": "evaluation.noise_k",
    "search.min_improvement": "evaluation.min_improvement",
    # 0.5: how memory behaves was the user's `learning:` block, and the graph
    # module sat beside `memory:`; both are the memory's params now
    **{f"learning.{name}": f"climber.memory_params.{name}" for name in (
        "max_cards", "live", "complexity_prior", "claims", "graph_retrieval", "credit", "playbooks", "skills",
    )},
    "climber.graph": "climber.memory_params.graph",
}
# settings that have no new home, and what to do instead
REMOVED_SETTINGS = {
    "paths.prompts_dir": (
        "prompts belong to the climber: name the directory in its block, `climber: {prompts: prompts/}`"
    ),
}


# The coding agent used to be called the backend, and the coding agents run in
# parallel used to be counted as operators: keys in either spelling load.
RENAMED_KEYS = {
    "backend": "agent",
    "backend_auth": "agent_auth",
    "parallel_operators": "parallel_agents",
    "machine_max_operators": "machine_max_agents",
}


# a setting whose VALUES changed spelling along with its key: the old words
# read as the new numbers (`replicate_mode: serial` is `parallel_replicates: 1`)
LEGACY_VALUES: dict[str, dict[str, object]] = {
    "concurrency.parallel_replicates": {"parallel": 0, "serial": 1},
}


def legacy_value(key: str, value):
    """`value` for the setting `key` (in today's spelling) with an old word
    read as what it means now; anything else comes back as it is."""
    words = LEGACY_VALUES.get(key)
    if words is not None and isinstance(value, str) and value.strip().lower() in words:
        return words[value.strip().lower()]
    return value


def renamed_keys(data):
    """A mapping with the old spellings moved to the new (new wins on a clash)."""
    if not isinstance(data, dict) or not any(k in data for k in RENAMED_KEYS):
        return data
    data = dict(data)
    for old, new in RENAMED_KEYS.items():
        if old in data:
            value = data.pop(old)
            data.setdefault(new, value)
    return data


def current_setting(key: str) -> str:
    """A dotted setting in today's spelling (longest legacy prefix wins)."""
    for old in sorted(LEGACY_SETTINGS, key=len, reverse=True):
        if key == old or key.startswith(old + "."):
            key = LEGACY_SETTINGS[old] + key[len(old):]
            break
    for old, advice in REMOVED_SETTINGS.items():
        if key == old or key.startswith(old + "."):
            raise KeyError(f"{old} is gone: {advice}")
    if key == "climber.ref":
        return "climber"  # 0.5 named a climber by `ref`; naming one replaces the block
    # the two decisions were `climber.policy` and `climber.select` (its knobs
    # `climber.select_params`) until 0.7
    from hillclimb.modules.spec import RENAMED_BLOCK_KEYS

    for old, new in RENAMED_BLOCK_KEYS.items():
        if key == f"climber.{old}" or key.startswith(f"climber.{old}."):
            key = f"climber.{new}" + key[len(f"climber.{old}"):]
            break
    if key.startswith("climber.operators."):
        # `operators` is the LIST of what may run; one operator's params are
        # addressed by its name
        return "climber.operator_params." + key[len("climber.operators."):]
    if key.startswith("climber.params."):
        # the schedule is the selector policy's: `climber.params.num_drafts`
        # (every spelling before 0.7) is `climber.selector_params.num_drafts`
        from hillclimb.modules.spec import SCHEDULE_KNOBS

        knob = key[len("climber.params."):].split(".", 1)[0]
        if knob in SCHEDULE_KNOBS:
            return "climber.selector_params." + key[len("climber.params."):]
    return key


def _climber_from_override(key: str, value) -> ClimberSpec:
    """`--set climber.<field>=…` on a folder that names no climber: naming
    one (`operator_policy`, `loop`) starts the block; a knob alone has
    nothing to land on, and the error says how to fetch a climber."""
    from hillclimb.modules.refs import NoClimber

    leaf = key.split(".", 1)[1]
    if leaf in ("operator_policy", "loop"):
        return ClimberSpec.model_validate({leaf: value})
    raise NoClimber(
        f"no climber to edit with `--set {key}=…`: name one first "
        "(`hillclimb climber get greedy`, or `--climber <file.py>`)"
    )


class EvaluationConfig(BaseModel):
    """How a candidate is measured (the `evaluation:` block) — the harness's,
    never a climber's."""

    # seeded executions per trial (one parameter set); the trial's score is
    # their MEDIAN. Replicate variance is noise, never something to climb.
    # How many of them run at once is `concurrency.parallel_replicates`.
    n_replicates: int = 1
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

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data):
        return renamed_keys(data)

    parallel_agents: int = 1  # attempts in flight per search; 1 = serial (default)
    # how many of a trial's replicates (`evaluation.n_replicates` seeded runs)
    # execute at once: 0 = all of them (the default, right for seed variance),
    # 1 = one after another — REQUIRED for a metric that measures the machine
    # (time, throughput, memory): runs sharing it contend, and the contention
    # is the measurement. `replicate_mode: parallel | serial` was the pre-0.7.2
    # spelling of 0 | 1.
    parallel_replicates: int = Field(default=0, ge=0)
    # Machine-wide cap on concurrent coding agents across every search on this
    # machine (flock slots in ~/.cache/hillclimb/agent-slots/). Operators
    # beyond it wait (`waiting-slot` in watch). 0 = off; None = default_machine_max_agents().
    machine_max_agents: int | None = None
    # CPU cores each run of a solution may use: verifier and holdout runs and
    # the coding agents' own test runs get it as $HILLCLIMB_CPUS, with the math
    # libraries' thread pools capped to match (executor.single_threaded). 1 =
    # one core per run, so parallel agents and replicates never contend.
    # Nothing enforces it beyond that; a run whose CPU time exceeds its wall
    # time well past the allotment is flagged (`Replicate.oversubscribed`).
    solution_cpus: int = Field(default=1, ge=1)

    def effective_machine_max_agents(self) -> int:
        if self.machine_max_agents is None:
            return default_machine_max_agents()
        return self.machine_max_agents

    def replicates_at_once(self, n_replicates: int) -> int:
        """How many of a trial's `n_replicates` runs execute at once."""
        n = max(1, n_replicates)
        return n if self.parallel_replicates == 0 else min(n, self.parallel_replicates)


class RouteConfig(BaseModel):
    """Per-operator coding agent/model override (the `routing:` config block).
    None fields inherit the global `agent`/`model`/`agent_auth` scalars."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data):
        return renamed_keys(data)

    agent: str | None = None
    model: str | None = None
    agent_auth: str | None = None
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
    # Relative runs_dir/problems_dir resolve at load time against the
    # hillclimb dir (the folder holding hillclimb.yaml).
    runs_dir: Path = Path("runs")
    problems_dir: Path = Path("problems")
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
    sqlite_path: Path = Path("store.sqlite")


class LearningConfig(BaseModel):
    """Cross-search learning — what is the USER's about it. Which memory a
    search uses and how it behaves (cards in the prompt, claims, skills, the
    graph module, ...) is the climber's: `climber.memory` and
    `climber.memory_params`."""

    # the master switch: false runs every climber without cross-search
    # memory, whatever its block says (`--no-learning` is the same)
    enabled: bool = True
    # advertise the read-only `hillclimb knowledge query` lookup to every
    # coding agent (a clause of the contract; needs `enabled`)
    tool: bool = True
    # default: <hillclimb dir>/knowledge (git-versionable); explicit
    # path overrides; None + no hillclimb dir = learning off
    dir: Path | None = None
    # wall clock for the one coding agent pass a memory may make after a search
    # (claim distillation; routed via `routing: distill:`)
    claims_timeout_s: int = 300


class AgentContextConfig(BaseModel):
    """What operators know beyond the prompt (`harness/agent_context.py`):
    the skills and AGENTS.md of the global layer (~/.config/hillclimb/agent/)
    and the hillclimb dir's own (agent/), rendered for whichever coding agent
    runs. Skills an operator creates are added to the local layer."""

    model_config = ConfigDict(extra="forbid")

    # use the global layer too (false keeps this folder to its own)
    include_global: bool = True
    skip: list[str] = Field(default_factory=list)  # skills left out, by name
    # Claude Code plugins operators run with (`--plugin-dir`), relative to
    # the hillclimb dir; none by default: the user's own plugins never apply
    claude_plugins: list[Path] = Field(default_factory=list)


# the `learning:` settings that described how memory BEHAVES moved into the
# climber's block in 0.6: they are its memory's params
MEMORY_PARAMS_FROM_LEARNING = (
    "max_cards", "live", "complexity_prior", "claims", "graph_retrieval", "credit", "playbooks", "skills",
)


class ReportConfig(BaseModel):
    """Trial evaluation reports: per-zone/horizon/quantile breakdowns from
    emflow validation evals."""

    # gates ONLY prompt injection — computation and journal storage always
    # run, so A/B experiments record identical data and differ only in what the
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
    """pi coding agent settings shared by routed pi instances."""

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


class SandboxConfig(BaseModel):
    """The OS sandbox coding agents and verifiers run in (`harness/sandbox.py`):
    sandbox-exec on macOS, bubblewrap on Linux. On by default; a search
    refuses to start when it is on and cannot be started. `sandbox: off`
    is the short form of `sandbox: {enabled: false}`."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    write: list[Path] = Field(default_factory=list)  # writable beyond the candidate dir
    deny_read: list[Path] = Field(default_factory=list)  # unreadable beyond the built-in secrets
    # hosts coding agents still reach with allow_internet_for_agents: false, beyond
    # their model provider's
    allow_hosts: list[str] = Field(default_factory=list)
    local_ports: list[int] = Field(default_factory=list)  # localhost ports left open without network


AGENT_AUTHS = ("subscription", "api-key", "openrouter")


class Config(BaseModel):
    agent: str = "claude-code"
    agent_auth: str = "subscription"  # one of AGENT_AUTHS
    model: str = "sonnet"
    # may the coding agents reach the internet (web search/fetch, curl, pip
    # from their shell)? False turns their web tools off, runs every shell
    # command they issue in an OS network jail and drops the draft's
    # web-research cue. Whether the SOLUTION may use the internet is the
    # problem's `allow_internet_during_solution`, not this.
    allow_internet_for_agents: bool = True
    # per-operator routing; keys: draft | debug | improve | ensemble |
    # distill | default. Missing keys (or an absent block) fall back to the
    # scalars above (except distill's model, which defaults to haiku).
    routing: dict[str, RouteConfig] = Field(default_factory=dict)
    budget: BudgetConfig = BudgetConfig()
    # the climber: the SAME block a run spec entry takes (`modules/spec.py`);
    # here it is the folder's default — and there is none until the folder
    # names one (`hillclimb climber get greedy` pins the catalog's copy): the
    # engine ships no climber of its own
    climber: ClimberSpec | None = None
    evaluation: EvaluationConfig = EvaluationConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    paths: PathsConfig = PathsConfig()
    store: StoreConfig = StoreConfig()
    emflow: EmflowConfig = EmflowConfig()
    einsteinarena: EinsteinArenaConfig = EinsteinArenaConfig()
    sandbox: SandboxConfig = SandboxConfig()
    pi: PiConfig = PiConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    learning: LearningConfig = LearningConfig()
    agent_context: AgentContextConfig = AgentContextConfig()
    report: ReportConfig = ReportConfig()
    # Resolved at load time; None for embedders that construct Config()
    # directly and set absolute paths themselves (e.g. the hosted container).
    hillclimb_dir: Path | None = Field(default=None, exclude=True)
    # a `hillclimb.Climber` composed in Python from classes that exist only
    # in this process: it cannot be a block, so it rides beside the config
    # (in-process runs only — see `api.run`)
    _live_climber: Any = PrivateAttr(default=None)
    # the block is a RECORD's (a resumed search from before snapshots): its
    # pre-0.9 registry names resolve to the catalog's files
    _climber_legacy_names: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _renamed_scalars(cls, data):
        return renamed_keys(data)

    @model_validator(mode="before")
    @classmethod
    def _sandbox_switch(cls, data):
        """`sandbox: off` (YAML reads it as false) and `sandbox: on`."""
        if isinstance(data, dict) and isinstance(data.get("sandbox"), (bool, str)):
            value = data["sandbox"]
            if isinstance(value, str):
                if value.strip().lower() not in {"on", "off"}:
                    raise ValueError(f"sandbox: {value!r} is neither on nor off")
                value = value.strip().lower() == "on"
            data = {**data, "sandbox": {"enabled": value}}
        return data

    @model_validator(mode="before")
    @classmethod
    def _replicate_mode(cls, data):
        """`replicate_mode: parallel | serial` (under `evaluation:`, or a
        0.3 `search:` block's `replicate_mode`/`trial_mode`) is
        `concurrency.parallel_replicates: 0 | 1` now. Read wherever the
        validators' order left it, the words mapped to the numbers; an
        explicit `parallel_replicates` wins."""
        if not isinstance(data, dict):
            return data
        found = None
        for block, keys in (("search", ("replicate_mode", "trial_mode")), ("evaluation", ("replicate_mode",))):
            section = data.get(block)
            if isinstance(section, dict) and any(k in section for k in keys):
                data = {**data, block: dict(section)}
                for k in keys:
                    if k in data[block]:
                        found = data[block].pop(k)
        concurrency = data.get("concurrency")
        if isinstance(concurrency, dict) and isinstance(concurrency.get("parallel_replicates"), str):
            data = {**data, "concurrency": {**concurrency, "parallel_replicates": legacy_value(
                "concurrency.parallel_replicates", concurrency["parallel_replicates"])}}
        if found is not None:
            concurrency = dict(data.get("concurrency") or {})
            concurrency.setdefault("parallel_replicates", legacy_value("concurrency.parallel_replicates", found))
            data = {**data, "concurrency": concurrency}
        return data

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_blocks(cls, data):
        """A config file written for 0.3 (`search:`, `ensemble:`,
        `operators:` blocks) still loads: every setting is moved to where it
        lives now (LEGACY_SETTINGS). A setting with no new home raises with
        what to do instead."""
        moved_learning = isinstance(data, dict) and isinstance(data.get("learning"), dict) and any(
            name in data["learning"] for name in MEMORY_PARAMS_FROM_LEARNING
        )
        if not isinstance(data, dict) or not any(k in data for k in ("search", "ensemble", "operators")) and not (
            isinstance(data.get("paths"), dict) and "prompts_dir" in data["paths"]
        ) and not moved_learning:
            return data
        data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in data.items()}
        moved: dict[str, object] = {}
        if moved_learning:  # a 0.5 `learning:` block: its behaviour flags are the memory's params
            for name in MEMORY_PARAMS_FROM_LEARNING:
                if name in data["learning"]:
                    moved[f"climber.memory_params.{name}"] = data["learning"].pop(name)
        for block in ("search", "ensemble", "operators"):
            for name, value in (data.pop(block, None) or {}).items():
                key = current_setting(f"{block}.{name}")
                # the climber's name goes in as the 0.5 `ref` key, which the
                # block expands under whatever params were moved beside it
                moved["climber.ref" if key == "climber" else key] = value
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
    def _check_agent_auth(self):
        """Reject auth/sampling combinations a coding agent would silently ignore.

        `openrouter` is implemented by codex and pi. Sampling is implemented
        only by pi's explicitly loaded provider-payload extension.
        """
        default_route = self.routing.get("default")

        def effective_agent(name: str, route: RouteConfig) -> str:
            if route.agent:
                return route.agent
            if name != "default" and default_route and default_route.agent:
                return default_route.agent
            return self.agent

        def effective_auth(name: str, route: RouteConfig) -> str:
            if route.agent_auth:
                return route.agent_auth
            if name != "default" and default_route and default_route.agent_auth:
                return default_route.agent_auth
            return self.agent_auth

        layers = [("agent_auth", self.agent, self.agent_auth)]
        for name, route in self.routing.items():
            layers.append(
                (
                    f"routing.{name}",
                    effective_agent(name, route),
                    effective_auth(name, route),
                )
            )
        for where, agent, auth in layers:
            if auth not in AGENT_AUTHS:
                raise ValueError(
                    f"{where}: unknown agent_auth {auth!r} "
                    f"(one of {', '.join(AGENT_AUTHS)})"
                )
            if auth == "openrouter" and agent not in {"codex", "pi"}:
                raise ValueError(
                    f"{where}: agent_auth: openrouter needs agent: codex or pi, "
                    f"not {agent!r}"
                )
        for name, route in self.routing.items():
            agent = effective_agent(name, route)
            sampling = route.sampling
            if sampling is None and default_route is not None:
                sampling = default_route.sampling
            if sampling and agent != "pi":
                raise ValueError(
                    f"routing.{name}: sampling needs agent: pi, not {agent!r}"
                )
        return self

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        require_dir: bool = True,
        start: Path | None = None,
        **overrides,
    ) -> Config:
        """Resolve configuration. Precedence (highest wins): keyword
        overrides > the hillclimb dir's `hillclimb.yaml` > user
        `~/.config/hillclimb/config.yaml` > built-in defaults.

        An explicit `path` reads only that file (no discovery, no user
        config) — the escape hatch for tests and embedders. Otherwise the
        hillclimb dir is `start` (default CWD) or its `hillclimb/` subfolder —
        no upward search;
        with `require_dir` (the default) a missing one raises
        HillclimbDirNotFound."""
        if path is not None:
            config = cls.model_validate(_read_yaml(path))
            config.hillclimb_dir = None
            if (path.parent / ".env").exists():
                _load_dotenv(path.parent / ".env")
            if config.pi.models_file is not None:
                config.pi.models_file = path.parent / config.pi.models_file.expanduser()
        else:
            found = find_hillclimb_dir(start)
            if found is None and require_dir:
                raise HillclimbDirNotFound((start or Path.cwd()).resolve())
            # old spellings are renamed per level BEFORE the levels merge: a
            # folder's `backend: dummy` must beat the user level's `agent:`,
            # not lose to it as "the new spelling wins" would after merging
            data = renamed_keys(_read_yaml(user_config_path()))
            if isinstance(data.get("climber"), (dict, str)):
                # a block means what it says where it was written: its file
                # refs resolve from the user config's own folder
                data["climber"] = ClimberSpec.model_validate(data["climber"]).anchored(
                    user_config_path().parent
                ).block()
            if found is not None:
                folder = renamed_keys(_read_yaml(found / MARKER_FILE))
                data = _deep_merge(data, folder)
                if "climber" in folder:
                    # a climber is ONE block: the folder's replaces the user
                    # level's whole, never merges into it (its params belong
                    # to its policy, not to whichever policy ends up chosen).
                    # A string names a file or a climber folder
                    # (looked for under the hillclimb dir); its refs stay
                    # relative, anchored at the hillclimb dir when used
                    data["climber"] = (
                        ClimberSpec.model_validate(folder["climber"], context={"base_dir": found}).block()
                        if isinstance(folder["climber"], str) else folder["climber"]
                    )
            config = cls.model_validate(data)
            config.hillclimb_dir = found
            if found is not None and (found / ".env").exists():
                _load_dotenv(found / ".env")
            # The user-level .env last: `_load_dotenv` never overrides, so
            # the shell wins over the folder's file, which wins over this
            # one — the same order as the config files.
            if user_env_path().exists():
                _load_dotenv(user_env_path())
        for key, value in overrides.items():
            if value is None:
                continue
            key = current_setting(key)
            value = legacy_value(key, value)
            *parents, leaf = key.split(".")
            if not parents and leaf == "climber":
                # a name, a file, a climber folder (under the hillclimb dir) or a block
                value = ClimberSpec.model_validate(value, context={"base_dir": config.hillclimb_dir})
            elif parents and parents[0] == "climber" and config.climber is None:
                config.climber = _climber_from_override(key, value)
                continue
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
        """Set dotted config paths (`climber`, `learning.enabled`,
        `climber.params.population_size`, top-level `model`) with
        pydantic validation at each level — the one way a study experiment
        or `hillclimb run --set` changes a setting. Unknown paths raise
        KeyError naming the offending key."""
        working = self.model_copy(deep=True)
        for key, value in overrides.items():
            key = current_setting(key)
            value = legacy_value(key, value)
            if key == "climber":
                # naming a climber (a file, a folder, a whole block)
                # replaces the block; later `climber.<field>` overrides then edit it
                working.climber = ClimberSpec.model_validate(value, context={"base_dir": self.hillclimb_dir})
                continue
            if key.startswith("climber.") and working.climber is None:
                # no climber to edit: naming one starts the block, a knob alone has nothing to land on
                working.climber = _climber_from_override(key, value)
                continue
            if key in ("climber.operator_policy", "climber.loop"):
                # a climber has one or the other: naming one drops the other
                working.climber.operator_policy = working.climber.loop = None
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


    def climber_block(self) -> dict:
        """The climber block as it travels — into a run's spec, to a child
        engine: every file ref absolute (relative ones resolve from the
        hillclimb dir), so it means the same thing wherever it is read.
        `NoClimber` when the folder names none."""
        if self.climber is None:
            from hillclimb.modules.refs import NO_CLIMBER_HINT, NoClimber

            raise NoClimber(NO_CLIMBER_HINT)
        return self.climber.anchored(self.hillclimb_dir).block()

    def climber_params(self, field: str) -> dict:
        """One of the block's param maps (`tuner_params`, `operator_params`,
        `memory_params`, …), `{}` when the folder names no climber: what the
        harness reads without needing a climber to exist."""
        return dict(getattr(self.climber, field) or {}) if self.climber is not None else {}

    def climber_label(self) -> str | None:
        return self.climber.label if self.climber is not None else None

    def _resolve_paths(self) -> None:
        """Anchor relative runs_dir/problems_dir at the hillclimb dir, so
        commands work from any subdirectory. Without one
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
                setattr(section, name, self.hillclimb_dir / value)


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
