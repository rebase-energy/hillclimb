from __future__ import annotations

from pathlib import Path

from typing import Literal

import yaml
from pydantic import BaseModel, Field

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
    # hard agent-spend ceiling; the search parks (resumable) when cumulative
    # backend cost reaches it. 0 = no ceiling.
    max_cost_usd: float = 0.0


class SearchConfig(BaseModel):
    """Policy knobs for one Search (the `search:` config block), not the
    Search entity itself — that lives in run.py as SearchMeta."""

    num_drafts: int = 3
    max_debug_depth: int = 3
    parallel_agents: int = 1  # >1 enables the worker pool; 1 = serial (default)
    n_trials: int = 1  # validation evals per candidate (median val is the climbing score)
    # how repeated trials run. "parallel" is right for seed variance (and 3x
    # faster); "serial" is REQUIRED for anything that measures time — trials
    # sharing a machine contend, and the contention is the measurement.
    trial_mode: Literal["parallel", "serial"] = "parallel"
    # Noise guard. A candidate is only better than the incumbent when it beats
    # it by more than the band, so the search cannot climb measurement noise.
    #   min_improvement: absolute floor, in metric units
    #   noise_k: multiples of the observed noise floor (the median per-candidate
    #            trial spread); needs n_trials > 1 to have anything to measure
    # Both default to 0 = off, which is the strict comparison.
    min_improvement: float = 0.0
    noise_k: float = 0.0
    machine_max_agents: int = 0  # machine-wide concurrent-agent cap across searches; 0 = off
    policy: str = "greedy"  # search policy (policies registry)
    policy_params: dict = Field(default_factory=dict)  # opaque; validated by the policy factory


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

    backend: str | None = None
    model: str | None = None
    backend_auth: str | None = None
    # model POOL: when set (2+ entries), a UCB1 bandit picks the model per
    # call, rewarded by whether the candidate improved on its parent
    # (bandit.py). Takes precedence over `model` in the same layer; a
    # single-entry pool behaves like `model`.
    models: list[str] | None = None


class HoldoutConfig(BaseModel):
    """Selection hygiene for problems whose verifier ships a hidden split
    (`holdout: true`). `enabled: false` turns the split off search-wide."""

    enabled: bool = True
    climb_on: str = "val"  # seam only; 'holdout' climbing is a future experiment
    selection: str = "rank-blend"  # rank-blend | holdout | val
    # holdout hygiene: only candidates whose val score ranks top-k get a
    # holdout evaluation (0 = score every ok candidate). Non-top-k candidates
    # climb on val but cannot win rank-blend selection.
    top_k: int = 5


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
    # None = shared machine venv under ~/.cache/hillclimb/venvs/, keyed by a
    # hash of the requirements (+ emflow source). Set explicitly to pin.
    runtime_python: Path | None = None
    emflow_runtime_python: Path | None = None
    mlebench_python: Path = Path("../mle-bench/.venv/bin/python")
    mlebench_data_dir: Path | None = None
    kaggle_bin: Path = Path("../mle-bench/.venv/bin/kaggle")


class LearningConfig(BaseModel):
    """Cross-search learning: distill a knowledge card from every finished
    search and inject prior experience into draft prompts."""

    enabled: bool = True
    # default: <workspace>/hillclimb/knowledge (git-versionable); explicit
    # path overrides; None + no workspace = learning off
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


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config(BaseModel):
    backend: str = "claude-code"
    backend_auth: str = "subscription"  # subscription | api-key
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
    emflow: EmflowConfig = EmflowConfig()
    learning: LearningConfig = LearningConfig()
    report: ReportConfig = ReportConfig()
    operators: OperatorsConfig = OperatorsConfig()
    # Resolved at load time; None for embedders that construct Config()
    # directly and set absolute paths themselves (e.g. the hosted container).
    hillclimb_dir: Path | None = Field(default=None, exclude=True)

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
        else:
            found = find_hillclimb_dir()
            if found is None and require_dir:
                raise HillclimbDirNotFound(Path.cwd())
            data = _read_yaml(user_config_path())
            if found is not None:
                data = _deep_merge(data, _read_yaml(found / MARKER_FILE))
            config = cls.model_validate(data)
            config.hillclimb_dir = found
        for key, value in overrides.items():
            if value is None:
                continue
            if "." in key:
                section, field = key.split(".", 1)
                setattr(getattr(config, section), field, value)
            else:
                setattr(config, key, value)
        config._resolve_paths()
        return config

    def _resolve_paths(self) -> None:
        """Anchor relative runs_dir/problems_dir at the folder holding the
        hillclimb dir, so commands work from any subdirectory. Without one
        (explicit-path loads, embedders) relative paths keep their CWD
        meaning."""
        if self.hillclimb_dir is None:
            return
        for name in ("runs_dir", "problems_dir"):
            value: Path = getattr(self.paths, name)
            if not value.is_absolute():
                setattr(self.paths, name, self.hillclimb_dir.parent / value)
