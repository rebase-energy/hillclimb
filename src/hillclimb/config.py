from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.project import (
    find_workspace_root,
    marker_path,
    user_config_path,
    WorkspaceNotFound,
)


class BudgetConfig(BaseModel):
    total_s: int = 7200
    agent_timeout_s: int = 1800
    exec_timeout_s: int = 1800
    stop_margin_s: int = 300


class SearchConfig(BaseModel):
    """Policy knobs for one Search (the `search:` config block), not the
    Search entity itself — that lives in run.py as SearchMeta."""

    num_drafts: int = 3
    max_debug_depth: int = 3
    parallel_agents: int = 1  # >1 enables the worker pool; 1 = serial (default)
    n_trials: int = 1  # validation evals per candidate (mean val is the climbing score)
    machine_max_agents: int = 0  # machine-wide concurrent-agent cap across searches; 0 = off


class HoldoutConfig(BaseModel):
    enabled: bool = True
    fraction: float = 0.1
    seed: int = 42
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
    # Relative runs_dir/problems_dir resolve against the workspace root at
    # load time; everything hillclimb writes stays inside the workspace's
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
    budget: BudgetConfig = BudgetConfig()
    search: SearchConfig = SearchConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    paths: PathsConfig = PathsConfig()
    emflow: EmflowConfig = EmflowConfig()
    # Resolved at load time; None for embedders that construct Config()
    # directly and set absolute paths themselves (e.g. the hosted container).
    workspace_root: Path | None = Field(default=None, exclude=True)

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        require_workspace: bool = True,
        **overrides,
    ) -> Config:
        """Resolve configuration. Precedence (highest wins): keyword
        overrides > workspace `hillclimb/config.yaml` > user
        `~/.config/hillclimb/config.yaml` > built-in defaults.

        An explicit `path` reads only that file (no discovery, no user
        config) — the escape hatch for tests and embedders. Otherwise the
        workspace is found by upward search; with `require_workspace` (the
        default) a missing workspace raises WorkspaceNotFound."""
        if path is not None:
            config = cls.model_validate(_read_yaml(path))
            config.workspace_root = None
        else:
            root = find_workspace_root()
            if root is None and require_workspace:
                raise WorkspaceNotFound(Path.cwd())
            data = _read_yaml(user_config_path())
            if root is not None:
                data = _deep_merge(data, _read_yaml(marker_path(root)))
            config = cls.model_validate(data)
            config.workspace_root = root
        for key, value in overrides.items():
            if value is None:
                continue
            if "." in key:
                section, field = key.split(".", 1)
                setattr(getattr(config, section), field, value)
            else:
                setattr(config, key, value)
        config._resolve_workspace_paths()
        return config

    def _resolve_workspace_paths(self) -> None:
        """Anchor relative runs_dir/problems_dir at the workspace root so
        commands work from any subdirectory. Without a root (explicit-path
        loads, embedders) relative paths keep their CWD meaning."""
        if self.workspace_root is None:
            return
        for name in ("runs_dir", "problems_dir"):
            value: Path = getattr(self.paths, name)
            if not value.is_absolute():
                setattr(self.paths, name, self.workspace_root / value)
