from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


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


class HoldoutConfig(BaseModel):
    enabled: bool = True
    fraction: float = 0.1
    seed: int = 42
    climb_on: str = "val"  # seam only; 'holdout' climbing is a future experiment
    selection: str = "rank-blend"  # rank-blend | holdout | val


class EnsembleConfig(BaseModel):
    enabled: bool = True
    reserve_fraction: float = 0.2  # final slice of budget reserved for ensembling
    top_k: int = 3
    max_attempts: int = 2


class PathsConfig(BaseModel):
    runs_dir: Path = Path("runs")
    problems_dir: Path = Path("problems")
    runtime_python: Path = Path(".runtime-venv/bin/python")
    mlebench_python: Path = Path("../mle-bench/.venv/bin/python")
    mlebench_data_dir: Path | None = None
    kaggle_bin: Path = Path("../mle-bench/.venv/bin/kaggle")


class Config(BaseModel):
    backend: str = "claude-code"
    model: str = "sonnet"
    budget: BudgetConfig = BudgetConfig()
    search: SearchConfig = SearchConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    paths: PathsConfig = PathsConfig()

    @classmethod
    def load(cls, path: Path | None = None, **overrides) -> Config:
        """Load YAML config; non-None keyword overrides win over file values."""
        config_path = path or DEFAULT_CONFIG_PATH
        data = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text()) or {}
        config = cls.model_validate(data)
        for key, value in overrides.items():
            if value is None:
                continue
            if "." in key:
                section, field = key.split(".", 1)
                setattr(getattr(config, section), field, value)
            else:
                setattr(config, key, value)
        return config
