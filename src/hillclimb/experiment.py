from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.node import utcnow

EXPERIMENTS_DIR = "experiments"
LEGACY_EXPERIMENT_ID = "__legacy__"
LEGACY_EXPERIMENT_NAME = "Legacy"


class ExperimentMeta(BaseModel):
    experiment_id: str
    name: str
    kind: str = "problem"
    target: str
    problem_ids: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utcnow)


def experiments_dir(runs_dir: Path) -> Path:
    return runs_dir / EXPERIMENTS_DIR


def experiment_path(runs_dir: Path, experiment_id: str) -> Path:
    return experiments_dir(runs_dir) / f"{experiment_id}.yaml"


def write_experiment(runs_dir: Path, meta: ExperimentMeta) -> Path:
    path = experiment_path(runs_dir, meta.experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False))
    return path


def load_experiments(runs_dir: Path) -> dict[str, ExperimentMeta]:
    root = experiments_dir(runs_dir)
    if not root.exists():
        return {}
    loaded = {}
    for path in sorted(root.glob("*.yaml")):
        try:
            meta = ExperimentMeta.model_validate(yaml.safe_load(path.read_text()) or {})
        except Exception:
            continue
        loaded[meta.experiment_id] = meta
    return loaded
