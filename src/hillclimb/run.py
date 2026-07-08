from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from hillclimb.candidate import utcnow

SCHEMA_VERSION = 2
SEARCHES_DIRNAME = "searches"
RUN_META_FILE = "run.yaml"
SEARCH_META_FILE = "search.yaml"


class RunMeta(BaseModel):
    """runs/<run-id>/run.yaml — one invocation of hillclimb."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    name: str
    kind: str = "problem"  # problem | suite
    target: str
    spec: str | None = None  # workspace-relative run-spec file, when launched from one
    problem_ids: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utcnow)


class SearchMeta(BaseModel):
    """runs/<run-id>/searches/<search-id>/search.yaml — one search worker on
    one problem."""

    schema_version: int = SCHEMA_VERSION
    search_id: str
    run_id: str
    problem: str  # path to the problem definition dir
    problem_id: str
    backend: str
    model: str
    # additive with defaults on purpose: bumping SCHEMA_VERSION would hide
    # every existing run dir from the scanners (exact-match gate below)
    policy: str = "greedy"
    policy_params: dict = Field(default_factory=dict)
    routing: dict = Field(default_factory=dict)  # RouteConfig dumps by operator
    metric: str
    lower_is_better: bool = False
    budget_s: int = 0
    holdout_enabled: bool = False
    holdout_seed: int | None = None
    holdout_fraction: float | None = None
    holdout_strategy: str = "random"
    seed_from: str | None = None  # incumbent solution the search was seeded with
    started_at: str = Field(default_factory=utcnow)


def _load_meta(path: Path, model: type[BaseModel]):
    """Parse a metadata file; None unless it is valid AND schema v2. This is
    the gate that makes pre-v2 run dirs invisible to every scanner."""
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        return model.model_validate(data)
    except Exception:
        return None


def write_run_meta(run_dir: Path, meta: RunMeta) -> Path:
    path = run_dir / RUN_META_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False))
    return path


def load_run_meta(run_dir: Path) -> RunMeta | None:
    return _load_meta(run_dir / RUN_META_FILE, RunMeta)


def write_search_meta(search_dir: Path, meta: SearchMeta) -> Path:
    path = search_dir / SEARCH_META_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False))
    return path


def load_search_meta(search_dir: Path) -> SearchMeta | None:
    return _load_meta(search_dir / SEARCH_META_FILE, SearchMeta)


def iter_run_dirs(runs_dir: Path) -> list[Path]:
    """v2 run dirs, newest first by run.yaml mtime."""
    if not runs_dir.exists():
        return []
    found = [d for d in runs_dir.iterdir() if d.is_dir() and load_run_meta(d) is not None]
    return sorted(found, key=lambda d: (d / RUN_META_FILE).stat().st_mtime, reverse=True)


def iter_search_dirs(run_dir: Path) -> list[Path]:
    """v2 search dirs within a run, sorted by name."""
    root = run_dir / SEARCHES_DIRNAME
    if not root.exists():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and load_search_meta(d) is not None
    )


def latest_search_dir(runs_dir: Path) -> Path | None:
    """The most recently active search across all v2 runs: newest status.json
    mtime, falling back to search.yaml mtime for searches that never started."""

    def activity(search_dir: Path) -> float:
        status = search_dir / "status.json"
        marker = status if status.exists() else search_dir / SEARCH_META_FILE
        return marker.stat().st_mtime

    searches = [s for run in iter_run_dirs(runs_dir) for s in iter_search_dirs(run)]
    return max(searches, key=activity) if searches else None
