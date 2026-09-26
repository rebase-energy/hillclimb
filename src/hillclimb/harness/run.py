from __future__ import annotations

from pathlib import Path

import uuid

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from hillclimb.harness.candidate import utcnow
from hillclimb.harness.direction import legacy_direction_key

SCHEMA_VERSION = 3
# what `_load_meta` still reads: v2 search records name a `policy`; they are
# mapped onto `climber` on load (SearchMeta._from_v2), so runs made before
# the climber vocabulary stay visible to every view
READABLE_SCHEMA_VERSIONS = (2, 3)
SEARCHES_DIRNAME = "searches"
RUN_META_FILE = "run.yaml"
SEARCH_META_FILE = "search.yaml"


class RunMeta(BaseModel):
    """runs/<run-id>/run.yaml — one invocation of hillclimb."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    name: str
    kind: str = "problem"  # problem | suite | experiment
    target: str
    spec: str | None = None  # hillclimb-dir-relative run-spec file, when launched from one
    problem_ids: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utcnow)


class SearchMeta(BaseModel):
    """runs/<run-id>/searches/<search-id>/search.yaml — one search worker on
    one problem."""

    schema_version: int = SCHEMA_VERSION
    search_id: str
    run_id: str
    problem: str  # path to the problem definition dir, or the provider target
    problem_id: str
    # The problem is an attribute of the search, and this is its canonical
    # identity across runs: `emflow://gefcom2014:solar`, `mlebench://<comp>`,
    # or the local problem id. Every problem-scoped view (the climb chart,
    # best-ever, knowledge) groups on it. Empty in pre-key search.yaml files
    # and backfilled on read — same rule as hillclimb-go's EffectiveProblemKey.
    problem_key: str = ""
    # Globally unique identity of this search (uuid4 at creation) so records
    # can live in a store shared across hillclimb dirs/machines without a
    # migration; `<run-id>/<search-id>` stays the human address. Pre-uid
    # search.yaml files get a deterministic uuid5 of that address on read.
    search_uid: str = ""
    backend: str
    model: str
    # additive with defaults on purpose: a field without one would hide every
    # existing run dir from the scanners
    #
    # The climber this search ran: the reference as written (a bundled name,
    # a directory, one file), the hash of its files at search start (the
    # identity of an edited exploration process, like seed_sha256), the
    # manifest as loaded, and what the USER laid over its params. The files
    # themselves are snapshotted into `<search_dir>/climber/`, which is what
    # the engine — and a resume — loads.
    climber: str = "greedy"
    climber_sha256: str | None = None
    climber_manifest: dict = Field(default_factory=dict)
    climber_params: dict = Field(default_factory=dict)
    hillclimb_version: str | None = None
    # the user's tuner override (None = the manifest's)
    tuner: str | None = None
    tuner_params: dict = Field(default_factory=dict)
    routing: dict = Field(default_factory=dict)  # RouteConfig dumps by operator
    metric: str
    higher_is_better: bool = True
    # Snapshot of problem.yaml's named chart reference lines. The chart also
    # reloads a reachable local problem so edits apply to existing searches.
    chart_baselines: dict[str, float] = Field(default_factory=dict)
    # Provider provenance and candidate outputs are snapshotted so offline
    # control operations and summit never need to rematerialize the problem.
    provider_revision: str | None = None
    output_artifacts: list[str] = Field(default_factory=lambda: ["submission.csv"])
    # Optional run-frozen correctness suite. The bundle path is relative to
    # the run dir; command and digest are copied here so resume never consults
    # an edited live problem.yaml.
    unit_tests_bundle: str | None = None
    unit_tests_command: list[str] = Field(default_factory=list)
    unit_tests_sha256: str | None = None

    @field_validator("output_artifacts")
    @classmethod
    def _safe_output_artifacts(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for name in value:
            path = Path(name)
            if not name or path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
                raise ValueError(f"output artifact must be a file name, got {name!r}")
            if name not in out:
                out.append(name)
        return out

    @model_validator(mode="before")
    @classmethod
    def _legacy_direction_key(cls, data):
        return legacy_direction_key(data)

    @model_validator(mode="before")
    @classmethod
    def _from_v2(cls, data):
        """A record written before the climber vocabulary (schema v2) named a
        `policy`. Same thing, older word: map it, so every view keeps
        showing runs made before the rename. In every store backend — this
        runs wherever a SearchMeta is validated."""
        if not isinstance(data, dict) or "policy" not in data:
            return data
        data = dict(data)
        data.setdefault("climber", data.pop("policy"))
        data.setdefault("climber_params", data.pop("policy_params", None) or {})
        data.setdefault("climber_sha256", data.pop("policy_sha256", None))
        for gone in ("policy", "policy_params", "policy_sha256", "templates_sha256", "templates_overridden"):
            data.pop(gone, None)
        data["schema_version"] = SCHEMA_VERSION
        return data
    budget_s: int = 0
    holdout_enabled: bool = False
    seed_from: str | None = None  # incumbent solution the search was seeded with
    # sha256 of that seed file's bytes at search start: the identity views
    # compare when several searches claim to share one seed
    seed_sha256: str | None = None
    # whether cross-search memory was active
    learning_enabled: bool = True
    # Experiment tags (experiment.py): which experiment and arm this search
    # belongs to, its repeat index, and the config overrides the arm applied
    # — the grouping keys for every setup-vs-setup comparison
    experiment: str | None = None
    arm: str | None = None
    repeat: int = 0
    arm_overrides: dict = Field(default_factory=dict)
    started_at: str = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _backfill_problem_key(self):
        if not self.problem_key:
            self.problem_key = problem_key_for(self.problem, self.problem_id)
        if not self.search_uid:
            self.search_uid = uuid.uuid5(SEARCH_UID_NAMESPACE, f"{self.run_id}/{self.search_id}").hex
        return self


SEARCH_UID_NAMESPACE = uuid.UUID("6f1c2a3e-7b0d-4d5e-9a8f-1d2c3b4a5e6f")


def new_search_uid() -> str:
    return uuid.uuid4().hex


def problem_key_for(problem: str, problem_id: str) -> str:
    """Canonical problem identity: a provider target (`emflow://…`,
    `mlebench://…`) is already canonical; a local problem is its id. The one
    place this rule lives — ProblemSpec.problem_key and the search.yaml
    backfill both come here (hillclimb-go: EffectiveProblemKey)."""
    if problem.startswith("einsteinarena://"):
        return problem.partition("@sha256:")[0]
    return problem if "://" in problem else problem_id


def _load_meta(path: Path, model: type[BaseModel]):
    """Parse a metadata file; None unless it is valid AND schema v2. This is
    the gate that makes pre-v2 run dirs invisible to every scanner."""
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("schema_version") not in READABLE_SCHEMA_VERSIONS:
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


def run_display_name(run_dir: Path) -> str:
    """The name the user gave the run, falling back to its id."""
    meta = load_run_meta(run_dir)
    return meta.name if meta and meta.name else run_dir.name


def search_ref(search_dir: Path) -> str:
    """Human-facing `<run-id>/<search-id>` address of a search dir."""
    return f"{search_dir.parents[1].name}/{search_dir.name}"


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
    """`store.latest_search` of the folder backend, as a dir."""
    from hillclimb.harness.store import FileDataStore, latest_search

    record = latest_search(FileDataStore(runs_dir))
    return record.search_dir if record else None


def running_search_dirs(runs_dir: Path) -> list[Path]:
    """`store.running_searches` of the folder backend, as dirs."""
    from hillclimb.harness.store import FileDataStore, running_searches

    return [r.search_dir for r in running_searches(FileDataStore(runs_dir))]


def resolve_search_dir(runs_dir: Path, ref: str | None) -> Path:
    """`store.resolve_search` of the folder backend, as a dir."""
    from hillclimb.harness.store import FileDataStore, resolve_search

    return resolve_search(FileDataStore(runs_dir), ref).search_dir
