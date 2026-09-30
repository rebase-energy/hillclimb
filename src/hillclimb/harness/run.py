from __future__ import annotations

from pathlib import Path

import uuid
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from hillclimb.harness.candidate import utcnow
from hillclimb.harness.direction import legacy_direction_key

SCHEMA_VERSION = 4
# what `_load_meta` still reads. v2 search records name a `policy`; v3 ones
# a climber by reference, with its manifest and the user's overlay beside
# it. Both are mapped onto v4's one block on load
# (SearchMeta._from_older_schemas), so runs made before stay visible to
# every view
READABLE_SCHEMA_VERSIONS = (2, 3, 4)
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

    @model_validator(mode="before")
    @classmethod
    def _legacy_agent_key(cls, data):
        """A search recorded before the rename names its coding agent `backend`."""
        if isinstance(data, dict) and "backend" in data:
            data = dict(data)
            data.setdefault("agent", data.pop("backend"))
        return data

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
    agent: str   # search.yaml files from before the rename say `backend`
    model: str
    # additive with defaults on purpose: a field without one would hide every
    # existing run dir from the scanners
    #
    # The climber this search ran: what views call it (its label), its
    # identity at search start (the block, the bytes of its local files and
    # its prompts — the identity of an edited exploration process, like
    # seed_sha256) and the block itself as it was launched, file refs
    # absolute. The block and its files are snapshotted into
    # `<search_dir>/climber/`, which is what the engine — and a resume — loads.
    climber: str = "greedy"
    # The role the climber plays here, derived from the problem at
    # `create_search` and never declared by the climber: `solver` when the
    # problem's solution is a program, `improver` when it is a climber (a
    # meta-problem). The same bundle may run in either role.
    role: Literal["solver", "improver"] = "solver"
    climber_sha256: str | None = None
    climber_spec: dict = Field(default_factory=dict)
    # how a search recorded before 0.6 named its climber (a bundled name, a
    # directory, one file); None on every search since
    climber_ref: str | None = None
    # param values memory learned for this search before its first step (a
    # draft-complexity offset), laid under the block's params. Recorded when
    # the search first runs, so a resume starts from the same ones; None
    # until then
    memory_priors: dict | None = None
    # False for a climber composed in Python from classes that existed only
    # in the launching process: the search ran, but cannot be resumed
    climber_portable: bool = True
    hillclimb_version: str | None = None
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
    def _legacy_arm_tags(cls, data):
        """A record written before the study vocabulary named the comparison
        the `experiment` and each setup in it an `arm`. Same things, older
        words: map them, so reports and charts keep grouping old runs."""
        if not isinstance(data, dict) or ("arm" not in data and "arm_overrides" not in data):
            return data
        data = dict(data)
        if "arm" in data:
            data.setdefault("study", data.get("experiment"))
            data["experiment"] = data.pop("arm")
        overrides = data.pop("arm_overrides", None)
        if overrides is not None:
            data.setdefault("experiment_overrides", overrides)
        return data

    @model_validator(mode="before")
    @classmethod
    def _from_older_schemas(cls, data):
        """Records written before 0.6. Schema v2 named a `policy`; v3 named
        a climber by reference and kept its manifest, the user's params
        overlay and the user's tuner override in separate fields. Same
        things, older shapes: fold them into the one block v4 holds, so every
        view keeps showing those runs. In every store backend — this runs
        wherever a SearchMeta is validated. (One validator on purpose: the
        v2 renames must happen before the v3 folding.)"""
        if not isinstance(data, dict) or "climber_spec" in data:
            return data
        legacy_keys = ("policy", "policy_params", "policy_sha256", "templates_sha256", "templates_overridden",
                       "climber_manifest", "climber_params", "tuner", "tuner_params")
        if not any(key in data for key in legacy_keys) and data.get("schema_version", SCHEMA_VERSION) >= SCHEMA_VERSION:
            return data
        data = dict(data)
        if "policy" in data:  # v2 -> v3
            data.setdefault("climber", data.pop("policy"))
            data.setdefault("climber_params", data.pop("policy_params", None) or {})
            data.setdefault("climber_sha256", data.pop("policy_sha256", None))
        for gone in ("policy", "policy_params", "policy_sha256", "templates_sha256", "templates_overridden"):
            data.pop(gone, None)
        # v3 -> v4: the manifest, with the user's overlay and tuner override on top
        manifest = dict(data.pop("climber_manifest", None) or {})
        overlay = data.pop("climber_params", None) or {}
        tuner = data.pop("tuner", None)
        tuner_params = data.pop("tuner_params", None) or {}
        block = {
            key: value for key, value in manifest.items()
            if value is not None and key not in ("description", "similarity", "holdout_timing")
        }
        if not manifest and isinstance(data.get("climber"), str):
            # no manifest was kept: the name is all there is (a preset, a
            # file), with the params laid over it — the 0.5 config shape, and
            # read the same way (an openevolve search's MAP-Elites settings
            # among them go to the selector)
            from hillclimb.modules.spec import block_from_05

            try:
                block = block_from_05({"ref": data["climber"], "params": dict(overlay)})
            except ValueError:
                block = {"params": dict(overlay)} if overlay else {}
        else:
            params = {**(block.get("params") or {}), **overlay}
            if params:
                block["params"] = params
        if tuner:
            block["tuner"] = tuner
        merged_tuner_params = {**(manifest.get("tuner_params") or {}), **tuner_params}
        if merged_tuner_params:
            block["tuner_params"] = merged_tuner_params
        graph = block.pop("graph", None)
        if graph and graph != "knowledge-graph":  # the graph module is a setting of the memory now
            block["memory_params"] = {"graph": graph}
        data["climber_spec"] = block
        data.setdefault("climber_ref", data.get("climber"))
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
    # Study tags (experiment.py): which study and experiment this search
    # belongs to, its repeat index, and the config overrides the experiment
    # applied — the grouping keys for every setup-vs-setup comparison
    study: str | None = None
    experiment: str | None = None
    repeat: int = 0
    experiment_overrides: dict = Field(default_factory=dict)
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
