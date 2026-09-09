from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from hillclimb.budget import BudgetManager
from hillclimb.candidate import utcnow

STATUS_FILE = "status.json"

# States the engine writes. `crashed` and `unknown` are never written — they
# are derived by readers (effective_state) from a stale heartbeat / dead pid
# or a missing file.
WRITTEN_STATES = ("running", "parked", "stopped", "done", "failed")

HEARTBEAT_INTERVAL_S = 15
# A heartbeat older than this many intervals while state says "running"
# means the engine died without finalizing.
STALE_FACTOR = 6


class BudgetStatus(BaseModel):
    total_s: float = 0.0
    spent_s: float = 0.0
    remaining_s: float = 0.0


class CandidateCounts(BaseModel):
    """Counts candidates, not trials — revisit if a tuning loop adds
    multiple trials per candidate."""

    total: int = 0
    ok: int = 0
    buggy: int = 0
    pruned: int = 0


class CurrentCandidate(BaseModel):
    candidate_id: str
    operator: str
    phase: str  # agent | exec
    candidate_dir: str
    agent_pid: int | None = None
    started_at: str = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def _legacy_workspace_key(cls, data):
        """status.json written before the rename carries `workspace`."""
        if isinstance(data, dict) and "workspace" in data:
            data = dict(data)
            data.setdefault("candidate_dir", data.pop("workspace"))
            data.pop("workspace", None)
        return data


class ScoreRef(BaseModel):
    candidate_id: str
    val_score: float | None = None
    holdout_score: float | None = None


class SearchStatus(BaseModel):
    search_id: str
    run_id: str = ""
    state: str = "running"  # one of WRITTEN_STATES
    pid: int | None = None
    started_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    heartbeat_interval_s: int = HEARTBEAT_INTERVAL_S
    budget: BudgetStatus = Field(default_factory=BudgetStatus)
    # cumulative agent spend (sum of per-candidate backend cost); the hosted
    # platform settles credits from this at completion via the GCS state sync
    cost_usd: float = 0.0
    candidates: CandidateCounts = Field(default_factory=CandidateCounts)
    current: list[CurrentCandidate] = Field(default_factory=list)  # in-flight operators
    best: ScoreRef | None = None
    selected: ScoreRef | None = None
    last_error: str | None = None


def write_status(search_dir: Path, status: SearchStatus) -> None:
    """Atomic write so readers never see a half-written file."""
    tmp = search_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(status.model_dump_json(indent=2))
    os.replace(tmp, search_dir / STATUS_FILE)


def read_status(search_dir: Path) -> SearchStatus | None:
    path = search_dir / STATUS_FILE
    if not path.exists():
        return None
    try:
        return SearchStatus.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def live_remaining_s(status: "SearchStatus", state: str) -> float:
    """Budget left right now. The engine writes `remaining_s` at each
    heartbeat; for a running search the clock has kept ticking since, so
    subtract the heartbeat's age — that is what makes a seconds display
    move between writes."""
    remaining = status.budget.remaining_s
    if state == "running":
        remaining -= _age_s(status.updated_at)
    return max(0.0, remaining)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _age_s(iso: str) -> float:
    then = datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def effective_state(search_dir: Path) -> str:
    """`derive_state` of the search dir's status.json."""
    return derive_state(read_status(search_dir))


REMOTE_STATE_ENV = "HILLCLIMB_REMOTE_STATE"


def remote_state() -> bool:
    """True when the records being read were written on another machine
    (a mirror of a hosted search): the engine's pid means nothing here, so
    liveness rests on the heartbeat alone. Read per call so a viewer can be
    switched by its launcher without an import-order dance."""
    return os.environ.get(REMOTE_STATE_ENV) == "1"


def derive_state(status: SearchStatus | None) -> str:
    """What a reader should believe about a search from its last status record.

    `running` requires the engine's own claim AND a live pid AND a fresh
    heartbeat (defends against PID reuse); otherwise the search `crashed`.
    Searches with no status record yet report `unknown`. Under
    `HILLCLIMB_REMOTE_STATE=1` the pid check is skipped (see `remote_state`).
    """
    if status is None:
        return "unknown"
    if status.state != "running":
        return status.state
    if not remote_state() and not pid_alive(status.pid):
        return "crashed"
    try:
        stale_after = status.heartbeat_interval_s * STALE_FACTOR
        if _age_s(status.updated_at) > stale_after:
            return "crashed"
    except ValueError:
        return "crashed"
    return "running"


StatusSink = Callable[[SearchStatus], None]


class StatusWriter:
    """Single writer of a search's status record, owned by the engine process.

    `sink` is where each write goes — `store.write_status` bound to the
    search's key; a search dir is accepted as shorthand for its status.json.
    Thread-safe because the heartbeat runs on a daemon thread while the
    search loop updates fields from the main thread.
    """

    def __init__(self, sink: StatusSink | Path, status: SearchStatus, budget: BudgetManager | None = None):
        self.sink: StatusSink = (lambda s, d=sink: write_status(d, s)) if isinstance(sink, Path) else sink
        self.status = status
        self.budget = budget
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._write()

    def _write(self) -> None:
        self.status.updated_at = utcnow()
        if self.budget is not None:
            self.status.budget = BudgetStatus(
                total_s=self.budget.total_s,
                spent_s=round(self.budget.elapsed(), 1),
                remaining_s=round(self.budget.remaining(), 1),
            )
        self.sink(self.status)

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self.status, key, value)
            self._write()

    def start_heartbeat(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._beat, daemon=True, name="status-heartbeat")
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(self.status.heartbeat_interval_s):
            with self._lock:
                self._refresh_agent_pid()
                self._write()

    def _refresh_agent_pid(self) -> None:
        """Pick up the operator backends' child pids by filesystem convention."""
        for current in self.status.current:
            pid_file = Path(current.candidate_dir) / "agent.pid"
            try:
                current.agent_pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
            except (ValueError, OSError):
                current.agent_pid = None

    # --- in-flight registry (thread-safe: called by scheduler and workers) ---

    def add_current(self, entry: CurrentCandidate) -> None:
        with self._lock:
            self.status.current = [
                c for c in self.status.current if c.candidate_id != entry.candidate_id
            ] + [entry]
            self._write()

    def update_current(self, candidate_id: str, **fields) -> None:
        with self._lock:
            for entry in self.status.current:
                if entry.candidate_id == candidate_id:
                    for key, value in fields.items():
                        setattr(entry, key, value)
            self._write()

    def remove_current(self, candidate_id: str) -> None:
        with self._lock:
            self.status.current = [
                c for c in self.status.current if c.candidate_id != candidate_id
            ]
            self._write()

    def finalize(self, state: str, last_error: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            self.status.state = state
            self.status.current = []
            if last_error is not None:
                self.status.last_error = last_error
            self._write()
