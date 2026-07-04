from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from hillclimb.budget import BudgetManager
from hillclimb.node import utcnow

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


class NodeCounts(BaseModel):
    total: int = 0
    ok: int = 0
    buggy: int = 0
    pruned: int = 0


class CurrentNode(BaseModel):
    node_id: str
    operator: str
    phase: str  # agent | exec
    workspace: str
    agent_pid: int | None = None
    started_at: str = Field(default_factory=utcnow)


class ScoreRef(BaseModel):
    node_id: str
    val_score: float | None = None
    holdout_score: float | None = None


class RunStatus(BaseModel):
    run_id: str
    state: str = "running"  # one of WRITTEN_STATES
    pid: int | None = None
    started_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    heartbeat_interval_s: int = HEARTBEAT_INTERVAL_S
    budget: BudgetStatus = Field(default_factory=BudgetStatus)
    nodes: NodeCounts = Field(default_factory=NodeCounts)
    current: CurrentNode | None = None
    best: ScoreRef | None = None
    selected: ScoreRef | None = None
    last_error: str | None = None


def write_status(run_dir: Path, status: RunStatus) -> None:
    """Atomic write so readers never see a half-written file."""
    tmp = run_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(status.model_dump_json(indent=2))
    os.replace(tmp, run_dir / STATUS_FILE)


def read_status(run_dir: Path) -> RunStatus | None:
    path = run_dir / STATUS_FILE
    if not path.exists():
        return None
    try:
        return RunStatus.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


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


def effective_state(run_dir: Path) -> str:
    """What a reader should believe about this run.

    `running` requires the engine's own claim AND a live pid AND a fresh
    heartbeat (defends against PID reuse); otherwise the run `crashed`.
    Runs predating status.json report `unknown`.
    """
    status = read_status(run_dir)
    if status is None:
        return "unknown"
    if status.state != "running":
        return status.state
    if not pid_alive(status.pid):
        return "crashed"
    try:
        stale_after = status.heartbeat_interval_s * STALE_FACTOR
        if _age_s(status.updated_at) > stale_after:
            return "crashed"
    except ValueError:
        return "crashed"
    return "running"


class StatusWriter:
    """Single writer of a run's status.json, owned by the engine process.

    Thread-safe because the heartbeat runs on a daemon thread while the
    search loop updates fields from the main thread.
    """

    def __init__(self, run_dir: Path, status: RunStatus, budget: BudgetManager | None = None):
        self.run_dir = run_dir
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
        write_status(self.run_dir, self.status)

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
        """Pick up the operator backend's child pid by filesystem convention."""
        current = self.status.current
        if current is None:
            return
        pid_file = Path(current.workspace) / "agent.pid"
        try:
            current.agent_pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
        except (ValueError, OSError):
            current.agent_pid = None

    def finalize(self, state: str, last_error: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            self.status.state = state
            self.status.current = None
            if last_error is not None:
                self.status.last_error = last_error
            self._write()
