from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher, ParkedRun
from hillclimb.status import (
    RunStatus,
    StatusWriter,
    effective_state,
    pid_alive,
    read_status,
    write_status,
)
from hillclimb.workspace import create_run_dir
from tests.conftest import ok_script

DEAD_PID = 2**22  # above macOS/Linux pid ranges


def iso_ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_write_read_roundtrip(tmp_path: Path):
    status = RunStatus(run_id="r1", state="running", pid=os.getpid())
    write_status(tmp_path, status)
    loaded = read_status(tmp_path)
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.pid == os.getpid()
    assert not (tmp_path / "status.json.tmp").exists()


def test_read_missing_and_corrupt(tmp_path: Path):
    assert read_status(tmp_path) is None
    (tmp_path / "status.json").write_text("{not json")
    assert read_status(tmp_path) is None


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(DEAD_PID)
    assert not pid_alive(None)


def test_effective_state_running(tmp_path: Path):
    write_status(tmp_path, RunStatus(run_id="r", state="running", pid=os.getpid()))
    assert effective_state(tmp_path) == "running"


def test_effective_state_crashed_dead_pid(tmp_path: Path):
    write_status(tmp_path, RunStatus(run_id="r", state="running", pid=DEAD_PID))
    assert effective_state(tmp_path) == "crashed"


def test_effective_state_crashed_stale_heartbeat(tmp_path: Path):
    status = RunStatus(run_id="r", state="running", pid=os.getpid())
    status.updated_at = iso_ago(600)
    tmp = tmp_path / "status.json"
    tmp.write_text(status.model_dump_json())
    assert effective_state(tmp_path) == "crashed"


def test_effective_state_terminal_states_pass_through(tmp_path: Path):
    for state in ("parked", "stopped", "done", "failed"):
        write_status(tmp_path, RunStatus(run_id="r", state=state, pid=DEAD_PID))
        assert effective_state(tmp_path) == state


def test_effective_state_unknown_without_file(tmp_path: Path):
    assert effective_state(tmp_path) == "unknown"


def test_status_writer_update_and_finalize(tmp_path: Path):
    writer = StatusWriter(
        tmp_path,
        RunStatus(run_id="r", state="running", pid=os.getpid()),
        budget=BudgetManager(100, stop_margin_s=0),
    )
    assert read_status(tmp_path).state == "running"
    assert read_status(tmp_path).budget.total_s == 100
    writer.finalize("done")
    loaded = read_status(tmp_path)
    assert loaded.state == "done"
    assert loaded.current is None


def make_searcher_with_status(task, config, backend, budget_s=3600):
    run_dir = create_run_dir(config.paths.runs_dir, "test-run")
    budget = BudgetManager(budget_s, stop_margin_s=1)
    status = StatusWriter(
        run_dir, RunStatus(run_id="test-run", state="running", pid=os.getpid()), budget=budget
    )
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=Journal(run_dir / "journal.jsonl"),
        backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=budget,
        run_dir=run_dir,
        max_nodes=4,
        log=lambda *_: None,
        status=status,
    )
    return searcher, status, run_dir


def test_search_updates_status(task, config):
    backend = FakeBackend()
    for score in (0.6, 0.7, 0.5):
        backend.queue(script=ok_script(score), notes="d\n")
    searcher, status, run_dir = make_searcher_with_status(task, config, backend)

    searcher.run()
    status.finalize("done")

    loaded = read_status(run_dir)
    assert loaded.state == "done"
    assert loaded.nodes.total == 4  # baseline + 3 drafts
    assert loaded.nodes.ok == 4  # baseline counts as ok
    assert loaded.best is not None and loaded.best.val_score == 0.7
    assert loaded.selected is not None


def test_rate_limited_run_can_finalize_parked(task, config):
    backend = FakeBackend()
    backend.queue(result={"ok": False, "error_kind": "rate_limited", "error_message": "limit"})
    searcher, status, run_dir = make_searcher_with_status(task, config, backend)

    with pytest.raises(ParkedRun):
        searcher.run()
    status.finalize("parked", last_error="limit")

    loaded = read_status(run_dir)
    assert loaded.state == "parked"
    assert loaded.last_error == "limit"
