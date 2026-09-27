from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from tests.conftest import local_executor
from hillclimb.harness.journal import Journal
from tests.harness_factory import SearchRig
from hillclimb.harness.glue import ParkedSearch
from hillclimb.harness.status import (
    SearchStatus,
    StatusWriter,
    effective_state,
    pid_alive,
    read_status,
    write_status,
)
from hillclimb.harness.dirs import create_search_dir
from tests.conftest import ok_script

DEAD_PID = 2**22  # above macOS/Linux pid ranges


def iso_ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_write_read_roundtrip(tmp_path: Path):
    status = SearchStatus(search_id="s1", state="running", pid=os.getpid())
    write_status(tmp_path, status)
    loaded = read_status(tmp_path)
    assert loaded is not None
    assert loaded.search_id == "s1"
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
    write_status(tmp_path, SearchStatus(search_id="s", state="running", pid=os.getpid()))
    assert effective_state(tmp_path) == "running"


def test_effective_state_crashed_dead_pid(tmp_path: Path):
    write_status(tmp_path, SearchStatus(search_id="s", state="running", pid=DEAD_PID))
    assert effective_state(tmp_path) == "crashed"


def test_effective_state_remote_ignores_pid(tmp_path: Path, monkeypatch):
    """A mirrored status.json carries the hosted engine's pid: meaningless
    locally, so the heartbeat alone decides under HILLCLIMB_REMOTE_STATE=1."""
    write_status(tmp_path, SearchStatus(search_id="s", state="running", pid=DEAD_PID))
    monkeypatch.setenv("HILLCLIMB_REMOTE_STATE", "1")
    assert effective_state(tmp_path) == "running"
    stale = SearchStatus(search_id="s", state="running", pid=DEAD_PID)
    stale.updated_at = iso_ago(600)
    write_status(tmp_path, stale)
    assert effective_state(tmp_path) == "crashed"


def test_effective_state_crashed_stale_heartbeat(tmp_path: Path):
    status = SearchStatus(search_id="s", state="running", pid=os.getpid())
    status.updated_at = iso_ago(600)
    tmp = tmp_path / "status.json"
    tmp.write_text(status.model_dump_json())
    assert effective_state(tmp_path) == "crashed"


def test_effective_state_terminal_states_pass_through(tmp_path: Path):
    for state in ("parked", "stopped", "done", "failed"):
        write_status(tmp_path, SearchStatus(search_id="s", state=state, pid=DEAD_PID))
        assert effective_state(tmp_path) == state


def test_effective_state_unknown_without_file(tmp_path: Path):
    assert effective_state(tmp_path) == "unknown"


def test_status_writer_update_and_finalize(tmp_path: Path):
    writer = StatusWriter(
        tmp_path,
        SearchStatus(search_id="s", state="running", pid=os.getpid()),
        budget=BudgetManager(100, stop_margin_s=0),
    )
    assert read_status(tmp_path).state == "running"
    assert read_status(tmp_path).budget.total_s == 100
    writer.finalize("done")
    loaded = read_status(tmp_path)
    assert loaded.state == "done"
    assert loaded.current == []


def make_searcher_with_status(task, config, agent, budget_s=3600):
    search_dir = create_search_dir(config.paths.runs_dir, "test-search")
    budget = BudgetManager(budget_s, stop_margin_s=1)
    status = StatusWriter(
        search_dir, SearchStatus(search_id="test-search", state="running", pid=os.getpid()), budget=budget
    )
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=Journal(search_dir / "journal.jsonl"),
        agent=agent,
        executor=local_executor(),
        budget=budget,
        search_dir=search_dir,
        max_candidates=4,
        log=lambda *_: None,
        status=status,
    )
    return searcher, status, search_dir


def test_search_updates_status(task, config):
    agent = FakeAgent()
    for score in (0.6, 0.7, 0.5):
        agent.queue(script=ok_script(score), notes="d\n")
    searcher, status, search_dir = make_searcher_with_status(task, config, agent)

    searcher.run()
    status.finalize("done")

    loaded = read_status(search_dir)
    assert loaded.state == "done"
    assert loaded.candidates.total == 4  # baseline + 3 drafts
    assert loaded.candidates.passing == 4  # baseline counts as ok
    assert loaded.best is not None and loaded.best.val_score == 0.7
    assert loaded.selected is not None


def test_rate_limited_run_can_finalize_parked(task, config):
    agent = FakeAgent()
    agent.queue(result={"ok": False, "error_kind": "rate_limited", "error_message": "limit"})
    searcher, status, search_dir = make_searcher_with_status(task, config, agent)

    with pytest.raises(ParkedSearch):
        searcher.run()
    status.finalize("parked", last_error="limit")

    loaded = read_status(search_dir)
    assert loaded.state == "parked"
    assert loaded.last_error == "limit"
