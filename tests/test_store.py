"""The DataStore contract: FileDataStore (the hillclimb folder) and
SqliteDataStore must answer identically — runs/searches, the journal in
append order, status with derived state, and the consume-once command
queue — and the engine must run entirely through whichever is configured."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import os
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hillclimb.candidate import Candidate
from hillclimb.chart import climb_curves
from hillclimb.config import Config
from hillclimb.control import ControlCommand
from hillclimb.journal import FileJournal, Journal
from hillclimb.run import RunMeta, SearchMeta, load_search_meta
from hillclimb.status import SearchStatus
from hillclimb.store import DataStore, FileDataStore, SqliteDataStore, key_for, open_store, sync_store


def _meta(run_id: str, search_id: str, key: str = "p", started: str = "2026-08-22T10:00:00+00:00") -> SearchMeta:
    return SearchMeta(
        search_id=search_id, run_id=run_id, problem=key, problem_id=key, backend="dummy",
        model="m", metric="score", started_at=started,
    )


def _populate(store: DataStore) -> None:
    """The same history written through the contract into any backend."""
    store.record_run(RunMeta(run_id="r1", name="first", kind="problem", target="p", started_at="2026-08-22T09:00:00+00:00"))
    store.record_run(RunMeta(run_id="r2", name="second", kind="problem", target="p", started_at="2026-08-22T09:30:00+00:00"))
    for run_id, search_id, key, started, scores in (
        ("r1", "p", "p", "2026-08-22T10:02:00+00:00", [1.0, 0.5]),
        ("r2", "p", "p", "2026-08-22T10:01:00+00:00", [2.0]),
        ("r2", "q", "q", "2026-08-22T10:01:00+00:00", [9.0]),
    ):
        store.search_dir((run_id, search_id)).mkdir(parents=True, exist_ok=True)
        store.record_search(_meta(run_id, search_id, key, started))
        journal = Journal(store.journal((run_id, search_id)))
        for index, score in enumerate(scores):
            cand = Candidate(candidate_id=f"c{index:03d}", operator="draft")
            journal.candidate_created(cand)
            cand.status = "passing"
            cand.trials = [mk_trial(val_score=score)]
            cand.finished_at = f"2026-08-22T10:0{index + 1}:00+00:00"
            journal.candidate_result(cand)


@pytest.fixture(params=["files", "sqlite"])
def store(request, tmp_path: Path) -> DataStore:
    runs = tmp_path / "runs"
    if request.param == "files":
        store = FileDataStore(runs)
    else:
        store = SqliteDataStore(tmp_path / "store.sqlite", runs_dir=runs)
    _populate(store)
    yield store
    store.close()


def test_backends_agree_on_runs_and_searches(store: DataStore):
    assert [r.run_id for r in store.runs()] == ["r1", "r2"]
    # started_at order, ties by address — never folder or insertion order
    assert [r.ref for r in store.searches()] == ["r2/p", "r2/q", "r1/p"]
    assert [r.ref for r in store.searches(problem_key="p")] == ["r2/p", "r1/p"]
    assert [r.ref for r in store.searches(run_id="r2")] == ["r2/p", "r2/q"]
    assert store.searches(problem_key="nope") == []
    record = store.search(("r2", "q"))
    assert record is not None and record.run_name == "second"
    assert record.meta.problem_key == "q"
    assert record.state == "unknown"  # no status record yet
    assert record.activity_at == record.meta.started_at
    assert record.search_dir == store.runs_dir / "r2" / "searches" / "q"
    assert store.search(("r2", "missing")) is None
    assert len(record.meta.search_uid) == 32  # backfilled deterministically
    assert record.meta.search_uid == store.search(("r2", "q")).meta.search_uid


def test_backends_agree_on_the_journal(store: DataStore):
    records = store.journal(("r1", "p")).records()
    assert [(r["event"], r["candidate_id"]) for r in records] == [
        ("candidate_created", "c000"), ("candidate_result", "c000"),
        ("candidate_created", "c001"), ("candidate_result", "c001"),
    ]
    journal = Journal(store.journal(("r1", "p")))
    assert [c.val_score for c in journal.candidates.values()] == [1.0, 0.5]
    journal.control_event("prune", candidate_id="c001")  # audit line survives, replay skips it
    assert store.journal(("r1", "p")).records()[-1]["event"] == "control"
    assert list(Journal(store.journal(("r1", "p"))).candidates) == ["c000", "c001"]
    assert store.journal(("r1", "missing")).records() == []


def test_backends_agree_on_status_and_derived_state(store: DataStore):
    key = ("r2", "p")
    store.write_status(key, SearchStatus(search_id="p", run_id="r2", state="running", pid=os.getpid()))
    assert store.read_status(key).pid == os.getpid()
    assert store.search(key).state == "running"
    assert store.search(key).activity_at > store.search(key).meta.started_at
    # the engine's claim alone is not enough: a dead pid means crashed
    store.write_status(key, SearchStatus(search_id="p", run_id="r2", state="running", pid=2**22 - 1))
    assert store.search(key).state == "crashed"
    store.write_status(key, SearchStatus(search_id="p", run_id="r2", state="done"))
    assert store.search(key).state == "done"
    assert store.read_status(("r1", "missing")) is None


def test_backends_agree_on_the_command_queue(store: DataStore):
    key = ("r2", "p")
    store.enqueue_command(key, ControlCommand(action="stop"))
    store.enqueue_command(key, ControlCommand(action="prune", candidate_id="c000", source="tui"))
    store.clear_stale_stops(key)  # engine startup: stale stops go, prunes stay
    store.enqueue_command(key, ControlCommand(action="stop", source="cli"))
    drained = store.drain_commands(key)
    assert [(c.action, c.candidate_id, c.source) for c in drained] == [
        ("prune", "c000", "tui"), ("stop", None, "cli"),
    ]
    assert store.drain_commands(key) == []  # consumed once
    assert store.drain_commands(("r1", "missing")) == []


def test_chart_reads_the_same_curves_from_both_backends(store: DataStore):
    curves = climb_curves(store, "p")
    assert [c.label for c in curves] == ["second", "first"]
    assert [c.ys for c in curves] == [[2.0], [1.0, 1.0]]


def test_file_backend_is_the_folder_layout(store: DataStore):
    if not isinstance(store, FileDataStore):
        pytest.skip("folder layout is the file backend's representation")
    search_dir = store.runs_dir / "r1" / "searches" / "p"
    assert load_search_meta(search_dir).search_id == "p"
    assert (search_dir / "journal.jsonl").exists()
    assert Journal(search_dir / "journal.jsonl").candidates.keys() == Journal(store.journal(("r1", "p"))).candidates.keys()
    assert key_for(search_dir) == ("r1", "p")


def test_sqlite_backend_writes_no_yaml(tmp_path: Path):
    store = SqliteDataStore(tmp_path / "s.sqlite", runs_dir=tmp_path / "runs")
    _populate(store)
    assert not (tmp_path / "runs" / "r1" / "searches" / "p" / "search.yaml").exists()
    assert not (tmp_path / "runs" / "r1" / "searches" / "p" / "journal.jsonl").exists()
    # re-recording is an upsert
    store.record_search(_meta("r1", "p").model_copy(update={"budget_s": 42}))
    assert store.search(("r1", "p")).meta.budget_s == 42
    store.close()


def test_sqlite_store_is_thread_and_process_safe(tmp_path: Path):
    path = tmp_path / "s.sqlite"
    store = SqliteDataStore(path, runs_dir=tmp_path / "runs")

    def worker(index: int):
        journal = store.journal(("r", f"p-{index}"))
        for n in range(20):
            journal.append({"event": "candidate_created", "candidate_id": f"c{n:03d}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # a second connection (another engine process in the demo) sees it all, in order
    other = SqliteDataStore(path, runs_dir=tmp_path / "runs")
    for i in range(4):
        ids = [r["candidate_id"] for r in other.journal(("r", f"p-{i}")).records()]
        assert ids == [f"c{n:03d}" for n in range(20)]
    other.close()
    store.close()


def test_sync_imports_the_folder_into_sqlite_once(tmp_path: Path):
    files = FileDataStore(tmp_path / "runs")
    _populate(files)
    files.write_status(("r2", "p"), SearchStatus(search_id="p", run_id="r2", state="done"))
    sqlite = SqliteDataStore(tmp_path / "s.sqlite", runs_dir=tmp_path / "runs")
    assert sync_store(files, sqlite) == {"runs": 2, "searches": 3, "records": 8}
    assert sync_store(files, sqlite) == {"runs": 0, "searches": 0, "records": 0}  # already there
    assert [r.ref for r in sqlite.searches()] == [r.ref for r in files.searches()]
    assert sqlite.search(("r2", "p")).state == "done"
    assert sqlite.journal(("r1", "p")).records() == files.journal(("r1", "p")).records()
    sqlite.close()


def test_config_store_section_and_open_store(tmp_path: Path, monkeypatch):
    hc = tmp_path / "proj" / "hillclimb"
    hc.mkdir(parents=True)
    (hc / "config.yaml").write_text("store:\n  backend: sqlite\n")
    monkeypatch.chdir(tmp_path / "proj")
    config = Config.load()
    assert config.store.backend == "sqlite"
    assert config.store.sqlite_path == hc / "store.sqlite"  # anchored like runs_dir
    store = open_store(config)
    assert isinstance(store, SqliteDataStore)
    store.close()
    assert isinstance(open_store(Config()), FileDataStore)
    bad = Config()
    bad.store.backend = "postgres"
    with pytest.raises(ValueError, match="postgres"):
        open_store(bad)


def test_store_cli_sync_and_searches(tmp_path: Path, monkeypatch):
    from hillclimb.cli import app

    folder = tmp_path / "runs"
    _populate(FileDataStore(folder))
    hc = tmp_path / "hillclimb"
    hc.mkdir()
    (hc / "config.yaml").write_text(f"store:\n  backend: sqlite\npaths:\n  runs_dir: {folder}\n")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["store", "sync"])
    assert result.exit_code == 0, result.output
    assert "2 run(s), 3 search(es), 8 journal record(s)" in result.output
    result = runner.invoke(app, ["store", "searches", "--problem", "p"])
    assert result.exit_code == 0, result.output
    assert "r2/p" in result.output and "r1/p" in result.output and "r2/q" not in result.output

    (hc / "config.yaml").write_text(f"paths:\n  runs_dir: {folder}\n")
    result = runner.invoke(app, ["store", "sync"])
    assert result.exit_code == 0 and "nothing to import" in result.output


@pytest.mark.parametrize("backend", ["files", "sqlite"])
@pytest.mark.slow
def test_engine_runs_entirely_through_the_store(config, tmp_path: Path, backend: str):
    from hillclimb.api import run_search

    config.store.backend = backend
    config.store.sqlite_path = tmp_path / "store.sqlite"
    outcome = run_search(
        "circle-packing", budget_s=10, name="store-test", config=config,
        backend="dummy", holdout=False, log=lambda _: None,
    )
    store = open_store(config)
    try:
        (run,) = store.runs()
        assert run.name == "store-test"
        (record,) = store.searches(problem_key="circle-packing")
        assert record.state == outcome.state == "done"
        assert record.search_dir == outcome.search_dir
        assert len(record.meta.search_uid) == 32
        journal = Journal(store.journal(record.key))
        assert journal.scored_candidates()
        assert store.read_status(record.key).state == "done"
        assert climb_curves(store, "circle-packing")[0].ys
        on_disk = outcome.search_dir / "journal.jsonl"
        assert on_disk.exists() == (backend == "files")
        assert (outcome.search_dir / "candidates").exists()  # agents always work on disk
    finally:
        store.close()
