from __future__ import annotations

from tests.factories import trial as mk_trial

from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.control import (
    ControlCommand,
    apply_prune,
    clear_stale_stops,
    read_commands,
    request_prune,
    resync_best,
    write_command,
)
from hillclimb.journal import Journal
from hillclimb.store import FileDataStore, key_for


def make_candidate(candidate_id: str, **kwargs) -> Candidate:
    val_score = kwargs.pop("val_score", None)
    if val_score is not None:
        kwargs["trials"] = [mk_trial(val_score=val_score)]
    return Candidate(candidate_id=candidate_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def make_search(tmp_path: Path) -> tuple[Path, Journal]:
    search_dir = tmp_path / "runs" / "r" / "searches" / "search"  # the real layout: keys resolve
    (search_dir / "best").mkdir(parents=True)
    (search_dir / "candidates").mkdir()
    return search_dir, Journal(search_dir / "journal.jsonl")


def add_candidate(journal: Journal, search_dir: Path, candidate_id: str, **kwargs) -> Candidate:
    candidate_dir = search_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "submission.csv").write_text(f"id,target\n1,{candidate_id}\n")
    (candidate_dir / "solution.py").write_text(f"# {candidate_id}\n")
    candidate = make_candidate(candidate_id, candidate_dir=str(candidate_dir), **kwargs)
    journal.candidate_result(candidate)
    return candidate


def test_command_roundtrip(tmp_path: Path):
    search_dir, _ = make_search(tmp_path)
    write_command(search_dir, ControlCommand(action="prune", candidate_id="c003", source="tui"))
    write_command(search_dir, ControlCommand(action="stop"))
    commands = read_commands(search_dir)
    assert [c.action for _, c in commands] == ["prune", "stop"]
    assert commands[0][1].candidate_id == "c003"


def test_read_commands_drops_unparseable(tmp_path: Path):
    search_dir, _ = make_search(tmp_path)
    (search_dir / "control").mkdir()
    bad = search_dir / "control" / "zz-bad.json"
    bad.write_text("{nope")
    assert read_commands(search_dir) == []
    assert not bad.exists()


def test_clear_stale_stops_keeps_prunes(tmp_path: Path):
    search_dir, _ = make_search(tmp_path)
    write_command(search_dir, ControlCommand(action="stop"))
    write_command(search_dir, ControlCommand(action="prune", candidate_id="c001"))
    clear_stale_stops(search_dir)
    commands = read_commands(search_dir)
    assert [c.action for _, c in commands] == ["prune"]


def test_apply_prune_subtree(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)
    add_candidate(journal, search_dir, "c002", operator="improve", parent_id="c001", status="buggy")
    add_candidate(journal, search_dir, "c003", operator="debug", parent_id="c002", status="passing", val_score=0.6)
    add_candidate(journal, search_dir, "c004", status="passing", val_score=0.4)  # separate branch

    pruned = apply_prune(journal, "c002", reason="overfitting", source="cli")

    assert pruned == ["c002", "c003"]
    assert journal.get("c002").pruned and journal.get("c003").pruned
    assert not journal.get("c001").pruned and not journal.get("c004").pruned
    assert journal.get("c002").pruned_reason == "overfitting"
    # survives replay, audit line recorded
    reloaded = Journal(search_dir / "journal.jsonl")
    assert reloaded.get("c003").pruned
    assert '"event": "control"' in (search_dir / "journal.jsonl").read_text()


def test_apply_prune_refuses_baseline_and_unknown(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c000", operator="baseline", status="passing")
    with pytest.raises(ValueError, match="baseline"):
        apply_prune(journal, "c000")
    with pytest.raises(ValueError, match="No candidate"):
        apply_prune(journal, "c999")


def test_apply_prune_idempotent(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)
    assert apply_prune(journal, "c001") == ["c001"]
    assert apply_prune(journal, "c001") == []


def test_resync_best_repoints_after_pruning_selected(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)
    winner = add_candidate(journal, search_dir, "c002", status="passing", val_score=0.9, is_selected=True)
    (search_dir / "best" / "submission.csv").write_text(
        (Path(winner.candidate_dir) / "submission.csv").read_text()
    )

    apply_prune(journal, "c002")
    selected = resync_best(search_dir, journal, higher_is_better=True, selection_mode="rank-blend")

    assert selected == "c001"
    assert "c001" in (search_dir / "best" / "submission.csv").read_text()
    assert journal.get("c001").is_selected
    assert not journal.get("c002").is_selected


def test_resync_best_falls_back_to_baseline(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c000", operator="baseline", status="passing")
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)

    apply_prune(journal, "c001")
    selected = resync_best(search_dir, journal, higher_is_better=True, selection_mode="rank-blend")

    assert selected is None
    assert "c000" in (search_dir / "best" / "submission.csv").read_text()
    assert not (search_dir / "best" / "solution.py").exists()


def test_request_prune_offline_applies_directly(tmp_path: Path):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)

    store = FileDataStore(search_dir.parents[2])
    outcome = request_prune(
        store, key_for(search_dir), "c001", higher_is_better=True, selection_mode="rank-blend", source="cli"
    )

    assert "pruned c001" in outcome
    assert Journal(search_dir / "journal.jsonl").get("c001").pruned
    assert read_commands(search_dir) == []  # applied, not queued


def test_request_prune_queues_when_running(tmp_path: Path, monkeypatch):
    search_dir, journal = make_search(tmp_path)
    add_candidate(journal, search_dir, "c001", status="passing", val_score=0.5)
    monkeypatch.setattr("hillclimb.control.derive_state", lambda _: "running")

    store = FileDataStore(search_dir.parents[2])
    outcome = request_prune(
        store, key_for(search_dir), "c001", higher_is_better=True, selection_mode="rank-blend", source="tui"
    )

    assert "queued" in outcome
    assert not Journal(search_dir / "journal.jsonl").get("c001").pruned
    commands = read_commands(search_dir)
    assert len(commands) == 1 and commands[0][1].action == "prune"
