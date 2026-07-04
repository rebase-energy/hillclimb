from __future__ import annotations

from pathlib import Path

import pytest

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
from hillclimb.node import Node


def make_node(node_id: str, **kwargs) -> Node:
    return Node(node_id=node_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def make_run(tmp_path: Path) -> tuple[Path, Journal]:
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "nodes").mkdir()
    return run_dir, Journal(run_dir / "journal.jsonl")


def add_node(journal: Journal, run_dir: Path, node_id: str, **kwargs) -> Node:
    workspace = run_dir / "nodes" / node_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "submission.csv").write_text(f"id,target\n1,{node_id}\n")
    (workspace / "solution.py").write_text(f"# {node_id}\n")
    node = make_node(node_id, workspace=str(workspace), **kwargs)
    journal.node_result(node)
    return node


def test_command_roundtrip(tmp_path: Path):
    run_dir, _ = make_run(tmp_path)
    write_command(run_dir, ControlCommand(action="prune", node_id="n003", source="tui"))
    write_command(run_dir, ControlCommand(action="stop"))
    commands = read_commands(run_dir)
    assert [c.action for _, c in commands] == ["prune", "stop"]
    assert commands[0][1].node_id == "n003"


def test_read_commands_drops_unparseable(tmp_path: Path):
    run_dir, _ = make_run(tmp_path)
    (run_dir / "control").mkdir()
    bad = run_dir / "control" / "zz-bad.json"
    bad.write_text("{nope")
    assert read_commands(run_dir) == []
    assert not bad.exists()


def test_clear_stale_stops_keeps_prunes(tmp_path: Path):
    run_dir, _ = make_run(tmp_path)
    write_command(run_dir, ControlCommand(action="stop"))
    write_command(run_dir, ControlCommand(action="prune", node_id="n001"))
    clear_stale_stops(run_dir)
    commands = read_commands(run_dir)
    assert [c.action for _, c in commands] == ["prune"]


def test_apply_prune_subtree(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)
    add_node(journal, run_dir, "n002", operator="improve", parent_id="n001", status="buggy")
    add_node(journal, run_dir, "n003", operator="debug", parent_id="n002", status="ok", val_score=0.6)
    add_node(journal, run_dir, "n004", status="ok", val_score=0.4)  # separate branch

    pruned = apply_prune(journal, "n002", reason="overfitting", source="cli")

    assert pruned == ["n002", "n003"]
    assert journal.get("n002").pruned and journal.get("n003").pruned
    assert not journal.get("n001").pruned and not journal.get("n004").pruned
    assert journal.get("n002").pruned_reason == "overfitting"
    # survives replay, audit line recorded
    reloaded = Journal(run_dir / "journal.jsonl")
    assert reloaded.get("n003").pruned
    assert '"event": "control"' in (run_dir / "journal.jsonl").read_text()


def test_apply_prune_refuses_baseline_and_unknown(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n000", operator="baseline", status="ok")
    with pytest.raises(ValueError, match="baseline"):
        apply_prune(journal, "n000")
    with pytest.raises(ValueError, match="No node"):
        apply_prune(journal, "n999")


def test_apply_prune_idempotent(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)
    assert apply_prune(journal, "n001") == ["n001"]
    assert apply_prune(journal, "n001") == []


def test_resync_best_repoints_after_pruning_selected(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)
    winner = add_node(journal, run_dir, "n002", status="ok", val_score=0.9, is_selected=True)
    (run_dir / "best" / "submission.csv").write_text((Path(winner.workspace) / "submission.csv").read_text())

    apply_prune(journal, "n002")
    selected = resync_best(run_dir, journal, lower_is_better=False, selection_mode="rank-blend")

    assert selected == "n001"
    assert "n001" in (run_dir / "best" / "submission.csv").read_text()
    assert journal.get("n001").is_selected
    assert not journal.get("n002").is_selected


def test_resync_best_falls_back_to_baseline(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n000", operator="baseline", status="ok")
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)

    apply_prune(journal, "n001")
    selected = resync_best(run_dir, journal, lower_is_better=False, selection_mode="rank-blend")

    assert selected is None
    assert "n000" in (run_dir / "best" / "submission.csv").read_text()
    assert not (run_dir / "best" / "solution.py").exists()


def test_request_prune_offline_applies_directly(tmp_path: Path):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)

    outcome = request_prune(
        run_dir, "n001", lower_is_better=False, selection_mode="rank-blend", source="cli"
    )

    assert "pruned n001" in outcome
    assert Journal(run_dir / "journal.jsonl").get("n001").pruned
    assert read_commands(run_dir) == []  # applied, not queued


def test_request_prune_queues_when_running(tmp_path: Path, monkeypatch):
    run_dir, journal = make_run(tmp_path)
    add_node(journal, run_dir, "n001", status="ok", val_score=0.5)
    monkeypatch.setattr("hillclimb.control.effective_state", lambda _: "running")

    outcome = request_prune(
        run_dir, "n001", lower_is_better=False, selection_mode="rank-blend", source="tui"
    )

    assert "queued" in outcome
    assert not Journal(run_dir / "journal.jsonl").get("n001").pruned
    commands = read_commands(run_dir)
    assert len(commands) == 1 and commands[0][1].action == "prune"
