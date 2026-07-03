from pathlib import Path

from hillclimb.journal import Journal
from hillclimb.node import Node


def make_node(node_id: str, **kwargs) -> Node:
    return Node(node_id=node_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def test_append_and_replay(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    node = make_node("n001")
    journal.node_created(node)
    node.status = "ok"
    node.val_score = 0.5
    journal.node_result(node)

    reloaded = Journal(path)
    assert reloaded.get("n001").status == "ok"
    assert reloaded.get("n001").val_score == 0.5
    assert len(path.read_text().splitlines()) == 2  # append-only


def test_best_node_direction(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n001", status="ok", val_score=0.5))
    journal.node_result(make_node("n002", status="ok", val_score=0.9))
    journal.node_result(make_node("n003", status="buggy"))
    assert journal.best_node(lower_is_better=False).node_id == "n002"
    assert journal.best_node(lower_is_better=True).node_id == "n001"


def test_best_node_ignores_unscored_baseline(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n000", operator="baseline", status="ok"))
    assert journal.best_node(lower_is_better=False) is None


def test_debug_chain_and_siblings(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n001", status="buggy"))
    journal.node_result(make_node("n002", operator="debug", parent_id="n001", status="buggy"))
    journal.node_result(make_node("n003", operator="debug", parent_id="n002", status="buggy"))
    chain = journal.debug_chain("n003")
    assert [n.node_id for n in chain] == ["n001", "n002", "n003"]

    journal.node_result(make_node("n004", operator="improve", parent_id="n001"))
    journal.node_result(make_node("n005", operator="improve", parent_id="n001"))
    assert {n.node_id for n in journal.siblings("n005")} == {"n002", "n004"}


def test_next_node_id(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    assert journal.next_node_id() == "n000"
    journal.node_result(make_node("n000", operator="baseline"))
    assert journal.next_node_id() == "n001"
