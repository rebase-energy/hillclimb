import json
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


def test_rank_blend_selection_robust_to_holdout_outlier(tmp_path):
    """The leaf-classification failure: a lucky-holdout early node must not
    beat a node that ranks well on BOTH signals (lower_is_better metric)."""
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("nA", status="ok", val_score=0.046, holdout_score=0.093))
    journal.node_result(make_node("nB", status="ok", val_score=0.138, holdout_score=0.084))
    journal.node_result(make_node("nC", status="ok", val_score=0.054, holdout_score=0.099))
    assert journal.selected_node(True).node_id == "nA"           # rank-blend
    assert journal.selected_node(True, "holdout").node_id == "nB"  # naive argmax
    assert journal.selected_node(True, "val").node_id == "nA"


def test_rank_blend_vetoes_val_overfit(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("honest", status="ok", val_score=0.80, holdout_score=0.85))
    journal.node_result(make_node("overfit", status="ok", val_score=0.99, holdout_score=0.40))
    assert journal.selected_node(False).node_id == "honest"


def test_rank_blend_ties_and_missing_holdout(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n1", status="ok", val_score=0.5))
    assert journal.selected_node(False).node_id == "n1"  # no holdout anywhere → val


def test_replay_skips_non_node_events(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    journal.node_result(make_node("n001", status="ok", val_score=0.5))
    journal.control_event("prune", node_id="n001", pruned=["n001"], source="cli")

    reloaded = Journal(path)
    assert set(reloaded.nodes) == {"n001"}
    assert '"event": "control"' in path.read_text()  # audit line survives


def test_legacy_lines_without_pruned_field_replay(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    record = make_node("n001", status="ok", val_score=0.5).model_dump()
    record.pop("pruned")
    record.pop("pruned_reason")
    path.write_text(json.dumps({"event": "node_result", **record}) + "\n")
    journal = Journal(path)
    assert journal.get("n001").pruned is False


def test_pruned_reappend_wins_on_replay(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    node = make_node("n001", status="ok", val_score=0.5)
    journal.node_result(node)
    node.pruned = True
    journal.node_result(node)

    reloaded = Journal(path)
    assert reloaded.get("n001").pruned is True


def test_queries_exclude_pruned(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n001", status="ok", val_score=0.5))
    journal.node_result(make_node("n002", status="ok", val_score=0.9, pruned=True))
    journal.node_result(make_node("n003", operator="improve", parent_id="n001", pruned=True))

    assert [n.node_id for n in journal.scored_nodes()] == ["n001"]
    assert [n.node_id for n in journal.drafts()] == ["n001"]
    assert journal.best_node(lower_is_better=False).node_id == "n001"
    assert journal.children("n001") == []
    assert [n.node_id for n in journal.children("n001", include_pruned=True)] == ["n003"]


def test_descendants_walks_whole_subtree(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n001"))
    journal.node_result(make_node("n002", operator="improve", parent_id="n001"))
    journal.node_result(make_node("n003", operator="debug", parent_id="n002", pruned=True))
    journal.node_result(make_node("n004"))
    assert {n.node_id for n in journal.descendants("n001")} == {"n002", "n003"}


def test_next_node_id_never_collides_with_gaps(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(make_node("n005"))  # gap: only n005 exists
    assert journal.next_node_id() == "n006"  # count-based would say n001
