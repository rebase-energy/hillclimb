import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hillclimb.candidate import Candidate, Trial
from hillclimb.journal import Journal


def make_candidate(candidate_id: str, **kwargs) -> Candidate:
    val_score = kwargs.pop("val_score", None)
    holdout_score = kwargs.pop("holdout_score", None)
    if val_score is not None or holdout_score is not None:
        kwargs["trials"] = [Trial(val_score=val_score, holdout_score=holdout_score)]
    return Candidate(candidate_id=candidate_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def test_append_and_replay(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    candidate = make_candidate("c001")
    journal.candidate_created(candidate)
    candidate.status = "ok"
    candidate.trials.append(Trial(val_score=0.5))
    journal.candidate_result(candidate)

    reloaded = Journal(path)
    assert reloaded.get("c001").status == "ok"
    assert reloaded.get("c001").val_score == 0.5
    assert len(path.read_text().splitlines()) == 2  # append-only


def test_best_candidate_direction(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001", status="ok", val_score=0.5))
    journal.candidate_result(make_candidate("c002", status="ok", val_score=0.9))
    journal.candidate_result(make_candidate("c003", status="buggy"))
    assert journal.best_candidate(lower_is_better=False).candidate_id == "c002"
    assert journal.best_candidate(lower_is_better=True).candidate_id == "c001"


def test_best_candidate_ignores_unscored_baseline(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c000", operator="baseline", status="ok"))
    assert journal.best_candidate(lower_is_better=False) is None


def test_debug_chain_and_siblings(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001", status="buggy"))
    journal.candidate_result(make_candidate("c002", operator="debug", parent_id="c001", status="buggy"))
    journal.candidate_result(make_candidate("c003", operator="debug", parent_id="c002", status="buggy"))
    chain = journal.debug_chain("c003")
    assert [c.candidate_id for c in chain] == ["c001", "c002", "c003"]

    journal.candidate_result(make_candidate("c004", operator="improve", parent_id="c001"))
    journal.candidate_result(make_candidate("c005", operator="improve", parent_id="c001"))
    assert {c.candidate_id for c in journal.siblings("c005")} == {"c002", "c004"}


def test_next_candidate_id(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    assert journal.next_candidate_id() == "c000"
    journal.candidate_result(make_candidate("c000", operator="baseline"))
    assert journal.next_candidate_id() == "c001"


def test_rank_blend_selection_robust_to_holdout_outlier(tmp_path):
    """The leaf-classification failure: a lucky-holdout early candidate must
    not beat one that ranks well on BOTH signals (lower_is_better metric)."""
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("cA", status="ok", val_score=0.046, holdout_score=0.093))
    journal.candidate_result(make_candidate("cB", status="ok", val_score=0.138, holdout_score=0.084))
    journal.candidate_result(make_candidate("cC", status="ok", val_score=0.054, holdout_score=0.099))
    assert journal.selected_candidate(True).candidate_id == "cA"           # rank-blend
    assert journal.selected_candidate(True, "holdout").candidate_id == "cB"  # naive argmax
    assert journal.selected_candidate(True, "val").candidate_id == "cA"


def test_rank_blend_vetoes_val_overfit(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("honest", status="ok", val_score=0.80, holdout_score=0.85))
    journal.candidate_result(make_candidate("overfit", status="ok", val_score=0.99, holdout_score=0.40))
    assert journal.selected_candidate(False).candidate_id == "honest"


def test_rank_blend_ties_and_missing_holdout(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c1", status="ok", val_score=0.5))
    assert journal.selected_candidate(False).candidate_id == "c1"  # no holdout anywhere → val


def test_replay_skips_non_candidate_events(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    journal.candidate_result(make_candidate("c001", status="ok", val_score=0.5))
    journal.control_event("prune", candidate_id="c001", pruned=["c001"], source="cli")

    reloaded = Journal(path)
    assert set(reloaded.candidates) == {"c001"}
    assert '"event": "control"' in path.read_text()  # audit line survives


def test_replay_rejects_v1_node_records(tmp_path: Path):
    """Clean break: an old-schema record behind the v2 gate is corruption and
    must fail loudly, not silently drop fields."""
    path = tmp_path / "j.jsonl"
    record = {"event": "candidate_result", "node_id": "n001", "operator": "draft",
              "status": "ok", "val_score": 0.5}
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValidationError):
        Journal(path)


def test_pruned_reappend_wins_on_replay(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    candidate = make_candidate("c001", status="ok", val_score=0.5)
    journal.candidate_result(candidate)
    candidate.pruned = True
    journal.candidate_result(candidate)

    reloaded = Journal(path)
    assert reloaded.get("c001").pruned is True


def test_queries_exclude_pruned(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001", status="ok", val_score=0.5))
    journal.candidate_result(make_candidate("c002", status="ok", val_score=0.9, pruned=True))
    journal.candidate_result(make_candidate("c003", operator="improve", parent_id="c001", pruned=True))

    assert [c.candidate_id for c in journal.scored_candidates()] == ["c001"]
    assert [c.candidate_id for c in journal.drafts()] == ["c001"]
    assert journal.best_candidate(lower_is_better=False).candidate_id == "c001"
    assert journal.children("c001") == []
    assert [c.candidate_id for c in journal.children("c001", include_pruned=True)] == ["c003"]


def test_descendants_walks_whole_subtree(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001"))
    journal.candidate_result(make_candidate("c002", operator="improve", parent_id="c001"))
    journal.candidate_result(make_candidate("c003", operator="debug", parent_id="c002", pruned=True))
    journal.candidate_result(make_candidate("c004"))
    assert {c.candidate_id for c in journal.descendants("c001")} == {"c002", "c003"}


def test_next_candidate_id_never_collides_with_gaps(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c005"))  # gap: only c005 exists
    assert journal.next_candidate_id() == "c006"  # count-based would say c001
