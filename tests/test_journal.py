from tests.factories import trial as mk_trial
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.journal import Journal


def make_candidate(candidate_id: str, **kwargs) -> Candidate:
    val_score = kwargs.pop("val_score", None)
    holdout_score = kwargs.pop("holdout_score", None)
    if val_score is not None or holdout_score is not None:
        kwargs["trials"] = [mk_trial(val_score=val_score, holdout_score=holdout_score)]
    return Candidate(candidate_id=candidate_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def test_append_and_replay(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    candidate = make_candidate("c001")
    journal.candidate_created(candidate)
    candidate.status = "passing"
    candidate.trials.append(mk_trial(val_score=0.5))
    journal.candidate_result(candidate)

    reloaded = Journal(path)
    assert reloaded.get("c001").status == "passing"
    assert reloaded.get("c001").val_score == 0.5
    assert len(path.read_text().splitlines()) == 2  # append-only


def test_best_candidate_direction(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001", status="passing", val_score=0.5))
    journal.candidate_result(make_candidate("c002", status="passing", val_score=0.9))
    journal.candidate_result(make_candidate("c003", status="buggy"))
    assert journal.best_candidate(higher_is_better=True).candidate_id == "c002"
    assert journal.best_candidate(higher_is_better=False).candidate_id == "c001"


def test_best_candidate_ignores_unscored_baseline(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c000", operator="baseline", status="passing"))
    assert journal.best_candidate(higher_is_better=True) is None


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
    not beat one that ranks well on BOTH signals (higher_is_better metric)."""
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("cA", status="passing", val_score=0.046, holdout_score=0.093))
    journal.candidate_result(make_candidate("cB", status="passing", val_score=0.138, holdout_score=0.084))
    journal.candidate_result(make_candidate("cC", status="passing", val_score=0.054, holdout_score=0.099))
    assert journal.selected_candidate(False).candidate_id == "cA"           # rank-blend
    assert journal.selected_candidate(False, "holdout").candidate_id == "cB"  # naive argmax
    assert journal.selected_candidate(False, "val").candidate_id == "cA"


def test_rank_blend_vetoes_val_overfit(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("honest", status="passing", val_score=0.80, holdout_score=0.85))
    journal.candidate_result(make_candidate("overfit", status="passing", val_score=0.99, holdout_score=0.40))
    assert journal.selected_candidate(True).candidate_id == "honest"


def test_rank_blend_ties_and_missing_holdout(tmp_path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c1", status="passing", val_score=0.5))
    assert journal.selected_candidate(True).candidate_id == "c1"  # no holdout anywhere → val


def test_replay_skips_non_candidate_events(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    journal.candidate_result(make_candidate("c001", status="passing", val_score=0.5))
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
    candidate = make_candidate("c001", status="passing", val_score=0.5)
    journal.candidate_result(candidate)
    candidate.pruned = True
    journal.candidate_result(candidate)

    reloaded = Journal(path)
    assert reloaded.get("c001").pruned is True


def test_queries_exclude_pruned(tmp_path: Path):
    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(make_candidate("c001", status="passing", val_score=0.5))
    journal.candidate_result(make_candidate("c002", status="passing", val_score=0.9, pruned=True))
    journal.candidate_result(make_candidate("c003", operator="improve", parent_id="c001", pruned=True))

    assert [c.candidate_id for c in journal.scored_candidates()] == ["c001"]
    assert [c.candidate_id for c in journal.drafts()] == ["c001"]
    assert journal.best_candidate(higher_is_better=True).candidate_id == "c001"
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


def test_trial_report_roundtrip_and_prefeature_replay(tmp_path: Path):
    """New journals carry Trial.report; pre-feature lines (no report key on
    the trial) must replay unchanged under extra="forbid"."""
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    report = {"version": 1, "overall": {"score": 0.5}}
    candidate = make_candidate("c001", status="passing")
    candidate.trials.append(mk_trial(val_score=0.5, report=report))
    journal.candidate_result(candidate)

    record = json.loads(path.read_text().splitlines()[0])
    record["candidate_id"] = "c000"
    for entry in record["trials"]:
        for replicate in entry["replicates"]:
            replicate.pop("report")  # what a pre-feature engine wrote
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    reloaded = Journal(path)
    assert reloaded.get("c001").trials[-1].replicates[-1].report == report
    assert reloaded.get("c000").trials[-1].replicates[-1].report is None


def test_trial_cpu_roundtrip_and_prefeature_replay(tmp_path: Path):
    """New journals carry Trial.cpu_s/holdout_cpu_s; pre-feature lines (no cpu
    keys on the trial) must replay unchanged under extra="forbid"."""
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    candidate = make_candidate("c001", status="passing")
    candidate.trials.append(mk_trial(val_score=0.5, cpu_s=1.25, holdout_cpu_s=0.5))
    journal.candidate_result(candidate)

    record = json.loads(path.read_text().splitlines()[0])
    record["candidate_id"] = "c000"
    for entry in record["trials"]:
        entry.pop("holdout_cpu_s")  # what a pre-feature engine wrote
        for replicate in entry["replicates"]:
            replicate.pop("cpu_s")
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    reloaded = Journal(path)
    assert reloaded.get("c001").trials[-1].replicates[-1].cpu_s == 1.25
    assert reloaded.get("c001").trials[-1].holdout_cpu_s == 0.5
    assert reloaded.get("c000").trials[-1].replicates[-1].cpu_s is None
    assert reloaded.get("c000").trials[-1].holdout_cpu_s is None


# --- the Trial (params) → Replicate (seed) split: pre-split journals fold ---

FLAT_RECORD = {
    "candidate_id": "c001",
    "operator": "draft",
    "status": "ok",
    "candidate_dir": "/tmp/x",
}


def test_flat_trials_fold_into_one_trial_of_replicates(tmp_path: Path):
    """A pre-split record carries `trials` as seeded executions; they become
    ONE trial (params={}) whose replicates they are, holdout lifted from
    the last entry carrying it, and it is the candidate's best trial."""
    record = {**FLAT_RECORD, "trials": [
        {"seed": 0, "val_score": 1.0, "params": {}, "submission_ok": True,
         "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:00+00:00"},
        {"seed": 1, "val_score": 1.2, "submission_ok": True, "holdout_score": 0.9,
         "holdout_cpu_s": 2.0, "finished_at": "2026-01-01T00:02:00+00:00"},
        {"seed": 2, "val_score": 1.1, "submission_ok": True, "holdout_error": "late"},
    ]}
    path = tmp_path / "j.jsonl"
    path.write_text(json.dumps({"event": "candidate_result", **record}) + "\n")

    candidate = Journal(path).get("c001")
    assert candidate.status == "passing"  # historical `ok` migrates on replay
    assert len(candidate.trials) == 1
    trial = candidate.trials[0]
    assert trial.index == 0 and trial.params == {} and trial.is_best
    assert [r.seed for r in trial.replicates] == [0, 1, 2]
    assert trial.val_score == 1.1 and candidate.val_score == 1.1
    assert trial.holdout_score == 0.9 and trial.holdout_cpu_s == 2.0 and trial.holdout_error == "late"
    assert trial.started_at == "2026-01-01T00:00:00+00:00"
    assert trial.finished_at == "2026-01-01T00:02:00+00:00"
    assert candidate.holdout_score == 0.9
    assert "holdout_score" not in trial.replicates[1].model_dump()


def test_backfill_is_idempotent_and_new_shape_passes_through(tmp_path: Path):
    record = {**FLAT_RECORD, "trials": [{"seed": None, "val_score": 0.5, "submission_ok": True}]}
    once = Candidate.model_validate(record)
    twice = Candidate.model_validate(once.model_dump())
    assert twice.model_dump() == once.model_dump()
    assert twice.trials[0].replicates[0].val_score == 0.5

    new_shape = {**FLAT_RECORD, "trials": [
        {"index": 3, "params": {"lr": 0.1}, "replicates": [{"val_score": 0.7}], "is_best": True},
    ]}
    candidate = Candidate.model_validate(new_shape)
    assert candidate.trials[0].index == 3 and candidate.trials[0].params == {"lr": 0.1}
    assert candidate.val_score == 0.7


def test_old_and_new_records_for_one_candidate_replay_last_wins(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    old = {**FLAT_RECORD, "trials": [{"seed": None, "val_score": 0.5, "submission_ok": True}]}
    new = {**FLAT_RECORD, "trials": [
        {"index": 0, "params": {}, "replicates": [{"val_score": 0.5, "submission_ok": True}]},
        {"index": 1, "params": {"k": 2}, "replicates": [{"val_score": 0.8, "submission_ok": True}], "is_best": True},
    ]}
    path.write_text("".join(json.dumps({"event": "candidate_result", **r}) + "\n" for r in (old, new)))
    candidate = Journal(path).get("c001")
    assert len(candidate.trials) == 2 and candidate.val_score == 0.8


def test_flat_record_replays_through_the_sqlite_store(tmp_path: Path):
    from hillclimb.harness.store import SqliteDataStore

    store = SqliteDataStore(tmp_path / "store.sqlite", runs_dir=tmp_path / "runs")
    key = ("run-1", "search-1")
    store.journal(key).append({"event": "candidate_result", **FLAT_RECORD, "trials": [
        {"seed": 0, "val_score": 1.0, "submission_ok": True},
        {"seed": 1, "val_score": 1.4, "submission_ok": True, "holdout_score": 0.7},
    ]})
    candidate = Journal(store.journal(key)).get("c001")
    assert len(candidate.trials) == 1 and len(candidate.trials[0].replicates) == 2
    assert candidate.val_score == 1.2 and candidate.holdout_score == 0.7


def test_stamp_best_trial_follows_direction():
    candidate = Candidate.model_validate({**FLAT_RECORD, "trials": [
        {"index": 0, "params": {}, "replicates": [{"val_score": 0.5}]},
        {"index": 1, "params": {"k": 1}, "replicates": [{"val_score": 0.9}]},
        {"index": 2, "params": {"k": 2}, "replicates": [{"val_score": None}]},
    ]})
    assert candidate.best_trial.index == 1  # unstamped: last scored
    assert candidate.stamp_best_trial(True).index == 1
    assert [t.is_best for t in candidate.trials] == [False, True, False]
    assert candidate.stamp_best_trial(False).index == 0
    assert candidate.val_score == 0.5
