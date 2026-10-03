"""What `hc.run` returns and `hc.open_search` reads: a search, without opening runs/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import hillclimb as hc
from hillclimb.harness.journal import Journal
from hillclimb.harness.status import SearchStatus
from hillclimb.harness.store import key_for, open_store
from hillclimb.results import HistoryStep, SearchOutcome

EVALUATIONS = 8


@pytest.fixture
def outcome(config, monkeypatch) -> SearchOutcome:
    """One finished toy search on the terrain."""
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    return hc.run(
        "fitness-landscape", agent="toy", max_evaluations=EVALUATIONS, learning=False, holdout=False,
        config=config, log=lambda *_: None,
    )


def test_a_result_holds_the_candidates_and_who_won(outcome):
    assert outcome.state == "done" and outcome.error is None
    assert [c.candidate_id for c in outcome.candidates][:2] == ["c000", "c001"]
    assert outcome.candidates[0].operator == "baseline"
    scores = [c.val_score for c in outcome.candidates if c.val_score is not None]
    assert outcome.best.val_score == max(scores) and outcome.higher_is_better
    assert outcome.selected.candidate_id == outcome.best.candidate_id  # no holdout: what ships is the best
    assert outcome.meta.problem_id == "fitness-landscape" and outcome.meta.agent == "toy"
    assert outcome.status.state == "done"


def test_history_is_the_staircase_and_spend_counts_what_the_climber_caused(outcome):
    history = outcome.history
    assert all(isinstance(step, HistoryStep) for step in history)
    assert history[0].candidate_id == "c000"  # the floor is where the climb starts
    assert [s.score for s in history] == sorted(s.score for s in history) and len(history) > 1
    assert history[-1].candidate_id == outcome.best.candidate_id
    assert [s.minutes for s in history] == sorted(s.minutes for s in history)

    spend = outcome.spend
    assert spend.evaluations == EVALUATIONS  # the baseline's own trial is the harness's floor, free
    assert spend.tokens == 0 and spend.cost_usd == 0.0
    assert spend.seconds is not None and spend.seconds >= 0


def test_what_ships_is_readable_as_text_and_values(outcome):
    selected = outcome.selected
    assert "X, Y = " in outcome.solution
    assert outcome.solution == (outcome.search_dir / "best" / "solution.py").read_text()
    assert outcome.source(selected.candidate_id) == outcome.solution
    assert outcome.source("c000") is None and outcome.source("nobody") is None  # a declared floor has no code
    assert set(outcome.params) == {"dx", "dy"}
    assert outcome.params == selected.best_trial.params


def test_to_frame_has_one_row_per_candidate_with_the_verifiers_numbers(outcome):
    frame = outcome.to_frame()
    assert list(frame.index) == [c.candidate_id for c in outcome.candidates]
    assert {"parent_id", "operator", "kind", "status", "val_score", "best", "selected", "n_trials",
            "tokens", "cost_usd", "minutes", "summary", "x", "y"} <= set(frame.columns)
    assert frame["best"].sum() == 1 and frame.index[frame["best"]][0] == outcome.best.candidate_id
    assert frame.loc[outcome.best.candidate_id, "val_score"] == outcome.best.val_score
    assert frame["n_trials"].sum() == EVALUATIONS + 1  # + the baseline's
    drafts = frame[frame["operator"] == "draft"]
    assert drafts["x"].between(-5, 5).all() and drafts["kind"].eq("create").all()


def test_the_journal_of_a_result_refuses_writes(outcome):
    journal = outcome.journal
    assert journal is outcome.journal  # a finished search is read once
    with pytest.raises(TypeError, match="read-only"):
        journal.candidate_result(outcome.best)
    with pytest.raises(TypeError, match="read-only"):
        journal.control_event("prune")


def test_open_search_reads_an_earlier_search_by_ref_or_the_latest(outcome, config):
    by_ref = hc.open_search(outcome.ref, config=config)
    latest = hc.open_search(config=config)
    by_run = hc.open_search(outcome.run_dir.name, config=config)  # a run with one search
    for opened in (by_ref, latest, by_run):
        assert opened.search_dir == outcome.search_dir and opened.state == "done"
        assert opened.selected.candidate_id == outcome.selected.candidate_id
        assert opened.spend.evaluations == EVALUATIONS
        assert opened.to_frame().equals(outcome.to_frame())
    with pytest.raises(LookupError):
        hc.open_search("no-such-run/no-such-search", config=config)


def test_a_result_built_without_a_config_reads_the_folder(outcome):
    bare = SearchOutcome(outcome.run_dir, outcome.search_dir, outcome.selected, "done")
    assert [c.candidate_id for c in bare.candidates] == [c.candidate_id for c in outcome.candidates]
    assert bare.best.candidate_id == outcome.best.candidate_id


def test_a_running_search_is_followed_and_a_finished_one_is_not_read_again(outcome, config):
    key = key_for(outcome.search_dir)
    store = open_store(config)
    # the search as another process would see it mid-climb: a live engine's status
    store.write_status(key, SearchStatus(search_id=key[1], run_id=key[0], state="running", pid=os.getpid()))
    live = hc.open_search(outcome.ref, config=config)
    assert live.state == "running"
    before = len(live.candidates)

    def land(candidate_id: str, score_from: str) -> None:
        """The engine journals one more result (the single writer, played here)."""
        engine = Journal(store.journal(key))
        landed = engine.candidates[score_from].model_copy(update={"candidate_id": candidate_id}, deep=True)
        engine.candidate_result(landed)

    land("c900", outcome.best.candidate_id)
    assert len(live.candidates) == before + 1 and live.state == "running"

    land("c901", outcome.best.candidate_id)
    store.write_status(key, SearchStatus(search_id=key[1], run_id=key[0], state="parked", last_error="out of credits"))
    assert len(live.candidates) == before + 2
    assert live.state == "parked" and live.error == "out of credits"

    land("c902", outcome.best.candidate_id)
    assert len(live.candidates) == before + 2  # over: what was read stays read
    store.close()
