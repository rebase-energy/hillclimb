"""Credit assignment: rewards, event persistence, track folding, retirement."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.modules.memory.credit import (
    PRIOR_WEIGHT,
    RETIRE_MIN_INJECTIONS,
    REWARD_OK_NO_GAIN,
    CreditEvent,
    Track,
    adjusted_confidence,
    credit_event_path,
    fold_track,
    load_credit_events,
    read_injected_claims,
    record_injected_claims,
    search_reward,
    should_retire,
    write_credit_event,
)
from hillclimb.harness.journal import Journal
from hillclimb.modules.memory.knowledge import KnowledgeCard


class Problem:
    problem_id = "spaceship-titanic"
    metric_name = "accuracy"
    higher_is_better = True


class ProblemLower(Problem):
    problem_id = "gefcom2014-load"
    higher_is_better = False


def make_journal(tmp_path, entries) -> Journal:
    journal = Journal(tmp_path / "journal.jsonl")
    for e in entries:
        journal.candidate_result(Candidate(**e))
    return journal


def cand(cid, op, val, status="passing"):
    trials = [mk_trial(val_score=val)] if val is not None else []
    return dict(candidate_id=cid, operator=op, status=status, candidate_dir="w", trials=trials)


def prior(problem_id, val):
    return KnowledgeCard(problem_id=problem_id, family=problem_id, run_ref="r0/s0",
                         selected_val=val)


class TestSearchReward:
    def test_beats_prior_best(self, tmp_path):
        journal = make_journal(tmp_path, [cand("c001", "draft", 0.85)])
        reward, basis = search_reward(journal, Problem(), [prior("spaceship-titanic", 0.8)])
        assert (reward, basis) == (1.0, "prior-best")

    def test_scored_but_no_record(self, tmp_path):
        journal = make_journal(tmp_path, [cand("c001", "draft", 0.75)])
        reward, basis = search_reward(journal, Problem(), [prior("spaceship-titanic", 0.8)])
        assert (reward, basis) == (REWARD_OK_NO_GAIN, "prior-best")

    def test_direction_aware_lower_is_better(self, tmp_path):
        journal = make_journal(tmp_path, [cand("c001", "draft", 0.02)])
        reward, _ = search_reward(journal, ProblemLower(), [prior("gefcom2014-load", 0.03)])
        assert reward == 1.0
        reward, _ = search_reward(journal, ProblemLower(), [prior("gefcom2014-load", 0.01)])
        assert reward == REWARD_OK_NO_GAIN

    def test_prior_for_other_problem_ignored(self, tmp_path):
        # a family sibling's score is not this problem's bar
        journal = make_journal(tmp_path, [
            cand("c000", "baseline", 0.5),
            cand("c001", "draft", 0.7),
        ])
        reward, basis = search_reward(journal, Problem(), [prior("other-problem", 0.9)])
        assert (reward, basis) == (1.0, "own-baseline")

    def test_no_prior_beats_own_baseline(self, tmp_path):
        journal = make_journal(tmp_path, [
            cand("c000", "baseline", 0.5),
            cand("c001", "draft", 0.7),
        ])
        assert search_reward(journal, Problem(), []) == (1.0, "own-baseline")

    def test_no_prior_flat_vs_baseline(self, tmp_path):
        journal = make_journal(tmp_path, [
            cand("c000", "baseline", 0.7),
            cand("c001", "draft", 0.7),
        ])
        assert search_reward(journal, Problem(), []) == (REWARD_OK_NO_GAIN, "own-baseline")

    def test_first_search_unscored_baseline(self, tmp_path):
        journal = make_journal(tmp_path, [
            cand("c000", "baseline", None),
            cand("c001", "draft", 0.7),
        ])
        assert search_reward(journal, Problem(), []) == (1.0, "own-baseline")

    def test_nothing_scored(self, tmp_path):
        journal = make_journal(tmp_path, [
            cand("c000", "baseline", None),
            cand("c001", "draft", None, status="buggy"),
        ])
        assert search_reward(journal, Problem(), []) == (0.0, "none-scored")


class TestPersistence:
    def test_event_roundtrip_and_idempotent_path(self, tmp_path):
        event = CreditEvent(run_ref="r1/s1", problem_id="p", family="p",
                            claim_ids=["a", "b"], reward=1.0, basis="prior-best")
        path = write_credit_event(tmp_path, event)
        assert path == credit_event_path(tmp_path, "r1/s1")
        loaded = load_credit_events(tmp_path)
        assert len(loaded) == 1 and loaded[0].claim_ids == ["a", "b"]
        # same run_ref overwrites, never duplicates
        write_credit_event(tmp_path, event.model_copy(update={"reward": 0.25}))
        assert len(load_credit_events(tmp_path)) == 1

    def test_corrupt_event_skipped(self, tmp_path):
        write_credit_event(tmp_path, CreditEvent(run_ref="r1/s1", problem_id="p"))
        (tmp_path / "credit" / "bad.yaml").write_text("][ nope")
        assert len(load_credit_events(tmp_path)) == 1

    def test_injected_claims_sidecar_unions_on_resume(self, tmp_path):
        assert read_injected_claims(tmp_path) == []
        record_injected_claims(tmp_path, ["b", "a"])
        record_injected_claims(tmp_path, ["c", "a"])
        assert read_injected_claims(tmp_path) == ["a", "b", "c"]


class TestTrack:
    def test_fold_and_smoothing(self, tmp_path):
        events = [
            CreditEvent(run_ref="r1/s1", problem_id="p", claim_ids=["a", "b"],
                        reward=1.0, observed_at="2026-07-01T00:00:00Z"),
            CreditEvent(run_ref="r2/s1", problem_id="p", claim_ids=["a"],
                        reward=0.0, observed_at="2026-07-02T00:00:00Z"),
        ]
        tracks = fold_track(events)
        assert tracks["a"].injections == 2
        assert tracks["a"].mean_reward == pytest.approx(0.5)
        assert tracks["a"].last_observed_at == "2026-07-02T00:00:00Z"
        assert tracks["b"].injections == 1
        # authored 0.5 with rewards [1.0, 0.0]: (0.5*2 + 1.0) / (2 + 2) = 0.5
        assert adjusted_confidence(0.5, tracks["a"]) == pytest.approx(
            (0.5 * PRIOR_WEIGHT + 1.0) / (PRIOR_WEIGHT + 2)
        )

    def test_retirement_boundaries(self):
        losing_twice = Track(injections=2, reward_sum=0.0)
        assert not should_retire(adjusted_confidence(0.1, losing_twice), losing_twice)
        losing_thrice = Track(injections=RETIRE_MIN_INJECTIONS, reward_sum=0.0)
        # authored 0.3, three zero rewards: 0.6/5 = 0.12 < 0.15 -> retire
        assert should_retire(adjusted_confidence(0.3, losing_thrice), losing_thrice)
        # a win in the record keeps it alive: (0.6 + 1.0)/5 = 0.32
        mixed = Track(injections=3, reward_sum=1.0)
        assert not should_retire(adjusted_confidence(0.3, mixed), mixed)
