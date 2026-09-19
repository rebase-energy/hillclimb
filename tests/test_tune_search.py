"""Tune jobs end to end: a candidate declares params.json, the greedy
policy proposes tune actions, the harness runs extra trials with the
tuner's values (no agent), commits them onto the live candidate, and the
best trial's values ship in best/ and inherit into children."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate
from hillclimb.dirs import create_search_dir
from hillclimb.journal import Journal
from hillclimb.policies.greedy import GreedyPolicy
from hillclimb.policy import TUNE_ACTION, Action, BudgetView, InflightRef, PolicyInput
from hillclimb.search import GreedySearcher, OutcomeMsg
from tests.conftest import executor_for, ok_script
from tests.factories import candidate as make_candidate, trial
from tests.test_search import FileHoldoutScorer

PARAMS = json.dumps({"k": {"type": "int", "low": 0, "high": 9, "default": 1}})

# score = 0.5 + 0.01 * k, so a tuner can be seen finding larger k
TUNED_SCRIPT = """\
import shutil
from hillclimb import spaces
P = spaces.params({"k": 1})
shutil.copy("data/sample_submission.csv", "submission.csv")
print(f"val_score: {0.5 + 0.01 * P['k']:.4f}")
"""

TUNED_HOLDOUT_SCRIPT = """\
import shutil
from hillclimb import spaces
P = spaces.params({"k": 1})
shutil.copy("data/sample_submission.csv", "submission.csv")
open("holdout_predictions.csv", "w").write(str(0.9 - 0.01 * P['k']))
print(f"val_score: {0.5 + 0.01 * P['k']:.4f}")
"""


def make_searcher(task, config, backend, max_candidates=3, holdout=False, **policy_params):
    config.search.num_drafts = 1
    config.search.policy_params = {"tune_budget": 2, **policy_params}
    config.search.tuner_params = {"seed": 1}
    search_dir = create_search_dir(config.paths.runs_dir, "tune-run")
    journal = Journal(search_dir / "journal.jsonl")
    kwargs = {}
    if holdout:
        from hillclimb.evaluation import CandidateEvaluator

        task = task.model_copy(update={"holdout_cmd": task.verifier_cmd + ["--holdout"]})
        kwargs["evaluator"] = CandidateEvaluator(
            executor=executor_for(task), problem=task, config=config,
            holdout_scorer=FileHoldoutScorer(), journal=journal,
        )
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=executor_for(task), budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=max_candidates, log=lambda *_: None,
        policy=GreedyPolicy(params=config.search.policy_params), **kwargs,
    )
    return searcher, journal, search_dir


def test_tune_trials_land_on_the_declaring_candidate(task, config):
    backend = FakeBackend()
    backend.queue(script=TUNED_SCRIPT, notes="tunable draft\n", files={"params.json": PARAMS})
    backend.queue(script=ok_script(0.3), notes="worse improve\n")
    searcher, journal, search_dir = make_searcher(task, config, backend)
    searcher.run()

    draft = journal.get("c001")
    assert draft.tunable and draft.params_error is None
    assert len(draft.trials) == 3  # defaults + tune_budget
    assert [t.index for t in draft.trials] == [0, 1, 2]
    assert draft.trials[0].params == {"k": 1}
    for t in draft.trials:
        assert t.val_score == pytest.approx(0.5 + 0.01 * t.params["k"])
        assert set(t.params) == {"k"}
    best = draft.best_trial
    assert best.is_best and best.val_score == max(t.val_score for t in draft.trials)
    assert draft.val_score == best.val_score
    # the trial dir carries the values document; the root keeps the agent's defaults
    t1 = json.loads((Path(draft.candidate_dir) / "trials" / "t1" / "params.json").read_text())
    assert t1["k"]["value"] == draft.trials[1].params["k"]
    assert json.loads((Path(draft.candidate_dir) / "params.json").read_text())["k"]["default"] == 1
    # no agent ran for the tune jobs; the improve ran after the budget was spent
    assert [r.operator for r in backend.requests] == ["draft", "improve"]
    # re-journaled per tune commit; replay keeps the last record
    records = [json.loads(line) for line in (search_dir / "journal.jsonl").read_text().splitlines()]
    assert sum(1 for r in records if r["event"] == "candidate_result" and r["candidate_id"] == "c001") >= 3
    assert [r["event"] for r in records if r["event"].startswith("tune_")] == ["tune_started"] * 2
    replayed = Journal(search_dir / "journal.jsonl").get("c001")
    assert replayed.model_dump() == draft.model_dump()
    # best/ ships the winning values
    shipped = json.loads((search_dir / "best" / "params.json").read_text())
    assert shipped["k"]["value"] == best.params["k"]
    # the child inherited the best values as its defaults, and was told so
    child = journal.get("c002")
    inherited = json.loads((Path(child.candidate_dir) / "params.json").read_text())
    assert inherited["k"]["default"] == best.params["k"] and "value" not in inherited["k"]
    improve_prompt = backend.requests[1].prompt
    assert "The parent declared tunable parameters" in improve_prompt
    assert f"k: int in [0, 9], default 1 = {best.params['k']}" in improve_prompt
    assert "## Tunable parameters (optional)" in improve_prompt


def test_malformed_declaration_is_scored_on_defaults_and_never_tuned(task, config):
    backend = FakeBackend()
    backend.queue(script=TUNED_SCRIPT, notes="broken declaration\n", files={"params.json": "{oops"})
    backend.queue(script=ok_script(0.3), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    searcher.run()

    draft = journal.get("c001")
    assert draft.status == "passing" and draft.val_score == pytest.approx(0.51)
    assert not draft.tunable and draft.params_error
    assert len(draft.trials) == 1 and draft.trials[0].params == {}
    assert [r.operator for r in backend.requests] == ["draft", "improve"]


def test_holdout_runs_for_the_winning_trial_only(task, config):
    backend = FakeBackend()
    backend.queue(script=TUNED_HOLDOUT_SCRIPT, notes="tunable\n", files={"params.json": PARAMS})
    backend.queue(script=ok_script(0.3), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend, holdout=True)
    scorer = searcher.evaluator.holdout_scorer
    searcher.run()

    draft = journal.get("c001")
    scored = [t for t in draft.trials if t.holdout_score is not None]
    best = draft.best_trial
    assert best in scored  # the winning parameter set has its own holdout
    assert best.holdout_score == pytest.approx(0.9 - 0.01 * best.params["k"])
    assert draft.holdout_score == best.holdout_score
    # every holdout call named the trial it scored, never a losing one
    indices = [t.index for cid, t in scorer.trials if cid == "c001"]
    assert best.index in indices
    losers = [t for t in draft.trials if t.val_score < best.val_score]
    assert all(t.holdout_score is None for t in losers if t.index not in indices)


def test_tune_beats_the_incumbent_through_the_accept_band(task, config):
    """A tune trial that lifts a runner-up above the best promotes it."""
    backend = FakeBackend()
    backend.queue(script=ok_script(0.55), notes="plain best\n")
    backend.queue(script=TUNED_SCRIPT, notes="tunable runner-up\n", files={"params.json": PARAMS})
    backend.queue(script=ok_script(0.2), notes="later improve\n")
    # nine distinct draws over k in 0..9 must reach k >= 6; no interleave
    searcher, journal, _ = make_searcher(
        task, config, backend, max_candidates=4, tune_budget=9, tune_burst=9, tune_gate="always"
    )
    searcher.run()

    plain, tuned = journal.get("c001"), journal.get("c002")
    assert plain.val_score == 0.55 and tuned.trials[0].val_score == pytest.approx(0.51)
    assert tuned.val_score > 0.55  # some trial found k >= 6
    assert tuned.is_best  # promoted by a tune trial, not by an agent
    assert journal.best_candidate(True).candidate_id == "c002"
    assert journal.get("c003").parent_id == "c002"  # the later improve built on the promoted one


def test_parallel_tune_jobs_get_distinct_indices(task, config):
    """Two tune jobs on one candidate in flight at once reserve distinct
    trial indices and both land. The two drafts may land in either order
    (the free slot may take an improve first), so the candidate cap and the
    response queue leave room for both orders."""
    config.search.parallel_operators = 2
    backend = FakeBackend()
    backend.queue(script=TUNED_SCRIPT, notes="tunable\n", files={"params.json": PARAMS})
    backend.queue(script=ok_script(0.3), notes="second draft (pool fills it)\n")
    for _ in range(4):
        backend.queue(script=ok_script(0.2), notes="improve\n")
    searcher, journal, _ = make_searcher(
        task, config, backend, max_candidates=6, tune_budget=4, tune_parallel=2
    )
    searcher.run()

    tuned = [c for c in journal.candidates.values() if c.tunable and c.operator == "draft"]
    assert len(tuned) == 1
    assert sorted(t.index for t in tuned[0].trials) == [0, 1, 2, 3, 4]
    assert len({t.index for t in tuned[0].trials}) == 5


def test_aborted_tune_job_leaves_the_candidate_untouched(task, config):
    backend = FakeBackend()
    backend.queue(script=TUNED_SCRIPT, notes="tunable\n", files={"params.json": PARAMS})
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=2)
    searcher.run()  # baseline c000 + the draft c001, then the candidate cap stops it
    before = journal.get("c001").model_dump()

    job = searcher._prepare_tune(Action(operator=TUNE_ACTION, target_id="c001"))
    assert job is not None and job.kind == "tune" and job.trial_index == 1
    assert job.candidate is not journal.get("c001")  # the worker gets a copy
    searcher._commit(OutcomeMsg(job=job, kind="aborted"))

    assert journal.get("c001").model_dump() == before
    assert searcher._inflight == {}
    events = [json.loads(l)["event"] for l in (search_dir / "journal.jsonl").read_text().splitlines()]
    assert events[-2:] == ["tune_started", "tune_discarded"]


def test_run_operator_tune_refuses_an_untunable_target(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="plain\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=2)
    draft = searcher.run_operator("draft", None)  # c000: no baseline written by run_operator
    with pytest.raises(ValueError, match="cannot be tuned"):
        searcher.run_operator(TUNE_ACTION, draft)


# --- the greedy rule, as a pure function of journal + in-flight refs ---


def make_view(journal, config, inflight=(), remaining_s=3600.0) -> PolicyInput:
    return PolicyInput(
        journal=journal, inflight=tuple(inflight),
        budget=BudgetView(remaining_s=remaining_s, total_s=3600, stop_margin_s=1),
        config=config, higher_is_better=True,
    )


def tunable(cid, *scores, trials=None, **kw) -> Candidate:
    return make_candidate(cid, *scores, trials=trials, tunable=True, candidate_dir="/tmp/x", **kw)


def seeded_journal(tmp_path, *candidates) -> Journal:
    tmp_path.mkdir(parents=True, exist_ok=True)
    journal = Journal(tmp_path / "journal.jsonl")
    for cand in candidates:
        journal.candidate_result(cand)
    return journal


class TestGreedyTuneRule:
    def test_off_by_default_for_undeclared_candidates(self, tmp_path, config):
        journal = seeded_journal(tmp_path, make_candidate("c001", 0.5, candidate_dir="/tmp/x"))
        assert GreedyPolicy().tune_target(make_view(journal, config)) is None

    def test_budget_counts_trials_and_inflight(self, tmp_path, config):
        journal = seeded_journal(tmp_path, tunable("c001", 0.5))
        policy = GreedyPolicy(params={"tune_budget": 2})
        assert policy.tune_target(make_view(journal, config)).candidate_id == "c001"
        one_inflight = (InflightRef("c001", TUNE_ACTION, None, trial_index=1),)
        assert policy.tune_target(make_view(journal, config, one_inflight)) is None  # tune_parallel=1
        policy = GreedyPolicy(params={"tune_budget": 2, "tune_parallel": 2})
        assert policy.tune_target(make_view(journal, config, one_inflight)) is not None
        two = one_inflight + (InflightRef("c001", TUNE_ACTION, None, trial_index=2),)
        assert policy.tune_target(make_view(journal, config, two)) is None  # budget spent
        spent = seeded_journal(tmp_path / "b", tunable("c002", trials=[trial(0.5), trial(0.6, index=1), trial(0.4, index=2, is_best=False)]))
        assert policy.tune_target(make_view(spent, config)) is None
        assert GreedyPolicy(params={"tune_budget": 0}).tune_target(make_view(journal, config)) is None

    def test_gate_modes(self, tmp_path, config):
        config.search.min_improvement = 0.05
        journal = seeded_journal(
            tmp_path, tunable("c001", 0.9, is_best=True), tunable("c002", 0.87), tunable("c003", 0.5),
        )
        picks = lambda **p: GreedyPolicy(params={"tune_budget": 8, **p}).tune_target(make_view(journal, config))  # noqa: E731
        assert picks(tune_gate="best").candidate_id == "c001"
        assert picks(tune_gate="band").candidate_id == "c001"  # best first
        # once the best is spent, the band admits c002 (within 0.05) but not c003
        spent_best = tunable("c001", trials=[trial(0.9), *[trial(0.8, index=i, is_best=False) for i in range(1, 9)]], is_best=True)
        journal = seeded_journal(tmp_path / "b", spent_best, tunable("c002", 0.87), tunable("c003", 0.5))
        picks = lambda **p: GreedyPolicy(params={"tune_budget": 8, **p}).tune_target(make_view(journal, config))  # noqa: E731
        assert picks(tune_gate="band").candidate_id == "c002"
        assert picks(tune_gate="best") is None
        journal = seeded_journal(tmp_path / "c", spent_best, tunable("c003", 0.5))
        picks = lambda **p: GreedyPolicy(params={"tune_budget": 8, **p}).tune_target(make_view(journal, config))  # noqa: E731
        assert picks(tune_gate="band") is None
        assert picks(tune_gate="always").candidate_id == "c003"

    def test_burst_interleaves_with_agent_proposals(self, tmp_path, config):
        cand = tunable("c001", trials=[trial(0.5), trial(0.6, index=1), trial(0.4, index=2, is_best=False)])
        journal = seeded_journal(tmp_path, cand)
        policy = GreedyPolicy(params={"tune_budget": 8, "tune_burst": 2})
        assert policy.tune_target(make_view(journal, config)) is None  # 2 spent, no agent since
        later = seeded_journal(tmp_path / "b", cand, make_candidate("c002", 0.3, candidate_dir="/tmp/x"))
        assert policy.tune_target(make_view(later, config)).candidate_id == "c001"
        busy = (InflightRef("c002", "improve", "c001"),)
        assert policy.tune_target(make_view(journal, config, busy)).candidate_id == "c001"

    def test_no_tune_without_headroom_for_a_trial(self, tmp_path, config):
        slow = tunable("c001", trials=[trial(0.5, duration_s=100.0)])
        journal = seeded_journal(tmp_path, slow)
        policy = GreedyPolicy(params={"tune_budget": 8})
        assert policy.tune_target(make_view(journal, config, remaining_s=120.0)) is None  # 1.5x100 > 119
        assert policy.tune_target(make_view(journal, config, remaining_s=200.0)) is not None
        quick = seeded_journal(tmp_path / "b", tunable("c002", trials=[trial(0.5, duration_s=1.0)]))
        assert policy.tune_target(make_view(quick, config, remaining_s=40.0)) is None  # 30s floor x1.5
        assert policy.tune_target(make_view(quick, config, remaining_s=60.0)) is not None

    def test_propose_prefers_tune_over_improve_once_drafts_exist(self, tmp_path, config):
        config.search.num_drafts = 1
        journal = seeded_journal(tmp_path, tunable("c001", 0.5))
        action = GreedyPolicy(params={"tune_budget": 2}).propose(make_view(journal, config))
        assert action == Action(operator=TUNE_ACTION, target_id="c001")
        action = GreedyPolicy(params={"tune_budget": 0}).propose(make_view(journal, config))
        assert action.operator == "improve"

    def test_same_journal_same_proposals(self, tmp_path, config):
        config.search.num_drafts = 1
        journal = seeded_journal(tmp_path, tunable("c001", 0.5))
        a = GreedyPolicy(params={"tune_budget": 2}).propose(make_view(journal, config))
        b = GreedyPolicy(params={"tune_budget": 2}).propose(make_view(Journal(tmp_path / "journal.jsonl"), config))
        assert a == b
