"""Driving one search a step at a time: `api.start` / `Search`, and the same
on the climber (`climber.start`, `climber.step`)."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import closing

import pytest

import hillclimb as hc
from hillclimb import api
from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.control import ControlCommand
from hillclimb.harness.glue import ParkedSearch
from hillclimb.harness.store import key_for, open_store
from hillclimb.sdk import Action, Loop
from tests.conftest import ok_script
from tests.harness_factory import make_harness
from tests.test_parallel_search import GOLDEN_SCENARIOS, GOLDEN_SEQUENCES, journal_sequence

QUIET = dict(log=lambda *_: None)


@pytest.fixture
def agent(task, config, monkeypatch):
    """A search on the synthetic problem with a scripted agent, as `hc.run` and
    `climber.start` find it."""
    agent = FakeAgent()
    monkeypatch.setattr("hillclimb.api.load_problem", lambda target, config: task)
    monkeypatch.setattr("hillclimb.api.get_agent", lambda *a, **k: agent)
    config.holdout.enabled = False
    config.learning.enabled = False
    yield agent
    if api._STEPPING is not None:  # a test that failed mid-search must not block the next
        api._STEPPING.close()


def queue(agent: FakeAgent, *scores: float) -> None:
    for score in scores:
        agent.queue(script=ok_script(score), notes=f"{score}\n")


def heartbeats() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name == "status-heartbeat"]


# --- stepping is the climber's own loop, taken one move at a time ---


@pytest.mark.parametrize("scenario_name", sorted(GOLDEN_SCENARIOS))
def test_stepping_reproduces_the_recorded_sequences(agent, config, scenario_name):
    """`step()` by `step()` writes the journal the built-in loop writes."""
    scenario = GOLDEN_SCENARIOS[scenario_name]
    scenario.queue(agent)
    search = api._new_search("anything", config=config, budget_s=3600, **QUIET)
    search.budget = BudgetManager(3600, stop_margin_s=1, spent_s=scenario.budget_spent)
    assert search.open() is None
    search.begin()
    for _ in range(scenario.max_candidates - 1):  # the baseline is the harness's
        assert search.step() is not None
    assert journal_sequence(search.search_dir / "journal.jsonl") == GOLDEN_SEQUENCES[scenario_name]
    assert search.close().state == "stopped"


def test_finish_after_steps_by_hand_ends_like_a_whole_run(agent, config):
    GOLDEN_SCENARIOS["drafts-then-improve"].queue(agent)
    search = hc.Climber(policy="greedy").start("anything", config=config, max_evaluations=4, **QUIET)
    assert [search.step().candidate.candidate_id for _ in range(2)] == ["c001", "c002"]
    outcome = search.finish()
    assert outcome.state == "done" and outcome.selected.val_score == 0.8
    assert journal_sequence(search.search_dir / "journal.jsonl") == GOLDEN_SEQUENCES["drafts-then-improve"]
    assert outcome.status.state == "done" and search.close() is outcome and search.finish() is outcome


# --- propose / run / step ---


def test_propose_runs_nothing_and_run_takes_your_own_action(agent, config):
    queue(agent, 0.6, 0.7, 0.9, 0.5)
    climber = hc.Climber(select=hc.selectors.Best(num_drafts=2, ensemble=False), policy=hc.policies.Greedy())
    search = climber.start("anything", config=config, **QUIET)
    assert climber.session is search and search.is_open and search.closed_reason is None
    assert climber.select() is None  # π_sel: no scored root yet, a root step

    first = climber.propose()
    assert first.operator == "draft" and climber.propose() == first  # asking twice changes nothing
    assert [c.candidate_id for c in climber.candidates] == ["c000"] and not agent.requests

    outcome = climber.run(first)
    assert outcome.kind == "evaluated" and outcome.candidate.val_score == 0.6
    assert climber.step().candidate.candidate_id == "c002"  # the second draft
    assert climber.best.candidate_id == "c002" and climber.state.higher_is_better

    # the person's move: not what greedy would do (it improves the best, c002)
    assert climber.propose().target_id == "c002"
    mine = climber.run(hc.Action("improve", target_id="c001"))
    assert mine.candidate.parent_id == "c001" and mine.candidate.val_score == 0.9
    # the policy observed it: the next proposal builds on the new best
    assert climber.propose().target_id == mine.candidate.candidate_id
    assert search.source("c003") == ok_script(0.9)
    assert search.result.state == "running" and search.result.spend.evaluations == 3
    climber.close()


def test_an_action_that_cannot_run_is_a_rejected_outcome_never_a_strike(agent, config):
    queue(agent, 0.6)
    search = api.start("anything", config=config, **QUIET)
    for _ in range(4):  # three refusals in a row would end a POLICY's search
        typo = search.run(Action("improve", target_id="c999"))
        assert typo.kind == "rejected" and typo.ticket.rejected == "no candidate c999 in this search"
        unscored = search.run(Action("improve", target_id="c000"))  # a declared floor has no code
        assert unscored.kind == "rejected" and unscored.candidate is None
    unknown = search.run(Action("no-such-operator"))
    assert unknown.kind == "rejected" and "no-such-operator" in unknown.ticket.rejected
    with pytest.raises(TypeError, match="takes an Action"):
        search.run("draft")
    assert search.step().kind == "evaluated" and not agent.responses  # still open, nothing was spent
    search.close()


# --- how a stepped search ends ---


def test_closing_by_hand_with_budget_left_is_a_resumable_stop(agent, config):
    queue(agent, 0.6)
    search = api.start("anything", config=config, **QUIET)
    search.step()
    outcome = search.close()
    assert outcome.state == "stopped" and outcome.error == "closed by hand with budget left"
    assert outcome.selected.candidate_id == "c001"  # what it would ship so far
    assert outcome.status.state == "stopped" and search.close() is outcome
    assert not search.is_open and search.result is outcome
    with pytest.raises(RuntimeError, match="closed"):
        search.step()


def test_a_spent_budget_closes_the_search_as_done(agent, config):
    queue(agent, 0.6, 0.7)
    with api.start("anything", config=config, max_evaluations=2, **QUIET) as search:
        assert search.step() and search.step()
        assert search.step() is None and search.propose() is None
        assert search.closed_reason == "evaluation budget spent"
        late = search.run(Action("draft"))
        assert late.kind == "rejected" and late.ticket.rejected == "evaluation budget spent"
    assert search.outcome.state == "done" and search.outcome.selected.val_score == 0.7


def test_a_stop_from_outside_surfaces_at_the_next_step(agent, config):
    queue(agent, 0.6)
    search = api.start("anything", config=config, **QUIET)
    search.step()
    with closing(open_store(config)) as store:
        store.enqueue_command(key_for(search.search_dir), ControlCommand(action="stop", source="cli"))
    assert search.propose() is None and "stop requested by cli" in search.closed_reason
    assert search.run(Action("draft")).kind == "rejected"
    outcome = search.close()
    assert outcome.state == "stopped" and "stop requested by cli" in outcome.error


def test_a_park_closes_the_search_as_parked(agent, config):
    queue(agent, 0.6)
    search = api.start("anything", config=config, **QUIET)
    search.step()
    search.harness._close(ParkedSearch("out of credits"))
    assert search.step() is None and search.closed_reason == "out of credits"
    outcome = search.close()
    assert outcome.state == "parked" and outcome.error == "out of credits"


def test_a_setup_that_fails_leaves_nothing_running(task, config, monkeypatch):
    monkeypatch.setattr("hillclimb.api.load_problem", lambda target, config: task)
    config.learning.enabled = False
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="Unknown agent: nobody"):
        api.start("anything", config=config, agent="nobody", **QUIET)
    assert api._STEPPING is None and not heartbeats()
    assert signal.getsignal(signal.SIGTERM) == before
    failed = hc.open_search(config=config)
    assert failed.state == "failed" and "Unknown agent: nobody" in failed.error


# --- one at a time, and nothing left behind ---


def test_one_stepped_search_at_a_time(agent, config):
    queue(agent, 0.6, 0.7)
    climber = hc.Climber(policy="greedy")
    with pytest.raises(RuntimeError, match="no search yet"):
        climber.step()
    climber.start("anything", config=config, **QUIET)
    with pytest.raises(RuntimeError, match="still on search"):
        climber.start("anything", config=config, **QUIET)
    with pytest.raises(RuntimeError, match="still open in this process"):
        hc.Climber(policy="greedy").start("anything", config=config, **QUIET)
    climber.step()
    first = climber.close()
    climber.start("anything", config=config, **QUIET)  # closed: the next one may start
    assert climber.session.ref != first.ref and climber.step().candidate.val_score == 0.7
    climber.close()


def test_the_clock_runs_only_while_a_step_does_and_nothing_outlives_the_search(agent, config):
    queue(agent, 0.6)
    before = signal.getsignal(signal.SIGTERM)
    search = api.start("anything", config=config, **QUIET)
    assert search.budget.paused and heartbeats() and signal.getsignal(signal.SIGTERM) != before
    left = search.budget.remaining()
    time.sleep(0.2)
    assert search.budget.remaining() == left  # reading an outcome costs no budget
    search.step()
    assert search.budget.paused and search.budget.remaining() < left
    search.close()
    assert not heartbeats() and signal.getsignal(signal.SIGTERM) == before


def test_starting_a_search_leaves_the_climber_the_definition_it_was(agent, config):
    import copy

    queue(agent, 0.6)
    climber = hc.Climber(select=hc.selectors.Best(num_drafts=2), policy=hc.policies.Greedy(), tuner="random")
    identity, block = climber.sha256, climber.to_spec().block()
    climber.start("anything", config=config, **QUIET)
    climber.step()
    assert (climber.sha256, climber.to_spec().block()) == (identity, block)
    assert copy.deepcopy(climber) is climber and "sha256" in repr(climber)
    climber.close()
    queue(agent, 0.7)  # and it still runs a whole search, like any climber
    assert hc.run("anything", climber=climber, config=config, max_evaluations=1, **QUIET).state == "done"


def test_a_class_of_this_process_steps_too(agent, config):
    class DraftsOnly(hc.policies.Policy):
        def propose(self, state, selection):
            return Action("draft")

    queue(agent, 0.6, 0.7)
    climber = hc.Climber(policy=DraftsOnly)
    assert not climber.portable
    climber.start("anything", config=config, **QUIET)
    assert [climber.step().candidate.operator for _ in range(2)] == ["draft", "draft"]
    assert climber.close().state == "stopped"


def test_a_loop_climber_cannot_be_stepped_only_finished(agent, config):
    class TwoShots(Loop):
        name = "two-shots"

        def run(self, harness):
            for _ in range(2):
                harness.run(Action(operator="draft"))

    queue(agent, 0.6, 0.7)
    climber = hc.Climber(loop=TwoShots)
    climber.start("anything", config=config, **QUIET)
    for call in (climber.propose, climber.step, lambda: climber.run(Action("draft"))):
        with pytest.raises(RuntimeError, match=r"TwoShots owns its control flow.*finish\(\)"):
            call()
    assert [c.candidate_id for c in climber.candidates] == ["c000"]
    outcome = climber.finish()
    assert outcome.state == "done" and len(outcome.candidates) == 3


# --- the harness and the clock underneath ---


def test_the_clock_pauses_and_resumes():
    budget = BudgetManager(100, stop_margin_s=1)
    budget.pause()
    budget.pause()  # idempotent
    frozen = budget.elapsed()
    time.sleep(0.05)
    assert budget.paused and budget.elapsed() == frozen
    budget.resume()
    budget.resume()
    time.sleep(0.05)
    assert not budget.paused and frozen + 0.04 < budget.elapsed() < frozen + 1.0


def test_start_writes_the_floor_once(task, config):
    harness, journal, _ = make_harness(task, config, FakeAgent())
    harness.start()
    harness.start()
    assert [c.operator for c in journal.candidates.values()] == ["baseline"]
    harness.raise_latched()  # nothing latched: nothing raised
    harness._close(ParkedSearch("credits"))
    with pytest.raises(ParkedSearch, match="credits"):
        harness.raise_latched()


def test_an_interrupted_run_abandons_its_own_attempt(task, config):
    class Interrupted(FakeAgent):
        def invoke(self, request):
            raise KeyboardInterrupt

    harness, journal, _ = make_harness(task, config, Interrupted())
    harness.start()
    with pytest.raises(KeyboardInterrupt):
        harness.run(Action(operator="draft"))
    assert not harness.inflight and journal.candidates["c001"].status == "abandoned"
    assert not journal.pending_candidates()
