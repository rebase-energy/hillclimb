"""A plateau costs as little as possible: an attempt that hands back its
parent or code the search already scored is never scored again, the greedy
selector spreads ties instead of re-asking one candidate, and `patience`
ends a search that has stopped finding anything better."""

from __future__ import annotations

from hillclimb.agents.fake import FakeAgent
from hillclimb.modules.policies.base import INJECT_ACTION, Action
from tests.catalog_fixture import module
from tests.conftest import ok_script
from tests.harness_factory import make_harness


def inject(harness, source):
    return harness.run(Action(operator=INJECT_ACTION, payload={"source": source}))


def test_code_the_search_scored_already_is_not_scored_again(task, config):
    harness, journal, _ = make_harness(task, config, FakeAgent())
    first = inject(harness, ok_script(0.5)).candidate
    again = inject(harness, ok_script(0.5))
    assert again.kind == "duplicate" and again.candidate.status == "abandoned"
    assert again.candidate.summary.startswith(f"same code as {first.candidate_id}, val 0.5")
    assert harness.spend().evaluations == 1  # the copy cost no verifier run


def test_an_improve_that_changes_nothing_is_not_scored(task, config):
    agent = FakeAgent()
    agent.queue(script=None, notes="looked, changed nothing\n")  # the parent's copy, untouched
    harness, _journal, _ = make_harness(task, config, agent)
    parent = inject(harness, ok_script(0.5)).candidate
    lazy = harness.run(Action(operator="improve", target_id=parent.candidate_id))
    assert lazy.kind == "unchanged" and lazy.candidate.status == "abandoned"
    assert harness.spend().evaluations == 1


def test_on_a_tie_greedy_builds_on_the_candidate_built_on_least(task, config):
    agent = FakeAgent()
    agent.queue(script=None, notes="nothing\n")
    harness, _journal, _ = make_harness(task, config, agent)
    a = inject(harness, ok_script(0.5)).candidate
    b = inject(harness, ok_script(0.5) + "# another way to the same score\n").candidate
    harness.run(Action(operator="improve", target_id=a.candidate_id))  # a has a child now
    best = module("greedy").Best()
    assert best.select(harness.view()).target_id == b.candidate_id


def test_patience_ends_a_search_that_stopped_finding_better(task, config):
    config.budget.patience = 2
    harness, _journal, _ = make_harness(task, config, FakeAgent())
    inject(harness, ok_script(0.5))
    inject(harness, ok_script(0.3))
    assert harness.open
    inject(harness, ok_script(0.2))
    assert not harness.open and "budget.patience" in harness.closed_reason
