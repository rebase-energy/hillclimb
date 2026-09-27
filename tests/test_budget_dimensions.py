"""The budget is the user's in every dimension: the clock, evaluations,
tokens and cost. A climber sees what is left and can never set or overshoot
it; the counts are journal-derived, so a resumed search needs no counter."""

from __future__ import annotations

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import Spend, journal_spend
from hillclimb.harness.candidate import AgentInfo, Candidate
from hillclimb.harness.journal import Journal
from hillclimb.harness.loop import PolicyLoop, SearchLoop
from hillclimb.modules.policies.greedy import GreedyPolicy
from hillclimb.modules.policies.base import Action
from hillclimb.harness.status import SearchStatus, StatusWriter
from tests.conftest import ok_script
from tests.factories import trial as mk_trial
from tests.harness_factory import make_harness


def test_spend_counts_what_the_climber_caused(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")

    def add(cid, operator, n_trials, tokens=None):
        journal.candidate_result(
            Candidate(
                candidate_id=cid, operator=operator, status="passing",
                trials=[mk_trial(val_score=0.5, submission_ok=True, index=i) for i in range(n_trials)],
                agent=AgentInfo(total_tokens=tokens, cost_usd=0.5 if tokens else None),
            )
        )

    add("c000", "baseline", 1)             # the harness's floor: free
    add("c001", "seed", 3)                 # its first trial is free, its two tune trials are not
    add("c002", "draft", 1, tokens=1000)
    add("c003", "improve", 2, tokens=500)  # one attempt + one tune trial
    add("c004", "draft", 0, tokens=200)    # the agent failed: tokens spent, nothing evaluated
    assert journal_spend(journal) == Spend(evaluations=5, tokens=1700, cost_usd=1.5)


def test_evaluation_budget_ends_the_search_like_the_clock_does(task, config):
    config.budget.max_evaluations = 3
    agent = FakeAgent()
    for score in (0.5, 0.6, 0.7, 0.8, 0.9):
        agent.queue(script=ok_script(score), notes="d\n")
    harness, journal, _ = make_harness(task, config, agent)

    selected = harness.execute(PolicyLoop(GreedyPolicy()))

    assert len(agent.requests) == 3 and harness.spend().evaluations == 3
    assert selected.val_score == 0.7
    assert harness.closed_reason == "evaluation budget spent" and harness.capacity == 0
    assert harness.view().budget.evaluations_remaining == 0


class FillEverySlot(SearchLoop):
    """Asks for as much as the harness will take, then records what it saw."""

    def __init__(self):
        self.accepted, self.remaining_seen = 0, []

    def run(self, harness):
        while harness.capacity:
            self.remaining_seen.append(harness.view().budget.evaluations_remaining)
            assert not harness.submit(Action(operator="draft")).rejected
            self.accepted += 1
        while harness.inflight:
            harness.wait()


def test_in_flight_work_has_its_evaluation_reserved(task, config):
    """Four free slots, two evaluations left: only two attempts start — the
    cap is never overshot by work that was already running."""
    config.concurrency.parallel_agents = 4
    config.budget.max_evaluations = 2
    agent = FakeAgent()
    for score in (0.5, 0.6, 0.7, 0.8):
        agent.queue(script=ok_script(score), notes="d\n")
    harness, _journal, _ = make_harness(task, config, agent)
    loop = FillEverySlot()

    harness.execute(loop)

    assert loop.accepted == 2 and loop.remaining_seen == [2, 1]
    assert harness.spend().evaluations == 2


def test_token_budget_closes_the_harness(task, config):
    config.budget.max_tokens = 1500
    agent = FakeAgent()
    for score in (0.5, 0.6, 0.7):
        agent.queue(script=ok_script(score), notes="d\n", result={"total_tokens": 1000})
    harness, _journal, _ = make_harness(task, config, agent)
    assert harness.view().budget.tokens_remaining == 1500

    harness.execute(PolicyLoop(GreedyPolicy()))

    # the second call crossed the line; a third never starts
    assert len(agent.requests) == 2 and harness.spend().tokens == 2000
    assert harness.closed_reason == "token budget spent"
    assert harness.view().budget.tokens_remaining == 0


def test_no_limits_means_no_limits(task, config):
    harness, _journal, _ = make_harness(task, config, FakeAgent())
    budget = harness.view().budget
    assert (budget.evaluations_remaining, budget.tokens_remaining, budget.cost_remaining_usd) == (None, None, None)
    assert harness.open


def test_status_reports_every_dimension(task, config):
    config.budget.max_evaluations = 5
    config.budget.max_tokens = 10_000
    written: list[SearchStatus] = []
    agent = FakeAgent()
    agent.queue(script=ok_script(0.5), notes="d\n", result={"total_tokens": 1234})
    status = StatusWriter(lambda s: written.append(s.model_copy(deep=True)), SearchStatus(search_id="s", run_id="r"))
    harness, _journal, _ = make_harness(task, config, agent, status=status)

    harness.run(Action(operator="draft"))

    budget = written[-1].budget
    assert (budget.evaluations, budget.max_evaluations) == (1, 5)
    assert (budget.tokens, budget.max_tokens) == (1234, 10_000)
