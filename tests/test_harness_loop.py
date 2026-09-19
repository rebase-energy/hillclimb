"""The harness as a `SearchLoop` sees it: submit / wait / run, the closed
latch, refusals — without any policy in the picture."""

from __future__ import annotations

import json

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.control import ControlCommand
from hillclimb.loop import ClimberError, HarnessClosed, SearchLoop
from hillclimb.policy import Action
from hillclimb.search_strategy import StopRequested
from tests.conftest import ok_script
from tests.harness_factory import make_harness

CRASH = 'raise RuntimeError("boom")\n'


def audit_lines(search_dir, event):
    path = search_dir / "journal.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [r for r in map(json.loads, lines) if r.get("event") == event]


class Generations(SearchLoop):
    """Not a policy: synchronized generations — fill every slot with drafts,
    wait for ALL of them, then refine the generation's best."""

    name = "generations"

    def __init__(self):
        self.outcomes = []

    def run(self, harness):
        for _ in range(harness.info.parallelism):
            assert not harness.submit(Action(operator="draft")).rejected
        while harness.inflight:
            self.outcomes += harness.wait()
        scored = [o.candidate for o in self.outcomes if o.candidate.is_scored]
        best = max(scored, key=lambda c: c.val_score)
        self.outcomes.append(harness.run(Action(operator="improve", target_id=best.candidate_id)))


def test_a_loop_that_is_not_a_policy_drives_a_search(task, config):
    config.search.parallel_operators = 2
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="a\n")
    backend.queue(script=ok_script(0.7), notes="b\n")
    backend.queue(script=ok_script(0.9), notes="refined\n")
    harness, journal, _ = make_harness(task, config, backend)
    loop = Generations()

    selected = harness.execute(loop)

    assert [o.kind for o in loop.outcomes] == ["evaluated"] * 3
    assert selected.val_score == 0.9 and selected.operator == "improve"
    parent = journal.get(selected.parent_id)
    assert parent.val_score == 0.7  # the generation's best, not the first to land
    assert all(o.candidate.holdout_score is None for o in loop.outcomes)
    assert harness.info.metric_name == task.metric_name and harness.info.parallelism == 2


class Stubborn(SearchLoop):
    """Swallows everything and keeps asking for work."""

    name = "stubborn"

    def __init__(self):
        self.closed_errors = 0

    def run(self, harness):
        for _ in range(10):
            try:
                harness.run(Action(operator="draft"))
            except Exception:  # noqa: BLE001 — a loop (or the library it wraps) that eats exceptions
                self.closed_errors += 1


def test_a_loop_that_ignores_a_stop_cannot_spend_any_more(task, config):
    backend = FakeBackend()
    for score in (0.5, 0.6, 0.7):
        backend.queue(script=ok_script(score), notes="d\n")
    commands: list = []
    harness, journal, _ = make_harness(task, config, backend, drain_commands=lambda: [commands.pop()] if commands else [])
    loop = Stubborn()

    original = harness._execute_job

    def stop_after_first(job):
        commands.append(ControlCommand(action="stop"))
        return original(job)

    harness._execute_job = stop_after_first
    with pytest.raises(StopRequested):
        harness.execute(loop)

    assert len(backend.requests) == 1  # the stop landed during the first attempt
    assert loop.closed_errors == 9  # every later ask was refused with HarnessClosed
    assert not harness.open and "stop" in harness.closed_reason.lower()
    assert harness.capacity == 0
    assert len([c for c in journal.candidates.values() if c.operator == "draft"]) == 1


def test_refused_actions_come_back_as_rejected_tickets_and_leave_no_trace(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="d\n")
    harness, journal, search_dir = make_harness(task, config, backend)
    passing = harness.run(Action(operator="draft")).candidate
    ids_before = set(journal.candidates)

    outcome = harness.run(Action(operator="debug", target_id=passing.candidate_id))

    assert outcome.kind == "rejected" and outcome.candidate is None
    assert "whose status is passing" in outcome.ticket.rejected
    assert set(journal.candidates) == ids_before and len(backend.requests) == 1
    rejected = audit_lines(search_dir, "action_rejected")
    assert [(r["operator"], r["target_id"]) for r in rejected] == [("debug", passing.candidate_id)]
    # a dangling id and an unknown operator are refusals too, not crashes
    assert harness.run(Action(operator="improve", target_id="c999")).kind == "rejected"
    with pytest.raises(ClimberError, match="3 actions refused in a row"):
        harness.run(Action(operator="crossover"))


def test_a_good_action_resets_the_refusal_count(task, config):
    backend = FakeBackend()
    for score in (0.5, 0.6):
        backend.queue(script=ok_script(score), notes="d\n")
    harness, _journal, _ = make_harness(task, config, backend)
    for _ in range(2):
        assert harness.run(Action(operator="crossover")).kind == "rejected"
    assert harness.run(Action(operator="draft")).kind == "evaluated"
    for _ in range(2):
        assert harness.run(Action(operator="crossover")).kind == "rejected"
    assert harness.run(Action(operator="draft")).kind == "evaluated"


def test_run_reports_what_happened_to_the_attempt(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH, notes="broken\n")
    backend.queue(script=ok_script(0.6), notes="fixed\n")
    harness, journal, _ = make_harness(task, config, backend)

    broken = harness.run(Action(operator="draft"))
    assert broken.kind == "evaluated" and broken.candidate.status == "buggy"
    fixed = harness.run(Action(operator="debug", target_id=broken.candidate.candidate_id))
    assert fixed.candidate.status == "passing" and fixed.candidate.role == "repair"
    assert fixed.ticket.candidate_id == fixed.candidate.candidate_id
    assert harness.source(fixed.candidate.candidate_id) == ok_script(0.6)
    assert harness.source("c999") is None
    assert not harness.inflight


def test_a_closed_harness_refuses_work_and_says_why(task, config):
    harness, _journal, _ = make_harness(task, config, FakeBackend(), max_candidates=0)
    assert not harness.open and harness.capacity == 0
    assert harness.closed_reason == "evaluation cap reached"
    with pytest.raises(HarnessClosed, match="evaluation cap reached"):
        harness.run(Action(operator="draft"))
    with pytest.raises(HarnessClosed):
        harness.submit(Action(operator="draft"))


class LeavesWorkBehind(SearchLoop):
    name = "careless"

    def run(self, harness):
        harness.submit(Action(operator="draft"))  # ...and returns without waiting


def test_work_a_loop_left_in_flight_is_still_committed(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.8), notes="d\n")
    harness, journal, _ = make_harness(task, config, backend)

    selected = harness.execute(LeavesWorkBehind())

    assert selected.val_score == 0.8
    assert not journal.pending_candidates() and not harness.inflight


def test_wait_with_nothing_in_flight_is_a_tick(task, config):
    harness, _journal, _ = make_harness(task, config, FakeBackend())
    assert harness.wait(timeout=0.01) == []
    assert harness.open


def test_running_out_of_budget_is_a_quiet_refusal_not_an_error(task, config):
    """The clock moves off the loop's thread: a loop that saw capacity and
    then lost the race to the budget gets a refusal, never an exception."""
    from hillclimb.budget import BudgetManager

    backend = FakeBackend()
    harness, journal, search_dir = make_harness(
        task, config, backend, budget=BudgetManager(3600, stop_margin_s=1, spent_s=3600)
    )
    assert not harness.open and harness.closed_reason == "out of budget"
    for _ in range(5):  # never a ClimberError either: it is not the climber's fault
        outcome = harness.run(Action(operator="draft"))
        assert outcome.kind == "rejected" and outcome.ticket.rejected == "out of budget"
    assert harness.submit(Action(operator="draft")).rejected == "out of budget"
    assert not backend.requests and not journal.candidates
    assert not audit_lines(search_dir, "action_rejected")


# --- inject, require_change, Outcome.result (what a self-driving loop needs) ---

from hillclimb import operators  # noqa: E402
from hillclimb.candidate import source_hash  # noqa: E402
from hillclimb.policy import INJECT_ACTION  # noqa: E402
from hillclimb.sdk import Operator, Preparation  # noqa: E402


def test_inject_scores_a_text_the_loop_already_has(task, config):
    backend = FakeBackend()
    harness, journal, search_dir = make_harness(task, config, backend)

    marker = "# only-in-the-source-text\n"
    first = harness.run(Action(operator=INJECT_ACTION, args={"source": ok_script(0.4)}))
    child = harness.run(
        Action(
            operator=INJECT_ACTION, target_id=first.candidate.candidate_id,
            args={"source": ok_script(0.6) + marker}, policy_meta={"optimizer": "mine"},
        )
    )

    assert not backend.requests  # no agent was ever called
    assert (first.kind, first.candidate.val_score) == ("evaluated", 0.4)
    assert child.candidate.parent_id == first.candidate.candidate_id
    assert child.candidate.operator == "inject" and child.candidate.role == "inject"
    assert child.candidate.policy_meta == {"optimizer": "mine"}
    assert child.candidate.solution_sha256 == source_hash(ok_script(0.6) + marker)
    assert harness.source(child.candidate.candidate_id) == ok_script(0.6) + marker
    # the scored view a loop consumes — and the source text is never journaled
    assert child.result.valid and child.result.score == 0.6
    assert "only-in-the-source-text" not in (search_dir / "journal.jsonl").read_text()
    # spend: injected evaluations are the climber's, the agent cost is nil
    assert harness.spend().evaluations == 2 and harness.spend().tokens == 0


def test_inject_refuses_nonsense_before_creating_anything(task, config):
    harness, journal, _ = make_harness(task, config, FakeBackend())
    assert harness.run(Action(operator=INJECT_ACTION)).kind == "rejected"
    outcome = harness.run(Action(operator=INJECT_ACTION, target_id="c999", args={"source": "x=1\n"}))
    assert outcome.kind == "rejected" and not journal.candidates
    assert outcome.ticket.rejected == "references c999 which is not in the journal"


class Mutate(Operator):
    """A refine-role operator that insists on a real change and hands the
    agent a feedback file written from text."""

    name, role, needs_target = "mutate", "refine", True

    def prepare(self, ctx):
        return Preparation(
            prompt="Change solution.py.", copy_parent=True, require_change=True,
            texts={"feedback.json": ctx.action.args["feedback"]},
        )


def test_require_change_turns_an_untouched_parent_into_unchanged(task, config):
    operators.register_operator(Mutate)
    try:
        backend = FakeBackend()
        backend.queue(script=None, notes="looked, changed nothing\n")  # leaves the parent copy as it is
        backend.queue(script=ok_script(0.9), notes="a real change\n")
        harness, journal, _ = make_harness(task, config, backend)
        parent = harness.run(Action(operator=INJECT_ACTION, args={"source": ok_script(0.5)})).candidate
        action = Action(operator="mutate", target_id=parent.candidate_id, args={"feedback": '{"weak": "x"}'})

        lazy = harness.run(action)
        real = harness.run(action)

        assert lazy.kind == "unchanged" and lazy.candidate.status == "abandoned" and lazy.result is None
        assert lazy.candidate.summary == "agent returned the parent source unchanged"
        assert real.kind == "evaluated" and real.result.score == 0.9
        feedback = journal.get(real.candidate.candidate_id).candidate_dir + "/feedback.json"
        assert open(feedback).read() == '{"weak": "x"}'
        # an unchanged attempt was never scored: it costs no evaluation
        assert harness.spend().evaluations == 2
    finally:
        operators._OPERATORS.pop("mutate", None)


def test_outcome_result_never_carries_holdout(task_larger, config):
    """The scored view is projected from the holdout-blind copy."""
    harness, journal, _ = make_harness(task_larger, config, FakeBackend())
    outcome = harness.run(Action(operator=INJECT_ACTION, args={"source": ok_script(0.5)}))
    dumped = repr(outcome.result) + repr(outcome.candidate)
    assert "holdout_score=None" in dumped or "holdout" not in repr(outcome.result)
    assert outcome.candidate.holdout_score is None


def test_search_info_and_budget_view_read_like_the_prompts_do(task, config):
    harness, _journal, _ = make_harness(task, config, FakeBackend())
    assert harness.info.baseline_source is None  # this problem ships no baseline solution
    assert harness.view().budget.remaining_str() in ("1h 00m", "59 minutes")


def test_a_stop_from_outside_the_loop_closes_the_harness_even_if_swallowed(task, config):
    """SIGTERM lands at an arbitrary point. `request_stop` latches first, so a
    loop (or a library under it, like gepa's proposer) that swallows the
    StopRequested raised right after still cannot start anything."""
    backend = FakeBackend()
    for score in (0.5, 0.6, 0.7):
        backend.queue(script=ok_script(score), notes="d\n")
    harness, journal, _ = make_harness(task, config, backend)
    original = harness._execute_job

    def sigterm_during_first_attempt(job):
        harness.request_stop("signal SIGTERM")
        return original(job)

    harness._execute_job = sigterm_during_first_attempt
    loop = Stubborn()
    with pytest.raises(StopRequested, match="SIGTERM"):
        harness.execute(loop)
    assert harness.abort.is_set() and len(backend.requests) == 1 and loop.closed_errors == 9
