"""The operator seam: a climber's own operator runs end to end, the harness
keeps the contract and the evaluator out of its reach, and the journal speaks
in roles so nothing else needs to know the operator's name."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb import operators
from hillclimb.backends.fake import FakeBackend
from hillclimb.candidate import Candidate
from hillclimb.journal import Journal
from hillclimb.policy import Action
from hillclimb.sdk import Operator, OperatorContext, Preparation
from tests.conftest import ok_script
from tests.factories import trial as mk_trial
from tests.test_parallel_search import make_searcher

CRASH = 'raise RuntimeError("boom")\n'


class ReflectOperator(Operator):
    """A refine-role operator with a hand-written prompt: no template, no
    contract token, records what it was shown."""

    name = "reflect"
    role = "refine"
    needs_target = True
    seen: list[OperatorContext] = []

    def prepare(self, ctx: OperatorContext) -> Preparation:
        type(self).seen.append(ctx)
        return Preparation(prompt=f"Reflect on {ctx.target.candidate_id} and do better.", copy_parent=True)


@pytest.fixture
def reflect():
    ReflectOperator.seen = []
    operators.register_operator(ReflectOperator)
    yield ReflectOperator
    operators._OPERATORS.pop("reflect", None)


def test_custom_operator_runs_end_to_end_with_the_contract_appended(task, config, reflect):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="draft\n")
    backend.queue(script=ok_script(0.7), notes="reflected\n")
    searcher, journal, _ = make_searcher(task, config, backend)
    parent = searcher.run_operator("draft", None)

    job = searcher._prepare(Action(operator="reflect", target_id=parent.candidate_id))
    child = searcher._commit(searcher._execute_job(job))

    assert child.operator == "reflect" and child.role == "refine" and child.val_score == 0.7
    assert backend.requests[-1].operator == "reflect" and backend.requests[-1].role == "refine"
    prompt = Path(child.candidate_dir, "prompt.md").read_text()
    assert prompt.startswith(f"Reflect on {parent.candidate_id} and do better.\n\n")
    # the operator never mentioned the contract; the harness appended it
    assert "val_score" in prompt and "{{contract}}" not in prompt
    assert Path(child.candidate_dir, "solution.py").read_text() == ok_script(0.7)


def test_refused_action_leaves_no_trace(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="draft\n")
    searcher, journal, search_dir = make_searcher(task, config, backend)
    passing = searcher.run_operator("draft", None)
    before_ids = set(journal.candidates)
    before_dirs = {p.name for p in (search_dir / "candidates").iterdir()}

    with pytest.raises(ValueError, match=f"debug targets {passing.candidate_id} whose status is passing"):
        searcher._prepare(Action(operator="debug", target_id=passing.candidate_id))
    with pytest.raises(ValueError, match="improve without a target_id"):
        searcher._prepare(Action(operator="improve"))
    with pytest.raises(ValueError, match="ensemble needs inspiration_ids"):
        searcher._prepare(Action(operator="ensemble", target_id=passing.candidate_id))
    with pytest.raises(ValueError, match="Unknown operator 'crossover'"):
        searcher._prepare(Action(operator="crossover"))

    assert set(journal.candidates) == before_ids
    assert {p.name for p in (search_dir / "candidates").iterdir()} == before_dirs
    assert len(backend.requests) == 1  # nothing was spent


def test_operator_context_is_holdout_blind(task, config, reflect, tmp_path):
    searcher, journal, _ = make_searcher(task, config, FakeBackend())
    ws = tmp_path / "c001"
    ws.mkdir()
    (ws / "solution.py").write_text(ok_script(0.5))
    journal.candidate_result(
        Candidate(
            candidate_id="c001", operator="draft", status="passing", candidate_dir=str(ws),
            is_selected=True,
            trials=[mk_trial(val_score=0.5, submission_ok=True, holdout_score=0.123)],
        )
    )
    searcher._prepare(Action(operator="reflect", target_id="c001", inspiration_ids=("c001",)))

    ctx = reflect.seen[-1]
    for candidate in (ctx.target, *ctx.inspirations, *ctx.journal.candidates.values()):
        assert candidate.holdout_score is None and not candidate.is_selected
    assert journal.get("c001").holdout_score == 0.123


def test_role_is_journaled_and_backfilled_for_older_records(task, config, tmp_path):
    backend = FakeBackend()
    backend.queue(script=CRASH, notes="buggy\n")
    backend.queue(script=ok_script(0.6), notes="fixed\n")
    searcher, journal, search_dir = make_searcher(task, config, backend)
    broken = searcher.run_operator("draft", None)
    fixed = searcher.run_operator("debug", broken)
    assert (broken.role, fixed.role, fixed.debug_depth) == ("create", "repair", 1)
    chain = [broken.candidate_id, fixed.candidate_id]
    assert [c.candidate_id for c in journal.debug_chain(fixed.candidate_id)] == chain
    assert [c.candidate_id for c in journal.drafts()] == [broken.candidate_id]

    # a record written before roles existed gets its operator's role on load
    old = Candidate.model_validate({"candidate_id": "c9", "operator": "ensemble"})
    assert old.role == "combine"
    assert Candidate.model_validate({"candidate_id": "c0", "operator": "baseline"}).role == "baseline"
    # an operator this process does not know stays role-less instead of failing to load
    assert Candidate.model_validate({"candidate_id": "c8", "operator": "crossover"}).role is None
    replayed = Journal(search_dir / "journal.jsonl")
    assert replayed.get(fixed.candidate_id).role == "repair"


def test_operator_registry_rejects_an_unknown_role():
    class Odd(Operator):
        name, role = "odd", "measure"

        def prepare(self, ctx):  # pragma: no cover
            return Preparation(prompt="")

    with pytest.raises(ValueError, match="role 'measure'"):
        operators.register_operator(Odd)
    assert "odd" not in operators.operator_names()


def test_extra_files_must_be_bare_names(task, config):
    class Escaper(Operator):
        name, role = "escaper", "create"

        def prepare(self, ctx):
            return Preparation(prompt="x", files={"../evil.py": Path(__file__)})

    operators.register_operator(Escaper)
    try:
        searcher, _journal, _dir = make_searcher(task, config, FakeBackend())
        with pytest.raises(ValueError, match="must be a bare file name"):
            searcher._prepare(Action(operator="escaper"))
    finally:
        operators._OPERATORS.pop("escaper", None)
