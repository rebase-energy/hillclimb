from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher, ParkedRun
from hillclimb.workspace import create_run_dir
from tests.conftest import CRASH_SCRIPT, ok_script


def make_searcher(task, config, backend, max_nodes=10, budget_s=3600):
    run_dir = create_run_dir(config.paths.runs_dir, "test-run")
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        task=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(budget_s, stop_margin_s=1),
        run_dir=run_dir,
        max_nodes=max_nodes,
        log=lambda *_: None,
    )
    return searcher, journal, run_dir


def test_happy_path_draft_then_improve(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft one\n")
    backend.queue(script=ok_script(0.7), notes="draft two\n")
    backend.queue(script=ok_script(0.5), notes="draft three\n")
    backend.queue(script=ok_script(0.8), notes="improve best\n")
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=5)

    best = searcher.run()

    assert best is not None
    assert best.val_score == 0.8
    operators = [r.operator for r in backend.requests]
    assert operators == ["draft", "draft", "draft", "improve"]
    # improve targets the best draft (n002, score 0.7)
    improve_node = journal.get(best.node_id)
    assert improve_node.parent_id == "n002"
    assert (run_dir / "best" / "submission.csv").exists()
    # baseline was written first
    assert journal.get("n000").operator == "baseline"


def test_complexity_schedule(task, config):
    backend = FakeBackend()
    for score in (0.1, 0.2, 0.3):
        backend.queue(script=ok_script(score), notes="d\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=4)
    searcher.run()
    drafts = journal.drafts()
    assert [d.complexity for d in drafts] == ["minimal", "moderate", "advanced"]
    assert "MINIMAL" in backend.requests[0].prompt
    assert "MODERATE" in backend.requests[1].prompt
    assert "ADVANCED" in backend.requests[2].prompt


def test_debug_path(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH_SCRIPT, notes="buggy draft\n")
    backend.queue(script=ok_script(0.6), notes="fixed\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=3)
    searcher.run()

    assert backend.requests[1].operator == "debug"
    assert "boom" in backend.requests[1].prompt  # stderr tail injected
    debug_node = journal.get("n002")
    assert debug_node.parent_id == "n001"
    assert debug_node.status == "ok"


def test_debug_depth_cap_then_redraft(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH_SCRIPT, notes="buggy draft\n")
    for i in range(3):
        backend.queue(script=CRASH_SCRIPT, notes=f"failed fix {i}\n")
    backend.queue(script=ok_script(0.6), notes="fresh draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=6)
    searcher.run()

    operators = [r.operator for r in backend.requests]
    assert operators == ["draft", "debug", "debug", "debug", "draft"]


def test_rate_limit_parks_run(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft\n")
    backend.queue(script=None, result={"ok": False, "error_kind": "rate_limited",
                                       "error_message": "usage limit reached"})
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=5)

    with pytest.raises(ParkedRun):
        searcher.run()
    parked = [n for n in journal.nodes.values() if n.status == "parked"]
    assert len(parked) == 1

    # resume: fresh searcher over the same journal continues, ignoring the parked node
    backend2 = FakeBackend()
    backend2.queue(script=ok_script(0.7), notes="draft after resume\n")
    backend2.queue(script=ok_script(0.9), notes="another\n")
    run_dir2_journal = Journal(run_dir / "journal.jsonl")
    searcher2 = GreedySearcher(
        task=task, config=config, journal=run_dir2_journal, backend=backend2,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        run_dir=run_dir, max_nodes=len(run_dir2_journal.nodes) + 2, log=lambda *_: None,
    )
    best = searcher2.run()
    assert best is not None


def test_contract_violation_no_solution(task, config):
    backend = FakeBackend()
    backend.queue(script=None, notes="")  # agent "succeeds" but writes nothing
    backend.queue(script=ok_script(0.5), notes="ok draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=3)
    searcher.run()
    assert journal.get("n001").status == "abandoned"
    assert journal.get("n002").status == "ok"


def test_prompt_contents(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="tfidf baseline\n")
    searcher, _, _ = make_searcher(task, config, backend, max_nodes=2)
    searcher.run()
    prompt = backend.requests[0].prompt
    assert "Predict target from feature." in prompt
    assert "val_score: <float>" in prompt
    assert "accuracy" in prompt
    assert "sample_submission.csv" in prompt
    assert "{{" not in prompt  # all tokens rendered


def test_agent_failure_abandons_and_parks_after_three(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft\n")
    for _ in range(3):
        backend.queue(script=None, result={"ok": False, "error_kind": "error",
                                           "error_message": "ConnectionRefused"})
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=10)
    with pytest.raises(ParkedRun):
        searcher.run()
    abandoned = [n for n in journal.nodes.values() if n.status == "abandoned"]
    assert len(abandoned) == 3
    # crucially: the failed improves were never executed/scored
    assert all(n.val_score is None for n in abandoned)


def test_agent_failure_counter_resets_on_success(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft\n")
    backend.queue(script=None, result={"ok": False, "error_kind": "error", "error_message": "x"})
    backend.queue(script=ok_script(0.7), notes="draft two\n")
    backend.queue(script=None, result={"ok": False, "error_kind": "error", "error_message": "x"})
    backend.queue(script=ok_script(0.8), notes="draft three\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=6)
    best = searcher.run()  # must NOT raise ParkedRun
    assert best.val_score == 0.8
