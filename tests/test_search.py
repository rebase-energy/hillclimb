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


def holdout_script(val: float, holdout_pred_flip: bool = False) -> str:
    """Fixture script that writes all three artifacts. holdout_pred_flip
    controls holdout accuracy: False → perfect, True → all-wrong."""
    return f'''
import pandas as pd
sample = pd.read_csv("data/sample_submission.csv")
sample.to_csv("submission.csv", index=False)
hold = pd.read_csv("data/holdout.csv")
truth = (hold["feature"] > 0)  # matches the fixture's target rule
pred = ~truth if {holdout_pred_flip} else truth
pd.DataFrame({{"id": hold["id"], "target": pred.astype(int)}}).to_csv(
    "holdout_predictions.csv", index=False)
print("val_score: {val}")
'''


def make_holdout_searcher(task, config, backend, tmp_path, max_nodes=10):
    from hillclimb.holdout import build_data_view

    run_dir = create_run_dir(config.paths.runs_dir, "test-run")
    info = build_data_view(task.data_dir, run_dir, task.sample_submission, 0.3, 42)
    assert info is not None
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        task=task, config=config, journal=journal, backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        run_dir=run_dir, max_nodes=max_nodes, log=lambda *_: None, holdout=info,
    )
    return searcher, journal, run_dir


def test_selection_by_holdout_not_val(task_larger, config):
    """The leaf-classification scenario: highest val_score but bad holdout
    must NOT be selected; selection = argmax holdout."""
    backend = FakeBackend()
    backend.queue(script=holdout_script(0.70), notes="honest draft\n")
    backend.queue(script=holdout_script(0.99, holdout_pred_flip=True), notes="overfit draft\n")
    backend.queue(script=holdout_script(0.80), notes="honest draft 2\n")
    searcher, journal, run_dir = make_holdout_searcher(task_larger, config, backend, None, max_nodes=4)
    selected = searcher.run()

    overfit = journal.get("n002")
    assert overfit.val_score == 0.99 and overfit.holdout_score == 0.0  # flipped preds
    assert overfit.is_best  # it IS the val-best (climbing signal)
    assert not overfit.is_selected
    assert selected.holdout_score == 1.0
    assert selected.node_id in ("n001", "n003")
    # best/ holds the selected node's submission, not the val-best's
    assert (run_dir / "best" / "solution.py").read_text() == (
        Path(selected.workspace) / "solution.py").read_text()


def test_missing_holdout_predictions_is_buggy(task_larger, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.9), notes="wrote no holdout preds\n")
    backend.queue(script=holdout_script(0.6), notes="compliant\n")
    searcher, journal, _ = make_holdout_searcher(task_larger, config, backend, None, max_nodes=3)
    searcher.run()
    bad = journal.get("n001")
    assert bad.status == "buggy"
    assert "holdout_predictions.csv" in bad.execution.holdout_error
    # and the debug prompt explains it
    assert any("holdout_predictions.csv" in r.prompt for r in backend.requests if r.operator == "debug")


def test_holdout_prompt_mentions_contract(task_larger, config):
    backend = FakeBackend()
    backend.queue(script=holdout_script(0.7), notes="d\n")
    searcher, _, _ = make_holdout_searcher(task_larger, config, backend, None, max_nodes=2)
    searcher.run()
    prompt = backend.requests[0].prompt
    assert "holdout_predictions.csv" in prompt
    assert "holdout.csv" in prompt


def test_no_holdout_falls_back_to_val_selection(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.9), notes="b\n")
    backend.queue(script=ok_script(0.7), notes="c\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=4)
    selected = searcher.run()
    assert selected.val_score == 0.9
    assert journal.selected_node(False).node_id == selected.node_id
