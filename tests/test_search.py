from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.control import ControlCommand, write_command
from hillclimb.executor import LocalExecutor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher, ParkedRun, StopRequested
from hillclimb.workspace import create_run_dir
from tests.conftest import CRASH_SCRIPT, ok_script


def make_searcher(task, config, backend, max_nodes=10, budget_s=3600):
    run_dir = create_run_dir(config.paths.runs_dir, "test-run")
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task,
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
        problem=task, config=config, journal=run_dir2_journal, backend=backend2,
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


def test_stop_command_stops_before_any_operator(task, config):
    backend = FakeBackend()  # empty queue: any invoke would raise
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=5)
    write_command(run_dir, ControlCommand(action="stop", source="cli"))

    with pytest.raises(StopRequested):
        searcher.run()
    assert backend.requests == []
    assert journal.get("n000").operator == "baseline"  # baseline still written
    assert '"event": "control"' in (run_dir / "journal.jsonl").read_text()


def test_stop_is_graceful_current_operator_finishes(task, config):
    class StopDroppingBackend(FakeBackend):
        def __init__(self, run_dir):
            super().__init__()
            self.run_dir = run_dir

        def invoke(self, request):
            write_command(self.run_dir, ControlCommand(action="stop", source="tui"))
            return super().invoke(request)

    run_dir_probe = create_run_dir(config.paths.runs_dir, "graceful-run")
    backend = StopDroppingBackend(run_dir_probe)
    backend.queue(script=ok_script(0.6), notes="draft\n")
    journal = Journal(run_dir_probe / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        run_dir=run_dir_probe, max_nodes=5, log=lambda *_: None,
    )
    with pytest.raises(StopRequested):
        searcher.run()
    # the operator that was in flight when stop arrived completed and scored
    assert journal.get("n001").status == "ok"
    assert journal.get("n001").val_score == 0.6


def test_prune_buggy_tip_redirects_to_draft(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH_SCRIPT, notes="buggy draft\n")
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=5)
    searcher.run_operator("draft", None)  # first node: n000 (no baseline written here)
    assert searcher.decide()[0] == "debug"  # would keep debugging

    write_command(run_dir, ControlCommand(action="prune", node_id="n000", source="cli"))
    searcher._process_control()

    assert journal.get("n000").pruned
    assert searcher.decide() == ("draft", None)  # buggy tip no longer targeted


def test_prune_scored_branch_makes_engine_redraft(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.7), notes="b\n")
    backend.queue(script=ok_script(0.5), notes="c\n")
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=10)
    for _ in range(3):
        searcher.run_operator("draft", None)  # n000..n002, best/selected = n001 (0.7)
    assert searcher.decide()[0] == "improve"  # 3 scored branches → improve best

    write_command(run_dir, ControlCommand(action="prune", node_id="n001", source="cli"))
    searcher._process_control()

    assert searcher.decide() == ("draft", None)  # only 2 scored branches remain
    # selection repointed away from the pruned branch
    assert journal.selected_node(False).node_id == "n000"
    assert (run_dir / "best" / "submission.csv").exists()


def test_prune_unknown_node_is_rejected_not_fatal(task, config):
    backend = FakeBackend()
    searcher, journal, run_dir = make_searcher(task, config, backend, max_nodes=5)
    write_command(run_dir, ControlCommand(action="prune", node_id="n999", source="cli"))
    searcher._process_control()  # must not raise
    assert all(not n.pruned for n in journal.nodes.values())


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
        problem=task, config=config, journal=journal, backend=backend,
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


def test_time_tail_prompt_names_id_col_and_warns_chronological(tmp_path, config):
    import pandas as pd

    from hillclimb.holdout import HoldoutOverride, build_data_view
    from hillclimb.problem import ProblemSpec

    d = tmp_path / "ts-public"
    d.mkdir()
    ts = pd.date_range("2025-05-01", periods=100, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({"timestamp_utc": ts, "net_load_kwh": range(100)}).to_csv(
        d / "train.csv", index=False
    )
    pd.DataFrame({"row_id": [0], "net_load_kwh": [0.0]}).to_csv(
        d / "sample_submission.csv", index=False
    )
    (d / "description.md").write_text("forecast")
    override = HoldoutOverride(
        strategy="time-tail", time_col="timestamp_utc", id_col="timestamp_utc",
        target_cols=["net_load_kwh"], fraction=0.1,
    )
    ts_task = ProblemSpec(
        problem_id="ts", problem_dir=d, data_dir=d, description="forecast",
        metric_name="nrmse", lower_is_better=True,
        sample_submission=d / "sample_submission.csv", time_budget_s=3600,
        holdout=override,
    )
    run_dir = create_run_dir(config.paths.runs_dir, "ts-run")
    info = build_data_view(d, run_dir, ts_task.sample_submission, 0.1, 42, override=override)
    searcher = GreedySearcher(
        problem=ts_task, config=config, journal=Journal(run_dir / "journal.jsonl"),
        backend=FakeBackend(), executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        run_dir=run_dir, log=lambda *_: None, holdout=info,
    )
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "`timestamp_utc`" in prompt
    assert "chronological TAIL" in prompt
    assert "do not train on them" in prompt
    assert info.time_cutoff in prompt
    assert "{{" not in prompt


def test_no_holdout_falls_back_to_val_selection(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.9), notes="b\n")
    backend.queue(script=ok_script(0.7), notes="c\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_nodes=4)
    selected = searcher.run()
    assert selected.val_score == 0.9
    assert journal.selected_node(False).node_id == selected.node_id


def make_ensemble_searcher(task, config, backend, spent_frac=0.0, max_nodes=12):
    """Searcher with controllable budget position (spent_frac of total)."""
    run_dir = create_run_dir(config.paths.runs_dir, "test-run")
    journal = Journal(run_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(1000, stop_margin_s=1, spent_s=1000 * spent_frac),
        run_dir=run_dir, max_nodes=max_nodes, log=lambda *_: None,
    )
    return searcher, journal, run_dir


def distinct_script(val: float) -> str:
    return ok_script(val) + f"# variant {val}\n"


def test_ensemble_triggers_in_reserve_window(task, config):
    backend = FakeBackend()
    backend.queue(script=distinct_script(0.6), notes="draft a\n")
    backend.queue(script=distinct_script(0.7), notes="draft b\n")
    backend.queue(script=distinct_script(0.5), notes="draft c\n")
    backend.queue(script=distinct_script(0.9), notes="blended\n")
    # 85% spent -> inside the 20% reserve window from the start
    searcher, journal, run_dir = make_ensemble_searcher(task, config, backend, spent_frac=0.85)
    # not yet: fewer than 2 scored candidates
    assert searcher.decide() == ("draft", None)
    searcher.run_operator("draft", None)
    assert searcher.decide()[0] == "draft"  # still 1 candidate short? no: 1 scored
    searcher.run_operator("draft", None)
    op, tgt = searcher.decide()
    assert op == "ensemble"
    assert tgt.node_id == "n001"  # top candidate (val 0.7) is the lineage parent
    node = searcher.run_operator(op, tgt)
    assert node.operator == "ensemble"
    assert node.status == "ok"
    # candidates were seeded into the workspace
    ws = Path(node.workspace)
    assert (ws / "candidate_1.py").exists() and (ws / "candidate_2.py").exists()
    # prompt contains the table and the instruction
    prompt = backend.requests[-1].prompt
    assert "candidate_1.py" in prompt and "at least two" in prompt
    # no further ensemble once one succeeded
    assert searcher._should_ensemble() is False


def test_ensemble_not_triggered_outside_window_or_disabled(task, config):
    backend = FakeBackend()
    backend.queue(script=distinct_script(0.6), notes="a\n")
    backend.queue(script=distinct_script(0.7), notes="b\n")
    searcher, _, _ = make_ensemble_searcher(task, config, backend, spent_frac=0.0)
    searcher.run_operator("draft", None)
    searcher.run_operator("draft", None)
    assert searcher._should_ensemble() is False  # plenty of budget left
    config.ensemble.enabled = False
    searcher2, _, _ = make_ensemble_searcher(task, config, backend, spent_frac=0.9)
    assert searcher2._should_ensemble() is False


def test_ensemble_candidates_dedupe_identical_scripts(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.6), notes="identical twin\n")
    searcher, _, _ = make_ensemble_searcher(task, config, backend, spent_frac=0.85)
    searcher.run_operator("draft", None)
    searcher.run_operator("draft", None)
    assert len(searcher._ensemble_candidates()) == 1  # deduped
    assert searcher._should_ensemble() is False       # so no ensemble


def test_buggy_ensemble_gets_debugged_and_counts_as_success(task, config):
    backend = FakeBackend()
    backend.queue(script=distinct_script(0.6), notes="a\n")
    backend.queue(script=distinct_script(0.7), notes="b\n")
    backend.queue(script=CRASH_SCRIPT, notes="broken blend\n")
    backend.queue(script=distinct_script(0.9), notes="fixed blend\n")
    searcher, journal, _ = make_ensemble_searcher(task, config, backend, spent_frac=0.85)
    searcher.run_operator("draft", None)
    searcher.run_operator("draft", None)
    op, tgt = searcher.decide()
    assert op == "ensemble"
    bad = searcher.run_operator(op, tgt)
    assert bad.status == "buggy"
    op2, tgt2 = searcher.decide()
    assert op2 == "debug" and tgt2.node_id == bad.node_id
    fixed = searcher.run_operator(op2, tgt2)
    assert fixed.status == "ok"
    assert searcher._ensemble_succeeded() is True  # via the debug chain root


def test_holdout_clause_uses_submission_columns_for_class_targets(tmp_path, config):
    """spooky-style task: target col `author` is not a submission column, so
    the clause must direct predictions to submission format, not `author`."""
    from hillclimb.problem import ProblemSpec

    d = tmp_path / "pub"
    d.mkdir()
    rows = "".join(f"{i},x,{a}\n" for i, a in enumerate(["EAP", "HPL", "MWS"] * 10))
    (d / "train.csv").write_text("id,text,author\n" + rows)
    (d / "sample_submission.csv").write_text("id,EAP,HPL,MWS\n9,0.3,0.3,0.4\n")
    (d / "description.md").write_text("d")
    spec = ProblemSpec(
        problem_id="t", problem_dir=d, data_dir=d, description="d",
        metric_name="multi-class-log-loss", lower_is_better=True,
        sample_submission=d / "sample_submission.csv", time_budget_s=600,
    )
    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="d\n")
    run_dir = create_run_dir(config.paths.runs_dir, "clause-test")
    from hillclimb.holdout import build_data_view

    info = build_data_view(d, run_dir, spec.sample_submission, 0.2, 42)
    searcher = GreedySearcher(
        problem=spec, config=config, journal=Journal(run_dir / "journal.jsonl"),
        backend=backend, executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(600, stop_margin_s=1), run_dir=run_dir,
        max_nodes=2, log=lambda *_: None, holdout=info,
    )
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "same prediction columns as submission.csv" in prompt
    assert "`EAP`" in prompt
    assert "prediction column(s): `author`" not in prompt


def test_stale_pending_node_recovered_on_resume(task, config):
    """If the orchestrator dies mid-operator, the pending node must be
    abandoned at next construction, not block the tree forever."""
    from hillclimb.node import Node

    run_dir = create_run_dir(config.paths.runs_dir, "crash-test")
    journal = Journal(run_dir / "journal.jsonl")
    journal.node_created(Node(node_id="n001", operator="draft", workspace=str(run_dir)))
    assert journal.get("n001").status == "pending"

    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="post-crash draft\n")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=LocalExecutor(Path(sys.executable)),
        budget=BudgetManager(3600, stop_margin_s=1),
        run_dir=run_dir, max_nodes=2, log=lambda *_: None,
    )
    assert journal.get("n001").status == "abandoned"
    best = searcher.run()  # loop proceeds normally
    assert best is not None
