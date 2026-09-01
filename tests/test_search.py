from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.backends.fake import FakeBackend
from hillclimb.budget import BudgetManager
from hillclimb.control import ControlCommand, write_command
from tests.conftest import executor_for, local_executor
from hillclimb.journal import Journal
from hillclimb.search import GreedySearcher, ParkedSearch, StopRequested
from hillclimb.dirs import create_search_dir
from tests.conftest import CRASH_SCRIPT, ok_script


def make_searcher(task, config, backend, max_candidates=10, budget_s=3600):
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=executor_for(task),
        budget=BudgetManager(budget_s, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=max_candidates,
        log=lambda *_: None,
    )
    return searcher, journal, search_dir


def test_happy_path_draft_then_improve(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft one\n")
    backend.queue(script=ok_script(0.7), notes="draft two\n")
    backend.queue(script=ok_script(0.5), notes="draft three\n")
    backend.queue(script=ok_script(0.8), notes="improve best\n")
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=5)

    best = searcher.run()

    assert best is not None
    assert best.val_score == 0.8
    operators = [r.operator for r in backend.requests]
    assert operators == ["draft", "draft", "draft", "improve"]
    # improve targets the best draft (c002, score 0.7)
    improve_node = journal.get(best.candidate_id)
    assert improve_node.parent_id == "c002"
    assert (search_dir / "best" / "submission.csv").exists()
    # baseline was written first
    assert journal.get("c000").operator == "baseline"


def test_complexity_schedule(task, config):
    backend = FakeBackend()
    for score in (0.1, 0.2, 0.3):
        backend.queue(script=ok_script(score), notes="d\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=4)
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
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()

    assert backend.requests[1].operator == "debug"
    assert "boom" in backend.requests[1].prompt  # stderr tail injected
    debug_node = journal.get("c002")
    assert debug_node.parent_id == "c001"
    assert debug_node.status == "ok"


def test_debug_depth_cap_then_redraft(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH_SCRIPT, notes="buggy draft\n")
    for i in range(3):
        backend.queue(script=CRASH_SCRIPT, notes=f"failed fix {i}\n")
    backend.queue(script=ok_script(0.6), notes="fresh draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=6)
    searcher.run()

    operators = [r.operator for r in backend.requests]
    assert operators == ["draft", "debug", "debug", "debug", "draft"]


def test_rate_limit_parks_run(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="draft\n")
    backend.queue(script=None, result={"ok": False, "error_kind": "rate_limited",
                                       "error_message": "usage limit reached"})
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=5)

    with pytest.raises(ParkedSearch):
        searcher.run()
    parked = [n for n in journal.candidates.values() if n.status == "parked"]
    assert len(parked) == 1

    # resume: fresh searcher over the same journal continues, ignoring the parked node
    backend2 = FakeBackend()
    backend2.queue(script=ok_script(0.7), notes="draft after resume\n")
    backend2.queue(script=ok_script(0.9), notes="another\n")
    search_dir2_journal = Journal(search_dir / "journal.jsonl")
    searcher2 = GreedySearcher(
        problem=task, config=config, journal=search_dir2_journal, backend=backend2,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=len(search_dir2_journal.candidates) + 2, log=lambda *_: None,
    )
    best = searcher2.run()
    assert best is not None


def test_contract_violation_no_solution(task, config):
    backend = FakeBackend()
    backend.queue(script=None, notes="")  # agent "succeeds" but writes nothing
    backend.queue(script=ok_script(0.5), notes="ok draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()
    assert journal.get("c001").status == "abandoned"
    assert journal.get("c002").status == "ok"


def test_prompt_contents(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="tfidf baseline\n")
    searcher, _, _ = make_searcher(task, config, backend, max_candidates=2)
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
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=10)
    with pytest.raises(ParkedSearch):
        searcher.run()
    abandoned = [n for n in journal.candidates.values() if n.status == "abandoned"]
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
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=6)
    best = searcher.run()  # must NOT raise ParkedSearch
    assert best.val_score == 0.8


def test_stop_command_stops_before_any_operator(task, config):
    backend = FakeBackend()  # empty queue: any invoke would raise
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=5)
    write_command(search_dir, ControlCommand(action="stop", source="cli"))

    with pytest.raises(StopRequested):
        searcher.run()
    assert backend.requests == []
    assert journal.get("c000").operator == "baseline"  # baseline still written
    assert '"event": "control"' in (search_dir / "journal.jsonl").read_text()


def test_stop_is_graceful_current_operator_finishes(task, config):
    class StopDroppingBackend(FakeBackend):
        def __init__(self, search_dir):
            super().__init__()
            self.search_dir = search_dir

        def invoke(self, request):
            write_command(self.search_dir, ControlCommand(action="stop", source="tui"))
            return super().invoke(request)

    search_dir_probe = create_search_dir(config.paths.runs_dir, "graceful-run")
    backend = StopDroppingBackend(search_dir_probe)
    backend.queue(script=ok_script(0.6), notes="draft\n")
    journal = Journal(search_dir_probe / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir_probe, max_candidates=5, log=lambda *_: None,
    )
    with pytest.raises(StopRequested):
        searcher.run()
    # the operator that was in flight when stop arrived completed and scored
    assert journal.get("c001").status == "ok"
    assert journal.get("c001").val_score == 0.6


def test_prune_buggy_tip_redirects_to_draft(task, config):
    backend = FakeBackend()
    backend.queue(script=CRASH_SCRIPT, notes="buggy draft\n")
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=5)
    searcher.run_operator("draft", None)  # first node: c000 (no baseline written here)
    assert searcher.decide()[0] == "debug"  # would keep debugging

    write_command(search_dir, ControlCommand(action="prune", candidate_id="c000", source="cli"))
    searcher._process_control()

    assert journal.get("c000").pruned
    assert searcher.decide() == ("draft", None)  # buggy tip no longer targeted


def test_prune_scored_branch_makes_engine_redraft(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.7), notes="b\n")
    backend.queue(script=ok_script(0.5), notes="c\n")
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=10)
    for _ in range(3):
        searcher.run_operator("draft", None)  # c000..c002, best/selected = c001 (0.7)
    assert searcher.decide()[0] == "improve"  # 3 scored branches → improve best

    write_command(search_dir, ControlCommand(action="prune", candidate_id="c001", source="cli"))
    searcher._process_control()

    assert searcher.decide() == ("draft", None)  # only 2 scored branches remain
    # selection repointed away from the pruned branch
    assert journal.selected_candidate(True).candidate_id == "c000"
    assert (search_dir / "best" / "submission.csv").exists()


def test_prune_unknown_node_is_rejected_not_fatal(task, config):
    backend = FakeBackend()
    searcher, journal, search_dir = make_searcher(task, config, backend, max_candidates=5)
    write_command(search_dir, ControlCommand(action="prune", candidate_id="c999", source="cli"))
    searcher._process_control()  # must not raise
    assert all(not n.pruned for n in journal.candidates.values())


def holdout_script(val: float, holdout: float = 1.0) -> str:
    """Fixture script writing a submission, a val_score, and the artifact the
    stand-in holdout scorer below reads."""
    return f'''
import pandas as pd
sample = pd.read_csv("data/sample_submission.csv")
sample.to_csv("submission.csv", index=False)
open("holdout_predictions.csv", "w").write("{holdout}")
print("val_score: {val}")
'''


class FileHoldoutScorer:
    """Stand-in for a problem's `verifier.sh --holdout`: scores the candidate
    out of sight and reports (score, error, cpu_s) the same way."""

    def score(self, candidate_dir):
        path = Path(candidate_dir) / "holdout_predictions.csv"
        if not path.exists():
            return None, "`holdout_predictions.csv` was not written", 0.01
        return float(path.read_text()), None, 0.01


def make_holdout_searcher(task, config, backend, tmp_path, max_candidates=10):
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    task = task.model_copy(update={"holdout_cmd": task.verifier_cmd + ["--holdout"]})
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=executor_for(task),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=max_candidates, log=lambda *_: None,
        holdout_scorer=FileHoldoutScorer(),
    )
    return searcher, journal, search_dir


def test_selection_by_holdout_not_val(task, config):
    """The leaf-classification scenario: highest val_score but bad holdout
    must NOT be selected; selection = argmax holdout."""
    backend = FakeBackend()
    backend.queue(script=holdout_script(0.70), notes="honest draft\n")
    backend.queue(script=holdout_script(0.99, holdout=0.0), notes="overfit draft\n")
    backend.queue(script=holdout_script(0.80), notes="honest draft 2\n")
    searcher, journal, search_dir = make_holdout_searcher(task, config, backend, None, max_candidates=4)
    selected = searcher.run()

    overfit = journal.get("c002")
    assert overfit.val_score == 0.99 and overfit.holdout_score == 0.0  # fits val only
    assert overfit.is_best  # it IS the val-best (climbing signal)
    assert not overfit.is_selected
    # the scorer's cpu lands on the trial holdout ran against
    assert overfit.last_trial.holdout_cpu_s == 0.01
    assert selected.holdout_score == 1.0
    assert selected.candidate_id in ("c001", "c003")
    # best/ holds the selected node's submission, not the val-best's
    assert (search_dir / "best" / "solution.py").read_text() == (
        Path(selected.candidate_dir) / "solution.py").read_text()


def test_failed_holdout_evaluation_is_buggy(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.9), notes="wrote no holdout preds\n")
    backend.queue(script=holdout_script(0.6), notes="compliant\n")
    searcher, journal, _ = make_holdout_searcher(task, config, backend, None, max_candidates=3)
    searcher.run()
    bad = journal.get("c001")
    assert bad.status == "buggy"
    assert "holdout_predictions.csv" in bad.last_trial.holdout_error
    # an errored holdout still burned its cpu — recorded despite the failure
    assert bad.last_trial.holdout_cpu_s == 0.01
    # and the debug prompt explains it
    assert any("holdout_predictions.csv" in r.prompt for r in backend.requests if r.operator == "debug")


def test_holdout_prompt_warns_about_the_hidden_split(task, config):
    backend = FakeBackend()
    backend.queue(script=holdout_script(0.7), notes="d\n")
    searcher, _, _ = make_holdout_searcher(task, config, backend, None, max_candidates=2)
    searcher.run()
    prompt = backend.requests[0].prompt
    assert "Hidden holdout" in prompt
    assert "never see" in prompt
    assert "{{" not in prompt


def test_no_holdout_falls_back_to_val_selection(task, config):
    backend = FakeBackend()
    backend.queue(script=ok_script(0.6), notes="a\n")
    backend.queue(script=ok_script(0.9), notes="b\n")
    backend.queue(script=ok_script(0.7), notes="c\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=4)
    selected = searcher.run()
    assert selected.val_score == 0.9
    assert journal.selected_candidate(True).candidate_id == selected.candidate_id


def make_ensemble_searcher(task, config, backend, spent_frac=0.0, max_candidates=12):
    """Searcher with controllable budget position (spent_frac of total)."""
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=local_executor(),
        budget=BudgetManager(1000, stop_margin_s=1, spent_s=1000 * spent_frac),
        search_dir=search_dir, max_candidates=max_candidates, log=lambda *_: None,
    )
    return searcher, journal, search_dir


def distinct_script(val: float) -> str:
    return ok_script(val) + f"# variant {val}\n"


def test_ensemble_triggers_in_reserve_window(task, config):
    backend = FakeBackend()
    backend.queue(script=distinct_script(0.6), notes="draft a\n")
    backend.queue(script=distinct_script(0.7), notes="draft b\n")
    backend.queue(script=distinct_script(0.5), notes="draft c\n")
    backend.queue(script=distinct_script(0.9), notes="blended\n")
    # 85% spent -> inside the 20% reserve window from the start
    searcher, journal, search_dir = make_ensemble_searcher(task, config, backend, spent_frac=0.85)
    # not yet: fewer than 2 scored candidates
    assert searcher.decide() == ("draft", None)
    searcher.run_operator("draft", None)
    assert searcher.decide()[0] == "draft"  # still 1 candidate short? no: 1 scored
    searcher.run_operator("draft", None)
    op, tgt = searcher.decide()
    assert op == "ensemble"
    assert tgt.candidate_id == "c001"  # top candidate (val 0.7) is the lineage parent
    node = searcher.run_operator(op, tgt)
    assert node.operator == "ensemble"
    assert node.status == "ok"
    # candidates were seeded into the candidate_dir
    ws = Path(node.candidate_dir)
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
    assert op2 == "debug" and tgt2.candidate_id == bad.candidate_id
    fixed = searcher.run_operator(op2, tgt2)
    assert fixed.status == "ok"
    assert searcher._ensemble_succeeded() is True  # via the debug chain root


def test_stale_pending_node_recovered_on_resume(task, config):
    """If the orchestrator dies mid-operator, the pending node must be
    abandoned at next construction, not block the tree forever."""
    from hillclimb.candidate import Candidate

    search_dir = create_search_dir(config.paths.runs_dir, "crash-test")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(Candidate(candidate_id="c001", operator="draft", candidate_dir=str(search_dir)))
    assert journal.get("c001").status == "pending"

    backend = FakeBackend()
    backend.queue(script=ok_script(0.5), notes="post-crash draft\n")
    searcher = GreedySearcher(
        problem=task, config=config, journal=journal, backend=backend,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir, max_candidates=2, log=lambda *_: None,
    )
    assert journal.get("c001").status == "abandoned"
    best = searcher.run()  # loop proceeds normally
    assert best is not None


# --- trial evaluation reports ---


def report_script(score, split="validation", zones=None, body="json-report"):
    """Solution script that mimics an emflow eval: writes eval_result.json
    (with a report block) alongside the usual submission + val_score line."""
    payload = {
        "problem": "p", "split": split, "objective": "accuracy", "score": score,
        "n_origins": 2, "n_scored": 4, "model": "m",
        "report": {
            "version": 1, "split": split, "objective": "accuracy",
            "higher_is_better": True,
            "overall": {"score": score, "n_origins": 2, "n_scored": 4},
            "zones": zones or [
                {"zone": "z1", "score": 0.4, "n_origins": 1, "n_scored": 2},
                {"zone": "z2", "score": 0.8, "n_origins": 1, "n_scored": 2},
            ],
            "report_error": None,
        },
    }
    if body == "malformed":
        write = 'open("eval_result.json", "w").write("{not json")\n'
    else:
        write = f"import json\njson.dump({payload!r}, open('eval_result.json', 'w'))\n"
    return (
        "import shutil\n"
        'shutil.copy("data/sample_submission.csv", "submission.csv")\n'
        + write
        + f'print("val_score: {score}")\n'
    )


def test_trial_report_pickup_and_improve_injection(task, config):
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=report_script(0.6), notes="draft\n")
    backend.queue(script=ok_script(0.7), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()

    draft = journal.get("c001")
    assert draft.trials[0].report is not None
    assert draft.trials[0].report["version"] == 1
    assert draft.trials[0].report["overall"]["score"] == 0.6
    assert backend.requests[1].operator == "improve"
    prompt = backend.requests[1].prompt
    assert "# Evaluation breakdown (validation split)" in prompt
    assert "z2" in prompt  # worst zone surfaced to the operator
    assert "{{evaluation_report}}" not in prompt


def test_holdout_split_report_never_lands_on_trial(task, config):
    """Leakage guard: a holdout-split eval_result.json must not reach the
    journal (and therefore can never reach a prompt)."""
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=report_script(0.6, split="holdout"), notes="draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=2)
    searcher.run()
    draft = journal.get("c001")
    assert draft.status == "ok"
    assert draft.trials[0].report is None


def test_malformed_eval_result_ignored(task, config):
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=report_script(0.6, body="malformed"), notes="draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=2)
    searcher.run()
    draft = journal.get("c001")
    assert draft.status == "ok"
    assert draft.trials[0].report is None


def test_report_injection_gated_by_config(task, config):
    """report.enabled=false stops injection but not recording — both A/B
    arms journal identical data."""
    config.search.num_drafts = 1
    config.report.enabled = False
    backend = FakeBackend()
    backend.queue(script=report_script(0.6), notes="draft\n")
    backend.queue(script=ok_script(0.7), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()
    assert journal.get("c001").trials[0].report is not None  # still recorded
    assert "Evaluation breakdown" not in backend.requests[1].prompt


def test_improve_prompt_carries_delta_vs_parent(task, config):
    """Second-generation improve: the target and its parent both have
    reports, so the prompt shows where the score moved."""
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=report_script(0.6), notes="draft\n")
    backend.queue(
        script=report_script(0.7, zones=[
            {"zone": "z1", "score": 0.6, "n_origins": 1, "n_scored": 2},
            {"zone": "z2", "score": 0.7, "n_origins": 1, "n_scored": 2},
        ]),
        notes="improve one\n",
    )
    backend.queue(script=ok_script(0.8), notes="improve two\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=4)
    searcher.run()

    assert [r.operator for r in backend.requests] == ["draft", "improve", "improve"]
    prompt = backend.requests[2].prompt  # targets c002 (best), parent c001
    assert "Where this solution moved vs its parent (c001)" in prompt
    # accuracy is higher-is-better: z1 0.4->0.6 improved, z2 0.8->0.7 regressed
    assert "improved z1" in prompt
    assert "regressed z2" in prompt


VERIFIER_WITH_REPORT = """\
import json, os
report = {"version": 1, "split": "validation", "objective": "accuracy",
          "higher_is_better": True, "segment_label": "class",
          "overall": {"score": 0.66, "n_origins": 2, "n_scored": 4},
          "zones": [{"zone": "cat", "score": 0.4, "n_scored": 2},
                    {"zone": "dog", "score": 0.9, "n_scored": 2}]}
json.dump({"split": "validation", "score": 0.66, "report": report},
          open(os.environ["HILLCLIMB_RESULT"], "w"))
"""

# a real two-step verifier.sh, in python so the fixture stays portable:
# run the candidate, drop whatever it left behind, then score it
VERIFIER_WRAPPER = """\
import os, subprocess, sys
subprocess.run([sys.executable, os.environ["HILLCLIMB_SOLUTION"]], check=True)
if os.path.exists(os.environ["HILLCLIMB_RESULT"]):
    os.remove(os.environ["HILLCLIMB_RESULT"])   # trust boundary
subprocess.run([sys.executable, "problem/verify.py"], check=True)
"""

AGENT_FAKED_REPORT = (
    "import json, shutil\n"
    'shutil.copy("data/sample_submission.csv", "submission.csv")\n'
    "json.dump({'split': 'validation', 'report': {'version': 1, 'objective': 'accuracy',"
    " 'higher_is_better': True, 'overall': {'score': 0.99, 'n_origins': 1, 'n_scored': 1}}},"
    " open('eval_result.json', 'w'))\n"
    'print("val_score: 0.99")\n'
)


def add_verifier(task, script=VERIFIER_WITH_REPORT):
    """Turn the fixture problem into one whose verifier owns the score."""
    (task.problem_dir / "verify.py").write_text(script)
    (task.problem_dir / "wrapper.py").write_text(VERIFIER_WRAPPER)
    return task.model_copy(update={
        "verifier_cmd": ["{python}", "problem/wrapper.py"],
        "report_trusted": True,
    })


def test_verifier_report_is_trusted_and_overrides_agent_file(task, config):
    """Tier 1: on verifier problems the verifier's eval_result.json is the
    report (evaluator trust) — anything the solution wrote is discarded,
    exactly like its val_score line."""
    task = add_verifier(task)
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=AGENT_FAKED_REPORT, notes="draft\n")
    backend.queue(script=ok_script(0.7), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()

    draft = journal.get("c001")
    assert draft.val_score == 0.66  # verifier's line wins over the agent's
    report = draft.trials[0].report
    assert report["source"] == "evaluator"
    assert report["overall"]["score"] == 0.66  # not the agent's faked 0.99
    prompt = backend.requests[1].prompt
    assert "Per class" in prompt  # segment_label flows through
    assert "cat" in prompt
    assert "Self-reported" not in prompt


def test_verifier_without_report_discards_agent_file(task, config):
    """The trust boundary also holds when the verifier writes no report:
    the agent's file must not survive as a fake evaluator report."""
    task = add_verifier(task, script='import json, os\njson.dump({"score": 0.66}, open(os.environ["HILLCLIMB_RESULT"], "w"))\n')
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=AGENT_FAKED_REPORT, notes="draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=2)
    searcher.run()
    assert journal.get("c001").trials[0].report is None


def test_bare_number_result_has_no_report_and_does_not_crash(task, config):
    """The simplest verifier form (`echo 12.5 > $HILLCLIMB_RESULT`) carries no
    report block — the trial must score, not explode reading one."""
    task = add_verifier(
        task, script='import os\nopen(os.environ["HILLCLIMB_RESULT"], "w").write("12.5")\n'
    )
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=ok_script(0.9), notes="draft\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=2)
    searcher.run()
    draft = journal.get("c001")
    assert draft.status == "ok"
    assert draft.val_score == 12.5
    assert draft.trials[0].report is None


def test_agent_report_labelled_self_reported(task, config):
    """Tier 2: verifier-less problems store the agent's own report, stamped
    source=agent and rendered with the self-reported caveat."""
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script=report_script(0.6), notes="draft\n")
    backend.queue(script=ok_script(0.7), notes="improve\n")
    searcher, journal, _ = make_searcher(task, config, backend, max_candidates=3)
    searcher.run()
    assert journal.get("c001").trials[0].report["source"] == "agent"
    assert "Self-reported" in backend.requests[1].prompt


def test_report_clause_only_where_the_agent_reports_its_own_score(task, config):
    searcher, _, _ = make_searcher(task, config, FakeBackend())
    clause_prompt = searcher.build_prompt("draft", None, "minimal")
    assert "eval_result.json" in clause_prompt  # Tier-2 invitation present
    assert "{{report_clause}}" not in clause_prompt

    verifier_searcher, _, _ = make_searcher(add_verifier(task), config, FakeBackend())
    verifier_prompt = verifier_searcher.build_prompt("draft", None, "minimal")
    assert "eval_result.json" not in verifier_prompt  # the verifier owns it
    assert "{{report_clause}}" not in verifier_prompt


# --- evaluator-kind problems (kind: evaluator) ---

EVALUATOR_EVALUATE = """\
import json, os, sys
sys.path.insert(0, os.getcwd())
import solution

score = float(solution.answer())
if os.environ.get("HILLCLIMB_TRIAL_SEED"):
    score += 0.001 * int(os.environ["HILLCLIMB_TRIAL_SEED"])
report = {"version": 1, "split": "validation", "objective": "score",
          "higher_is_better": True,
          "overall": {"score": score, "n_origins": 2, "n_scored": 2},
          "zones": [{"zone": "easy", "score": score + 0.1, "n_scored": 1},
                    {"zone": "hard", "score": score - 0.1, "n_scored": 1}]}
json.dump({"split": "validation", "score": score, "report": report},
          open("eval_result.json", "w"))
print(f"val_score: {score}")
"""


def make_evaluator_searcher(config, tmp_path, backend, evaluate_py=EVALUATOR_EVALUATE, **searcher_kwargs):
    from hillclimb.executor import CommandExecutor
    from hillclimb.problem import ProblemSpec

    problem_dir = tmp_path / "eval-problem"
    problem_dir.mkdir(exist_ok=True)
    (problem_dir / "evaluate.py").write_text(evaluate_py)
    problem = ProblemSpec(
        problem_id="eval-problem",
        problem_dir=problem_dir,
        data_dir=problem_dir,
        description="Maximize answer().",
        metric_name="score",
        higher_is_better=True,
        time_budget_s=3600,
        verifier_cmd=["{python}", "problem/evaluate.py"],
        verifier_display="./problem/evaluate.py",
        contract="solution.py must define `answer() -> float`.",
    )
    search_dir = create_search_dir(config.paths.runs_dir, "eval-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=backend,
        executor=CommandExecutor(Path(sys.executable), problem.verifier_cmd),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        log=lambda *_: None,
        **searcher_kwargs,
    )
    return searcher, journal, search_dir


def test_evaluator_kind_full_loop(config, tmp_path):
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script="def answer():\n    return 0.6\n", notes="first answer\n")
    backend.queue(script="def answer():\n    return 0.7\n", notes="better answer\n")
    searcher, journal, search_dir = make_evaluator_searcher(
        config, tmp_path, backend, max_candidates=3
    )
    best = searcher.run()

    assert journal.get("c000").operator == "baseline"
    assert "unscored placeholder" in journal.get("c000").summary
    draft = journal.get("c001")
    assert draft.val_score == 0.6
    assert draft.trials[0].report["source"] == "evaluator"  # trusted by kind
    assert best.val_score == 0.7

    draft_prompt = backend.requests[0].prompt
    assert "./problem/evaluate.py" in draft_prompt  # display form, not the argv
    assert "`answer() -> float`" in draft_prompt  # problem-supplied contract
    assert "submission.csv" not in draft_prompt
    assert "{{" not in draft_prompt
    improve_prompt = backend.requests[1].prompt
    assert "# Evaluation breakdown (validation split)" in improve_prompt
    assert "hard" in improve_prompt  # worst zone surfaced
    assert "Self-reported" not in improve_prompt  # evaluator trust


def test_evaluator_missing_result_json_wording(config, tmp_path):
    """An evaluator that scores but writes no eval_result.json is a contract
    violation — the debug prompt must name the missing artifact, not
    submission.csv."""
    config.search.num_drafts = 1
    backend = FakeBackend()
    backend.queue(script="def answer():\n    return 0.5\n", notes="draft\n")
    backend.queue(script="def answer():\n    return 0.5\n", notes="fix attempt\n")
    searcher, journal, _ = make_evaluator_searcher(
        config, tmp_path, backend,
        evaluate_py='import sys, os\nsys.path.insert(0, os.getcwd())\n'
                    'import solution\nprint(f"val_score: {solution.answer()}")\n',
        max_candidates=3,
    )
    searcher.run()
    assert journal.get("c001").status == "buggy"
    assert backend.requests[1].operator == "debug"
    debug_prompt = backend.requests[1].prompt
    assert "the verifier reported no score" in debug_prompt
    assert "submission.csv" not in debug_prompt


def test_evaluator_multi_trial_seeds(config, tmp_path):
    config.search.num_drafts = 1
    config.search.n_trials = 2
    backend = FakeBackend()
    backend.queue(script="def answer():\n    return 0.6\n", notes="draft\n")
    searcher, journal, _ = make_evaluator_searcher(config, tmp_path, backend, max_candidates=2)
    searcher.run()
    draft = journal.get("c001")
    assert [t.seed for t in draft.trials] == [0, 1]
    assert draft.trials[0].val_score == 0.6
    assert draft.trials[1].val_score == 0.601  # HILLCLIMB_TRIAL_SEED reached the evaluator
    assert draft.val_score == pytest.approx(0.6005)  # mean climbs
    assert draft.trials[0].report is not None  # per-trial reports survive trial dirs
