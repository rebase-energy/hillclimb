"""The shared evaluation layer: trial execution, report trust, score
comparison, and the EvalResult projection engines consume."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import json
from math import isclose

from hillclimb.candidate import BackendInfo, Candidate
from hillclimb.dirs import create_candidate_dir, create_search_dir
from hillclimb.evaluation import (
    CandidateEvaluator,
    EvalResult,
    accept_band,
    eval_result_for,
    gate_passes,
    holdout_threshold,
    improves,
)
from hillclimb.executor import RESULT_FILE
from hillclimb.journal import Journal
from tests.conftest import CRASH_SCRIPT, executor_for, ok_script


def evaluator_for(task, config) -> CandidateEvaluator:
    return CandidateEvaluator(executor=executor_for(task), problem=task, config=config)


def fresh_candidate_dir(tmp_path, task, cid="c001"):
    search_dir = create_search_dir(tmp_path / "runs" / "r", "s")
    return create_candidate_dir(search_dir, cid, task.data_dir, task.problem_dir)


def scored(cid: str, *scores: float) -> Candidate:
    return Candidate(
        candidate_id=cid,
        operator="draft",
        status="passing",
        candidate_dir="/tmp/x",
        trials=[mk_trial(*scores, submission_ok=True)] if scores else [],
    )


# --- CandidateEvaluator.run_trial ---


def test_single_trial_scores_the_candidate(tmp_path, task, config):
    candidate_dir = fresh_candidate_dir(tmp_path, task)
    (candidate_dir / "solution.py").write_text(ok_script(0.7))
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))

    trial, ok = evaluator_for(task, config).run_trial(
        candidate, candidate_dir / "solution.py", candidate_dir, exec_timeout=30
    )

    assert ok
    assert candidate.trials == [trial]
    assert trial.index == 0 and trial.params == {} and trial.is_best
    assert [r.seed for r in trial.replicates] == [None]
    assert trial.val_score == 0.7 and candidate.val_score == 0.7
    # one replicate still lives in its own replicate dir; r0 is hoisted
    assert (candidate_dir / "trials" / "t0" / "replicates" / "r0" / "solution.py").exists()
    assert (candidate_dir / "submission.csv").exists()
    assert (candidate_dir / "exec_stdout.log").exists()


def test_multi_trial_runs_in_index_order_and_copies_trial0_artifacts(tmp_path, task, config):
    config.evaluation.n_replicates = 3
    config.evaluation.replicate_mode = "serial"
    candidate_dir = fresh_candidate_dir(tmp_path, task)
    (candidate_dir / "solution.py").write_text(ok_script(0.6))
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))

    trial, ok = evaluator_for(task, config).run_trial(
        candidate, candidate_dir / "solution.py", candidate_dir, exec_timeout=30
    )

    assert ok
    assert len(candidate.trials) == 1  # one parameter set, three seeded runs
    assert [r.seed for r in trial.replicates] == [0, 1, 2]
    assert candidate.val_score == 0.6  # median of identical replicates
    # r0 artifacts surface at the candidate-dir root
    assert (candidate_dir / "submission.csv").exists()
    for j in range(3):
        assert (candidate_dir / "trials" / "t0" / "replicates" / f"r{j}" / "eval_result.json").exists()


def test_parallel_trial_mode_also_scores(tmp_path, task, config):
    config.evaluation.n_replicates = 2
    config.evaluation.replicate_mode = "parallel"
    candidate_dir = fresh_candidate_dir(tmp_path, task)
    (candidate_dir / "solution.py").write_text(ok_script(0.4))
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))

    trial, ok = evaluator_for(task, config).run_trial(
        candidate, candidate_dir / "solution.py", candidate_dir, exec_timeout=30
    )

    assert ok
    assert len(trial.replicates) == 2
    assert candidate.val_score == 0.4


def test_crashing_solution_is_not_ok_and_surfaces_stderr(tmp_path, task, config):
    candidate_dir = fresh_candidate_dir(tmp_path, task)
    (candidate_dir / "solution.py").write_text(CRASH_SCRIPT)
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))

    trial, ok = evaluator_for(task, config).run_trial(
        candidate, candidate_dir / "solution.py", candidate_dir, exec_timeout=30
    )

    assert not ok
    assert trial.val_score is None and candidate.val_score is None
    assert "boom" in trial.replicates[0].stdout_tail


# --- CandidateEvaluator.read_replicate_report ---


def test_report_requires_validation_split(tmp_path, task, config):
    evaluator = evaluator_for(task, config)
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    (cwd / RESULT_FILE).write_text("0.5")  # bare number: no report possible
    assert evaluator.read_replicate_report(cwd) is None

    (cwd / RESULT_FILE).write_text(
        json.dumps({"score": 0.5, "split": "holdout", "report": {"rmse": 1.0}})
    )
    assert evaluator.read_replicate_report(cwd) is None  # never trust non-validation

    (cwd / RESULT_FILE).write_text(
        json.dumps({"score": 0.5, "split": "validation", "report": {"rmse": 1.0}})
    )
    report = evaluator.read_replicate_report(cwd)
    assert report is not None
    assert report["source"] == "agent"  # task fixture has report_trusted=False


def test_report_provenance_follows_problem_trust(tmp_path, task, config):
    task.report_trusted = True
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / RESULT_FILE).write_text(
        json.dumps({"score": 0.5, "split": "validation", "report": {"rmse": 1.0}})
    )
    report = evaluator_for(task, config).read_replicate_report(cwd)
    assert report["source"] == "evaluator"


def test_no_holdout_scorer_returns_none_triple(tmp_path, task, config):
    assert evaluator_for(task, config).score_holdout(tmp_path) == (None, None, None)


# --- score comparison semantics ---


def test_improves_is_direction_aware():
    assert improves(0.9, 0.8, higher_is_better=True, band=0.0)
    assert not improves(0.8, 0.9, higher_is_better=True, band=0.0)
    assert improves(0.8, 0.9, higher_is_better=False, band=0.0)
    assert not improves(0.9, 0.8, higher_is_better=False, band=0.0)
    # the band is exclusive: a delta equal to the band is not an improvement
    assert not improves(0.9, 0.8, higher_is_better=True, band=0.1)
    assert improves(0.91, 0.8, higher_is_better=True, band=0.1)


def test_gate_passes_on_win_tie_or_no_gate():
    assert gate_passes(0.5, None, higher_is_better=True)  # gate disabled
    assert gate_passes(None, 0.5, higher_is_better=True)  # unscored passes
    assert gate_passes(0.6, 0.5, higher_is_better=True)
    assert gate_passes(0.5, 0.5, higher_is_better=True)  # exact tie
    assert not gate_passes(0.4, 0.5, higher_is_better=True)
    assert gate_passes(0.4, 0.5, higher_is_better=False)


def test_accept_band_takes_max_of_floor_and_noise(tmp_path, config):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.candidate_result(scored("c1", 1.0, 1.2))  # spread 0.1

    config.evaluation.min_improvement = 0.0
    config.evaluation.noise_k = 0.0
    assert accept_band(config, journal) == 0.0  # strict default

    config.evaluation.noise_k = 2.0
    assert isclose(accept_band(config, journal), 0.2)  # 2 x measured floor

    config.evaluation.min_improvement = 0.5
    assert accept_band(config, journal) == 0.5  # author's floor wins


def test_holdout_threshold_is_kth_best(tmp_path, config):
    journal = Journal(tmp_path / "journal.jsonl")
    for cid, score in [("c1", 0.9), ("c2", 0.7), ("c3", 0.8)]:
        journal.candidate_result(scored(cid, score))

    assert holdout_threshold(journal, top_k=0, higher_is_better=True) is None  # disabled
    assert holdout_threshold(journal, top_k=5, higher_is_better=True) is None  # < k scored
    assert holdout_threshold(journal, top_k=2, higher_is_better=True) == 0.8
    assert holdout_threshold(journal, top_k=2, higher_is_better=False) == 0.8


# --- EvalResult projection ---


def test_eval_result_projects_a_terminal_candidate():
    candidate = scored("c007", 0.5, 0.6, 0.7)
    candidate.backend = BackendInfo(name="claude-code", cost_usd=1.25)
    for replicate, x in zip(candidate.trials[0].replicates, (1.0, 3.0, 2.0)):
        replicate.metrics = {"x": x}

    result = eval_result_for(candidate, feedback="try harder")

    assert result.candidate_id == "c007"
    assert result.score == 0.6  # median
    assert result.valid
    assert result.instance_scores == {}
    assert result.features == {"x": 2.0}  # per-key median
    assert result.feedback == "try harder"
    assert [t.val_score for t in result.trials] == [0.6]
    assert [r.val_score for r in result.trials[0].replicates] == [0.5, 0.6, 0.7]
    assert result.cost_usd == 1.25


def test_eval_result_for_buggy_candidate():
    candidate = Candidate(
        candidate_id="c008",
        operator="improve",
        status="buggy",
        candidate_dir="/tmp/x",
        trials=[mk_trial(returncode=1, stdout_tail="[stderr] boom")],
    )
    result = eval_result_for(candidate)
    assert not result.valid
    assert result.score is None
    assert result.cost_usd == 0.0


def test_eval_result_is_frozen():
    result = EvalResult(candidate_id="c1", score=None, valid=False)
    try:
        result.score = 1.0  # type: ignore[misc]
        raise AssertionError("EvalResult must be frozen")
    except AttributeError:
        pass
