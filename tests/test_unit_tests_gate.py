from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.api import create_search
from hillclimb.backends.fake import FakeBackend
from hillclimb.baseline import run_scored_baseline
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate
from hillclimb.dirs import create_candidate_dir, create_run_dir, create_search_dir
from hillclimb.evaluation import CandidateEvaluator, eval_result_for
from hillclimb.journal import Journal
from hillclimb.problem import UnitTestSpec
from hillclimb.run import load_search_meta
from hillclimb.search import GreedySearcher
from hillclimb.unit_tests import (
    UnitTestInfrastructureError,
    UnitTestRunner,
    bundle_relative,
    freeze_for_run,
    restore_frozen,
)
from tests.conftest import executor_for, ok_script
from tests.factories import trial as make_trial


def frozen_suite(tmp_path: Path, task, body: str) -> UnitTestSpec:
    source = tmp_path / "source-tests"
    source.mkdir()
    (source / "check.py").write_text(body)
    task.unit_tests = UnitTestSpec(
        root=source,
        command=["{python}", "{tests}/check.py"],
    )
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    task.unit_tests = freeze_for_run(task, run_dir)
    return task.unit_tests


def evaluate(
    tmp_path: Path,
    task,
    config,
    body: str,
    solution: str | None = None,
    timeout_s: int = 30,
):
    spec = frozen_suite(tmp_path, task, body)
    search_dir = create_search_dir(tmp_path / "search-root", "s")
    candidate_dir = create_candidate_dir(
        search_dir,
        "c001",
        task.data_dir,
        task.problem_dir,
        unit_tests_dir=spec.root,
    )
    script = candidate_dir / "solution.py"
    script.write_text(solution or ok_script(0.7))
    candidate = Candidate(candidate_id="c001", operator="draft", candidate_dir=str(candidate_dir))
    evaluator = CandidateEvaluator(
        executor=executor_for(task),
        problem=task,
        config=config,
        unit_test_runner=UnitTestRunner(Path(sys.executable), spec),
    )
    trial, ok = evaluator.run_trial(candidate, script, candidate_dir, timeout_s)
    return candidate, trial, ok, candidate_dir


def test_passing_suite_allows_scoring(tmp_path, task, config):
    candidate, trial, ok, candidate_dir = evaluate(
        tmp_path,
        task,
        config,
        "import os\nassert os.path.exists(os.environ['HILLCLIMB_SOLUTION'])\n",
    )

    assert ok and trial.verdict == "passing"
    assert trial.unit_tests is not None and trial.unit_tests.passed
    assert trial.val_score == pytest.approx(0.7)
    assert candidate.val_score == pytest.approx(0.7)
    assert (candidate_dir / "unit_tests" / "check.py").exists()


def test_completed_nonzero_suite_is_failing_and_unranked(tmp_path, task, config):
    candidate, trial, ok, _candidate_dir = evaluate(
        tmp_path,
        task,
        config,
        "import sys\nprint('expected 2, got 3')\nsys.exit(1)\n",
    )

    assert not ok and trial.verdict == "failing"
    assert trial.val_score == pytest.approx(0.7)  # retained on the raw trial
    assert candidate.val_score is None  # never eligible as a best/shipped trial
    candidate.status = "failing"
    result = eval_result_for(candidate)
    assert not result.valid and result.score is None
    assert "expected 2" in trial.unit_tests.stdout_tail


def test_verifier_crash_is_buggy_and_skips_tests(tmp_path, task, config):
    marker = tmp_path / "tests-ran"
    _candidate, trial, ok, _candidate_dir = evaluate(
        tmp_path,
        task,
        config,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n",
        solution="raise RuntimeError('boom')\n",
    )

    assert not ok and trial.verdict == "buggy"
    assert trial.unit_tests is None
    assert not marker.exists()


def test_test_timeout_is_buggy(tmp_path, task, config):
    _candidate, trial, ok, _candidate_dir = evaluate(
        tmp_path,
        task,
        config,
        "import time\ntime.sleep(5)\n",
        timeout_s=2,
    )

    assert not ok and trial.verdict == "buggy"
    assert trial.unit_tests is not None and trial.unit_tests.timed_out


def test_test_process_crash_is_buggy(tmp_path, task, config):
    _candidate, trial, ok, _candidate_dir = evaluate(
        tmp_path,
        task,
        config,
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )

    assert not ok and trial.verdict == "buggy"
    assert trial.unit_tests is not None and trial.unit_tests.returncode < 0


def test_failing_candidate_is_debugged_but_never_selected(tmp_path, task, config):
    spec = frozen_suite(tmp_path, task, "raise AssertionError('wrong answer')\n")
    backend = FakeBackend()
    backend.queue(script=ok_script(0.9), notes="runs but is wrong\n")
    backend.queue(script=ok_script(0.8), notes="attempted repair\n")
    search_dir = create_search_dir(tmp_path / "search-run", "s")
    journal = Journal(search_dir / "journal.jsonl")
    evaluator = CandidateEvaluator(
        executor=executor_for(task),
        problem=task,
        config=config,
        unit_test_runner=UnitTestRunner(Path(sys.executable), spec),
    )
    searcher = GreedySearcher(
        problem=task,
        config=config,
        journal=journal,
        backend=backend,
        executor=evaluator.executor,
        evaluator=evaluator,
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=3,
        log=lambda *_: None,
    )

    assert searcher.run() is None
    assert journal.get("c001").status == "failing"
    assert backend.requests[1].operator == "debug"
    assert "frozen unit-test suite failed" in backend.requests[1].prompt
    assert journal.best_candidate(True) is None


def test_scored_baseline_must_pass_the_frozen_tests(tmp_path, task, config):
    spec = frozen_suite(tmp_path, task, "raise AssertionError('baseline is wrong')\n")
    search_dir = create_search_dir(tmp_path / "baseline-run", "s")
    evaluator = CandidateEvaluator(
        executor=executor_for(task),
        problem=task,
        config=config,
        unit_test_runner=UnitTestRunner(Path(sys.executable), spec),
    )

    baseline = run_scored_baseline(
        task,
        search_dir,
        evaluator,
        30,
        solution_text=ok_script(0.7),
        summary="baseline",
    )

    assert baseline.status == "failing"
    assert baseline.val_score is None
    assert not (search_dir / "best" / "solution.py").exists()


def test_failing_tuned_trial_cannot_replace_passing_trial():
    candidate = Candidate(
        candidate_id="c001",
        operator="draft",
        status="passing",
        trials=[
            make_trial(0.5, verdict="passing", is_best=True),
            make_trial(0.9, verdict="failing", is_best=False, index=1),
        ],
    )

    candidate.stamp_best_trial(True)

    assert candidate.best_trial.index == 0
    assert candidate.val_score == pytest.approx(0.5)


def test_snapshot_ignores_live_edits_and_detects_bundle_tampering(tmp_path, task):
    spec = frozen_suite(tmp_path, task, "print('original')\n")
    source = tmp_path / "source-tests" / "check.py"
    source.write_text("print('edited later')\n")
    assert (spec.root / "check.py").read_text() == "print('original')\n"

    run_dir = tmp_path / "runs" / "r"
    restored = restore_frozen(
        task.model_copy(deep=True),
        run_dir,
        bundle_path=bundle_relative(spec, run_dir),
        command=spec.command,
        sha256=spec.sha256,
    )
    assert restored.unit_tests.sha256 == spec.sha256

    (spec.root / "check.py").write_text("print('tampered')\n")
    with pytest.raises(UnitTestInfrastructureError, match="bundle changed"):
        restore_frozen(
            task.model_copy(deep=True),
            run_dir,
            bundle_path=bundle_relative(spec, run_dir),
            command=spec.command,
            sha256=spec.sha256,
        )


def test_create_search_records_and_reuses_run_bundle(tmp_path, task, config):
    source = tmp_path / "declared-tests"
    source.mkdir()
    (source / "check.py").write_text("print('frozen')\n")
    task.unit_tests = UnitTestSpec(
        root=source,
        command=["{python}", "{tests}/check.py"],
    )
    run_dir = create_run_dir(config.paths.runs_dir, "r")

    first = create_search(config, task, run_dir, "r", 60)
    first_meta = load_search_meta(first)
    frozen_text = (task.unit_tests.root / "check.py").read_text()
    source.joinpath("check.py").write_text("print('edited')\n")
    second = create_search(config, task, run_dir, "r", 60)
    second_meta = load_search_meta(second)

    assert first_meta.unit_tests_sha256 == second_meta.unit_tests_sha256
    assert first_meta.unit_tests_bundle == second_meta.unit_tests_bundle
    assert frozen_text == "print('frozen')\n"
    assert (task.unit_tests.root / "check.py").read_text() == frozen_text
