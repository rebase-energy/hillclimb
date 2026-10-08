from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from hillclimb.cli import main as cli_main
from hillclimb.catalog import PROBLEM_IDS, install_problem
from hillclimb.harness.executor import CommandExecutor, CommandHoldoutScorer, RESULT_FILE
from hillclimb.problem import load_problem


def test_knapsack_is_a_bundled_problem_and_unknown_ids_are_rejected(tmp_path):
    assert "knapsack" in PROBLEM_IDS
    problem_dir, created = install_problem(tmp_path / "problems", "knapsack")
    assert created and problem_dir.name == "knapsack"
    assert (problem_dir / "problem.yaml").exists()
    # two steps: run.py calls select_items, verify.py scores; no verifier to run
    assert (problem_dir / "run.py").exists() and not (problem_dir / "verifier.sh").exists()

    with pytest.raises(ValueError, match="no catalog problem"):
        install_problem(tmp_path / "problems", "not-a-problem")


def test_knapsack_baseline_scores_validation_and_holdout(tmp_path, config):
    config.paths.problems_dir = tmp_path / "problems"
    install_problem(config.paths.problems_dir, "knapsack")
    problem = load_problem("knapsack", config)
    assert problem.metric_name == "mean-percent-of-upper-bound"
    assert problem.higher_is_better
    assert problem.holdout_cmd is not None
    assert problem.baseline_text is not None

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "problem").symlink_to(problem.problem_dir, target_is_directory=True)
    (candidate_dir / "data").symlink_to(problem.data_dir, target_is_directory=True)
    solution = candidate_dir / "solution.py"
    solution.write_text(problem.baseline_text)

    assert [p.name for p in problem.holdout_inputs] == ["holdout"]
    validation = CommandExecutor(
        Path(sys.executable), problem.verifier_cmd, score_argv=problem.score_cmd
    ).execute(solution, candidate_dir, timeout_s=30)
    assert validation.ok and 96.5 < validation.val_score < 98.0
    payload = json.loads((candidate_dir / RESULT_FILE).read_text())
    assert payload["report"]["source"] == "evaluator"
    assert len(payload["report"]["zones"]) == 6

    holdout = CommandHoldoutScorer(
        Path(sys.executable), problem.holdout_cmd, problem_dir=problem.problem_dir,
        data_dir=problem.data_dir, work_root=tmp_path / "holdout-eval", timeout_s=30,
        score_argv=problem.score_cmd,
    )
    score, error, _ = holdout.score(candidate_dir)
    assert error is None and 96.5 < score < 98.0
    assert json.loads((tmp_path / "holdout-eval" / "candidate" / "t0" / RESULT_FILE).read_text())["split"] == "holdout"


def test_fetch_knapsack_installs_and_lists_problem_files(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli_main(["problem", "get", "knapsack"])
    assert exc.value.code == 0
    problem_dir = tmp_path / "problems" / "knapsack"
    assert (problem_dir / "baseline.py").exists()
    assert (problem_dir / "run.py").exists()
    output = capsys.readouterr().out
    assert "Fetched knapsack" in output
    assert "contract.md" in output and "baseline.py" in output and "verify.py" in output
    assert "run.py" in output and "holdout inputs" in output
