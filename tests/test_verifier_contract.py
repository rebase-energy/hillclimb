"""The verifier contract end to end: CommandExecutor + CommandHoldoutScorer
against a tiny real evaluator (run with sys.executable — no venv build)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from hillclimb.executor import CommandExecutor, CommandHoldoutScorer

EVALUATE = textwrap.dedent(
    """
    import json, os, sys
    sys.path.insert(0, os.getcwd())
    import solution

    split = "holdout" if "--holdout" in sys.argv else "validation"
    score = float(solution.answer())
    if os.environ.get("HILLCLIMB_TRIAL_SEED"):
        score += 0.001 * int(os.environ["HILLCLIMB_TRIAL_SEED"])
    report = {"version": 1, "split": split, "overall": {"score": score}}
    if os.environ.get("X_SECRET_TOKEN"):
        report["saw_token"] = True
    json.dump({"split": split, "score": score, "report": report},
              open("eval_result.json", "w"))
    print(f"val_score: {score}")
    """
)


def make_problem(tmp_path: Path) -> Path:
    problem_dir = tmp_path / "problem-src"
    problem_dir.mkdir()
    (problem_dir / "evaluate.py").write_text(EVALUATE)
    return problem_dir


def make_workspace(tmp_path: Path, problem_dir: Path, solution: str = "def answer():\n    return 0.5\n") -> Path:
    workspace = tmp_path / "candidates" / "c001"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text(solution)
    (workspace / "problem").symlink_to(problem_dir, target_is_directory=True)
    (workspace / "data").symlink_to(problem_dir, target_is_directory=True)
    return workspace


COMMAND = ["{python}", "problem/evaluate.py"]


def test_happy_path_scores_and_result_json(tmp_path):
    problem_dir = make_problem(tmp_path)
    workspace = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60)
    assert result.ok, Path(result.stderr_path).read_text()[-300:]
    assert result.val_score == 0.5
    payload = json.loads((workspace / "eval_result.json").read_text())
    assert payload["split"] == "validation"
    assert payload["report"]["overall"]["score"] == 0.5


def test_seed_env_and_placeholder_substitution(tmp_path):
    problem_dir = make_problem(tmp_path)
    workspace = make_workspace(tmp_path, problem_dir)
    # {solution} placeholder: evaluator receives the explicit path
    (problem_dir / "evaluate.py").write_text(
        EVALUATE.replace("import solution", "import importlib.util\n"
                         "spec = importlib.util.spec_from_file_location('solution', sys.argv[1])\n"
                         "solution = importlib.util.module_from_spec(spec)\n"
                         "spec.loader.exec_module(solution)")
    )
    executor = CommandExecutor(Path(sys.executable), ["{python}", "problem/evaluate.py", "{solution}"])
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60, seed=3)
    assert result.ok
    assert result.val_score == 0.503  # seed propagated via HILLCLIMB_TRIAL_SEED


def test_no_result_file_is_a_contract_violation(tmp_path):
    """Exit 0 without a result file is not a zero — the printed line is a log,
    not a score (agent code shares that stream)."""
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text('print("val_score: 1.0")\n')
    workspace = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60)
    assert result.val_score is None
    assert not result.submission_ok
    assert not result.ok


def test_bare_number_result_file(tmp_path):
    """The simplest possible verifier: echo a number into $HILLCLIMB_RESULT."""
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text(
        'import os\nopen(os.environ["HILLCLIMB_RESULT"], "w").write(" 12.5\\n")\n'
    )
    workspace = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60)
    assert result.ok
    assert result.val_score == 12.5


def test_stale_result_json_scrubbed_and_crash_not_ok(tmp_path):
    problem_dir = make_problem(tmp_path)
    (problem_dir / "evaluate.py").write_text('raise RuntimeError("evaluator boom")\n')
    workspace = make_workspace(tmp_path, problem_dir)
    (workspace / "eval_result.json").write_text('{"split": "validation", "score": 9}')
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60)
    assert result.returncode != 0
    assert not result.submission_ok  # the stale file was unlinked pre-run
    assert not (workspace / "eval_result.json").exists()


def test_validation_env_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET_TOKEN", "sk-123")
    problem_dir = make_problem(tmp_path)
    workspace = make_workspace(tmp_path, problem_dir)
    executor = CommandExecutor(Path(sys.executable), COMMAND)
    result = executor.execute(workspace / "solution.py", workspace, timeout_s=60)
    assert result.ok
    payload = json.loads((workspace / "eval_result.json").read_text())
    assert "saw_token" not in payload["report"]  # *_TOKEN scrubbed


def test_holdout_scorer_hidden_dir_full_env(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET_TOKEN", "sk-123")
    problem_dir = make_problem(tmp_path)
    workspace = make_workspace(tmp_path, problem_dir)
    (workspace / "candidate_1.py").write_text("# ensemble input\n")
    scorer = CommandHoldoutScorer(
        Path(sys.executable), COMMAND + ["--holdout"],
        problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval", timeout_s=60,
    )
    score, error = scorer.score(workspace)
    assert error is None
    assert score == 0.5
    eval_dir = tmp_path / "holdout-eval" / "c001"
    # hidden dir recreated the run layout: solution + extras + symlinks
    assert (eval_dir / "solution.py").exists()
    assert (eval_dir / "candidate_1.py").exists()
    assert (eval_dir / "problem").is_symlink() and (eval_dir / "data").is_symlink()
    payload = json.loads((eval_dir / "eval_result.json").read_text())
    assert payload["split"] == "holdout"
    assert payload["report"].get("saw_token") is True  # full env: credentials flow
    # agent-visible workspace untouched by the holdout run
    assert not (workspace / "eval_result.json").exists()


def test_holdout_scorer_failure_mapping(tmp_path):
    problem_dir = make_problem(tmp_path)
    workspace = make_workspace(
        tmp_path, problem_dir, solution="def answer():\n    raise ValueError('nope')\n"
    )
    scorer = CommandHoldoutScorer(
        Path(sys.executable), COMMAND, problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval", timeout_s=60,
    )
    score, error = scorer.score(workspace)
    assert score is None
    assert "holdout evaluation failed" in error

    scorer_no_score = CommandHoldoutScorer(
        Path(sys.executable), ["{python}", "-c", "print(42)"],
        problem_dir=problem_dir, data_dir=problem_dir,
        work_root=tmp_path / "holdout-eval-2", timeout_s=60,
    )
    score, error = scorer_no_score.score(workspace)
    assert score is None
    assert "produced no score" in error
