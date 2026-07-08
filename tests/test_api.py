"""Public programmatic API: run_search creates the Run/Search layout and
returns a SearchOutcome (dummy backend, no agent spend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hillclimb.api import run_search
from hillclimb.run import load_run_meta, load_search_meta


@pytest.mark.slow
def test_run_search_end_to_end(config):
    config.paths.problems_dir = Path("problems")  # repo problems (circle-packing)
    logs: list[str] = []

    outcome = run_search(
        "circle-packing",
        budget_s=10,
        name="api-test",
        config=config,
        backend="dummy",
        holdout=False,
        log=logs.append,
    )

    assert outcome.state == "done"
    assert outcome.search_dir.name == "circle-packing"
    assert load_run_meta(outcome.run_dir) is not None
    assert load_search_meta(outcome.search_dir).budget_s == 10
    assert (outcome.search_dir / "journal.jsonl").exists()
    assert (outcome.search_dir / "best" / "submission.csv").exists()
    assert any("Search " in line for line in logs)
    assert outcome.selected is None or outcome.selected.val_score is not None


def test_runtime_packages_evaluator_kind_and_requirements_file(tmp_path):
    from hillclimb.runtime import runtime_packages

    assert runtime_packages("evaluator") == runtime_packages("csv")
    req = tmp_path / "requirements.txt"
    req.write_text("# solver deps\nnumpy\nnetworkx>=3\n")
    assert runtime_packages("evaluator", requirements_file=req) == ["numpy", "networkx>=3"]


def test_default_venv_python_keys_on_requirements_content(config, tmp_path):
    from hillclimb.api import default_venv_python

    req = tmp_path / "r.txt"
    req.write_text("numpy\n")
    first = default_venv_python(config, "evaluator", requirements=req)
    assert "problem-" in str(first)
    req.write_text("numpy\npandas\n")
    changed = default_venv_python(config, "evaluator", requirements=req)
    assert changed != first
    # identical content in a different file shares the venv
    twin = tmp_path / "r2.txt"
    twin.write_text("numpy\npandas\n")
    assert default_venv_python(config, "evaluator", requirements=twin) == changed


def test_ensure_runtime_venv_requirements_short_circuit(config, tmp_path, monkeypatch):
    from hillclimb import api

    req = tmp_path / "r.txt"
    req.write_text("numpy\n")
    python = tmp_path / "venvs" / "problem-abc" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(api, "default_venv_python", lambda *a, **k: python)

    def boom(*a, **k):
        raise AssertionError("existing venv must not trigger a build")

    monkeypatch.setattr(api.subprocess, "run", boom)
    assert api.ensure_runtime_venv(config, "evaluator", requirements=req) == python


# --- evaluator kind: dispatch + baseline ---

EVALUATE_PY = """\
import json, os, sys
sys.path.insert(0, os.getcwd())
import solution
score = float(solution.answer())
json.dump({"split": "validation", "score": score,
           "report": {"version": 1, "split": "validation", "overall": {"score": score}}},
          open("eval_result.json", "w"))
print(f"val_score: {score}")
"""


def make_evaluator_problem(tmp_path, **overrides):
    from hillclimb.problem import ProblemSpec

    problem_dir = tmp_path / "eval-problem"
    problem_dir.mkdir(exist_ok=True)
    (problem_dir / "evaluate.py").write_text(EVALUATE_PY)
    fields = dict(
        kind="evaluator",
        problem_id="eval-problem",
        problem_dir=problem_dir,
        data_dir=problem_dir,
        description="score the answer",
        metric_name="score",
        lower_is_better=False,
        time_budget_s=600,
        holdout_mode="evaluator",
        eval_command="{python} problem/evaluate.py",
    )
    fields.update(overrides)
    return ProblemSpec(**fields)


def test_build_executor_dispatches_command_executor(config, tmp_path, monkeypatch):
    import sys

    from hillclimb import api
    from hillclimb.command_executor import CommandExecutor

    monkeypatch.setattr(api, "ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    executor = api.build_executor(config, make_evaluator_problem(tmp_path), log=lambda *_: None)
    assert isinstance(executor, CommandExecutor)


def test_build_holdout_scorer_for_evaluator(config, tmp_path, monkeypatch):
    import sys

    from hillclimb import api
    from hillclimb.command_executor import CommandHoldoutScorer

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr(api, "ensure_runtime_venv", lambda *a, **k: Path(sys.executable))

    no_holdout = make_evaluator_problem(tmp_path)
    assert api.build_holdout_scorer(config, no_holdout, tmp_path / "s") is None

    # no HF-credential preflight for evaluator problems (that check is emflow's)
    with_holdout = make_evaluator_problem(
        tmp_path, holdout_command="{python} problem/evaluate.py --holdout"
    )
    scorer = api.build_holdout_scorer(config, with_holdout, tmp_path / "s")
    assert isinstance(scorer, CommandHoldoutScorer)


def test_evaluator_baseline_placeholder_and_scored(config, tmp_path):
    import sys

    from hillclimb.baseline import write_baseline
    from hillclimb.command_executor import CommandExecutor
    from hillclimb.workspace import create_search_dir

    problem = make_evaluator_problem(tmp_path)
    search_dir = create_search_dir(tmp_path / "runs" / "r1", "eval-problem")
    placeholder = write_baseline(problem, search_dir)  # no baseline shipped
    assert placeholder.candidate_id == "c000"
    assert not placeholder.trials
    assert "unscored placeholder" in placeholder.summary

    baseline_file = problem.problem_dir / "baseline.py"
    baseline_file.write_text("def answer():\n    return 0.25\n")
    problem = make_evaluator_problem(tmp_path, baseline_solution=baseline_file)
    search_dir = create_search_dir(tmp_path / "runs" / "r2", "eval-problem")
    executor = CommandExecutor(Path(sys.executable), problem.eval_command)
    scored = write_baseline(problem, search_dir, executor=executor, timeout_s=60)
    assert scored.val_score == 0.25
    assert scored.is_best
    assert (search_dir / "best" / "solution.py").exists()
