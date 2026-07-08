from pathlib import Path

import pandas as pd
import pytest

from hillclimb.problem import load_problem, load_suite, resolve_target, suite_problem_targets


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    problem = tmp_path / "problems" / "my-problem"
    problem.mkdir(parents=True)
    (problem / "description.md").write_text("Optimize the thing.")
    (problem / "verify.py").write_text('print("val_score: 1.0")\n')
    pd.DataFrame({"id": [0], "target": [0]}).to_csv(
        problem / "sample_submission.csv", index=False
    )
    (problem / "problem.yaml").write_text(
        """
problem_id: my-problem
metric: score
lower_is_better: false
description: description.md
sample_submission: sample_submission.csv
verifier: verify.py
time_budget_s: 123
"""
    )
    return problem


def test_load_problem_from_directory(problem_dir, config):
    config.paths.problems_dir = problem_dir.parent
    spec = load_problem("my-problem", config)
    assert spec.problem_id == "my-problem"
    assert spec.problem_dir == problem_dir
    assert spec.data_dir == problem_dir
    assert spec.description == "Optimize the thing."
    assert spec.metric_name == "score"
    assert not spec.lower_is_better
    assert spec.sample_submission == problem_dir / "sample_submission.csv"
    assert spec.verifier == problem_dir / "verify.py"
    assert spec.time_budget_s == 123
    assert spec.holdout is None
    assert not spec.allow_network


def test_load_problem_with_data_dir_and_holdout(problem_dir, config):
    data_dir = problem_dir / "data"
    data_dir.mkdir()
    (problem_dir / "problem.yaml").write_text(
        """
problem_id: my-problem
metric: nrmse
lower_is_better: true
description: description.md
sample_submission: sample_submission.csv
verifier: verify.py
data_dir: data
allow_network: true
holdout:
  strategy: time-tail
  time_col: timestamp_utc
  id_col: timestamp_utc
  target_cols: [net_load_kwh]
  fraction: 0.1
"""
    )
    spec = load_problem(problem_dir, config)
    assert spec.data_dir == data_dir
    assert spec.allow_network
    assert spec.holdout is not None
    assert spec.holdout.strategy == "time-tail"
    assert spec.holdout.id_col == "timestamp_utc"
    assert spec.holdout.target_cols == ["net_load_kwh"]
    assert spec.holdout.fraction == 0.1


def test_suite_resolution_relative_to_suite_file(problem_dir, config):
    config.paths.problems_dir = problem_dir.parent
    suite_yaml = problem_dir.parent / "demo-suite.yaml"
    suite_yaml.write_text(
        """
suite_id: demo
problems:
  - my-problem
"""
    )
    suite = load_suite(suite_yaml, config)
    assert suite.suite_id == "demo"
    assert suite_problem_targets(suite, config) == [str(problem_dir)]

    resolved = resolve_target(suite_yaml, config)
    assert resolved.kind == "suite"
    assert resolved.suite is not None


@pytest.fixture
def evaluator_dir(tmp_path: Path) -> Path:
    problem = tmp_path / "problems" / "my-eval"
    problem.mkdir(parents=True)
    (problem / "description.md").write_text("Pack the bins.")
    (problem / "contract.md").write_text("solution.py must define pack(items, capacity).")
    (problem / "evaluate.py").write_text('print("val_score: 1.0")\n')
    (problem / "problem.yaml").write_text(
        """
problem_id: my-eval
kind: evaluator
metric: mean-bins
lower_is_better: true
eval: "{python} problem/evaluate.py"
holdout_eval: "{python} problem/evaluate.py --holdout"
time_budget_s: 300
"""
    )
    return problem


def test_load_evaluator_problem(evaluator_dir, config):
    config.paths.problems_dir = evaluator_dir.parent
    spec = load_problem("my-eval", config)
    assert spec.kind == "evaluator"
    assert spec.eval_command == "{python} problem/evaluate.py"
    assert spec.holdout_command == "{python} problem/evaluate.py --holdout"
    assert spec.holdout_mode == "evaluator"
    assert spec.sample_submission is None
    assert spec.verifier is None
    # contract.md picked up by default even without a `contract:` key
    assert "pack(items, capacity)" in spec.contract
    assert spec.requirements_file is None
    assert spec.baseline_solution is None
    assert spec.metric_name == "mean-bins" and spec.lower_is_better


def test_load_evaluator_optional_files(evaluator_dir, config):
    (evaluator_dir / "requirements.txt").write_text("numpy\n")
    (evaluator_dir / "baseline.py").write_text("def pack(i, c): return [i]\n")
    (evaluator_dir / "problem.yaml").write_text(
        """
kind: evaluator
metric: mean-bins
lower_is_better: true
eval: "{python} problem/evaluate.py"
requirements: requirements.txt
baseline: baseline.py
"""
    )
    spec = load_problem(evaluator_dir, config)
    assert spec.requirements_file == evaluator_dir / "requirements.txt"
    assert spec.baseline_solution == evaluator_dir / "baseline.py"
    assert spec.holdout_command is None  # optional


def test_load_evaluator_validation_errors(evaluator_dir, config):
    yaml_path = evaluator_dir / "problem.yaml"

    yaml_path.write_text("kind: evaluator\nmetric: m\nlower_is_better: true\n")
    with pytest.raises(ValueError, match="non-empty `eval:`"):
        load_problem(evaluator_dir, config)

    yaml_path.write_text('kind: evaluator\nmetric: m\nlower_is_better: true\neval: "  "\n')
    with pytest.raises(ValueError, match="non-empty `eval:`"):
        load_problem(evaluator_dir, config)

    yaml_path.write_text(
        'kind: evaluator\nmetric: m\nlower_is_better: true\neval: "x"\n'
        "holdout:\n  strategy: time-tail\n"
    )
    with pytest.raises(ValueError, match="does not apply to"):
        load_problem(evaluator_dir, config)

    yaml_path.write_text(
        'kind: evaluator\nmetric: m\nlower_is_better: true\neval: "x"\nrequirements: nope.txt\n'
    )
    with pytest.raises(FileNotFoundError, match="requirements file not found"):
        load_problem(evaluator_dir, config)

    yaml_path.write_text('kind: mystery\nmetric: m\nlower_is_better: true\n')
    with pytest.raises(ValueError, match="unknown problem kind"):
        load_problem(evaluator_dir, config)


def test_bin_packing_repo_problem_loads_and_scores(config, tmp_path):
    """The shipped evaluator example is self-consistent: the loader accepts
    it and its baseline scores through the real command executor."""
    import sys

    from hillclimb.baseline import write_baseline
    from hillclimb.command_executor import CommandExecutor
    from hillclimb.workspace import create_search_dir

    spec = load_problem("bin-packing", config)
    assert spec.kind == "evaluator"
    assert spec.holdout_command == "{python} problem/evaluate.py --holdout"
    assert spec.baseline_solution is not None
    assert "pack(items" in spec.contract

    search_dir = create_search_dir(tmp_path / "runs" / "r1", "bin-packing")
    executor = CommandExecutor(Path(sys.executable), spec.eval_command)
    baseline = write_baseline(spec, search_dir, executor=executor, timeout_s=120)
    assert baseline.val_score is not None
    assert baseline.is_best
    # FFD lands close to the ceil(sum/capacity) floor on these instances
    assert 40 < baseline.val_score < 60
    assert (search_dir / "best" / "solution.py").exists()
