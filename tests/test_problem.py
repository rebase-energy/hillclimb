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
