from pathlib import Path

import pandas as pd
import pytest

from hillclimb.problem import load_problem, load_suite, resolve_target, suite_problem_targets


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    problem = tmp_path / "problems" / "my-problem"
    problem.mkdir(parents=True)
    (problem / "description.md").write_text("Optimize the thing.")
    verifier = problem / "verifier.sh"
    verifier.write_text('#!/bin/sh\necho 1.0 > "$HILLCLIMB_RESULT"\n')
    verifier.chmod(0o755)
    pd.DataFrame({"id": [0], "target": [0]}).to_csv(
        problem / "sample_submission.csv", index=False
    )
    (problem / "problem.yaml").write_text(
        """
problem_id: my-problem
metric: score
higher_is_better: true
description: description.md
time_budget_s: 123
chart_baselines:
  reference floor: 0.5
  OpenEvolve best: 0.75
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
    assert spec.higher_is_better
    assert spec.verifier_cmd == [str(problem_dir / "verifier.sh")]
    assert spec.verifier_display == "./problem/verifier.sh"
    assert spec.time_budget_s == 123
    assert spec.chart_baselines == {"reference floor": 0.5, "OpenEvolve best": 0.75}
    assert spec.holdout_cmd is None  # no `holdout: true`
    assert not spec.allow_network


def test_load_problem_with_data_dir_and_holdout(problem_dir, config):
    data_dir = problem_dir / "data"
    data_dir.mkdir()
    (problem_dir / "problem.yaml").write_text(
        """
problem_id: my-problem
metric: nrmse
higher_is_better: false
description: description.md
data_dir: data
allow_network: true
holdout: true
"""
    )
    spec = load_problem(problem_dir, config)
    assert spec.data_dir == data_dir
    assert spec.allow_network
    # the hidden split is the same verifier, told which side to score
    assert spec.holdout_cmd == [str(problem_dir / "verifier.sh"), "--holdout"]


def test_verifier_must_exist_and_be_executable(problem_dir, config):
    (problem_dir / "verifier.sh").chmod(0o644)
    with pytest.raises(PermissionError, match="chmod \\+x"):
        load_problem(problem_dir, config)

    (problem_dir / "verifier.sh").unlink()
    with pytest.raises(FileNotFoundError, match="verifier not found"):
        load_problem(problem_dir, config)


def test_kind_key_is_rejected(problem_dir, config):
    (problem_dir / "problem.yaml").write_text(
        "problem_id: my-problem\nmetric: score\nhigher_is_better: true\n"
        "description: description.md\nkind: evaluator\n"
    )
    with pytest.raises(ValueError, match="`kind:` is gone"):
        load_problem(problem_dir, config)


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
    """A problem whose verifier drives solution.py itself."""
    problem = tmp_path / "problems" / "my-eval"
    problem.mkdir(parents=True)
    (problem / "description.md").write_text("Pack the bins.")
    (problem / "contract.md").write_text("solution.py must define pack(items, capacity).")
    verifier = problem / "verifier.sh"
    verifier.write_text('#!/bin/sh\necho 1.0 > "$HILLCLIMB_RESULT"\n')
    verifier.chmod(0o755)
    (problem / "problem.yaml").write_text(
        """
problem_id: my-eval
metric: mean-bins
higher_is_better: false
holdout: true
time_budget_s: 300
"""
    )
    return problem


def test_load_evaluator_problem(evaluator_dir, config):
    config.paths.problems_dir = evaluator_dir.parent
    spec = load_problem("my-eval", config)
    verifier = str(evaluator_dir / "verifier.sh")
    assert spec.verifier_cmd == [verifier]
    assert spec.holdout_cmd == [verifier, "--holdout"]
    assert spec.report_trusted  # the verifier owns the score
    # contract.md picked up by default even without a `contract:` key
    assert "pack(items, capacity)" in spec.contract
    assert spec.requirements_file is None
    assert spec.baseline_text is None
    assert spec.metric_name == "mean-bins" and not spec.higher_is_better


def test_load_evaluator_optional_files(evaluator_dir, config):
    (evaluator_dir / "requirements.txt").write_text("numpy\n")
    (evaluator_dir / "baseline.py").write_text("def pack(i, c): return [i]\n")
    (evaluator_dir / "problem.yaml").write_text(
        """
metric: mean-bins
higher_is_better: false
requirements: requirements.txt
baseline: baseline.py
"""
    )
    spec = load_problem(evaluator_dir, config)
    assert spec.requirements_file == evaluator_dir / "requirements.txt"
    assert spec.baseline_text == "def pack(i, c): return [i]\n"
    assert spec.baseline_summary == "baseline: baseline.py"
    assert spec.holdout_cmd is None  # holdout is opt-in


def test_interface_absent_leaves_spec_untouched(problem_dir, config):
    spec = load_problem(problem_dir, config)
    assert spec.interface_path is None
    assert spec.interface_text is None


def test_interface_picked_up_by_default(problem_dir, config):
    (problem_dir / "interface.py").write_text(
        "from hillclimb import spaces\n"
        "output = spaces.Table('submission.csv',"
        " columns={'id': spaces.Int(unique=True)}, n_rows=1)\n"
    )
    spec = load_problem(problem_dir, config)
    assert spec.interface_path == problem_dir / "interface.py"
    assert "`submission.csv`" in spec.interface_text
    assert "exactly 1 row" in spec.interface_text


def test_interface_errors(problem_dir, config):
    yaml_text = (problem_dir / "problem.yaml").read_text()
    (problem_dir / "problem.yaml").write_text(yaml_text + "interface: nope.py\n")
    with pytest.raises(FileNotFoundError, match="interface file not found"):
        load_problem(problem_dir, config)

    (problem_dir / "problem.yaml").write_text(yaml_text)
    (problem_dir / "interface.py").write_text("raise ValueError('broken')\n")
    with pytest.raises(ValueError, match="invalid interface file.*broken"):
        load_problem(problem_dir, config)


def test_load_problem_validation_errors(evaluator_dir, config):
    yaml_path = evaluator_dir / "problem.yaml"

    yaml_path.write_text("metric: m\nhigher_is_better: false\nrequirements: nope.txt\n")
    with pytest.raises(FileNotFoundError, match="requirements file not found"):
        load_problem(evaluator_dir, config)

    yaml_path.write_text("metric: m\nhigher_is_better: false\nverifier: absent.sh\n")
    with pytest.raises(FileNotFoundError, match="verifier not found"):
        load_problem(evaluator_dir, config)


def test_bin_packing_repo_problem_loads_and_scores(config, tmp_path):
    """The shipped evaluator example is self-consistent: the loader accepts
    it and its baseline scores through the real command executor."""
    import sys

    from hillclimb.baseline import write_baseline
    from hillclimb.executor import CommandExecutor
    from hillclimb.dirs import create_search_dir

    spec = load_problem("bin-packing", config)
    assert spec.verifier_cmd[0].endswith("problems/bin-packing/verifier.sh")
    assert spec.holdout_cmd == spec.verifier_cmd + ["--holdout"]
    assert spec.baseline_text is not None
    assert "pack(items" in spec.contract

    search_dir = create_search_dir(tmp_path / "runs" / "r1", "bin-packing")
    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    baseline = write_baseline(spec, search_dir, executor=executor, timeout_s=120)
    assert baseline.val_score is not None
    assert baseline.is_best
    # FFD lands close to the ceil(sum/capacity) floor on these instances
    assert 40 < baseline.val_score < 60
    assert (search_dir / "best" / "solution.py").exists()


def test_load_scalar_baseline(evaluator_dir, config):
    """`baseline: 0.5` declares the floor as a number instead of a script."""
    (evaluator_dir / "problem.yaml").write_text(
        "metric: sum-radii\nhigher_is_better: true\nbaseline: 0.5  # one big circle\n"
    )
    spec = load_problem(evaluator_dir, config)
    assert spec.baseline_score == 0.5
    assert spec.baseline_text is None
    assert spec.baseline_summary == "baseline: 0.5 (declared)"
    assert spec.chart_baselines == {"baseline": 0.5}


def test_numeric_baseline_is_authoritative_chart_reference(evaluator_dir, config):
    (evaluator_dir / "problem.yaml").write_text(
        "metric: sum-radii\nhigher_is_better: true\nbaseline: 0.5\n"
        "chart_baselines:\n  baseline: 999\n  OpenEvolve best: 0.75\n"
    )
    spec = load_problem(evaluator_dir, config)
    assert spec.chart_baselines == {"baseline": 0.5, "OpenEvolve best": 0.75}


def test_declared_floor_is_scored_but_has_no_code(evaluator_dir, config, tmp_path):
    from hillclimb.baseline import write_baseline

    (evaluator_dir / "problem.yaml").write_text(
        "metric: sum-radii\nhigher_is_better: true\nbaseline: 0.5\n"
    )
    spec = load_problem(evaluator_dir, config)
    c000 = write_baseline(spec, tmp_path / "search")
    assert c000.candidate_id == "c000" and c000.is_best
    assert c000.val_score == 0.5
    assert not (tmp_path / "search" / "candidates" / "c000" / "solution.py").exists()
