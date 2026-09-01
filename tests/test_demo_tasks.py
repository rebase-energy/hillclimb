from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hillclimb.config import Config
from hillclimb.demo import BUNDLED_PROBLEM_IDS, install_demo_problem
from hillclimb.problem import load_problem


DEMO_PROBLEMS = [
    ("circle-packing", "sum-radii", True),
    ("heilbronn-11", "min-triangle-area", True),
    ("tsp-200", "tour-length", False),
    ("labs-60", "autocorrelation-energy", False),
]


def test_demo_problems_load(config: Config):
    for problem_id, metric, higher_is_better in DEMO_PROBLEMS:
        spec = load_problem(problem_id, config)
        assert spec.problem_id == problem_id
        assert spec.metric_name == metric
        assert spec.higher_is_better is higher_is_better
        assert Path(spec.verifier_cmd[0]).exists()
        # every demo problem has a floor: circle-packing declares it as a number
        # (one big circle), the others ship a sample submission
        if problem_id == "circle-packing":
            assert spec.baseline_score == 0.5
            assert len((spec.problem_dir / "sample_submission.csv").read_text().splitlines()) == 27
        else:
            assert spec.baseline_files["submission.csv"].exists()
        assert spec.time_budget_s == 900


def test_demo_verifiers_score_the_sample_submission(config: Config, tmp_path):
    """The shipped verifiers run end to end through the real contract: the
    sample submission is valid by construction, so each must report a score."""
    from hillclimb.executor import CommandExecutor

    for problem_id, _, _ in DEMO_PROBLEMS:
        spec = load_problem(problem_id, config)
        candidate_dir = tmp_path / problem_id
        candidate_dir.mkdir()
        (candidate_dir / "data").symlink_to(spec.data_dir, target_is_directory=True)
        (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
        # a "solution" that just ships the sample submission
        (candidate_dir / "solution.py").write_text(
            "import shutil\n"
            f'shutil.copy(r"{spec.problem_dir / 'sample_submission.csv'}", "submission.csv")\n'
        )
        executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
        result = executor.execute(candidate_dir / "solution.py", candidate_dir, timeout_s=120)
        assert result.ok, Path(result.stderr_path).read_text()[-400:]
        assert result.val_score is not None
        assert "INVALID" not in Path(result.stdout_path).read_text()
        if problem_id == "circle-packing":
            # reference producer of the per-instance contract: one instance
            # per circle, and the instances decompose the score exactly
            assert len(result.instance_scores) == 26
            assert all(k.startswith("circle-") for k in result.instance_scores)
            assert abs(sum(result.instance_scores.values()) - result.val_score) < 1e-4


def test_heilbronn_convex_13_loads_and_scores_baseline(config: Config, tmp_path):
    """The comparison demo is directly runnable with its published chart lines."""
    from hillclimb.executor import CommandExecutor

    spec = load_problem("heilbronn-convex-13", config)
    assert spec.metric_name == "normalized-min-triangle-area"
    assert spec.higher_is_better is True
    assert spec.time_budget_s == 1800
    assert spec.baseline_text is not None
    assert spec.baseline_files["submission.csv"].exists()
    assert spec.chart_baselines == {
        "OpenEvolve": 0.0267,
        "AdaEvolve": 0.029,
        "AlphaEvolve": 0.030936889034895654,
    }

    candidate_dir = tmp_path / "heilbronn-convex-13"
    candidate_dir.mkdir()
    (candidate_dir / "data").symlink_to(spec.data_dir, target_is_directory=True)
    (candidate_dir / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
    (candidate_dir / "solution.py").write_text(
        "import shutil\n"
        f'shutil.copy(r"{spec.problem_dir / "sample_submission.csv"}", "submission.csv")\n'
    )

    executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
    result = executor.execute(
        candidate_dir / "solution.py", candidate_dir, timeout_s=120
    )
    assert result.ok, Path(result.stderr_path).read_text()[-400:]
    assert result.val_score == pytest.approx(0.0003224584042678343)
    assert "INVALID" not in Path(result.stdout_path).read_text()


def test_heilbronn_convex_13_is_fetchable_and_matches_repo_problem(
    config: Config, tmp_path
):
    assert "heilbronn-convex-13" in BUNDLED_PROBLEM_IDS
    config.paths.problems_dir = tmp_path / "problems"
    installed, created = install_demo_problem(
        config.paths.problems_dir, "heilbronn-convex-13"
    )
    assert created
    assert (installed / "verifier.sh").stat().st_mode & 0o111

    repo = Path("problems/heilbronn-convex-13")
    for name in (
        "problem.yaml",
        "description.md",
        "baseline.py",
        "sample_submission.csv",
        "verifier.sh",
        "verify.py",
    ):
        assert (installed / name).read_text() == (repo / name).read_text(), name

    spec = load_problem("heilbronn-convex-13", config)
    assert spec.baseline_text is not None
    assert spec.chart_baselines["AlphaEvolve"] == pytest.approx(
        0.030936889034895654
    )
