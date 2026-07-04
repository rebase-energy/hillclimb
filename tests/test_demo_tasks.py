from __future__ import annotations

import shutil
import subprocess
import sys

from hillclimb.config import Config
from hillclimb.problem import load_problem


DEMO_PROBLEMS = [
    ("circle-packing", "sum-radii", False),
    ("heilbronn-11", "min-triangle-area", False),
    ("tsp-200", "tour-length", True),
    ("labs-60", "autocorrelation-energy", True),
]


def test_demo_problems_load(config: Config):
    for problem_id, metric, lower_is_better in DEMO_PROBLEMS:
        spec = load_problem(problem_id, config)
        assert spec.problem_id == problem_id
        assert spec.metric_name == metric
        assert spec.lower_is_better is lower_is_better
        assert spec.sample_submission.exists()
        assert spec.verifier is not None and spec.verifier.exists()
        assert spec.time_budget_s == 900


def test_demo_sample_submissions_pass_verifiers(config: Config, tmp_path):
    for problem_id, _, _ in DEMO_PROBLEMS:
        spec = load_problem(problem_id, config)
        workspace = tmp_path / problem_id
        workspace.mkdir()
        (workspace / "data").symlink_to(spec.data_dir, target_is_directory=True)
        (workspace / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
        shutil.copy(spec.sample_submission, workspace / "submission.csv")

        result = subprocess.run(
            [sys.executable, str(spec.verifier)],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "val_score:" in result.stdout
        assert "INVALID" not in result.stdout
