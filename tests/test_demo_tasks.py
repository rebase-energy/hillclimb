from __future__ import annotations

import sys
from pathlib import Path

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
        assert Path(spec.verifier_cmd[0]).exists()
        assert spec.baseline_files["submission.csv"].exists()
        assert spec.time_budget_s == 900


def test_demo_verifiers_score_the_sample_submission(config: Config, tmp_path):
    """The shipped verifiers run end to end through the real contract: the
    sample submission is valid by construction, so each must report a score."""
    from hillclimb.executor import CommandExecutor

    for problem_id, _, _ in DEMO_PROBLEMS:
        spec = load_problem(problem_id, config)
        workspace = tmp_path / problem_id
        workspace.mkdir()
        (workspace / "data").symlink_to(spec.data_dir, target_is_directory=True)
        (workspace / "problem").symlink_to(spec.problem_dir, target_is_directory=True)
        # a "solution" that just ships the sample submission
        (workspace / "solution.py").write_text(
            "import shutil\n"
            f'shutil.copy(r"{spec.baseline_files["submission.csv"]}", "submission.csv")\n'
        )
        executor = CommandExecutor(Path(sys.executable), spec.verifier_cmd)
        result = executor.execute(workspace / "solution.py", workspace, timeout_s=120)
        assert result.ok, Path(result.stderr_path).read_text()[-400:]
        assert result.val_score is not None
        assert "INVALID" not in Path(result.stdout_path).read_text()
