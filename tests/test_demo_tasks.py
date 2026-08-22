from __future__ import annotations

import sys
from pathlib import Path

from hillclimb.config import Config
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
