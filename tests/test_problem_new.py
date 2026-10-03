"""`hillclimb problem new`: the scaffold for a problem of your own."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from hillclimb import cli
from hillclimb.config import Config
from hillclimb.demo import scaffold_problem
from hillclimb.problem import load_problem


def test_problem_new_writes_a_problem_that_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    result = CliRunner().invoke(cli.app, ["problem", "new", "my-problem"])
    assert result.exit_code == 0, result.output
    problem_dir = tmp_path / "problems" / "my-problem"
    assert (tmp_path / "hillclimb.yaml").is_file()
    assert "__PROBLEM_ID__" not in (problem_dir / "problem.yaml").read_text()
    problem = load_problem("my-problem", Config.load(start=tmp_path))
    assert problem.problem_id == "my-problem"
    assert problem.higher_is_better is False
    assert problem.score_cmd and problem.baseline_text and problem.contract


def test_problem_new_never_overwrites_and_refuses_odd_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["problem", "new", "mine"]).exit_code == 0
    (tmp_path / "problems" / "mine" / "description.md").write_text("my own words")
    again = runner.invoke(cli.app, ["problem", "new", "mine"])
    assert again.exit_code == 1 and "already exists" in again.output
    assert (tmp_path / "problems" / "mine" / "description.md").read_text() == "my own words"
    for bad in ("Bad Id", "../escape", "-dash"):
        assert runner.invoke(cli.app, ["problem", "new", bad]).exit_code != 0
    assert sorted(p.name for p in (tmp_path / "problems").iterdir() if p.is_dir()) == ["mine"]


@pytest.mark.parametrize(
    ("solution", "valid"),
    [
        (None, True),  # the baseline
        ("def partition(numbers):\n    return [0, 0]\n", False),
        ("def partition(numbers):\n    return 'all'\n", False),
    ],
)
def test_the_scaffold_runs_and_scores(tmp_path, solution, valid):
    """run.py then score.py, the two steps the engine runs: the baseline
    scores, an invalid answer exits non-zero, and the score lands in
    $HILLCLIMB_RESULT."""
    problem_dir = scaffold_problem(tmp_path / "problems", "p")
    work = tmp_path / "work"
    work.mkdir()
    (work / "solution.py").write_text(solution or (problem_dir / "baseline.py").read_text())
    env = {
        **os.environ,
        "PYTHONPATH": str(problem_dir),
        "HILLCLIMB_SOLUTION": str(work / "solution.py"),
        "HILLCLIMB_RESULT": str(work / "result.json"),
    }
    run = subprocess.run([sys.executable, problem_dir / "run.py"], cwd=work, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    score = subprocess.run([sys.executable, problem_dir / "score.py"], cwd=work, env=env, capture_output=True, text=True)
    assert (score.returncode == 0) is valid, score.stderr
    if valid:
        result = json.loads((work / "result.json").read_text())
        assert 0 < result["score"] < 13 and len(result["instances"]) == 20
    else:
        assert not (work / "result.json").exists()
