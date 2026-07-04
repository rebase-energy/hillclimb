from __future__ import annotations

import yaml

from hillclimb.cli import _run_problem, _run_suite
from hillclimb.experiment import load_experiments


def write_problem(root, name):
    problem = root / name
    problem.mkdir(parents=True)
    (problem / "problem.yaml").write_text(
        f"""
problem_id: {name}
metric: score
lower_is_better: false
description: description.md
sample_submission: sample_submission.csv
verifier: verify.py
"""
    )
    (problem / "description.md").write_text(name)
    (problem / "sample_submission.csv").write_text("id,x\n0,0\n")
    (problem / "verify.py").write_text('print("val_score: 1")\n')
    return problem


def test_run_problem_creates_experiment_metadata(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    problem = write_problem(root, "a")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    executed = []

    def fake_execute(config_arg, problem_arg, run_dir, budget):
        executed.append((problem_arg.problem_id, run_dir))

    monkeypatch.setattr("hillclimb.cli._execute", fake_execute)

    _run_problem(str(problem), config, budget="10m", experiment_name="My Experiment")

    experiments = load_experiments(config.paths.runs_dir)
    assert len(experiments) == 1
    meta = next(iter(experiments.values()))
    assert meta.name == "My Experiment"
    assert meta.problem_ids == ["a"]
    run_meta = yaml.safe_load((executed[0][1] / "run.yaml").read_text())
    assert run_meta["experiment_id"] == meta.experiment_id
    assert run_meta["experiment_name"] == "My Experiment"


def test_run_suite_launches_one_child_per_problem(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    for name in ("a", "b"):
        write_problem(root, name)
    suite = root / "suite.yaml"
    suite.write_text("suite_id: demo\nproblems:\n  - a\n  - b\n")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    calls = []

    class DummyProc:
        pid = 123

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return DummyProc()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    _run_suite(
        str(suite),
        config,
        budget="10m",
        backend="dummy",
        model=None,
        holdout=True,
        name="Demo",
    )

    assert len(calls) == 2
    assert calls[0][0][2:4] == ["hillclimb.cli", "run"]
    experiment_ids = [cmd[cmd.index("--experiment-id") + 1] for cmd, _ in calls]
    assert len(set(experiment_ids)) == 1
    assert all(cmd[cmd.index("--experiment-name") + 1] == "Demo" for cmd, _ in calls)
    assert all("--backend" in cmd and "dummy" in cmd for cmd, _ in calls)
    assert all("--budget" in cmd and "10m" in cmd for cmd, _ in calls)
    experiments = load_experiments(config.paths.runs_dir)
    assert list(experiments.values())[0].name == "Demo"
