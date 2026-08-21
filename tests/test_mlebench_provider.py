"""mlebench:// provider: metadata from the mle-bench checkout, prepared
public data as the problem, split suites, and the one-shot post-search
grading hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hillclimb import api
from hillclimb.candidate import Candidate
from hillclimb.problem import load_problem, resolve_target


def fake_checkout(tmp_path: Path, comp_id: str = "fake-comp", scores=(0.99, 0.9, 0.5)) -> Path:
    repo = tmp_path / "mle-bench"
    comp = repo / "mlebench" / "competitions" / comp_id
    comp.mkdir(parents=True)
    (comp / "config.yaml").write_text(
        f"id: {comp_id}\ngrader:\n  name: accuracy\n  grade_fn: x:y\n"
    )
    (comp / "description.md").write_text(f"# {comp_id}\n\nPredict the thing.\n")
    (comp / "leaderboard.csv").write_text(
        "teamId,score\n" + "\n".join(f"t{i},{s}" for i, s in enumerate(scores)) + "\n"
    )
    splits = repo / "experiments" / "splits"
    splits.mkdir(parents=True)
    (splits / "low.txt").write_text(f"{comp_id}\nother-comp\n")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("")
    return repo


def prepare_data(tmp_path: Path, comp_id: str = "fake-comp") -> Path:
    public = tmp_path / "data" / comp_id / "prepared" / "public"
    public.mkdir(parents=True)
    (public / "sample_submission.csv").write_text("id,target\n1,0\n")
    (public / "train.csv").write_text("id,feature,target\n1,0.5,1\n")
    return tmp_path / "data"


@pytest.fixture
def mb_config(tmp_path, config):
    repo = fake_checkout(tmp_path)
    config.paths.mlebench_python = repo / ".venv" / "bin" / "python"
    config.paths.mlebench_data_dir = prepare_data(tmp_path)
    return config


def test_load_mlebench_problem(mb_config):
    problem = load_problem("mlebench://fake-comp", mb_config)
    assert not problem.report_trusted  # the agent reports its own score
    assert problem.mlebench_comp_id == "fake-comp"
    assert problem.metric_name == "accuracy"
    assert problem.lower_is_better is False  # leaderboard best-first, 0.99 on top
    assert problem.verifier_cmd[-2:] == ["--require", "submission.csv"]
    assert problem.data_dir.name == "public"
    assert "Predict the thing" in problem.description


def test_direction_inferred_lower_better(tmp_path, config):
    repo = fake_checkout(tmp_path, comp_id="rmse-comp", scores=(0.1, 1.0, 5.0))
    config.paths.mlebench_python = repo / ".venv" / "bin" / "python"
    config.paths.mlebench_data_dir = prepare_data(tmp_path, comp_id="rmse-comp")
    problem = load_problem("mlebench://rmse-comp", config)
    assert problem.lower_is_better is True


def test_unprepared_competition_names_the_fix(mb_config, tmp_path):
    comp = mb_config.paths.mlebench_python.parents[2] / "mlebench" / "competitions" / "bare"
    comp.mkdir()
    for name in ("config.yaml", "description.md", "leaderboard.csv"):
        src = comp.parent / "fake-comp" / name
        (comp / name).write_text(src.read_text())
    with pytest.raises(FileNotFoundError, match="mlebench prepare -c bare"):
        load_problem("mlebench://bare", mb_config)


def test_unknown_competition(mb_config):
    with pytest.raises(FileNotFoundError, match="unknown MLE-bench competition"):
        load_problem("mlebench://nope", mb_config)


def test_split_resolves_as_suite_with_lite_alias(mb_config):
    for name in ("mlebench://low", "mlebench://lite"):
        resolved = resolve_target(name, mb_config)
        assert resolved.kind == "suite"
        assert resolved.suite.suite_id == "mlebench-low"
        assert resolved.suite.problems[0].target == "mlebench://fake-comp"
        assert resolved.suite.problems[1].target == "mlebench://other-comp"


def test_comp_id_resolves_as_problem(mb_config):
    resolved = resolve_target("mlebench://fake-comp", mb_config)
    assert resolved.kind == "problem"
    assert resolved.problem.mlebench_comp_id == "fake-comp"


def test_search_meta_records_resumable_target(mb_config, tmp_path):
    from hillclimb.run import load_search_meta

    problem = load_problem("mlebench://fake-comp", mb_config)
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "searches").mkdir(parents=True)
    search_dir = api.create_search(mb_config, problem, run_dir, "r1", total_s=600)
    assert load_search_meta(search_dir).problem == "mlebench://fake-comp"


def test_post_search_grading_writes_report(mb_config, tmp_path, monkeypatch):
    problem = load_problem("mlebench://fake-comp", mb_config)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "submission.csv").write_text("id,target\n1,1\n")
    selected = Candidate(candidate_id="c003", operator="improve", workspace=str(workspace))
    search_dir = tmp_path / "search"
    search_dir.mkdir()

    graded = {}

    def fake_grade(submission, comp_id, config):
        graded["args"] = (submission, comp_id)
        return {"score": 0.91, "gold_medal": False, "silver_medal": True, "bronze_medal": False}

    monkeypatch.setattr("hillclimb.grading.grade_submission", fake_grade)
    logs = []
    api._mlebench_grade(mb_config, problem, search_dir, selected, logs.append)

    assert graded["args"] == (workspace / "submission.csv", "fake-comp")
    report = json.loads((search_dir / "mlebench-grade.json").read_text())
    assert report["score"] == 0.91
    assert any("medal=silver" in line for line in logs)


def test_grading_failure_never_raises(mb_config, tmp_path):
    problem = load_problem("mlebench://fake-comp", mb_config)
    selected = Candidate(
        candidate_id="c003", operator="improve", workspace=str(tmp_path / "missing")
    )
    logs = []
    api._mlebench_grade(mb_config, problem, tmp_path, selected, logs.append)
    assert any("grading failed" in line for line in logs)
