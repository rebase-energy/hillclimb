from __future__ import annotations

import pytest
import typer

from hillclimb.cli import _run_problem, _run_suite, resolve_search_dir
from hillclimb.run import (
    RunMeta,
    SearchMeta,
    iter_run_dirs,
    load_run_meta,
    load_search_meta,
    write_run_meta,
    write_search_meta,
)


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


def test_run_problem_creates_run_and_search_metadata(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    problem = write_problem(root, "a")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    executed = []

    def fake_execute(config_arg, problem_arg, search_dir, budget, seed_from=None):
        executed.append((problem_arg.problem_id, search_dir))

    monkeypatch.setattr("hillclimb.cli._execute", fake_execute)

    _run_problem(str(problem), config, budget="10m", run_name="My Run")

    run_dirs = iter_run_dirs(config.paths.runs_dir)
    assert len(run_dirs) == 1
    run_meta = load_run_meta(run_dirs[0])
    assert run_meta.schema_version == 2
    assert run_meta.name == "My Run"
    assert run_meta.problem_ids == ["a"]
    search_dir = executed[0][1]
    assert search_dir == run_dirs[0] / "searches" / "a"
    search_meta = load_search_meta(search_dir)
    assert search_meta.schema_version == 2
    assert search_meta.run_id == run_meta.run_id
    assert search_meta.problem_id == "a"
    assert search_meta.budget_s == 600
    assert (search_dir / "candidates").is_dir()
    assert (search_dir / "best").is_dir()


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
    run_ids = [cmd[cmd.index("--run-id") + 1] for cmd, _ in calls]
    assert len(set(run_ids)) == 1
    assert all(cmd[cmd.index("--run-name") + 1] == "Demo" for cmd, _ in calls)
    assert all("--backend" in cmd and "dummy" in cmd for cmd, _ in calls)
    assert all("--budget" in cmd and "10m" in cmd for cmd, _ in calls)
    run_dirs = iter_run_dirs(config.paths.runs_dir)
    assert len(run_dirs) == 1
    meta = load_run_meta(run_dirs[0])
    assert meta.name == "Demo"
    assert meta.kind == "suite"
    # suite logs live inside the run dir
    assert (run_dirs[0] / "logs").is_dir()


def test_run_suite_rejects_duplicate_problem_ids(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    write_problem(root, "a")
    suite = root / "suite.yaml"
    suite.write_text("suite_id: demo\nproblems:\n  - a\n  - a\n")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"

    with pytest.raises(typer.BadParameter, match="duplicate problem ids"):
        _run_suite(str(suite), config, budget=None, backend=None, model=None, holdout=True, name=None)


def make_search(runs_dir, run_id, search_id, run_kind="problem"):
    run_dir = runs_dir / run_id
    if load_run_meta(run_dir) is None:
        write_run_meta(
            run_dir,
            RunMeta(run_id=run_id, name=run_id, kind=run_kind, target="x", problem_ids=[]),
        )
    search_dir = run_dir / "searches" / search_id
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id=search_id, run_id=run_id, problem="p", problem_id=search_id,
            backend="dummy", model="m", metric="score",
        ),
    )
    return search_dir


def test_search_meta_defaults_for_pre_policy_files(tmp_path):
    """search.yaml written before the policy fields existed loads with greedy
    defaults — no SCHEMA_VERSION bump, no invisible runs."""
    import yaml

    search_dir = tmp_path / "s"
    search_dir.mkdir()
    (search_dir / "search.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2, "search_id": "a", "run_id": "r",
                "problem": "p", "problem_id": "a", "backend": "dummy",
                "model": "m", "metric": "score",
            }
        )
    )
    meta = load_search_meta(search_dir)
    assert meta.policy == "greedy"
    assert meta.policy_params == {}
    assert meta.routing == {}


def test_create_search_persists_policy_and_routing(task, config, tmp_path):
    from hillclimb.api import create_search
    from hillclimb.config import RouteConfig

    config.search.policy = "greedy"
    config.search.policy_params = {"beam": 3}
    config.routing = {"draft": RouteConfig(model="opus-4.8")}
    search_dir = create_search(config, task, tmp_path / "runs" / "r1", "r1", total_s=600)
    meta = load_search_meta(search_dir)
    assert meta.policy == "greedy"
    assert meta.policy_params == {"beam": 3}
    assert meta.routing == {"draft": {"model": "opus-4.8"}}


def test_resume_restores_policy_and_routing(config, tmp_path, monkeypatch):
    """A search resumes under the policy/routing it started with, regardless
    of what the live workspace config says."""
    from hillclimb.cli import resume

    config.paths.runs_dir = tmp_path / "runs"
    run_dir = config.paths.runs_dir / "run-1"
    write_run_meta(
        run_dir,
        RunMeta(run_id="run-1", name="run-1", kind="problem", target="x", problem_ids=["a"]),
    )
    search_dir = run_dir / "searches" / "a"
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id="a", run_id="run-1", problem="p", problem_id="a",
            backend="dummy", model="m", metric="score", budget_s=600,
            policy="scripted", policy_params={"depth": 2},
            routing={"draft": {"model": "opus-4.8"}},
        ),
    )
    (search_dir / "journal.jsonl").write_text("")
    captured = {}
    monkeypatch.setattr(
        "hillclimb.cli.load_config", lambda **kw: config.model_copy(deep=True)
    )
    monkeypatch.setattr("hillclimb.cli.load_problem", lambda *a, **k: object())
    monkeypatch.setattr(
        "hillclimb.cli._execute",
        lambda config_arg, *a, **k: captured.setdefault("config", config_arg),
    )

    resume("run-1/a")

    restored = captured["config"]
    assert restored.search.policy == "scripted"
    assert restored.search.policy_params == {"depth": 2}
    assert restored.routing["draft"].model == "opus-4.8"
    assert restored.routing["draft"].backend is None


def test_resolve_search_dir_exact_and_bare_run(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    s1 = make_search(config.paths.runs_dir, "run-1", "a")

    assert resolve_search_dir(config, "run-1/a") == s1
    assert resolve_search_dir(config, "run-1") == s1  # only search in the run


def test_resolve_search_dir_ambiguous_run_lists_choices(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    make_search(config.paths.runs_dir, "run-1", "a", run_kind="suite")
    make_search(config.paths.runs_dir, "run-1", "b", run_kind="suite")

    with pytest.raises(typer.BadParameter, match="run-1/a"):
        resolve_search_dir(config, "run-1")


def test_resolve_search_dir_latest_prefers_recent_status(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    older = make_search(config.paths.runs_dir, "run-1", "a")
    newer = make_search(config.paths.runs_dir, "run-2", "b")
    (older / "status.json").write_text("{}")
    (newer / "status.json").write_text("{}")
    import os
    past = 1_000_000_000
    os.utime(older / "status.json", (past, past))

    assert resolve_search_dir(config, "latest") == newer


def test_resolve_search_dir_skips_v1_layout(config, tmp_path):
    """Old flat-layout dirs contain a run.yaml without schema_version and must
    be invisible to resolution."""
    config.paths.runs_dir = tmp_path / "runs"
    old = config.paths.runs_dir / "20260101-000000-legacy"
    old.mkdir(parents=True)
    (old / "run.yaml").write_text("run_id: legacy\nproblem_id: x\nbudget_s: 60\n")

    with pytest.raises(typer.BadParameter, match="No searches found"):
        resolve_search_dir(config, "latest")
    with pytest.raises(typer.BadParameter, match="No run named"):
        resolve_search_dir(config, "20260101-000000-legacy")


def test_spent_seconds_sums_agent_and_trial_time(tmp_path):
    from hillclimb.candidate import BackendInfo, Candidate, Trial
    from hillclimb.api import spent_seconds
    from hillclimb.journal import Journal

    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(
        Candidate(
            candidate_id="c001",
            operator="draft",
            backend=BackendInfo(agent_duration_s=100.0),
            trials=[Trial(duration_s=30.0), Trial(duration_s=20.0)],
        )
    )
    journal.candidate_result(
        Candidate(candidate_id="c002", operator="draft")  # no agent time, no trials
    )
    assert spent_seconds(journal) == 150.0


def test_resolve_search_dir_unknown_refs(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    make_search(config.paths.runs_dir, "run-1", "a")

    with pytest.raises(typer.BadParameter, match="No search at"):
        resolve_search_dir(config, "run-1/nope")
    with pytest.raises(typer.BadParameter, match="No run named"):
        resolve_search_dir(config, "nope")


def test_knowledge_live_renders_run_cards(config, tmp_path, monkeypatch, capsys):
    from hillclimb.cli import knowledge_live
    from hillclimb.knowledge import KnowledgeCard, write_live_card
    from hillclimb.workspace import create_run_dir

    config.paths.runs_dir = tmp_path / "runs"
    run_dir = create_run_dir(config.paths.runs_dir, "r1")
    write_run_meta(
        run_dir,
        RunMeta(run_id="r1", name="r1", kind="suite", target="t", problem_ids=["solar"]),
    )
    write_live_card(
        run_dir,
        KnowledgeCard(
            problem_id="gefcom2014-solar", family="gefcom2014", run_ref="r1/solar",
            metric="PinballLoss", n_candidates=3, n_ok=2, selected_val=0.014,
            top_approaches=[dict(candidate_id="c002", operator="draft",
                                 summary="clearsky trick")],
        ),
        "gefcom2014-solar",
    )
    monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)

    knowledge_live("r1")
    out = capsys.readouterr().out
    assert "r1/solar" in out and "clearsky trick" in out and "CONCURRENTLY" in out

    with pytest.raises(typer.BadParameter, match="No run named"):
        knowledge_live("nope")


def test_show_renders_report_diff_and_notes(config, tmp_path, monkeypatch, capsys):
    from hillclimb.candidate import Candidate, Trial
    from hillclimb.cli import show
    from hillclimb.journal import Journal

    config.paths.runs_dir = tmp_path / "runs"
    search_dir = make_search(config.paths.runs_dir, "run-1", "a")
    parent_ws = tmp_path / "ws" / "c001"
    child_ws = tmp_path / "ws" / "c002"
    for ws in (parent_ws, child_ws):
        ws.mkdir(parents=True)
    (parent_ws / "solution.py").write_text("model = 'gbm'\n")
    (child_ws / "solution.py").write_text("model = 'gbm with lags'\n")
    (child_ws / "notes.md").write_text("added lag features\n")
    (child_ws / "exec_stdout.log").write_text("val_score: 0.7\n")

    report = {
        "version": 1, "split": "validation", "objective": "score",
        "lower_is_better": False,
        "overall": {"score": 0.7, "n_origins": 3, "n_scored": 30},
        "zones": [
            {"zone": "z1", "score": 0.6, "n_origins": 1, "n_scored": 10},
            {"zone": "z2", "score": 0.9, "n_origins": 2, "n_scored": 20},
        ],
    }
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(
        Candidate(candidate_id="c001", operator="draft", status="ok",
                  workspace=str(parent_ws), trials=[Trial(val_score=0.6)])
    )
    journal.candidate_result(
        Candidate(candidate_id="c002", operator="improve", parent_id="c001", status="ok",
                  workspace=str(child_ws), summary="added lag features",
                  trials=[Trial(val_score=0.7, report=report)])
    )
    monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)

    show("run-1/a", "c002")
    out = capsys.readouterr().out
    assert "c002  improve" in out and "<- c001" in out
    assert "# Evaluation breakdown (validation split)" in out
    assert "z1" in out
    assert "+++ c002/solution.py" in out and "gbm with lags" in out
    assert "added lag features" in out
    assert "val_score: 0.7" in out

    # report-less candidate degrades gracefully
    show("run-1/a", "c001")
    assert "(no evaluation report" in capsys.readouterr().out

    with pytest.raises(typer.BadParameter, match="No candidate"):
        show("run-1/a", "c999")
