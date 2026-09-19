from __future__ import annotations

from tests.factories import trial as mk_trial

import pytest
import typer
from typer.testing import CliRunner

from hillclimb.cli import BANNER_LINES, LOGO_LINES, WORDMARK_LINES, _run_problem, _run_suite, resolve_search_dir
from hillclimb.cli import main as cli_main
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
higher_is_better: true
description: description.md
"""
    )
    (problem / "description.md").write_text(name)
    (problem / "sample_submission.csv").write_text("id,x\n0,0\n")
    verifier = problem / "verifier.sh"
    verifier.write_text('#!/bin/sh\necho 1 > "$HILLCLIMB_RESULT"\n')
    verifier.chmod(0o755)
    return problem


def test_run_problem_creates_run_and_search_metadata(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    problem = write_problem(root, "a")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    executed = []

    def fake_execute(config_arg, problem_arg, search_dir, budget, seed_from=None, knowledge_context=None):
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


def test_run_suite_threads_no_learning_flag(config, tmp_path, monkeypatch):
    root = tmp_path / "problems"
    write_problem(root, "a")
    suite = root / "suite.yaml"
    suite.write_text("suite_id: demo\nproblems:\n  - a\n")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    calls = []

    class DummyProc:
        pid = 123

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: (calls.append(cmd), DummyProc())[1])

    _run_suite(str(suite), config, budget=None, backend="dummy", model=None,
               holdout=True, name="Demo", learning=False)
    assert all("--no-learning" in cmd for cmd in calls)

    calls.clear()
    _run_suite(str(suite), config, budget=None, backend="dummy", model=None,
               holdout=True, name="Demo2", learning=True)
    assert all("--no-learning" not in cmd for cmd in calls)


def test_run_suite_allows_the_same_problem_twice(config, tmp_path, monkeypatch):
    """The problem is an attribute of a search: a suite may list one problem
    several times (two models on it, say) and each entry is its own search."""
    root = tmp_path / "problems"
    write_problem(root, "a")
    suite = root / "suite.yaml"
    suite.write_text("suite_id: demo\nproblems:\n  - a\n  - a\n")
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    calls = []

    class DummyProc:
        pid = 123

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd) or DummyProc())
    _run_suite(str(suite), config, budget=None, backend=None, model=None, holdout=True, name=None)
    assert len(calls) == 2
    meta = load_run_meta(iter_run_dirs(config.paths.runs_dir)[0])
    assert meta.problem_ids == ["a"]


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
    of what the live candidate_dir config says."""
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


def test_resume_all_spawns_only_resumable_searches(config, tmp_path, monkeypatch):
    """`resume --all` restarts every parked/stopped/crashed search detached
    and leaves running/done ones alone."""
    import os

    from hillclimb.cli import resume
    from hillclimb.status import SearchStatus, write_status

    config.paths.runs_dir = tmp_path / "runs"
    states = {"a": "stopped", "b": "parked", "c": "done", "d": "running"}
    for search_id, state in states.items():
        search_dir = make_search(config.paths.runs_dir, "run-1", search_id)
        pid = os.getpid() if state == "running" else None
        write_status(search_dir, SearchStatus(search_id=search_id, run_id="run-1", state=state, pid=pid))
    monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config.model_copy(deep=True))
    spawned = []
    monkeypatch.setattr(
        "hillclimb.cli._spawn_resume",
        lambda cfg, record: spawned.append(record.ref) or (123, tmp_path / "log"),
    )

    resume(all_=True)

    assert sorted(spawned) == ["run-1/a", "run-1/b"]


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
    from hillclimb.candidate import BackendInfo, Candidate
    from hillclimb.api import spent_seconds
    from hillclimb.journal import Journal

    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(
        Candidate(
            candidate_id="c001",
            operator="draft",
            backend=BackendInfo(agent_duration_s=100.0),
            trials=[mk_trial(duration_s=30.0), mk_trial(duration_s=20.0)],
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
    from hillclimb.dirs import create_run_dir

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


def test_show_lists_trials_with_their_params(config, tmp_path, monkeypatch, capsys):
    from hillclimb.candidate import Candidate
    from hillclimb.cli import show
    from hillclimb.journal import Journal

    config.paths.runs_dir = tmp_path / "runs"
    search_dir = make_search(config.paths.runs_dir, "run-1", "a")
    ws = tmp_path / "ws" / "c001"
    ws.mkdir(parents=True)
    (ws / "solution.py").write_text("x = 1\n")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(Candidate(
        candidate_id="c001", operator="draft", status="passing", candidate_dir=str(ws), tunable=True,
        trials=[
            mk_trial(0.5, params={"k": 1}, is_best=False),
            mk_trial(0.6, 0.62, params={"k": 4}, index=1, holdout_score=0.55),
        ],
    ))
    monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config)

    show("run-1/a", "c001")
    out = capsys.readouterr().out
    assert 'trial 0: val=0.5  params={"k": 1}' in out
    assert 'trial 1*: val=0.61  params={"k": 4}  holdout=0.55' in out
    assert "  replicate 1: val=0.62  seed=1" in out
    assert "tunable: yes" in out


def test_show_renders_report_diff_and_notes(config, tmp_path, monkeypatch, capsys):
    from hillclimb.candidate import Candidate
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
        "higher_is_better": True,
        "overall": {"score": 0.7, "n_origins": 3, "n_scored": 30},
        "zones": [
            {"zone": "z1", "score": 0.6, "n_origins": 1, "n_scored": 10},
            {"zone": "z2", "score": 0.9, "n_origins": 2, "n_scored": 20},
        ],
    }
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(
        Candidate(candidate_id="c001", operator="draft", status="passing",
                  candidate_dir=str(parent_ws), trials=[mk_trial(val_score=0.6)])
    )
    journal.candidate_result(
        Candidate(candidate_id="c002", operator="improve", parent_id="c001", status="passing",
                  candidate_dir=str(child_ws), summary="added lag features",
                  trials=[mk_trial(val_score=0.7, report=report)])
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


def test_bare_invocation_prints_banner_and_command_list(capsys):
    """`hillclimb` with no arguments is a request for the menu, not a usage error."""
    with pytest.raises(SystemExit) as exc:
        cli_main([])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert WORDMARK_LINES[0] in out
    assert "Usage: hillclimb" in out
    for command in ("run", "status", "watch", "knowledge", "experiment"):
        assert command in out


def test_help_flags_print_the_banner_too(capsys):
    for flag in ("--help", "-h"):
        with pytest.raises(SystemExit) as exc:
            cli_main([flag])
        assert exc.value.code == 0
        assert WORDMARK_LINES[0] in capsys.readouterr().out


def test_subcommand_help_skips_the_banner(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert WORDMARK_LINES[0] not in out
    assert "Usage: hillclimb run" in out


def test_banner_lines_are_uniform_width():
    assert len({len(line) for line in BANNER_LINES}) == 1
    assert len({len(line) for line in WORDMARK_LINES}) == 1


def test_arrow_logo_closes_its_bottom_shadow():
    assert LOGO_LINES[-2].startswith("██╔╝")
    assert LOGO_LINES[-1].startswith("╚═╝")
    assert len(BANNER_LINES) == len(LOGO_LINES)
    assert BANNER_LINES[-1].rstrip().endswith(LOGO_LINES[-1].rstrip())


def test_wide_terminal_prints_the_mark(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "120")
    with pytest.raises(SystemExit):
        cli_main([])
    assert BANNER_LINES[0] in capsys.readouterr().out


def test_legacy_parallel_agents_key_maps_to_parallel_operators():
    from hillclimb.config import SearchConfig
    from hillclimb.problem import SuiteEntry

    assert SearchConfig(parallel_agents=4).parallel_operators == 4
    assert SuiteEntry(target="x", parallel_agents=2).parallel_operators == 2
    assert SearchConfig(machine_max_agents=5).effective_machine_max_operators() == 5


def test_machine_max_operators_defaults_to_cores_minus_two_capped(monkeypatch):
    from hillclimb.config import SearchConfig

    monkeypatch.setattr("os.cpu_count", lambda: 10)
    assert SearchConfig().effective_machine_max_operators() == 8
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    assert SearchConfig().effective_machine_max_operators() == 8
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert SearchConfig().effective_machine_max_operators() == 2
    assert SearchConfig(machine_max_operators=0).effective_machine_max_operators() == 0  # off


def test_orphan_engines_are_those_whose_dir_is_gone(tmp_path, monkeypatch):
    from hillclimb import orphans

    alive = tmp_path / "hillclimb"
    alive.mkdir()
    envs = {11: alive, 12: tmp_path / "deleted" / "hillclimb", 13: None}
    monkeypatch.setattr(orphans, "_environ_dir", lambda pid: envs[pid])
    listing = (
        "11 11 /venv/bin/python3 -m hillclimb.cli run circle-packing\n"
        "12 12 /venv/bin/python3 -m hillclimb.cli run circle-packing --budget 5m\n"
        "13 13 /venv/bin/python3 -m hillclimb.cli run other\n"
        "14 14 claude -p --output-format stream-json\n"
        "15 15 /usr/bin/python3 -m hillclimb.cli watch\n"
    )
    engines = orphans.live_engines(listing)
    assert [e.pid for e in engines] == [11, 12, 13]
    assert [e.pid for e in orphans.orphan_engines(engines)] == [12]  # unknown env is not an orphan


def test_kill_engines_takes_the_whole_process_group(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    from hillclimb.orphans import Engine, kill_engines

    # a session leader that spawns a detached child (its own session, like the
    # engine's verifiers and agents) and ignores SIGTERM, like a wedged engine
    code = (
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    child_pid = int(proc.stdout.readline())
    forced = kill_engines([Engine(pid=proc.pid, pgid=proc.pid, hillclimb_dir=tmp_path / "gone")], grace_s=0.5)
    assert [e.pid for e in forced] == [proc.pid]
    proc.wait(timeout=5)
    for _ in range(50):  # the grandchild was killed through the group, not reaped by anyone yet
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        try:
            os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("detached grandchild survived the tree kill")


def test_descendants_walks_the_ps_tree():
    from hillclimb.orphans import _descendants

    listing = "1 0\n10 1\n11 10\n12 11\n13 10\n20 1\n"
    assert sorted(_descendants(10, listing)) == [11, 12, 13]
    assert _descendants(20, listing) == []


def test_stop_all_without_a_dir_reaps_orphaned_engines(tmp_path, monkeypatch, capsys):
    from hillclimb import cli as cli_module
    from hillclimb.orphans import Engine

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)
    gone = tmp_path / "deleted" / "hillclimb"
    killed = []
    monkeypatch.setattr("hillclimb.orphans.orphan_engines", lambda: [Engine(pid=7, pgid=7, hillclimb_dir=gone)])
    monkeypatch.setattr("hillclimb.orphans.kill_engines", lambda engines, grace_s=5.0: killed.extend(engines) or [])
    with pytest.raises(SystemExit) as exc:
        cli_main(["stop", "--all"])
    assert exc.value.code == 0
    assert [e.pid for e in killed] == [7]
    assert "pid 7" in capsys.readouterr().out

    # without --all the original error stands; with --all and no orphans it is reported
    monkeypatch.setattr("hillclimb.orphans.orphan_engines", lambda: [])
    with pytest.raises(SystemExit) as exc:
        cli_main(["kill", "--all"])
    assert exc.value.code == 1
    assert "No orphaned engines" in capsys.readouterr().out


def test_engines_for_matches_only_this_hillclimb_dir(tmp_path, monkeypatch):
    from hillclimb import orphans

    mine = tmp_path / "a" / "hillclimb"
    other = tmp_path / "b" / "hillclimb"
    mine.mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "a")
    envs = {21: mine, 22: other, 23: None, 24: tmp_path / "link" / "hillclimb"}
    monkeypatch.setattr(orphans, "_environ_dir", lambda pid: envs[pid])
    listing = "\n".join(f"{pid} {pid} /venv/bin/python3 -m hillclimb.cli run x" for pid in envs)
    engines = orphans.live_engines(listing)
    assert [e.pid for e in orphans.engines_for(mine, engines)] == [21, 24]  # 24: same dir via symlink


def test_reset_kills_this_dirs_engines_and_deletes_it(tmp_path, monkeypatch, capsys):
    from hillclimb.orphans import Engine

    root = tmp_path / "hillclimb"
    root.mkdir()
    (root / "config.yaml").write_text("")
    (root / "runs").mkdir()
    other = tmp_path / "elsewhere" / "hillclimb"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)
    engines = [
        Engine(pid=31, pgid=31, hillclimb_dir=root),
        Engine(pid=32, pgid=32, hillclimb_dir=other),
        Engine(pid=33, pgid=33, hillclimb_dir=None),
    ]
    killed = []
    monkeypatch.setattr("hillclimb.orphans.live_engines", lambda: engines)
    monkeypatch.setattr("hillclimb.orphans.kill_engines", lambda engines, grace_s=5.0: killed.extend(engines) or [])

    # without --yes a declined prompt aborts and deletes nothing
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    with pytest.raises(SystemExit) as exc:
        cli_main(["reset"])
    assert exc.value.code == 1
    assert root.exists() and killed == []

    with pytest.raises(SystemExit) as exc:
        cli_main(["reset", "--yes"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert [e.pid for e in killed] == [31]  # not the other folder's engine, not the unreadable one
    assert not root.exists()
    assert "pid 33" in out and "left alone" in out


def test_ps_lists_engines_with_their_process_trees(tmp_path, monkeypatch, capsys):
    from hillclimb import orphans

    listing = (
        "100 1 100 0.5 40000 05:00 /venv/bin/python3 -m hillclimb.cli run circle-packing --budget 5m\n"
        "101 100 101 1.0 200000 04:00 claude -p --output-format stream-json --model sonnet\n"
        "102 100 102 95.0 60000 00:10 /bin/bash verifier.sh\n"
        "103 102 102 90.0 50000 00:09 /venv/bin/python3 solution.py\n"
        "200 1 200 0.0 3000 1-02:00:00 /usr/bin/python3 -m hillclimb.cli watch\n"
    )
    monkeypatch.setattr(orphans, "_ps", lambda args: listing)
    monkeypatch.setattr(orphans, "_environ_dir", lambda pid: tmp_path / "gone" / "hillclimb")
    with pytest.raises(SystemExit) as exc:
        cli_main(["ps"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "engine pid 100" in out and "[orphan: dir deleted]" in out
    assert "agent    pid 101" in out and "verifier pid 102" in out and "child    pid 103" in out
    assert "1 engine(s), 4 processes" in out
    assert "watch" not in out

    monkeypatch.setattr(orphans, "_ps", lambda args: "")
    with pytest.raises(SystemExit):
        cli_main(["ps"])
    assert "No hillclimb engines running." in capsys.readouterr().out


def test_is_engine_matches_the_launcher_argv_only():
    from hillclimb.orphans import is_engine

    assert is_engine("/venv/bin/python3 -m hillclimb.cli run circle-packing")
    assert is_engine("/venv/bin/python3 -m hillclimb.cli resume run-1/search-1")
    assert not is_engine("/bin/zsh -c 'grep hillclimb.cli run'")
    assert not is_engine("/venv/bin/python3 -m hillclimb.cli watch")


# --- summit -----------------------------------------------------------------


def _summit_search(
    runs_dir,
    run_id,
    search_id,
    problem_id,
    scores,
    higher_is_better=True,
    output_artifacts=None,
):
    """A finished-looking search: metadata, a journal of scored drafts, and a
    best/ dir stamped with its own address so tests can see whose files won."""
    from hillclimb.candidate import Candidate
    from hillclimb.journal import Journal

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(
        run_dir,
        RunMeta(run_id=run_id, name=run_id, kind="problem", target=problem_id, problem_ids=[problem_id]),
    )
    search_dir = run_dir / "searches" / search_id
    (search_dir / "best").mkdir(parents=True)
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id=search_id,
            run_id=run_id,
            problem=problem_id,
            problem_id=problem_id,
            problem_key=problem_id,
            backend="dummy",
            model="",
            metric="score",
            higher_is_better=higher_is_better,
            output_artifacts=output_artifacts or ["submission.csv"],
        ),
    )
    journal = Journal(search_dir / "journal.jsonl")
    for i, score in enumerate(scores):
        journal.candidate_result(
            Candidate(
                candidate_id=f"c{i:03d}",
                operator="draft",
                status="passing",
                trials=[mk_trial(val_score=score, submission_ok=True)],
            )
        )
    (search_dir / "best" / "solution.py").write_text(f"# {run_id}/{search_id}\n")
    for artifact in output_artifacts or ["submission.csv"]:
        contents = f"id\n{run_id}/{search_id}\n" if artifact == "submission.csv" else f"{run_id}/{search_id}\n"
        (search_dir / "best" / artifact).write_text(contents)
    return search_dir


def test_summit_copies_the_best_search_across_runs(config, tmp_path):
    from hillclimb.cli import _summit

    _summit_search(config.paths.runs_dir, "r1", "p", "p", [0.1, 0.3])
    _summit_search(config.paths.runs_dir, "r2", "p", "p", [0.2])
    dest = tmp_path / "root"
    dest.mkdir()

    record, candidate, copied = _summit(config, None, dest)

    assert record.run_id == "r1"
    assert candidate.val_score == 0.3
    assert copied == ["solution.py", "submission.csv"]
    assert (dest / "solution.py").read_text() == "# r1/p\n"
    assert (dest / "submission.csv").read_text() == "id\nr1/p\n"


def test_summit_respects_lower_is_better(config, tmp_path):
    from hillclimb.cli import _summit

    _summit_search(config.paths.runs_dir, "r1", "p", "p", [0.4], higher_is_better=False)
    _summit_search(config.paths.runs_dir, "r2", "p", "p", [0.2], higher_is_better=False)
    dest = tmp_path / "root"
    dest.mkdir()

    record, candidate, _ = _summit(config, None, dest)

    assert record.run_id == "r2"
    assert candidate.val_score == 0.2


def test_summit_copies_provider_declared_json_artifact(config, tmp_path):
    from hillclimb.cli import _summit

    _summit_search(
        config.paths.runs_dir,
        "r1",
        "arena",
        "einsteinarena://toy",
        [0.5],
        output_artifacts=["submission.json"],
    )
    dest = tmp_path / "root"
    dest.mkdir()

    _record, _candidate, copied = _summit(config, None, dest)

    assert copied == ["solution.py", "submission.json"]
    assert (dest / "submission.json").read_text() == "r1/arena\n"
    assert not (dest / "submission.csv").exists()


def test_summit_command_accepts_a_new_destination(config, tmp_path, monkeypatch):
    from hillclimb import cli

    _summit_search(
        config.paths.runs_dir,
        "r1",
        "arena",
        "einsteinarena://toy",
        [0.5],
        output_artifacts=["submission.json"],
    )
    dest = tmp_path / "not-created-yet"
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["summit", "--to", str(dest)])

    assert result.exit_code == 0
    assert (dest / "solution.py").is_file()
    assert (dest / "submission.json").is_file()


def test_summit_requires_a_problem_when_several_exist(config, tmp_path):
    from hillclimb.cli import _summit

    _summit_search(config.paths.runs_dir, "r1", "a", "a", [0.1])
    _summit_search(config.paths.runs_dir, "r1", "b", "b", [0.2])
    dest = tmp_path / "root"
    dest.mkdir()

    with pytest.raises(typer.BadParameter, match="a, b"):
        _summit(config, None, dest)

    record, _, _ = _summit(config, "b", dest)
    assert record.search_id == "b"


def test_summit_with_no_scored_candidate_explains_itself(config, tmp_path):
    from hillclimb.cli import _summit

    _summit_search(config.paths.runs_dir, "r1", "p", "p", [])
    dest = tmp_path / "root"
    dest.mkdir()

    with pytest.raises(typer.BadParameter, match="no scored candidate"):
        _summit(config, None, dest)


# --- hillclimb verify: interface lint ---

VERIFY_BASELINE_OK = (
    'open("submission.csv", "w").write("x\\n0.1\\n0.9\\n")\n'
)
VERIFY_BASELINE_BAD = (
    'open("submission.csv", "w").write("x\\n0.1\\n5.0\\n")\n'
)


def _lintable_problem(tmp_path, baseline_code: str, with_interface: bool = True):
    problem = tmp_path / "problems" / "fmt"
    problem.mkdir(parents=True)
    (problem / "description.md").write_text("format demo")
    verifier = problem / "verifier.sh"
    verifier.write_text(
        '#!/bin/sh\n"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"\n'
        'echo \'{"score": 1.0}\' > "$HILLCLIMB_RESULT"\n'
    )
    verifier.chmod(0o755)
    (problem / "baseline.py").write_text(baseline_code)
    (problem / "problem.yaml").write_text(
        "metric: score\nhigher_is_better: true\nbaseline: baseline.py\n"
    )
    if with_interface:
        (problem / "interface.py").write_text(
            "from hillclimb import spaces\n"
            "output = spaces.Table('submission.csv',"
            " columns={'x': spaces.Float(low=0.0, high=1.0)}, n_rows=2)\n"
        )
    return problem


def _run_verify(config, tmp_path, monkeypatch):
    from hillclimb.cli import verify

    config.paths.problems_dir = tmp_path / "problems"
    monkeypatch.setattr(
        "hillclimb.cli.load_config", lambda **kw: config.model_copy(deep=True)
    )
    verify("fmt", solution=None, repeat=1, holdout=False)


def test_verify_lints_declared_interface(config, tmp_path, monkeypatch, capsys):
    _lintable_problem(tmp_path, VERIFY_BASELINE_OK)
    _run_verify(config, tmp_path, monkeypatch)
    assert "interface: OK" in capsys.readouterr().out


def test_verify_reports_interface_violations(config, tmp_path, monkeypatch, capsys):
    _lintable_problem(tmp_path, VERIFY_BASELINE_BAD)
    with pytest.raises(typer.Exit):
        _run_verify(config, tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert "interface:" in err and "above high=1" in err


def test_verify_without_interface_stays_silent(config, tmp_path, monkeypatch, capsys):
    _lintable_problem(tmp_path, VERIFY_BASELINE_OK, with_interface=False)
    _run_verify(config, tmp_path, monkeypatch)
    assert "interface:" not in capsys.readouterr().out


def test_verify_runs_declared_unit_tests(config, tmp_path, monkeypatch, capsys):
    problem = _lintable_problem(tmp_path, VERIFY_BASELINE_OK, with_interface=False)
    tests = problem / "tests"
    tests.mkdir()
    (tests / "check.py").write_text("print('suite passed')\n")
    with (problem / "problem.yaml").open("a") as fh:
        fh.write(
            "unit_tests:\n"
            "  root: tests\n"
            '  command: ["{python}", "{tests}/check.py"]\n'
        )

    _run_verify(config, tmp_path, monkeypatch)

    assert "unit tests: PASSING" in capsys.readouterr().out


def test_verify_reports_completed_test_failure(config, tmp_path, monkeypatch, capsys):
    problem = _lintable_problem(tmp_path, VERIFY_BASELINE_OK, with_interface=False)
    tests = problem / "tests"
    tests.mkdir()
    (tests / "check.py").write_text("raise AssertionError('wrong answer')\n")
    with (problem / "problem.yaml").open("a") as fh:
        fh.write(
            "unit_tests:\n"
            "  root: tests\n"
            '  command: ["{python}", "{tests}/check.py"]\n'
        )

    with pytest.raises(typer.Exit):
        _run_verify(config, tmp_path, monkeypatch)

    assert "unit tests: FAILING" in capsys.readouterr().err


def _capture_fleet(monkeypatch, config, tmp_path):
    """`hillclimb run` with the fleet launcher replaced: records run_fleet's
    keyword arguments instead of spawning engines."""
    from types import SimpleNamespace

    from hillclimb import cli

    calls: list[dict] = []

    def fake_run_fleet(target, **kwargs):
        calls.append({"target": target, **kwargs})
        return SimpleNamespace(run_id="20260910-120000-cp", run_dir=tmp_path / "runs" / "20260910-120000-cp")

    monkeypatch.setattr(cli, "load_config", lambda backend=None, model=None: config)
    monkeypatch.setattr(cli, "run_fleet", fake_run_fleet)
    return calls


def test_run_with_several_policies_launches_a_mixed_fleet(config, monkeypatch, tmp_path):
    from hillclimb import cli
    from hillclimb.api import FleetEngine

    calls = _capture_fleet(monkeypatch, config, tmp_path)
    result = CliRunner().invoke(cli.app, [
        "run", "circle-packing", "--budget", "1m", "--backend", "dummy",
        "--policy", "greedy", "--policy", "openevolve", "--policy", "gepa",
        "--arm-set", "gepa:search.parallel_operators=1", "--set", "learning.enabled=false",
        "--experiment", "three-way",
    ])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["target"] == "circle-packing" and call["policy"] is None
    assert call["engines"] == [
        FleetEngine(arm="greedy", policy="greedy"),
        FleetEngine(arm="openevolve", policy="openevolve"),
        FleetEngine(arm="gepa", policy="gepa", overrides=("search.parallel_operators=1",)),
    ]
    assert call["experiment"] == "three-way" and call["overrides"] == ["learning.enabled=false"]
    assert config.search.policy == "greedy"  # the parent's config is not bent to any one arm
    assert "3 searches (greedy, openevolve, gepa)" in result.output
    assert "hillclimb experiment report three-way" in result.output


def test_run_mixed_fleet_repeats_every_arm_and_rejects_stray_flags(config, monkeypatch, tmp_path):
    from hillclimb import cli
    calls = _capture_fleet(monkeypatch, config, tmp_path)
    result = CliRunner().invoke(cli.app, [
        "run", "circle-packing", "--policy", "greedy", "--policy", "gepa", "--parallel-searches", "2",
    ])
    assert result.exit_code == 0, result.output
    assert [(e.arm, e.repeat) for e in calls[0]["engines"]] == [("greedy", 1), ("gepa", 1), ("greedy", 2), ("gepa", 2)]
    assert "hillclimb experiment report 20260910-120000-cp" in result.output

    for extra, message in (
        (["--arm", "x"], "--arm/--run-id do not apply"),
        (["--arm-set", "openevolve:search.parallel_operators=1"], "unknown arm"),
        (["--arm-set", "gepa-search.parallel_operators=1"], "ARM:KEY=VALUE"),
    ):
        # a wide terminal: rich wraps (and elides) usage errors in narrow boxes
        result = CliRunner().invoke(
            cli.app, ["run", "circle-packing", "--policy", "greedy", "--policy", "gepa", *extra], env={"COLUMNS": "300"}
        )
        assert result.exit_code != 0 and message in result.output, (extra, result.output)
    # a single policy is the classic path; --arm-set has nothing to attach to
    result = CliRunner().invoke(
        cli.app, ["run", "circle-packing", "--policy", "gepa", "--arm-set", "gepa:x=1"], env={"COLUMNS": "300"}
    )
    assert result.exit_code != 0 and "needs a mixed fleet" in result.output
    assert len(calls) == 1


def test_resume_warns_when_the_policy_file_changed(config, tmp_path, monkeypatch, capsys):
    """A file policy resumes from its recorded path; a changed hash is said
    out loud (replay may diverge), a missing file is a usage error."""
    from hillclimb.cli import resume
    from tests.test_policy import FILE_POLICY

    policy_file = tmp_path / "drafts_only.py"
    policy_file.write_text(FILE_POLICY)
    config.paths.runs_dir = tmp_path / "runs"
    run_dir = config.paths.runs_dir / "run-1"
    write_run_meta(run_dir, RunMeta(run_id="run-1", name="run-1", kind="problem", target="x", problem_ids=["a"]))
    search_dir = run_dir / "searches" / "a"
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id="a", run_id="run-1", problem="p", problem_id="a",
            backend="dummy", model="m", metric="score", budget_s=600,
            policy=str(policy_file), policy_sha256="0" * 64,
        ),
    )
    (search_dir / "journal.jsonl").write_text("")
    captured = {}
    monkeypatch.setattr("hillclimb.cli.load_config", lambda **kw: config.model_copy(deep=True))
    monkeypatch.setattr("hillclimb.cli.load_problem", lambda *a, **k: object())
    monkeypatch.setattr("hillclimb.cli._execute", lambda config_arg, *a, **k: captured.setdefault("config", config_arg))

    resume("run-1/a")
    assert captured["config"].search.policy == str(policy_file)
    err = capsys.readouterr().err
    assert "changed since the search started" in err and "000000000000 ->" in err

    policy_file.unlink()
    with pytest.raises(typer.BadParameter, match="is gone"):
        resume("run-1/a")
