from __future__ import annotations

from tests.factories import trial as mk_trial, name_climber

import pytest
import typer
from typer.testing import CliRunner

from hillclimb.cli import BANNER_LINES, LOGO_LINES, WORDMARK_LINES, common
from hillclimb.cli.common import resolve_search_dir
from hillclimb.cli.run import _run_problem, _run_suite
from hillclimb.cli import main as cli_main
from hillclimb.harness.run import (
    SCHEMA_VERSION,
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

    monkeypatch.setattr("hillclimb.cli.run._execute", fake_execute)

    _run_problem(str(problem), config, budget="10m", run_name="My Run")

    run_dirs = iter_run_dirs(config.paths.runs_dir)
    assert len(run_dirs) == 1
    run_meta = load_run_meta(run_dirs[0])
    assert run_meta.schema_version == SCHEMA_VERSION
    assert run_meta.name == "My Run"
    assert run_meta.problem_ids == ["a"]
    search_dir = executed[0][1]
    assert search_dir == run_dirs[0] / "searches" / "a"
    search_meta = load_search_meta(search_dir)
    assert search_meta.schema_version == SCHEMA_VERSION
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
        agent="dummy",
        model=None,
        holdout=True,
        name="Demo",
    )

    assert len(calls) == 2
    assert calls[0][0][2:4] == ["hillclimb.cli", "run"]
    run_ids = [cmd[cmd.index("--run-id") + 1] for cmd, _ in calls]
    assert len(set(run_ids)) == 1
    assert all(cmd[cmd.index("--run-name") + 1] == "Demo" for cmd, _ in calls)
    assert all("--agent" in cmd and "dummy" in cmd for cmd, _ in calls)
    assert all("--budget" in cmd and "10m" in cmd for cmd, _ in calls)
    run_dirs = iter_run_dirs(config.paths.runs_dir)
    assert len(run_dirs) == 1
    meta = load_run_meta(run_dirs[0])
    assert meta.name == "Demo"
    assert meta.kind == "suite"
    # suite logs live inside the run dir
    assert (run_dirs[0] / "logs").is_dir()
    # and so does the run's own spec: the entries as launched, rerunnable
    from hillclimb.problem import load_suite

    rerun = load_suite(run_dirs[0] / "spec.yaml", config)
    assert [(e.target, e.budget, e.agent) for e in rerun.problems] == [
        (str(root / "a"), "10m", "dummy"), (str(root / "b"), "10m", "dummy"),
    ]
    assert "# launched from:" in (run_dirs[0] / "spec.yaml").read_text()


def test_run_suite_hands_each_child_the_climber_its_entry_defines(config, tmp_path, monkeypatch):
    """The spec defines the climber per search: a child whose entry carries
    a block gets it (before the entry's `set` pairs), the run's own spec
    records every search's FULL block, and `--climber` on the command line
    overrides the spec for every entry."""
    import json

    import yaml

    root = tmp_path / "problems"
    for name in ("a", "b"):
        write_problem(root, name)
    (root / "mine.py").write_text(
        "class Mine:\n    def propose(self, view):\n        return None\n"
        "    def observe(self, view, candidate):\n        pass\n"
    )
    suite = root / "suite.yaml"
    suite.write_text(yaml.safe_dump({"problems": [
        {"target": "a", "climber": {"policy": "mine.py", "params": {"k": 1}}, "set": ["climber.params.k=2"]},
        "b",
    ]}))
    config.paths.problems_dir = root
    config.paths.runs_dir = tmp_path / "runs"
    config.apply_overrides({"climber": "openevolve"})  # the folder's default, for the entry that names none
    calls = []

    class DummyProc:
        pid = 123

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kwargs: calls.append(cmd) or DummyProc())

    _run_suite(str(suite), config, budget="10m", agent="dummy", model=None, holdout=True, name="Demo")

    first, second = calls
    sets = [first[i + 1] for i, word in enumerate(first) if word == "--set"]
    assert json.loads(sets[0].removeprefix("climber="))["policy"] == str(root / "mine.py")  # relative to the spec
    assert sets[1:] == ["climber.params.k=2"] and "--climber" not in first
    assert "--set" not in second and "--climber" not in second  # no climber named: the child reads the folder's
    written = yaml.safe_load((iter_run_dirs(config.paths.runs_dir)[0] / "spec.yaml").read_text())["problems"]
    assert written[0]["climber"]["policy"] == str(root / "mine.py") and written[0]["climber"]["params"] == {"k": 1}
    assert written[1]["climber"]["select"] == "map-elites"  # the full block, though the entry named none

    calls.clear()
    _run_suite(str(suite), config, budget="10m", agent="dummy", model=None, holdout=True, name="Demo2", climber="gepa")
    assert all(cmd[cmd.index("--climber") + 1] == "gepa" for cmd in calls)
    assert not any(pair.startswith("climber={") for cmd in calls for pair in cmd)


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

    _run_suite(str(suite), config, budget=None, agent="dummy", model=None,
               holdout=True, name="Demo", learning=False)
    assert all("--no-learning" in cmd for cmd in calls)

    calls.clear()
    _run_suite(str(suite), config, budget=None, agent="dummy", model=None,
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
    _run_suite(str(suite), config, budget=None, agent=None, model=None, holdout=True, name=None)
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
            agent="dummy", model="m", metric="score",
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
                "problem": "p", "problem_id": "a", "agent": "dummy",
                "model": "m", "metric": "score",
            }
        )
    )
    meta = load_search_meta(search_dir)
    assert meta.climber == "greedy"
    assert meta.climber_spec == {} and meta.climber_ref is None  # nothing recorded: the default
    assert meta.routing == {}


def test_create_search_persists_policy_and_routing(task, config, tmp_path):
    from hillclimb.api import create_search
    from hillclimb.config import RouteConfig

    name_climber(config, "greedy")
    config.climber.params = {"num_drafts": 2}
    config.routing = {"draft": RouteConfig(model="opus-4.8")}
    search_dir = create_search(config, task, tmp_path / "runs" / "r1", "r1", total_s=600)
    meta = load_search_meta(search_dir)
    assert meta.climber == "greedy" and meta.schema_version == SCHEMA_VERSION
    assert meta.climber_spec["policy"] == "greedy" and meta.climber_spec["params"] == {"num_drafts": 2}
    assert meta.climber_ref is None  # only a pre-0.6 record names its climber by reference
    assert meta.routing == {"draft": {"model": "opus-4.8"}}


def test_resume_restores_policy_and_routing(config, tmp_path, monkeypatch):
    """A search resumes under the policy/routing it started with, regardless
    of what the live config says. (This one predates climber snapshots: the
    record is all there is.)"""
    from hillclimb.cli.run import resume

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
            agent="dummy", model="m", metric="score", budget_s=600,
            policy="openevolve", policy_params={"num_drafts": 2},
            routing={"draft": {"model": "opus-4.8"}},
        ),
    )
    (search_dir / "journal.jsonl").write_text("")
    captured = {}
    monkeypatch.setattr(
        "hillclimb.cli.common.load_config", lambda **kw: config.model_copy(deep=True)
    )
    monkeypatch.setattr("hillclimb.cli.run.load_problem", lambda *a, **k: object())
    monkeypatch.setattr(
        "hillclimb.cli.run._execute",
        lambda config_arg, *a, **k: captured.setdefault("config", config_arg),
    )

    resume("run-1/a")

    restored = captured["config"]
    assert (restored.climber.label, restored.climber.select) == ("openevolve", "map-elites")
    assert restored.climber.params == {"ensemble": False, "tune_budget": 0, "num_drafts": 2}
    assert restored.routing["draft"].model == "opus-4.8"
    assert restored.routing["draft"].agent is None


def test_resume_all_spawns_only_resumable_searches(config, tmp_path, monkeypatch):
    """`resume --all` restarts every parked/stopped/crashed search detached
    and leaves running/done ones alone."""
    import os

    from hillclimb.cli.run import resume
    from hillclimb.harness.status import SearchStatus, write_status

    config.paths.runs_dir = tmp_path / "runs"
    states = {"a": "stopped", "b": "parked", "c": "done", "d": "running"}
    for search_id, state in states.items():
        search_dir = make_search(config.paths.runs_dir, "run-1", search_id)
        pid = os.getpid() if state == "running" else None
        write_status(search_dir, SearchStatus(search_id=search_id, run_id="run-1", state=state, pid=pid))
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config.model_copy(deep=True))
    spawned = []
    monkeypatch.setattr(
        "hillclimb.cli.run._spawn_resume",
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
    from hillclimb.harness.candidate import AgentInfo, Candidate
    from hillclimb.api import spent_seconds
    from hillclimb.harness.journal import Journal

    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(
        Candidate(
            candidate_id="c001",
            operator="draft",
            agent=AgentInfo(agent_duration_s=100.0),
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
    from hillclimb.cli.knowledge import knowledge_live
    from hillclimb.modules.memory.knowledge import KnowledgeCard, write_live_card
    from hillclimb.harness.dirs import create_run_dir

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
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)

    knowledge_live("r1")
    out = capsys.readouterr().out
    assert "r1/solar" in out and "clearsky trick" in out and "CONCURRENTLY" in out

    with pytest.raises(typer.BadParameter, match="No run named"):
        knowledge_live("nope")


def test_show_lists_trials_with_their_params(config, tmp_path, monkeypatch, capsys):
    from hillclimb.harness.candidate import Candidate
    from hillclimb.cli.views import show
    from hillclimb.harness.journal import Journal

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
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)

    show("run-1/a", "c001")
    out = capsys.readouterr().out
    assert 'trial 0: val=0.5  params={"k": 1}' in out
    assert 'trial 1*: val=0.61  params={"k": 4}  holdout=0.55' in out
    assert "  replicate 1: val=0.62  seed=1" in out
    assert "tunable: yes" in out


def test_show_renders_report_diff_and_notes(config, tmp_path, monkeypatch, capsys):
    from hillclimb.harness.candidate import Candidate
    from hillclimb.cli.views import show
    from hillclimb.harness.journal import Journal

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
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)

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
    listed = {line.split()[1] for line in out.splitlines() if line.startswith("│ ") and len(line.split()) > 1}
    for command in ("init", "run", "watch", "experiment"):
        assert command in listed
    for command in ("status", "knowledge", "tree2"):  # still runs, just not on the first screen
        assert command not in listed
    assert "more commands: hillclimb --help --all" in out


def test_help_all_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help", "--all"])
    out = capsys.readouterr().out
    listed = {line.split()[1] for line in out.splitlines() if line.startswith("│ ") and len(line.split()) > 1}
    assert {"status", "knowledge", "tree2", "run"} <= listed
    assert "--help --all" not in out


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


def test_legacy_parallel_agents_key_maps_to_parallel_agents():
    from hillclimb.config import Config
    from hillclimb.problem import SuiteEntry

    old = Config.model_validate({"search": {"parallel_agents": 4, "machine_max_agents": 5}})
    assert old.concurrency.parallel_agents == 4
    assert SuiteEntry(target="x", parallel_agents=2).parallel_agents == 2
    assert old.concurrency.effective_machine_max_agents() == 5


def test_machine_max_agents_defaults_to_cores_minus_two_capped(monkeypatch):
    from hillclimb.config import ConcurrencyConfig

    monkeypatch.setattr("os.cpu_count", lambda: 10)
    assert ConcurrencyConfig().effective_machine_max_agents() == 8
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    assert ConcurrencyConfig().effective_machine_max_agents() == 8
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert ConcurrencyConfig().effective_machine_max_agents() == 2
    assert ConcurrencyConfig(machine_max_agents=0).effective_machine_max_agents() == 0  # off


def test_orphan_engines_are_those_whose_dir_is_gone(tmp_path, monkeypatch):
    from hillclimb.harness import orphans

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

    from hillclimb.harness.orphans import Engine, kill_engines

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
    from hillclimb.harness.orphans import _descendants

    listing = "1 0\n10 1\n11 10\n12 11\n13 10\n20 1\n"
    assert sorted(_descendants(10, listing)) == [11, 12, 13]
    assert _descendants(20, listing) == []


def test_stop_all_without_a_dir_reaps_orphaned_engines(tmp_path, monkeypatch, capsys):
    from hillclimb import cli as cli_module
    from hillclimb.harness.orphans import Engine

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    gone = tmp_path / "deleted" / "hillclimb"
    killed = []
    monkeypatch.setattr("hillclimb.harness.orphans.orphan_engines", lambda: [Engine(pid=7, pgid=7, hillclimb_dir=gone)])
    monkeypatch.setattr("hillclimb.harness.orphans.kill_engines", lambda engines, grace_s=5.0: killed.extend(engines) or [])
    with pytest.raises(SystemExit) as exc:
        cli_main(["stop", "--all"])
    assert exc.value.code == 0
    assert [e.pid for e in killed] == [7]
    assert "pid 7" in capsys.readouterr().out

    # without --all the original error stands; with --all and no orphans it is reported
    monkeypatch.setattr("hillclimb.harness.orphans.orphan_engines", lambda: [])
    with pytest.raises(SystemExit) as exc:
        cli_main(["kill", "--all"])
    assert exc.value.code == 1
    assert "No orphaned engines" in capsys.readouterr().out


def test_engines_for_matches_only_this_hillclimb_dir(tmp_path, monkeypatch):
    from hillclimb.harness import orphans

    mine = tmp_path / "a" / "hillclimb"
    other = tmp_path / "b" / "hillclimb"
    mine.mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "a")
    envs = {21: mine, 22: other, 23: None, 24: tmp_path / "link" / "hillclimb"}
    monkeypatch.setattr(orphans, "_environ_dir", lambda pid: envs[pid])
    listing = "\n".join(f"{pid} {pid} /venv/bin/python3 -m hillclimb.cli run x" for pid in envs)
    engines = orphans.live_engines(listing)
    assert [e.pid for e in orphans.engines_for(mine, engines)] == [21, 24]  # 24: same dir via symlink


def test_reset_kills_this_dirs_engines_and_deletes_what_hillclimb_made(tmp_path, monkeypatch, capsys):
    """The hillclimb dir may be a code repo's root: reset removes
    hillclimb.yaml and hillclimb's folders beside it, and nothing else."""
    from hillclimb.harness.orphans import Engine

    root = tmp_path / "proj"
    root.mkdir()
    (root / "hillclimb.yaml").write_text("")
    for owned in ("runs/r1", "problems/p", "knowledge", "climbers", "experiments"):
        (root / owned).mkdir(parents=True)
    (root / "store.sqlite").write_text("x")
    (root / "store.sqlite-wal").write_text("x")
    mine = ("main.py", ".env", ".gitignore", "src/lib.py", "config.yaml")
    for name in mine:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("keep")
    other = tmp_path / "elsewhere"
    monkeypatch.chdir(root)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    engines = [
        Engine(pid=31, pgid=31, hillclimb_dir=root),
        Engine(pid=32, pgid=32, hillclimb_dir=other),
        Engine(pid=33, pgid=33, hillclimb_dir=None),
    ]
    killed = []
    monkeypatch.setattr("hillclimb.harness.orphans.live_engines", lambda: engines)
    monkeypatch.setattr("hillclimb.harness.orphans.kill_engines", lambda engines, grace_s=5.0: killed.extend(engines) or [])

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
    assert root.exists()
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == sorted(mine)
    assert "pid 33" in out and "left alone" in out


def test_ps_lists_engines_with_their_process_trees(tmp_path, monkeypatch, capsys):
    from hillclimb.harness import orphans

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
    from hillclimb.harness.orphans import is_engine

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
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.journal import Journal

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
            agent="dummy",
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
    from hillclimb.cli.problem import _summit

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
    from hillclimb.cli.problem import _summit

    _summit_search(config.paths.runs_dir, "r1", "p", "p", [0.4], higher_is_better=False)
    _summit_search(config.paths.runs_dir, "r2", "p", "p", [0.2], higher_is_better=False)
    dest = tmp_path / "root"
    dest.mkdir()

    record, candidate, _ = _summit(config, None, dest)

    assert record.run_id == "r2"
    assert candidate.val_score == 0.2


def test_summit_copies_provider_declared_json_artifact(config, tmp_path):
    from hillclimb.cli.problem import _summit

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
    monkeypatch.setattr(common, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["summit", "--to", str(dest)])

    assert result.exit_code == 0
    assert (dest / "solution.py").is_file()
    assert (dest / "submission.json").is_file()


def test_summit_requires_a_problem_when_several_exist(config, tmp_path):
    from hillclimb.cli.problem import _summit

    _summit_search(config.paths.runs_dir, "r1", "a", "a", [0.1])
    _summit_search(config.paths.runs_dir, "r1", "b", "b", [0.2])
    dest = tmp_path / "root"
    dest.mkdir()

    with pytest.raises(typer.BadParameter, match="a, b"):
        _summit(config, None, dest)

    record, _, _ = _summit(config, "b", dest)
    assert record.search_id == "b"


def test_summit_with_no_scored_candidate_explains_itself(config, tmp_path):
    from hillclimb.cli.problem import _summit

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
    from hillclimb.cli.problem import verify

    config.paths.problems_dir = tmp_path / "problems"
    monkeypatch.setattr(
        "hillclimb.cli.common.load_config", lambda **kw: config.model_copy(deep=True)
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

    monkeypatch.setattr(common, "load_config", lambda agent=None, model=None: config)
    monkeypatch.setattr("hillclimb.cli.run.run_fleet", fake_run_fleet)
    return calls


def test_run_detaches_by_default_and_points_at_watch(config, monkeypatch, tmp_path):
    """A plain `hillclimb run` is a one-search detached engine: the terminal
    comes back with the run summary and `hillclimb watch` as the next step;
    `--no-detach` runs it in-process as before."""
    from hillclimb import cli

    calls = _capture_fleet(monkeypatch, config, tmp_path)
    result = CliRunner().invoke(cli.app, ["run", "circle-packing", "--budget", "1m", "--agent", "dummy"])
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["target"] == "circle-packing" and call["parallel_searches"] == 1
    assert "1 search x" in result.output and "running in the background" in result.output
    assert "hillclimb watch" in result.output

    ran = []
    monkeypatch.setattr("hillclimb.cli.run._run_problem", lambda *a, **k: ran.append(a))
    result = CliRunner().invoke(cli.app, ["run", "circle-packing", "--budget", "1m", "--agent", "dummy", "--no-detach"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and len(ran) == 1


def test_run_with_several_policies_launches_a_mixed_fleet(config, monkeypatch, tmp_path):
    from hillclimb import cli
    from hillclimb.api import FleetEngine

    calls = _capture_fleet(monkeypatch, config, tmp_path)
    result = CliRunner().invoke(cli.app, [
        "run", "circle-packing", "--budget", "1m", "--agent", "dummy",
        "--climber", "greedy", "--climber", "openevolve", "--climber", "gepa",
        "--experiment-set", "gepa:concurrency.parallel_agents=1", "--set", "learning.enabled=false",
        "--study", "three-way",
    ])

    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["target"] == "circle-packing" and call["climber"] is None
    assert call["engines"] == [
        FleetEngine(experiment="greedy", climber="greedy"),
        FleetEngine(experiment="openevolve", climber="openevolve"),
        FleetEngine(experiment="gepa", climber="gepa", overrides=("concurrency.parallel_agents=1",)),
    ]
    assert call["study"] == "three-way" and call["overrides"] == ["learning.enabled=false"]
    assert config.climber.policy == "greedy"  # the parent's config is not bent to any one experiment
    assert "3 searches (greedy, openevolve, gepa)" in result.output
    assert "hillclimb experiment report three-way" in result.output


def test_run_mixed_fleet_repeats_every_arm_and_rejects_stray_flags(config, monkeypatch, tmp_path):
    from hillclimb import cli
    calls = _capture_fleet(monkeypatch, config, tmp_path)
    result = CliRunner().invoke(cli.app, [
        "run", "circle-packing", "--climber", "greedy", "--climber", "gepa", "--parallel-searches", "2",
    ])
    assert result.exit_code == 0, result.output
    assert [(e.experiment, e.repeat) for e in calls[0]["engines"]] == [("greedy", 1), ("gepa", 1), ("greedy", 2), ("gepa", 2)]
    assert "hillclimb experiment report 20260910-120000-cp" in result.output

    for extra, message in (
        (["--experiment", "x"], "--experiment/--run-id do not apply"),
        (["--experiment-set", "openevolve:concurrency.parallel_agents=1"], "unknown experiment"),
        (["--experiment-set", "gepa-concurrency.parallel_agents=1"], "EXPERIMENT:KEY=VALUE"),
    ):
        # a wide terminal: rich wraps (and elides) usage errors in narrow boxes
        result = CliRunner().invoke(
            cli.app, ["run", "circle-packing", "--climber", "greedy", "--climber", "gepa", *extra], env={"COLUMNS": "300"}
        )
        assert result.exit_code != 0 and message in result.output, (extra, result.output)
    # a single policy is the classic path; --experiment-set has nothing to attach to
    result = CliRunner().invoke(
        cli.app, ["run", "circle-packing", "--climber", "gepa", "--experiment-set", "gepa:x=1"], env={"COLUMNS": "300"}
    )
    assert result.exit_code != 0 and "needs a mixed fleet" in result.output
    assert len(calls) == 1


def _resumable(config, tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config.model_copy(deep=True))
    monkeypatch.setattr("hillclimb.cli.common.require_sandbox", lambda *a, **k: None)
    monkeypatch.setattr("hillclimb.cli.run.load_problem", lambda *a, **k: object())
    monkeypatch.setattr("hillclimb.cli.run._execute", lambda config_arg, *a, **k: captured.__setitem__("config", config_arg))
    return captured


def test_resume_runs_the_snapshot_and_says_when_the_live_climber_changed(task, config, tmp_path, monkeypatch, capsys):
    """A search resumes as the climber it started as — its snapshot, the
    whole block: policy, params, operators, tuner, memory — whatever the live
    config says by now. An edited live file is said out loud; a deleted one
    changes nothing."""
    from hillclimb.api import create_run, create_search
    from hillclimb.cli.run import resume
    from tests.test_policy import FILE_POLICY

    policy_file = tmp_path / "drafts_only.py"
    policy_file.write_text(FILE_POLICY)
    config.apply_overrides({"climber": {
        "policy": str(policy_file), "params": {"num_drafts": 2}, "operators": ["draft", "debug"],
        "operator_params": {"draft": {"retrieval": False}}, "tuner_params": {"seed": 5}, "memory": "none",
    }})
    run_dir = create_run(config, RunMeta(run_id="run-1", name="run-1", kind="problem", target="x", problem_ids=[task.problem_id]))
    search_dir = create_search(config, task, run_dir, "run-1", 600)
    (search_dir / "journal.jsonl").write_text("")
    config.apply_overrides({"climber": "gepa"})  # the live config has moved on since
    captured = _resumable(config, tmp_path, monkeypatch)

    resume(f"run-1/{search_dir.name}")
    restored = captured["config"].climber
    assert restored.policy == str(search_dir / "climber" / "files" / "drafts_only.py") and restored.loop is None
    assert (restored.params, restored.operators, restored.memory) == ({"num_drafts": 2}, ["draft", "debug"], "none")
    assert restored.operator_params == {"draft": {"retrieval": False}} and restored.tuner_params == {"seed": 5}
    assert "changed since" not in capsys.readouterr().err

    policy_file.write_text(FILE_POLICY + "# edited\n")
    resume(f"run-1/{search_dir.name}")
    assert "changed since the search started" in capsys.readouterr().err
    assert captured["config"].climber.policy == restored.policy  # still the version it started with

    policy_file.unlink()
    resume(f"run-1/{search_dir.name}")  # gone: the snapshot is what runs
    assert captured["config"].climber.policy == restored.policy


def test_resume_of_a_search_without_a_snapshot_needs_the_live_climber(config, tmp_path, monkeypatch, capsys):
    """A search from before snapshots resumes from the live file its record
    names; a missing file is a usage error."""
    from hillclimb.cli.run import resume
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
            agent="dummy", model="m", metric="score", budget_s=600,
            policy=str(policy_file), policy_sha256="0" * 64,
        ),
    )
    (search_dir / "journal.jsonl").write_text("")
    captured = _resumable(config, tmp_path, monkeypatch)

    resume("run-1/a")
    assert captured["config"].climber.policy == str(policy_file)
    assert "predates climber snapshots" in capsys.readouterr().err

    policy_file.unlink()
    with pytest.raises(typer.BadParameter, match="is gone"):
        resume("run-1/a")


def test_engine_lines_speak_in_the_cli_voice():
    """A foreground engine's log lines get the theme's markup by shape —
    clock dim, `word:` lead bold, candidate ids in the path colour, trouble
    warned whole — and nothing in the text itself is read as markup."""
    from hillclimb.cli.common import engine_line

    # rich escapes only what could read as a tag, so the digit-led clock stays bare
    assert engine_line("[59 minutes left] draft (c002)") == "[note][59 minutes left][/] draft ([path]c002[/])"
    assert engine_line("  new selection: c002 val_score=0.0355") == "  [head]new selection:[/] [path]c002[/] val_score=0.0355"
    assert engine_line("learning: 3 prior search card(s) inform this search") == (
        "[head]learning:[/] 3 prior search card(s) inform this search"
    )
    assert engine_line("baseline written: baseline: none shipped") == "[head]baseline written:[/] baseline: none shipped"
    assert engine_line("  worker for c009 crashed: KeyError") == "[warn]  worker for c009 crashed: KeyError[/]"
    # an agent's own words, brackets included, print as written
    assert engine_line("draft (c003): uses [x, y] grid") == "draft ([path]c003[/]): uses \\[x, y] grid"
    assert engine_line("Done.") == "Done."


def test_engine_lines_split_into_a_clock_gutter_and_a_message():
    """On a terminal the foreground log is two columns: the clock (without
    its brackets) and the message, marked up; a line without a clock has an
    empty gutter and keeps its own indent."""
    from hillclimb.cli.common import split_engine_line

    assert split_engine_line("[9:42 left] draft (c001)") == ("9:42 left", "draft ([path]c001[/])")
    assert split_engine_line("[1:05:00 left] tune c003 t1 restarts=7 lr=0.0196") == (
        "1:05:00 left", "tune [path]c003[/] t1 restarts=7 lr=0.0196"
    )
    assert split_engine_line("  new selection: c001 val=0.028911") == (
        "", "  [head]new selection:[/] [path]c001[/] val=0.028911"
    )
    assert split_engine_line("learning: 5 claim(s) distilled") == ("", "[head]learning:[/] 5 claim(s) distilled")


def test_version_flag_prints_the_version(capsys):
    from hillclimb import __version__

    for flag in ("--version", "-V"):
        with pytest.raises(SystemExit) as exc:
            cli_main([flag])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"hillclimb {__version__}"
