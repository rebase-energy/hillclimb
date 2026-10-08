from __future__ import annotations

from tests.factories import trial as mk_trial

import os
import sys
from pathlib import Path

import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.chart import climb_curve, climb_curves
from hillclimb.cli import main as cli_main
from hillclimb.catalog import DEMO_PROBLEM_ID, install_problem
from hillclimb.harness.journal import Journal
from hillclimb.problem import load_problem
from hillclimb.harness.run import RunMeta, SearchMeta, write_run_meta, write_search_meta


def test_install_demo_problem_copies_once(tmp_path):
    problems = tmp_path / "problems"
    problem_dir, created = install_problem(problems)
    assert created and problem_dir == problems / DEMO_PROBLEM_ID
    assert (problem_dir / "verifier.sh").exists()
    assert os.access(problem_dir / "verifier.sh", os.X_OK)
    (problem_dir / "description.md").write_text("edited")
    again, created = install_problem(problems)
    assert not created and (again / "description.md").read_text() == "edited"


def test_catalog_problem_loads(tmp_path, config):
    """The catalog IS the repo's problems/: one copy, nothing to keep in sync."""
    config.paths.problems_dir = tmp_path / "problems"
    installed, _ = install_problem(config.paths.problems_dir)
    spec = load_problem(DEMO_PROBLEM_ID, config)
    assert spec.metric_name == "sum-radii" and spec.higher_is_better
    assert spec.requirements_file is not None  # the lean runtime: a first run starts in seconds
    repo = Path("problems") / DEMO_PROBLEM_ID
    for name in ("verifier.sh", "verify.py", "description.md", "sample_submission.csv", "interface.py"):
        assert (installed / name).read_text() == (repo / name).read_text(), name


def _search(runs_dir: Path, run_id: str, name: str, scores: list[tuple[str, float]], lower=False):
    run_dir = runs_dir / run_id
    write_run_meta(run_dir, RunMeta(run_id=run_id, name=name, kind="problem", target="p", problem_ids=["p"]))
    search_dir = run_dir / "searches" / "p"
    search_dir.mkdir(parents=True)
    write_search_meta(search_dir, SearchMeta(
        search_id="p", run_id=run_id, problem="p", problem_id="p", agent="dummy",
        model="m", metric="score", higher_is_better=not lower, started_at="2026-08-22T10:00:00+00:00",
    ))
    journal = Journal(search_dir / "journal.jsonl")
    for index, (finished, score) in enumerate(scores):
        journal.candidate_result(Candidate(
            candidate_id=f"c{index:03d}", operator="draft", status="passing",
            trials=[mk_trial(val_score=score)], finished_at=finished,
        ))
    return search_dir


def test_climb_curve_is_best_so_far_by_tested_candidate(tmp_path):
    search_dir = _search(tmp_path / "runs", "r1", "one", [
        ("2026-08-22T10:02:00+00:00", 1.0),
        ("2026-08-22T10:01:00+00:00", 0.5),  # out of order on disk
        ("2026-08-22T10:05:00+00:00", 0.8),  # worse: the curve stays flat
        ("2026-08-22T10:06:00+00:00", 1.4),
    ])
    curve = climb_curve(search_dir, "one")
    assert curve.xs == [1.0, 2.0, 3.0, 4.0]
    assert curve.ys == [0.5, 1.0, 1.0, 1.4]
    assert curve.best == 1.4


def test_climb_curve_respects_lower_is_better_and_skips_unscored(tmp_path):
    search_dir = _search(tmp_path / "runs", "r1", "one", [
        ("2026-08-22T10:01:00+00:00", 3.0),
        ("2026-08-22T10:02:00+00:00", 2.0),
    ], lower=True)
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(Candidate(candidate_id="c999", operator="debug", status="buggy"))
    assert climb_curve(search_dir).ys == [3.0, 2.0]


def test_climb_curves_groups_by_problem_oldest_first(tmp_path):
    runs = tmp_path / "runs"
    _search(runs, "r2", "demo-2", [("2026-08-22T10:01:00+00:00", 2.0)])
    _search(runs, "r1", "demo-1", [("2026-08-22T10:01:00+00:00", 1.0)])
    # same started_at: sort is stable on the (started_at, dir) key -> r1 then r2
    curves = climb_curves(runs, "p")
    assert [c.label for c in curves] == ["demo-1", "demo-2"]
    assert climb_curves(runs, "other") == []


def test_agent_preflight_names_the_missing_tool(monkeypatch):
    from hillclimb.cli.run import _agent_preflight
    import typer

    monkeypatch.setattr("shutil.which", lambda name: None if name == "claude" else "/usr/bin/x")
    with pytest.raises(typer.Exit):
        _agent_preflight("claude-code")
    _agent_preflight("dummy")


def test_problem_get_installs_the_problem_and_lists_its_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli_main(["problem", "get", DEMO_PROBLEM_ID])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert (tmp_path / "problems" / DEMO_PROBLEM_ID / "verifier.sh").exists()
    assert (tmp_path / "hillclimb.yaml").exists()  # this folder became the hillclimb dir
    assert "Fetched circle-packing" in out and "verifier.sh" in out and "verify.py" in out
    # getting it again never overwrites; `fetch` stays as a deprecated alias
    with pytest.raises(SystemExit):
        cli_main(["fetch", DEMO_PROBLEM_ID])
    captured = capsys.readouterr()
    assert "Already have" in captured.out
    assert "hillclimb problem get" in captured.err
    with pytest.raises(SystemExit) as exc:
        cli_main(["problem", "get", "no-such-problem"])
    assert exc.value.code == 1


def test_problem_list_shows_every_bundled_problem_without_a_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli_main(["problem", "list"])
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert "problem" in output and "metric" in output and "direction" in output and "best known" in output
    assert "budget" not in output  # the clock is the user's `--budget`, not the problem's size
    assert "circle-packing" in output and "sum-radii" in output and "AlphaEvolve" in output
    assert "heilbronn-convex-13" in output and "normalized-min-triangle-area" in output
    assert "knapsack" in output and "mean-percent-of-upper-bound" in output
    assert "hillclimb problem get <problem>" in output
    assert not (tmp_path / "hillclimb.yaml").exists()


def test_run_parallel_searches_spawns_detached_engines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    with pytest.raises(SystemExit):
        cli_main(["problem", "get", DEMO_PROBLEM_ID])
    with pytest.raises(SystemExit):
        cli_main(["climber", "get", "greedy"])  # the engine ships no climber: the folder fetches one
    launched = []

    class FakeProc:
        pid = 4242
        returncode = 0

        def poll(self):  # the launch's startup check: it exited cleanly
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.append(cmd) or FakeProc())
    # the fleet pre-builds the problem's venv; a cold cache must not make this test build one
    monkeypatch.setattr("hillclimb.api.ensure_runtime_venv", lambda *a, **k: Path(sys.executable))
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "run", DEMO_PROBLEM_ID, "--budget", "10m",
            "--parallel-searches", "3", "--parallel-agents", "2", "--agent", "dummy",
        ])
    assert exc.value.code == 0
    assert len(launched) == 3
    from hillclimb.harness.run import iter_run_dirs, load_run_meta

    (run_dir,) = iter_run_dirs(tmp_path / "runs")
    assert load_run_meta(run_dir).name == DEMO_PROBLEM_ID
    # the fleet's own spec: one entry per search, with what it launched with
    import yaml

    from hillclimb.config import Config

    spec = yaml.safe_load((run_dir / "spec.yaml").read_text())
    assert len(spec["problems"]) == 3
    assert spec["problems"][0] == {
        "target": DEMO_PROBLEM_ID, "budget": "10m", "agent": "dummy", "model": "sonnet",
        "climber": Config.load().climber_block(), "parallel_agents": 2,  # the full block, as resolved: the folder's fetched greedy
    }
    assert launched[0][3:] == [
        "run", DEMO_PROBLEM_ID, "--run-id", run_dir.name, "--run-name", DEMO_PROBLEM_ID,
        "--budget", "10m", "--agent", "dummy", "--parallel-agents", "2",
    ]


def test_stop_all_reaches_every_running_search(tmp_path, config, monkeypatch):
    from hillclimb.harness.control import read_commands
    from hillclimb.harness.status import SearchStatus, write_status

    runs = config.paths.runs_dir
    a = _search(runs, "r1", "demo-1", [])
    b = _search(runs, "r2", "demo-2", [])
    for s in (a, b):
        write_status(s, SearchStatus(search_id="p", state="running", pid=os.getpid()))
    monkeypatch.setattr("hillclimb.cli.common.load_config", lambda **kw: config)
    with pytest.raises(SystemExit) as exc:
        cli_main(["stop", "--all"])
    assert exc.value.code == 0
    # immediate by default: in-flight operators are aborted, not waited for
    assert [(c.action, c.graceful) for _, c in read_commands(a)] == [("stop", False)]
    assert [(c.action, c.graceful) for _, c in read_commands(b)] == [("stop", False)]

    with pytest.raises(SystemExit) as exc:
        cli_main(["stop", "--all", "--graceful"])
    assert exc.value.code == 0
    assert [c.graceful for _, c in read_commands(a)] == [False, True]


@pytest.mark.asyncio
async def test_watch_candidates_opens_on_the_search(tmp_path, config):
    from hillclimb.tui.watch import CandidateScreen, WatchApp

    search_dir = _search(config.paths.runs_dir, "r1", "demo-1", [("2026-08-22T10:01:00+00:00", 1.0)])
    app = WatchApp(config, search_dir=search_dir)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, CandidateScreen)
        assert app.screen.search_dir == search_dir
        assert [type(s).__name__ for s in app.screen_stack[1:]] == [
            "RunsScreen", "SearchesScreen", "CandidateScreen",
        ]
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert type(app.screen).__name__ == "SearchesScreen"


def test_search_ids_suffix_within_a_run(tmp_path):
    from hillclimb.harness.dirs import allocate_search_dir

    run_dir = tmp_path / "run"
    ids = [allocate_search_dir(run_dir, "circle-packing").name for _ in range(3)]
    assert ids == ["circle-packing", "circle-packing-2", "circle-packing-3"]
    assert (run_dir / "searches" / "circle-packing-3" / "candidates").is_dir()
    # a different problem in the same run starts its own sequence
    assert allocate_search_dir(run_dir, "tsp-200").name == "tsp-200"


def test_search_meta_problem_key_backfills_like_hillclimb_go():
    from hillclimb.harness.run import SearchMeta

    def meta(**kw):
        base = dict(search_id="s", run_id="r", agent="b", model="m", metric="score")
        return SearchMeta(**{**base, **kw})

    assert meta(problem="/x/toy", problem_id="toy", problem_key="toy@ab12cd34").problem_key == "toy@ab12cd34"
    assert meta(problem="emflow://gefcom2014:solar", problem_id="gefcom2014-solar").problem_key == "emflow://gefcom2014:solar"
    assert meta(problem="mlebench://spaceship-titanic", problem_id="spaceship-titanic").problem_key == "mlebench://spaceship-titanic"
    assert meta(
        problem=f"einsteinarena://circle-packing@sha256:{'a' * 64}",
        problem_id="circle-packing",
    ).problem_key == "einsteinarena://circle-packing"
    assert meta(problem="/abs/problems/circle-packing", problem_id="circle-packing").problem_key == "circle-packing"

    with pytest.raises(ValueError, match="output artifact must be a file name"):
        meta(problem="p", problem_id="p", output_artifacts=["../escape"])


def test_create_search_records_problem_key_and_unique_ids(tmp_path, config):
    from hillclimb.api import create_search
    from hillclimb.harness.run import load_search_meta

    config.paths.problems_dir = tmp_path / "problems"
    install_problem(config.paths.problems_dir)
    problem = load_problem(DEMO_PROBLEM_ID, config)
    run_dir = config.paths.runs_dir / "r1"
    first = create_search(config, problem, run_dir, "r1", total_s=60)
    second = create_search(config, problem, run_dir, "r1", total_s=60)
    assert (first.name, second.name) == ("circle-packing", "circle-packing-2")
    meta = load_search_meta(second)
    assert meta.search_id == "circle-packing-2"
    assert meta.problem_id == "circle-packing"
    assert meta.problem_key == "circle-packing"
    assert meta.chart_baselines == {
        "baseline": 0.5,
        "OpenEvolve": 2.6359773947566274,
        "AlphaEvolve": 2.6359830849176067,
    }


def test_chart_baselines_reload_current_problem_config(tmp_path, config):
    from hillclimb.tui.chart import chart_baselines
    from hillclimb.harness.run import SearchMeta

    config.paths.problems_dir = tmp_path / "problems"
    problem_dir, _ = install_problem(config.paths.problems_dir)
    meta = SearchMeta(
        search_id="p",
        run_id="r",
        problem=str(problem_dir),
        problem_id="circle-packing",
        agent="dummy",
        model="m",
        metric="sum-radii",
        chart_baselines={"stale snapshot": 0.1},
    )
    assert chart_baselines(config, meta) == {
        "baseline": 0.5,
        "OpenEvolve": 2.6359773947566274,
        "AlphaEvolve": 2.6359830849176067,
    }

    yaml_path = problem_dir / "problem.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("baseline: 0.5", "baseline: 0.6", 1))
    assert chart_baselines(config, meta) == {
        "baseline": 0.6,
        "OpenEvolve": 2.6359773947566274,
        "AlphaEvolve": 2.6359830849176067,
    }

    meta.problem = str(tmp_path / "removed-problem")
    assert chart_baselines(config, meta) == {"stale snapshot": 0.1}

    meta.problem = "emflow://some-provider-problem"
    assert chart_baselines(config, meta) == {"stale snapshot": 0.1}


def test_chart_groups_searches_by_problem_key_across_runs(tmp_path):
    runs = tmp_path / "runs"
    _search(runs, "r1", "demo", [("2026-08-22T10:01:00+00:00", 1.0)])
    # a second search on the same problem inside the same run
    run_dir = runs / "r1"
    second = run_dir / "searches" / "p-2"
    second.mkdir(parents=True)
    write_search_meta(second, SearchMeta(
        search_id="p-2", run_id="r1", problem="p", problem_id="p", agent="dummy",
        model="m", metric="score", started_at="2026-08-22T10:00:30+00:00",
    ))
    Journal(second / "journal.jsonl").candidate_result(Candidate(
        candidate_id="c000", operator="draft", status="passing",
        trials=[mk_trial(val_score=2.0)], finished_at="2026-08-22T10:02:00+00:00",
    ))
    curves = climb_curves(runs, "p")
    # a run with several searches on the problem labels each by search id
    assert [c.label for c in curves] == ["demo/p", "demo/p-2"]


def test_budget_margin_scales_with_short_budgets():
    from hillclimb.harness.budget import BudgetManager

    assert BudgetManager(600, stop_margin_s=300).stop_margin_s == 60
    assert BudgetManager(7200, stop_margin_s=300).stop_margin_s == 300


def test_step_points_hold_each_score_until_the_next():
    from hillclimb.tui.chart import step_points

    assert step_points([], []) == ([], [])
    assert step_points([1.0], [2.0]) == ([1.0], [2.0])
    assert step_points([1.0], [2.0], extent=3.0) == ([1.0, 3.0], [2.0, 2.0])
    xs, ys = step_points([0.0, 2.0, 5.0], [0.5, 3.0, 3.6], extent=9.0)
    # flat to each rise, vertical at it, flat to the extent
    assert xs == [0.0, 2.0, 2.0, 5.0, 5.0, 9.0]
    assert ys == [0.5, 0.5, 3.0, 3.0, 3.6, 3.6]
    # an extent before the last x never shortens the line
    assert step_points([0.0, 4.0], [1.0, 2.0], extent=1.0)[0] == [0.0, 4.0, 4.0]


def test_climb_folds_every_search_into_one_staircase(tmp_path):
    """Three parallel searches are one climb: `best` is judged against what
    any of them had landed so far, x counts candidates across searches, and
    the misses are kept as dots."""
    from hillclimb.tui.chart import climb_for_problem

    runs = tmp_path / "runs"
    _search(runs, "r1", "demo", [
        ("2026-08-22T10:02:00+00:00", 1.0),
        ("2026-08-22T10:06:00+00:00", 1.4),
    ])
    _search(runs, "r2", "demo", [
        ("2026-08-22T10:04:00+00:00", 2.0),   # best of everything so far
        ("2026-08-22T10:08:00+00:00", 1.9),   # a miss against r2's own 2.0
    ])
    climb = climb_for_problem(runs, "p")
    assert climb.searches == 2
    assert [(e.x, e.y, e.best) for e in climb.events] == [
        (1.0, 1.0, True), (2.0, 2.0, True), (3.0, 1.4, False), (4.0, 1.9, False),
    ]
    assert climb.best == 2.0
    assert climb.hits == 2
    assert climb.extent == 4.0
    assert climb.staircase() == ([1.0, 2.0, 2.0, 4.0], [1.0, 1.0, 2.0, 2.0])
    assert climb_for_problem(runs, "other").events == []


def test_climb_respects_lower_is_better(tmp_path):
    from hillclimb.tui.chart import climb_for_problem

    runs = tmp_path / "runs"
    _search(runs, "r1", "demo", [
        ("2026-08-22T10:01:00+00:00", 5.0),
        ("2026-08-22T10:02:00+00:00", 6.0),
        ("2026-08-22T10:03:00+00:00", 4.0),
    ], lower=True)
    assert [e.best for e in climb_for_problem(runs, "p").events] == [True, False, True]


def test_build_climb_plot_renders_steps_and_dots(tmp_path):
    from hillclimb.tui.chart import build_climb_plot, climb_for_problem

    runs = tmp_path / "runs"
    _search(runs, "r1", "demo", [
        ("2026-08-22T10:01:00+00:00", 1.0),
        ("2026-08-22T10:02:00+00:00", 0.5),
        ("2026-08-22T10:03:00+00:00", 2.0),
    ])
    plot = build_climb_plot(climb_for_problem(runs, "p"))
    assert len(plot.render_rgba(200, 100)) == 200 * 100 * 4
    # a single scored candidate still draws as a point without inventing a
    # fractional candidate count merely to make a line segment visible
    solo_runs = tmp_path / "solo-runs"
    _search(solo_runs, "r2", "solo", [("2026-08-22T10:01:00+00:00", 1.0)])
    solo = climb_for_problem(solo_runs, "p")
    assert [e.x for e in solo.events] == [1.0]
    build_climb_plot(solo).render_rgba(200, 100)


def test_problem_get_asks_before_creating_a_hillclimb_dir(tmp_path, monkeypatch):
    """Outside a hillclimb dir, `problem get` says what it would create and
    asks; a decline creates nothing and points at `hillclimb init`. A yes
    (or no terminal to ask on) creates the dir with only that problem."""
    from typer.testing import CliRunner

    from hillclimb import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.setattr("hillclimb.cli.problem._stdin_is_tty", lambda: True)
    result = CliRunner().invoke(cli.app, ["problem", "get", "golomb-20"], input="n\n")
    assert result.exit_code == 1 and "hillclimb init" in result.output
    assert not (tmp_path / "hillclimb.yaml").exists() and not (tmp_path / "problems").exists()

    result = CliRunner().invoke(cli.app, ["problem", "get", "golomb-20"], input="y\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "hillclimb.yaml").exists()
    assert (tmp_path / "problems" / "golomb-20" / "interface.py").exists()
    assert not (tmp_path / "problems" / "example").exists()
    assert not (tmp_path / "specs" / "example.yaml").exists()


def test_problem_get_wont_mix_into_a_folders_own_problems(tmp_path, monkeypatch):
    """A folder with its own problems/ (and no hillclimb.yaml) gets its
    hillclimb dir in ./hillclimb/, and the problem goes there; the folder's
    own problems/ is never written to."""
    from typer.testing import CliRunner

    from hillclimb import cli

    (tmp_path / "problems").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    result = CliRunner().invoke(cli.app, ["problem", "get", "golomb-20"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "hillclimb.yaml").exists()
    assert list((tmp_path / "problems").iterdir()) == []
    assert (tmp_path / "hillclimb" / "problems" / "golomb-20" / "problem.yaml").is_file()
