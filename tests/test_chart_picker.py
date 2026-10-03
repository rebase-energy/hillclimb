"""A bare `hillclimb chart` on a folder with several charts lists them
(one row per problem worked in a run); enter opens the chart, esc comes
back. The watch tables reach the same chart with `c`."""

from tests.factories import trial as mk_trial
from pathlib import Path

import pytest

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.chart import ChartApp, ChartPickerScreen, ChartScreen, chart_index, chart_run_scope, climb_curves
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import RunMeta, SearchMeta, load_search_meta, write_run_meta, write_search_meta
from hillclimb.harness.status import ScoreRef, SearchStatus, write_status
from hillclimb.harness.store import FileDataStore


def _search(
    runs_dir: Path, run_id: str, search_id: str, problem_id: str, *,
    started: str, experiment: str | None = None, best: float | None = None, state: str = "done",
) -> Path:
    run_dir = runs_dir / run_id
    if not (run_dir / "run.yaml").exists():
        write_run_meta(run_dir, RunMeta(
            run_id=run_id, name=run_id, target="p", problem_ids=[problem_id], started_at=started,
        ))
    search_dir = run_dir / "searches" / search_id
    (search_dir / "candidates").mkdir(parents=True)
    write_search_meta(search_dir, SearchMeta(
        search_id=search_id, run_id=run_id, problem=problem_id, problem_id=problem_id,
        agent="dummy", model="m", metric="score", higher_is_better=True,
        started_at=started, experiment=experiment, study="exp" if experiment else None,
    ))
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(Candidate(
        candidate_id="c001", operator="draft", status="passing",
        trials=[mk_trial(val_score=best)] if best is not None else [],
        created_at=started, finished_at=started,
    ))
    status = SearchStatus(search_id=search_id, run_id=run_id, state=state, updated_at=started)
    if best is not None:
        status.best = ScoreRef(candidate_id="c001", val_score=best)
    write_status(search_dir, status)
    return search_dir


def test_chart_index_one_row_per_run_and_problem(tmp_path: Path):
    runs = tmp_path / "runs"
    _search(runs, "r1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00", best=0.5)
    # an experiment run: two experiments on beta, one on gamma
    _search(runs, "r2", "beta", "beta", started="2026-09-02T10:00:00+00:00", experiment="greedy", best=0.3)
    _search(runs, "r2", "beta-2", "beta", started="2026-09-02T10:00:01+00:00", experiment="gepa", best=0.4)
    _search(runs, "r2", "gamma", "gamma", started="2026-09-02T10:00:02+00:00", experiment="greedy")

    rows = chart_index(FileDataStore(runs))

    assert [(r.run_id, r.problem_key) for r in rows] == [("r2", "gamma"), ("r2", "beta"), ("r1", "alpha")]
    beta = rows[1]
    assert beta.anchor == "r2/beta-2"  # the newest search of the pair
    assert beta.searches == 2 and beta.running == 0
    assert beta.experiments == ("greedy", "gepa")
    assert beta.best == 0.4  # best across the pair's searches
    assert rows[0].best is None and rows[0].experiments == ("greedy",)
    assert rows[2].run_name == "r1"


def test_chart_index_empty_folder(tmp_path: Path):
    assert chart_index(FileDataStore(tmp_path / "runs")) == []


def test_experiment_chart_stays_inside_its_run(tmp_path: Path):
    """Two runs of one experiment on the same problem each carry a
    `greedy r1`; the chart anchored on one run must not overlay the other's."""
    runs = tmp_path / "runs"
    _search(runs, "pilot", "beta", "beta", started="2026-09-04T10:00:00+00:00", experiment="greedy", best=0.3)
    _search(runs, "pilot", "beta-2", "beta", started="2026-09-04T10:00:01+00:00", experiment="gepa", best=0.2)
    _search(runs, "real", "beta", "beta", started="2026-09-10T10:00:00+00:00", experiment="greedy", best=0.1)
    _search(runs, "real", "beta-2", "beta", started="2026-09-10T10:00:01+00:00", experiment="gepa", best=0.4)
    store = FileDataStore(runs)

    anchor = load_search_meta(runs / "real" / "searches" / "beta-2")
    assert chart_run_scope(anchor) == "real"
    curves = climb_curves(store, "beta", run_id=chart_run_scope(anchor))
    assert [(c.label, c.best) for c in curves] == [("greedy", 0.1), ("gepa", 0.4)]
    # every run of the problem, as before, when nothing scopes it
    assert [c.label for c in climb_curves(store, "beta")] == ["greedy", "gepa", "greedy", "gepa"]

    # a plain search anchors the cross-run climb
    plain = _search(runs, "solo", "beta", "beta", started="2026-09-11T10:00:00+00:00", best=0.5)
    assert chart_run_scope(load_search_meta(plain)) is None


@pytest.mark.asyncio
async def test_bare_chart_lists_then_opens_then_comes_back(config):
    runs = config.paths.runs_dir
    _search(runs, "r1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00", best=0.5)
    _search(runs, "r2", "beta", "beta", started="2026-09-02T10:00:00+00:00", best=0.3)

    app = ChartApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartPickerScreen)
        table = app.screen.query_one("#charts")
        assert table.row_count == 2
        assert table.cursor_row == 0  # newest activity first: r2/beta
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        assert app.screen.search == "r2/beta"
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ChartPickerScreen)
        # `b` is back too, on every screen that has an esc
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        await pilot.press("b")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ChartPickerScreen)


@pytest.mark.asyncio
async def test_picked_experiment_run_charts_only_its_own_arms(config):
    runs = config.paths.runs_dir
    _search(runs, "pilot", "beta", "beta", started="2026-09-04T10:00:00+00:00", experiment="greedy", best=0.3)
    _search(runs, "real", "beta", "beta", started="2026-09-10T10:00:00+00:00", experiment="greedy", best=0.1)
    _search(runs, "real", "beta-2", "beta", started="2026-09-10T10:00:01+00:00", experiment="gepa", best=0.4)

    app = ChartApp(config, "real/beta-2")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        line = str(app.screen.query_one("#chartline").render())
        assert line.count("greedy") == 1 and "gepa" in line  # the pilot's greedy stays off this chart


@pytest.mark.asyncio
async def test_single_chart_opens_directly_and_esc_is_inert(config):
    _search(config.paths.runs_dir, "r1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00", best=0.5)

    app = ChartApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ChartScreen)  # nothing underneath to go back to


@pytest.mark.asyncio
async def test_explicit_search_skips_the_picker(config):
    runs = config.paths.runs_dir
    _search(runs, "r1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00", best=0.5)
    _search(runs, "r2", "beta", "beta", started="2026-09-02T10:00:00+00:00", best=0.3)

    app = ChartApp(config, "r1/alpha")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        assert app.screen.search == "r1/alpha"


@pytest.mark.asyncio
async def test_watch_c_opens_the_chart_and_esc_returns(config):
    from hillclimb.tui.watch import RunsScreen, SearchesScreen, WatchApp

    runs = config.paths.runs_dir
    _search(runs, "r1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00", best=0.5)
    _search(runs, "r2", "beta", "beta", started="2026-09-02T10:00:00+00:00", best=0.3)

    app = WatchApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, RunsScreen)
        await pilot.press("c")
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        assert app.screen.search == "r2/beta"  # the highlighted (newest) run's search
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, RunsScreen)

        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, SearchesScreen)
        await pilot.press("c")
        await pilot.pause(0.3)
        assert isinstance(app.screen, ChartScreen)
        assert app.screen.search == "r2/beta"
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, SearchesScreen)


def test_plain_runs_of_one_problem_are_one_chart(tmp_path: Path):
    """Four runs on one problem are four lines on ONE chart, not four rows."""
    runs = tmp_path / "runs"
    for n, hour in enumerate((10, 11, 12), start=1):
        _search(runs, f"r{n}", "alpha", "alpha", started=f"2026-09-01T{hour}:00:00+00:00", best=0.1 * n)
    _search(runs, "s1", "beta", "beta", started="2026-09-02T10:00:00+00:00", experiment="greedy", best=0.3)

    rows = chart_index(FileDataStore(runs))

    assert [(r.problem_key, r.run_id, r.run_name) for r in rows] == [("beta", "s1", "s1"), ("alpha", "", "3 runs")]
    assert rows[1].anchor == "r3/alpha" and rows[1].best == pytest.approx(0.3)


def test_run_labels_number_runs_and_keep_chosen_names(tmp_path: Path):
    from datetime import datetime, timezone

    from hillclimb.tui.chart import run_labels

    runs = tmp_path / "runs"
    _search(runs, "alpha-1", "alpha", "alpha", started="2026-09-01T10:00:00+00:00")
    _search(runs, "alpha-2", "alpha", "alpha", started="2026-09-02T10:05:00+00:00")
    _search(runs, "alpha-3", "alpha", "alpha", started="2026-09-02T11:00:00+00:00")
    # a run named by hand keeps its name
    write_run_meta(runs / "alpha-3", RunMeta(
        run_id="alpha-3", name="longer-budget", target="p", problem_ids=["alpha"],
        started_at="2026-09-02T11:00:00+00:00",
    ))
    store = FileDataStore(runs)
    records = store.searches(problem_key="alpha")
    # the test fixture names a run after its id; the default name is the problem's
    today = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc).astimezone()
    labels = run_labels(records, today=today)

    assert labels["alpha-3"] == "#3 longer-budget"
    local = lambda iso: datetime.fromisoformat(iso).astimezone()  # noqa: E731
    first, second = local("2026-09-01T10:00:00+00:00"), local("2026-09-02T10:05:00+00:00")
    assert labels["alpha-1"] == f"#1 {first.strftime('%m-%d %H:%M')}"  # another day: dated
    assert labels["alpha-2"] == f"#2 {second.strftime('%H:%M')}"


@pytest.mark.asyncio
async def test_several_runs_open_as_the_plain_climb_and_v_cycles_the_views(config):
    runs = config.paths.runs_dir
    for n, hour in enumerate((10, 11, 12), start=1):
        _search(runs, f"r{n}", "alpha", "alpha", started=f"2026-09-01T{hour}:00:00+00:00", best=0.1 * n)

    app = ChartApp(config)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.3)
        screen = app.screen
        assert isinstance(screen, ChartScreen)  # one problem: no picker
        line = lambda: str(screen.query_one("#chartline").render())  # noqa: E731
        legend = lambda: [e[0] for e in screen._legend_entries]  # noqa: E731

        # the default: the plain climb — one staircase, no run told apart
        assert screen.view == "climb" and "3 searches" in line() and "#" not in line()
        assert legend()[:2] == ["best so far", "new best"]  # every run improved: no misses

        # v: the same climb, each run's dots in its own colour
        await pilot.press("v")
        await pilot.pause(0.2)
        assert screen.view == "runs" and line().count("runs: ") == 1 and "(by #3 " in line()
        entries = screen._legend_entries
        assert [e[0][:3] for e in entries[:3]] == ["#1 ", "#2 ", "#3 "]
        assert len({e[1] for e in entries[:3]}) == 3  # three distinct run colours

        # ] focuses run 1; the others fade
        await pilot.press("right_square_bracket")
        await pilot.pause(0.2)
        assert screen.run_focus == "r1" and "(1 of 3)" in line() and "overall best" in line()
        faded = {e[1] for e in screen._legend_entries[1:3]}
        assert len(faded) == 1 and screen._legend_entries[0][1] not in faded

        # v: one line per run, the focus kept
        await pilot.press("v")
        await pilot.pause(0.2)
        assert screen.view == "compare" and line().count("compare: run #1 ") == 1

        # v: back to the plain climb, the focus dropped
        await pilot.press("v")
        await pilot.pause(0.2)
        assert screen.view == "climb" and screen.run_focus is None and "#" not in line()

        # ] in the plain climb opens the runs view on run 1
        await pilot.press("right_square_bracket")
        await pilot.pause(0.2)
        assert screen.view == "runs" and screen.run_focus == "r1"

        # d follows the focused run
        await pilot.press("d")
        await pilot.pause(0.2)
        assert "run #1 " in line() and "· detail" in line()


def test_a_tie_to_float_precision_is_not_a_new_best():
    """Two runs at the same score to float precision: the credit stays with
    the run that got there first. A small but real gain still counts."""
    from hillclimb.tui.chart import climb_from_searches

    def run(score: float, hour: int) -> list[Candidate]:
        when = f"2026-09-01T{hour}:00:00+00:00"
        return [Candidate(
            candidate_id="c001", operator="draft", status="passing",
            trials=[mk_trial(val_score=score)], created_at=when, finished_at=when,
        )]

    tie = climb_from_searches([
        ("a", run(0.0370370370370, 10), None, "r1"),
        ("b", run(0.0370370370370 + 1e-14, 11), None, "r2"),
    ])
    assert [(e.run, e.best) for e in tie.events] == [("r1", True), ("r2", False)]

    # 0.0370370107 → 0.0370370370 is 2.6e-8: tiny, but closer to 1/27 — a new best
    gain = climb_from_searches([
        ("a", run(0.037037010743, 10), None, "r1"),
        ("b", run(0.03703703701089958, 11), None, "r2"),
    ])
    assert [(e.run, e.best) for e in gain.events] == [("r1", True), ("r2", True)]


def test_the_line_over_the_chart_wraps_between_items_not_inside_them():
    from hillclimb.tui.chart import flow_items

    items = ["[bold]heilbronn-11[/]", "best=0.037037", "4 searches", "2 of 9 candidates improved"]
    cost = ["1.36M tok", "54.4 cpu-min", "85.6 wall-min", "11 evals"]
    # everything fits: one line
    assert "\n" not in flow_items(items, cost, 200)
    # it does not: the climb first, the cost on a line of its own, kept together
    lines = flow_items(items, cost, 80).split("\n")
    assert lines == [
        "[bold]heilbronn-11[/] · best=0.037037 · 4 searches · 2 of 9 candidates improved",
        "[dim]1.36M tok[/] · [dim]54.4 cpu-min[/] · [dim]85.6 wall-min[/] · [dim]11 evals[/]",
    ]
    # narrower still: broken between items, never inside one
    narrow = flow_items(items, cost, 45).split("\n")
    assert narrow[:2] == ["[bold]heilbronn-11[/] · best=0.037037 · 4 searches", "2 of 9 candidates improved"]


def test_the_tree_views_line_wraps_like_the_charts():
    """`tree` and `treeclimb` flow the same items over their canvas."""
    from hillclimb.tui.lines import flow_items
    from hillclimb.tui.treeview import statusline, statusline_items

    class Tree:
        nodes = {"c000": None, "c001": None, "c002": None}
        depth = 1
        best_id = None

    items = statusline_items("20261003-155741-heilbronn-11/heilbronn-11", "done", Tree(), "score", True, (3, 4))
    assert statusline("20261003-155741-heilbronn-11/heilbronn-11", "done", Tree(), "score", True, (3, 4)) == " · ".join(items)
    lines = flow_items(items, [], 70).split("\n")
    assert len(lines) == 2 and lines[1] == "3 candidates · depth 1"
