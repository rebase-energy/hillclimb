"""A bare `hillclimb chart` on a folder with several charts lists them
(one row per problem worked in a run); enter opens the chart, esc comes
back. The watch tables reach the same chart with `c`."""

from tests.factories import trial as mk_trial
from pathlib import Path

import pytest

from hillclimb.candidate import Candidate
from hillclimb.chart import ChartApp, ChartPickerScreen, ChartScreen, chart_index, chart_run_scope, climb_curves
from hillclimb.journal import Journal
from hillclimb.run import RunMeta, SearchMeta, load_search_meta, write_run_meta, write_search_meta
from hillclimb.status import ScoreRef, SearchStatus, write_status
from hillclimb.store import FileDataStore


def _search(
    runs_dir: Path, run_id: str, search_id: str, problem_id: str, *,
    started: str, arm: str | None = None, best: float | None = None, state: str = "done",
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
        backend="dummy", model="m", metric="score", higher_is_better=True,
        started_at=started, arm=arm, experiment="exp" if arm else None,
    ))
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(Candidate(
        candidate_id="c001", operator="draft", status="ok",
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
    # an experiment run: two arms on beta, one on gamma
    _search(runs, "r2", "beta", "beta", started="2026-09-02T10:00:00+00:00", arm="greedy", best=0.3)
    _search(runs, "r2", "beta-2", "beta", started="2026-09-02T10:00:01+00:00", arm="gepa", best=0.4)
    _search(runs, "r2", "gamma", "gamma", started="2026-09-02T10:00:02+00:00", arm="greedy")

    rows = chart_index(FileDataStore(runs))

    assert [(r.run_id, r.problem_key) for r in rows] == [("r2", "gamma"), ("r2", "beta"), ("r1", "alpha")]
    beta = rows[1]
    assert beta.anchor == "r2/beta-2"  # the newest search of the pair
    assert beta.searches == 2 and beta.running == 0
    assert beta.arms == ("greedy", "gepa")
    assert beta.best == 0.4  # best across the pair's searches
    assert rows[0].best is None and rows[0].arms == ("greedy",)
    assert rows[2].run_name == "r1"


def test_chart_index_empty_folder(tmp_path: Path):
    assert chart_index(FileDataStore(tmp_path / "runs")) == []


def test_experiment_chart_stays_inside_its_run(tmp_path: Path):
    """Two runs of one experiment on the same problem each carry a
    `greedy r1`; the chart anchored on one run must not overlay the other's."""
    runs = tmp_path / "runs"
    _search(runs, "pilot", "beta", "beta", started="2026-09-04T10:00:00+00:00", arm="greedy", best=0.3)
    _search(runs, "pilot", "beta-2", "beta", started="2026-09-04T10:00:01+00:00", arm="gepa", best=0.2)
    _search(runs, "real", "beta", "beta", started="2026-09-10T10:00:00+00:00", arm="greedy", best=0.1)
    _search(runs, "real", "beta-2", "beta", started="2026-09-10T10:00:01+00:00", arm="gepa", best=0.4)
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
    _search(runs, "pilot", "beta", "beta", started="2026-09-04T10:00:00+00:00", arm="greedy", best=0.3)
    _search(runs, "real", "beta", "beta", started="2026-09-10T10:00:00+00:00", arm="greedy", best=0.1)
    _search(runs, "real", "beta-2", "beta", started="2026-09-10T10:00:01+00:00", arm="gepa", best=0.4)

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
    from hillclimb.watch import RunsScreen, SearchesScreen, WatchApp

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
