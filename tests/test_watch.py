from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hillclimb.candidate import BackendInfo, Candidate, Trial
from hillclimb.config import Config
from hillclimb.control import read_commands
from hillclimb.journal import Journal
from hillclimb.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.status import SearchStatus, write_status
from hillclimb.watch import (
    DETAIL_MIN_HEIGHT,
    DETAIL_STEP,
    WatchApp,
    candidate_detail_lines,
    candidate_detail_renderables,
    candidate_rows,
    render_stream_line,
    scan_runs,
    scan_searches,
    stream_tail,
)

DEAD_PID = 2**22


def make_candidate(candidate_id: str, **kwargs) -> Candidate:
    val_score = kwargs.pop("val_score", None)
    if val_score is not None:
        kwargs["trials"] = [Trial(val_score=val_score)]
    return Candidate(candidate_id=candidate_id, operator=kwargs.pop("operator", "draft"), **kwargs)


def make_run_with_search(
    runs_dir: Path,
    run_id: str,
    status: SearchStatus | None = None,
    run_name: str = "Demo",
    search_id: str = "circle-packing",
) -> Path:
    """Create a v2 run with one search; returns the search dir."""
    run_dir = runs_dir / run_id
    write_run_meta(
        run_dir,
        RunMeta(run_id=run_id, name=run_name, target="demo", problem_ids=[search_id]),
    )
    search_dir = run_dir / "searches" / search_id
    (search_dir / "candidates").mkdir(parents=True)
    (search_dir / "best").mkdir()
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id=search_id,
            run_id=run_id,
            problem="/tmp/problem",
            problem_id=search_id,
            backend="claude-code",
            model="sonnet",
            metric="score",
            lower_is_better=False,
            budget_s=3600,
        ),
    )
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_result(make_candidate("c000", operator="baseline", status="ok"))
    c001 = make_candidate("c001", operator="draft", status="ok", val_score=0.7)
    c001.backend = BackendInfo(name="claude-code", total_tokens=240_000)
    journal.candidate_result(c001)
    c002 = make_candidate("c002", operator="improve", parent_id="c001", status="buggy", pruned=True)
    c002.backend = BackendInfo(name="claude-code", total_tokens=1_000_000)
    journal.candidate_result(c002)
    if status is not None:
        write_status(search_dir, status)
    return search_dir


def make_demo_search(
    tmp_path: Path,
    run_id: str,
    status: SearchStatus | None = None,
    *,
    wide: bool = False,
) -> tuple[Path, Config]:
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir, run_id, status or SearchStatus(search_id="circle-packing", run_id=run_id, state="done")
    )
    if wide:
        journal = Journal(search_dir / "journal.jsonl")
        for i in range(3, 12):
            journal.candidate_result(
                make_candidate(
                    f"c{i:03d}",
                    status="ok",
                    val_score=float(i),
                    summary="wide summary " * 20,
                )
            )
    config = Config()
    config.paths.runs_dir = runs_dir
    return search_dir, config


async def open_candidate_detail(pilot) -> None:
    await pilot.press("enter")
    await pilot.press("enter")
    await pilot.press("enter")
    await pilot.pause()


def test_scan_runs_and_searches_with_status(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    make_run_with_search(
        runs_dir,
        "20260701-run",
        SearchStatus(search_id="circle-packing", run_id="20260701-run", state="done"),
    )
    rows = scan_runs(runs_dir)
    assert len(rows) == 1
    assert rows[0].state == "done"
    assert rows[0].name == "Demo"
    assert rows[0].searches == "1"
    assert rows[0].candidates == "3"

    search_rows = scan_searches(runs_dir / "20260701-run")
    assert len(search_rows) == 1
    assert search_rows[0].problem == "circle-packing"
    assert search_rows[0].candidates == "3 (2 ok)"
    assert search_rows[0].tokens == "1.24M"  # 240k + 1.0M, summed across candidates


def test_fmt_tokens():
    from hillclimb.watch import _fmt_tokens

    assert _fmt_tokens(None) == "-"
    assert _fmt_tokens(0) == "-"
    assert _fmt_tokens(812) == "812"
    assert _fmt_tokens(24_500) == "24.5k"
    assert _fmt_tokens(1_240_000) == "1.24M"


def test_stream_tokens_live_and_final(tmp_path: Path):
    import json

    from hillclimb.watch import _stream_tokens

    ws = tmp_path / "cand"
    ws.mkdir()
    assert _stream_tokens(ws) == 0  # no stream yet

    def turn(mid, out, cc, cr):
        return json.dumps({"type": "assistant", "message": {"id": mid, "usage": {
            "input_tokens": 1, "output_tokens": out,
            "cache_creation_input_tokens": cc, "cache_read_input_tokens": cr}}})

    # each turn streams twice under one id (partial then final) -> deduped
    lines = [turn("m1", 2, 100, 500), turn("m1", 2, 100, 500), turn("m2", 3, 50, 900)]
    (ws / "agent_stream.jsonl").write_text("\n".join(lines) + "\n")
    # unique turns: (1+2+100+500) + (1+3+50+900)
    assert _stream_tokens(ws) == 603 + 954

    # once the result lands it is authoritative, replacing the per-turn estimate
    result = json.dumps({"type": "result", "usage": {
        "input_tokens": 10, "output_tokens": 8209,
        "cache_creation_input_tokens": 29297, "cache_read_input_tokens": 199902}})
    (ws / "agent_stream.jsonl").write_text("\n".join(lines + [result]) + "\n")
    assert _stream_tokens(ws) == 10 + 8209 + 29297 + 199902

    # a half-written trailing line (agent still streaming) is ignored
    (ws / "agent_stream.jsonl").write_text("\n".join(lines) + "\n{\"type\": \"assis")
    assert _stream_tokens(ws) == 603 + 954


def test_search_row_counts_in_flight_tokens(tmp_path: Path):
    import json

    from hillclimb.status import CurrentCandidate, SearchStatus
    from hillclimb.watch import _search_row

    runs_dir = tmp_path / "runs"
    make_run_with_search(
        runs_dir, "20260701-run",
        SearchStatus(search_id="circle-packing", run_id="20260701-run", state="running"),
    )
    search_dir = runs_dir / "20260701-run" / "searches" / "circle-packing"
    # an in-flight candidate the journal does not know about yet
    live = search_dir / "candidates" / "c003"
    live.mkdir(parents=True)
    (live / "agent_stream.jsonl").write_text(json.dumps({"type": "result", "usage": {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 500_000}}) + "\n")
    status = SearchStatus(
        search_id="circle-packing", run_id="20260701-run", state="running",
        current=[CurrentCandidate(candidate_id="c003", operator="improve", phase="agent",
                                  workspace=str(live))],
    )
    from hillclimb.status import write_status
    write_status(search_dir, status)
    row = _search_row(search_dir)
    # 240k + 1.0M finished (from make_run_with_search) + 500k in-flight = 1.74M
    assert row.tokens == "1.74M"


def test_scan_searches_detects_crash(tmp_path: Path):
    make_run_with_search(
        tmp_path / "runs",
        "crashed-run",
        SearchStatus(search_id="circle-packing", run_id="crashed-run", state="running", pid=DEAD_PID),
    )
    assert scan_searches(tmp_path / "runs" / "crashed-run")[0].state == "crashed"


def test_scan_runs_skips_v1_layout(tmp_path: Path):
    """Old flat-layout dirs (run.yaml without schema_version) are invisible
    and must not crash the scanner."""
    runs_dir = tmp_path / "runs"
    old = runs_dir / "20260101-000000-legacy"
    (old / "nodes").mkdir(parents=True)
    (old / "run.yaml").write_text("run_id: legacy\nproblem_id: x\nbudget_s: 60\n")
    make_run_with_search(runs_dir, "20260701-run")

    rows = scan_runs(runs_dir)
    assert [row.run_id for row in rows] == ["20260701-run"]


def test_candidate_rows_tree_order_and_pruned(tmp_path: Path):
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    rows = candidate_rows(Journal(search_dir / "journal.jsonl"))
    assert [r.candidate_id for r in rows] == ["c000", "c001", "c002"]
    assert rows[2].label == "  c002"  # child indented under c001
    assert "PRUNED" in rows[2].marks
    assert "strike" in rows[2].style


def test_candidate_detail_lines_include_scores_lineage_and_notes(tmp_path: Path):
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    workspace = search_dir / "candidates" / "c001"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("tried nearest-neighbor seed\nkept deterministic order\n")
    (workspace / "exec_stdout.log").write_text("val_score: 0.7\n")
    (workspace / "exec_stderr.log").write_text("warning: local search plateau\n")
    (workspace / "agent_stream.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I will try a constructive heuristic."},
        ]}})
        + "\n"
    )

    detail = "\n".join(
        candidate_detail_lines(search_dir, Journal(search_dir / "journal.jsonl"), "c001")
    )

    assert "Candidate c001 | draft | ok" in detail
    assert "Score: val=0.7  holdout=-  metric=score (higher is better)" in detail
    assert "Parent: root  Children: 1  Path: c001" in detail
    assert "c002  improve  buggy  val=-  PRUNED" in detail
    assert "Trial: returncode=" in detail
    assert "Notes:" in detail
    assert "tried nearest-neighbor seed" in detail
    assert "Stderr:" in detail
    assert "warning: local search plateau" in detail
    assert "Stdout:" in detail
    assert "val_score: 0.7" in detail
    assert "Agent stream:" in detail
    assert "I will try a constructive heuristic." in detail


def test_candidate_detail_lines_baseline_without_trial(tmp_path: Path):
    """The baseline has no trials; the detail panel must not crash."""
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    detail = "\n".join(
        candidate_detail_lines(search_dir, Journal(search_dir / "journal.jsonl"), "c000")
    )
    assert "Candidate c000 | baseline | ok" in detail
    assert "Trial: (not executed)" in detail


def test_candidate_detail_renderables_are_sectioned(tmp_path: Path):
    from rich.console import Console

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    workspace = search_dir / "candidates" / "c001"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("tried nearest-neighbor seed\n")
    (workspace / "exec_stdout.log").write_text("val_score: 0.7\n")
    (workspace / "exec_stderr.log").write_text("warning: local search plateau\n")

    renderables = candidate_detail_renderables(
        search_dir,
        Journal(search_dir / "journal.jsonl"),
        "c001",
    )
    console = Console(record=True, width=100)
    for renderable in renderables:
        console.print(renderable)
    rendered = console.export_text()

    assert len(renderables) >= 4
    assert "Candidate c001" in rendered
    assert "val" in rendered
    assert "0.7" in rendered
    assert "Children" in rendered
    assert "c002" in rendered
    assert "Notes" in rendered
    assert "tried nearest-neighbor seed" in rendered
    assert "Stderr" in rendered
    assert "Stdout" in rendered


def test_render_stream_line_shapes():
    assistant = json.dumps(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Let me look at the data."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "head train.csv"}},
        ]}}
    )
    rendered = render_stream_line(assistant)
    assert "Let me look at the data." in rendered
    assert "→ Bash(head train.csv)" in rendered

    result = json.dumps({"type": "result", "subtype": "success", "num_turns": 5, "total_cost_usd": 1.25})
    assert render_stream_line(result) == "[result] success turns=5 cost=$1.25"

    system = json.dumps({"type": "system", "subtype": "init", "session_id": "s1"})
    assert "init" in render_stream_line(system)

    assert render_stream_line("not json at all") == "not json at all"
    assert render_stream_line(json.dumps({"type": "user"})) is None


def test_stream_tail(tmp_path: Path):
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": "s"}) + "\n"
        + json.dumps({"type": "result", "subtype": "success", "num_turns": 1, "total_cost_usd": 0.1}) + "\n"
    )
    lines = stream_tail(tmp_path)
    assert len(lines) == 2
    assert lines[-1].startswith("[result]")
    assert stream_tail(tmp_path / "missing") == []


@pytest.mark.asyncio
async def test_watch_app_lists_searches_and_stops(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir,
        "live-run",
        SearchStatus(
            search_id="circle-packing", run_id="live-run", state="running", pid=os.getpid()
        ),
    )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        table = app.screen.query_one("#runs")
        assert table.row_count == 1
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#searches")
        assert table.row_count == 1
        await pilot.press("s")  # stop highlighted search…
        await pilot.press("y")  # …confirm
        await pilot.pause()
    commands = read_commands(search_dir)
    assert len(commands) == 1
    assert commands[0][1].action == "stop"
    assert commands[0][1].source == "tui"


@pytest.mark.asyncio
async def test_searches_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_id = "20260704-many-searches"
    for i in range(30):
        make_run_with_search(
            runs_dir,
            run_id,
            SearchStatus(search_id=f"problem-{i:02d}", run_id=run_id, state="done"),
            search_id=f"problem-{i:02d}-with-a-very-wide-name-for-columns",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#searches")
        await pilot.pause()
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_runs_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    for i in range(30):
        make_run_with_search(
            runs_dir,
            f"20260704-run-{i:02d}",
            SearchStatus(search_id="circle-packing", run_id=f"20260704-run-{i:02d}", state="done"),
            run_name=f"Long run name {i:02d} with wide columns",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        table = app.screen.query_one("#runs")
        await pilot.pause()
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless runs table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_candidate_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    search_dir = make_run_with_search(
        runs_dir,
        "wide-candidate-run",
        SearchStatus(search_id="circle-packing", run_id="wide-candidate-run", state="done"),
    )
    journal = Journal(search_dir / "journal.jsonl")
    for i in range(3, 40):
        journal.candidate_result(
            make_candidate(
                f"c{i:03d}",
                status="ok",
                val_score=float(i),
                summary="wide summary " * 20,
            )
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#candidates")
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless candidate table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_candidate_detail_panel_opens_updates_and_closes(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "detail-run")
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c001").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c000" / "notes.md").write_text("baseline copy\n")
    (search_dir / "candidates" / "c001" / "notes.md").write_text("draft heuristic\n")

    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause()
        detail = app.screen.query_one("#candidate-detail")
        divider = app.screen.query_one("#detail-divider")
        assert str(detail.styles.display) == "none"
        assert str(divider.styles.display) == "none"

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen._detail_candidate_id == "c000"
        assert str(detail.styles.display) != "none"
        assert str(divider.styles.display) != "none"

        await pilot.press("down")
        await pilot.pause()
        assert app.screen._detail_candidate_id == "c001"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen._detail_candidate_id is None
        assert str(detail.styles.display) == "none"
        assert str(divider.styles.display) == "none"
        assert app.screen.__class__.__name__ == "CandidateScreen"


@pytest.mark.asyncio
async def test_candidate_detail_panel_resizes_and_clamps(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "resizable-run")

    app = WatchApp(config)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        initial = screen._detail_height

        await pilot.press("+")
        await pilot.pause()
        assert screen._detail_height == initial + DETAIL_STEP

        await pilot.press("-")
        await pilot.pause()
        assert screen._detail_height == initial

        screen._set_detail_height(1)
        assert screen._detail_height == DETAIL_MIN_HEIGHT

        screen._set_detail_height(999)
        assert screen._detail_height == screen._max_detail_height()


@pytest.mark.asyncio
async def test_candidate_detail_divider_drag_resizes(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "drag-run")

    app = WatchApp(config)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        divider = screen.query_one("#detail-divider")
        initial = screen._detail_height

        assert await pilot.mouse_down("#detail-divider", offset=(1, 0))
        await pilot.hover(offset=(divider.region.x + 1, divider.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3

        await pilot.mouse_up(offset=(divider.region.x + 1, divider.region.y - 3))
        assert not screen._dragging_detail


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_row_drag_resizes(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "table-bottom-drag-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial = screen._detail_height

        assert hbar.__class__.__name__ == "CandidateHorizontalScrollBar"
        assert table.max_scroll_x > 0
        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 1, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3

        await pilot.mouse_up(offset=(hbar.region.x + 1, hbar.region.y - 3))
        assert not screen._dragging_detail


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_horizontal_drag_scrolls(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "table-horizontal-scroll-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert hbar.__class__.__name__ == "CandidateHorizontalScrollBar"
        assert table.max_scroll_x > 0
        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y))
        await pilot.pause()

        assert table.scroll_x > initial_scroll
        assert screen._detail_height == initial_height

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y))


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_switches_scroll_then_resize_in_one_drag(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "scroll-then-resize-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y))
        await pilot.pause()
        assert table.scroll_x > initial_scroll
        assert screen._detail_height == initial_height

        after_scroll = table.scroll_x
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y - 3))
        await pilot.pause()
        assert table.scroll_x == after_scroll
        assert screen._detail_height == initial_height + 3

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y - 3))


@pytest.mark.asyncio
async def test_candidate_table_scrollbar_switches_resize_then_scroll_in_one_drag(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "resize-then-scroll-run", wide=True)

    app = WatchApp(config)
    async with app.run_test(size=(50, 24)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        hbar = table.horizontal_scrollbar
        initial_height = screen._detail_height
        initial_scroll = table.scroll_x

        assert await pilot.mouse_down(offset=(hbar.region.x + 1, hbar.region.y))
        await pilot.hover(offset=(hbar.region.x + 1, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial_height + 3
        assert table.scroll_x == initial_scroll

        after_resize = screen._detail_height
        await pilot.hover(offset=(hbar.region.x + 12, hbar.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == after_resize
        assert table.scroll_x > initial_scroll

        await pilot.mouse_up(offset=(hbar.region.x + 12, hbar.region.y - 3))


def test_budget_left_shows_seconds():
    from hillclimb.watch import _format_budget_left

    assert _format_budget_left(None) == "-"
    assert _format_budget_left(247.9) == "4m 07s"
    assert _format_budget_left(3727) == "1h 02m 07s"
    assert _format_budget_left(-5) == "0m 00s"


def test_live_remaining_counts_down_between_heartbeats():
    from datetime import datetime, timedelta, timezone

    from hillclimb.status import BudgetStatus, SearchStatus, live_remaining_s

    written = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    status = SearchStatus(search_id="p", state="running", budget=BudgetStatus(remaining_s=100), updated_at=written)
    assert 89 <= live_remaining_s(status, "running") <= 91
    # a finished search's clock stopped at the last write
    assert live_remaining_s(status, "done") == 100
    status.budget.remaining_s = 3
    assert live_remaining_s(status, "running") == 0.0
