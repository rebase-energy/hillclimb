from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from hillclimb.config import Config
from hillclimb.control import read_commands
from hillclimb.experiment import ExperimentMeta, LEGACY_EXPERIMENT_ID, write_experiment
from hillclimb.journal import Journal
from hillclimb.node import Node
from hillclimb.status import RunStatus, write_status
from hillclimb.watch import (
    DETAIL_MIN_HEIGHT,
    DETAIL_STEP,
    WatchApp,
    candidate_detail_lines,
    candidate_detail_renderables,
    node_rows,
    render_stream_line,
    scan_experiments,
    scan_problem_runs,
    stream_tail,
)

DEAD_PID = 2**22


def make_run(
    runs_dir: Path,
    run_id: str,
    status: RunStatus | None = None,
    experiment_id: str | None = "demo-exp",
    experiment_name: str | None = "demo",
) -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "nodes").mkdir(parents=True)
    (run_dir / "best").mkdir()
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "problem": "/tmp/problem",
                "problem_id": "circle-packing",
                "model": "sonnet",
                "backend": "claude-code",
                "budget_s": 3600,
                "lower_is_better": False,
            }
        )
    )
    journal = Journal(run_dir / "journal.jsonl")
    journal.node_result(Node(node_id="n000", operator="baseline", status="ok"))
    journal.node_result(Node(node_id="n001", operator="draft", status="ok", val_score=0.7))
    journal.node_result(
        Node(node_id="n002", operator="improve", parent_id="n001", status="buggy", pruned=True)
    )
    if status is not None:
        write_status(run_dir, status)
    return run_dir


def make_demo_run(
    tmp_path: Path,
    run_id: str,
    status: RunStatus | None = None,
    *,
    wide: bool = False,
) -> tuple[Path, Config]:
    runs_dir = tmp_path / "runs"
    write_experiment(
        runs_dir,
        ExperimentMeta(
            experiment_id="demo-exp",
            name="Demo",
            target="demo",
            problem_ids=["circle-packing"],
        ),
    )
    run_dir = make_run(runs_dir, run_id, status or RunStatus(run_id=run_id, state="done"))
    if wide:
        journal = Journal(run_dir / "journal.jsonl")
        for i in range(3, 12):
            journal.node_result(
                Node(
                    node_id=f"n{i:03d}",
                    operator="draft",
                    status="ok",
                    val_score=float(i),
                    summary="wide summary " * 20,
                )
            )
    config = Config()
    config.paths.runs_dir = runs_dir
    return run_dir, config


async def open_candidate_detail(pilot) -> None:
    await pilot.press("enter")
    await pilot.press("enter")
    await pilot.press("enter")
    await pilot.pause()


def test_scan_experiments_with_status(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    write_experiment(
        runs_dir,
        ExperimentMeta(
            experiment_id="demo-exp",
            name="Demo",
            target="problems/demo-suite.yaml",
            problem_ids=["circle-packing"],
        ),
    )
    make_run(runs_dir, "20260701-run", RunStatus(run_id="20260701-run", state="done"))
    rows = scan_experiments(runs_dir)
    assert len(rows) == 1
    assert rows[0].state == "done"
    assert rows[0].name == "Demo"
    assert rows[0].problem_runs == "1"
    assert rows[0].candidates == "3"

    run_rows = scan_problem_runs(runs_dir, "demo-exp")
    assert len(run_rows) == 1
    assert run_rows[0].problem == "circle-packing"
    assert run_rows[0].candidates == "3 (2 ok)"


def test_scan_runs_detects_crash(tmp_path: Path):
    make_run(
        tmp_path / "runs",
        "crashed-run",
        RunStatus(run_id="crashed-run", state="running", pid=DEAD_PID),
    )
    assert scan_problem_runs(tmp_path / "runs", "demo-exp")[0].state == "crashed"


def test_scan_experiments_groups_legacy_runs(tmp_path: Path):
    make_run(tmp_path / "runs", "old-run", experiment_id=None, experiment_name=None)
    row = scan_experiments(tmp_path / "runs")[0]
    assert row.experiment_id == LEGACY_EXPERIMENT_ID
    assert row.name == "Legacy"
    run_row = scan_problem_runs(tmp_path / "runs", LEGACY_EXPERIMENT_ID)[0]
    assert row.state == "unknown"
    assert run_row.best_val == "-"


def test_node_rows_tree_order_and_pruned(tmp_path: Path):
    run_dir = make_run(tmp_path / "runs", "r")
    rows = node_rows(Journal(run_dir / "journal.jsonl"))
    assert [r.node_id for r in rows] == ["n000", "n001", "n002"]
    assert rows[2].label == "  n002"  # child indented under n001
    assert "PRUNED" in rows[2].marks
    assert "strike" in rows[2].style


def test_candidate_detail_lines_include_scores_lineage_and_notes(tmp_path: Path):
    run_dir = make_run(tmp_path / "runs", "r")
    workspace = run_dir / "nodes" / "n001"
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

    detail = "\n".join(candidate_detail_lines(run_dir, Journal(run_dir / "journal.jsonl"), "n001"))

    assert "Candidate n001 | draft | ok" in detail
    assert "Score: val=0.7  holdout=-  metric=score (higher is better)" in detail
    assert "Parent: root  Children: 1  Path: n001" in detail
    assert "n002  improve  buggy  val=-  PRUNED" in detail
    assert "Notes:" in detail
    assert "tried nearest-neighbor seed" in detail
    assert "Stderr:" in detail
    assert "warning: local search plateau" in detail
    assert "Stdout:" in detail
    assert "val_score: 0.7" in detail
    assert "Agent stream:" in detail
    assert "I will try a constructive heuristic." in detail


def test_candidate_detail_renderables_are_sectioned(tmp_path: Path):
    from rich.console import Console

    run_dir = make_run(tmp_path / "runs", "r")
    workspace = run_dir / "nodes" / "n001"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.md").write_text("tried nearest-neighbor seed\n")
    (workspace / "exec_stdout.log").write_text("val_score: 0.7\n")
    (workspace / "exec_stderr.log").write_text("warning: local search plateau\n")

    renderables = candidate_detail_renderables(
        run_dir,
        Journal(run_dir / "journal.jsonl"),
        "n001",
    )
    console = Console(record=True, width=100)
    for renderable in renderables:
        console.print(renderable)
    rendered = console.export_text()

    assert len(renderables) >= 4
    assert "Candidate n001" in rendered
    assert "val" in rendered
    assert "0.7" in rendered
    assert "Children" in rendered
    assert "n002" in rendered
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
async def test_watch_app_lists_runs_and_stops(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    write_experiment(
        runs_dir,
        ExperimentMeta(
            experiment_id="demo-exp",
            name="Demo",
            target="demo",
            problem_ids=["circle-packing"],
        ),
    )
    run_dir = make_run(
        runs_dir,
        "live-run",
        RunStatus(run_id="live-run", state="running", pid=os.getpid()),
    )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test() as pilot:
        table = app.screen.query_one("#experiments")
        assert table.row_count == 1
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#problem-runs")
        assert table.row_count == 1
        await pilot.press("s")  # stop highlighted run…
        await pilot.press("y")  # …confirm
        await pilot.pause()
    commands = read_commands(run_dir)
    assert len(commands) == 1
    assert commands[0][1].action == "stop"
    assert commands[0][1].source == "tui"


@pytest.mark.asyncio
async def test_runs_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    write_experiment(
        runs_dir,
        ExperimentMeta(
            experiment_id="demo-exp",
            name="Demo",
            target="demo",
            problem_ids=["circle-packing"],
        ),
    )
    for i in range(30):
        make_run(
            runs_dir,
            f"20260704-very-long-run-name-{i:02d}-with-wide-columns",
            RunStatus(run_id=f"r-{i}", state="done"),
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        table = app.screen.query_one("#problem-runs")
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
async def test_experiments_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    for i in range(30):
        experiment_id = f"experiment-{i:02d}-with-a-long-name"
        write_experiment(
            runs_dir,
            ExperimentMeta(
                experiment_id=experiment_id,
                name=f"Long experiment name {i:02d}",
                target="demo",
                problem_ids=["circle-packing"],
            ),
        )
        make_run(
            runs_dir,
            f"20260704-run-{i:02d}",
            RunStatus(run_id=f"r-{i}", state="done"),
            experiment_id=experiment_id,
            experiment_name=f"Long experiment name {i:02d}",
        )
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(50, 10)) as pilot:
        table = app.screen.query_one("#experiments")
        await pilot.pause()
        if table.max_scroll_x == 0 or table.max_scroll_y == 0:
            pytest.skip("headless experiments table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_node_table_refresh_preserves_scroll_offsets(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    write_experiment(
        runs_dir,
        ExperimentMeta(
            experiment_id="demo-exp",
            name="Demo",
            target="demo",
            problem_ids=["circle-packing"],
        ),
    )
    run_dir = make_run(
        runs_dir,
        "wide-node-run",
        RunStatus(run_id="wide-node-run", state="done"),
    )
    journal = Journal(run_dir / "journal.jsonl")
    for i in range(3, 40):
        journal.node_result(
            Node(
                node_id=f"n{i:03d}",
                operator="draft",
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
            pytest.skip("headless node table did not overflow in both axes")

        table.scroll_to(x=table.max_scroll_x, y=8, immediate=True, force=True)
        await pilot.pause()
        before = (table.scroll_x, table.scroll_y)

        app.screen.refresh_data()
        await pilot.pause()

        assert (table.scroll_x, table.scroll_y) == before


@pytest.mark.asyncio
async def test_candidate_detail_panel_opens_updates_and_closes(tmp_path: Path):
    run_dir, config = make_demo_run(tmp_path, "detail-run")
    (run_dir / "nodes" / "n000").mkdir(parents=True, exist_ok=True)
    (run_dir / "nodes" / "n001").mkdir(parents=True, exist_ok=True)
    (run_dir / "nodes" / "n000" / "notes.md").write_text("baseline copy\n")
    (run_dir / "nodes" / "n001" / "notes.md").write_text("draft heuristic\n")

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
        assert app.screen._detail_node_id == "n000"
        assert str(detail.styles.display) != "none"
        assert str(divider.styles.display) != "none"

        await pilot.press("down")
        await pilot.pause()
        assert app.screen._detail_node_id == "n001"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen._detail_node_id is None
        assert str(detail.styles.display) == "none"
        assert str(divider.styles.display) == "none"
        assert app.screen.__class__.__name__ == "CandidateScreen"


@pytest.mark.asyncio
async def test_candidate_detail_panel_resizes_and_clamps(tmp_path: Path):
    _, config = make_demo_run(tmp_path, "resizable-run")

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
    _, config = make_demo_run(tmp_path, "drag-run")

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
    _, config = make_demo_run(tmp_path, "table-bottom-drag-run", wide=True)

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
    _, config = make_demo_run(tmp_path, "table-horizontal-scroll-run", wide=True)

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
    _, config = make_demo_run(tmp_path, "scroll-then-resize-run", wide=True)

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
    _, config = make_demo_run(tmp_path, "resize-then-scroll-run", wide=True)

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
