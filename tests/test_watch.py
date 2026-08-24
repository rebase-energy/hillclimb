from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hillclimb.candidate import BackendInfo, Candidate, Trial
from hillclimb.config import Config
from hillclimb.control import read_commands
from hillclimb.store import FileDataStore, key_for
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


def _record(search_dir):
    """The search's store record, the way the TUI hands it to the detail renderers."""
    return FileDataStore(search_dir.parents[2]).search(key_for(search_dir))


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
            higher_is_better=True,
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


DETAIL_MIN_HEIGHT_FOR_TESTS = 6  # == watch.DETAIL_MIN_HEIGHT


async def open_candidate_detail(pilot) -> None:
    await pilot.press("enter")  # runs -> searches
    await pilot.press("o")  # searches -> full candidate screen
    await pilot.press("enter")  # candidate -> detail
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

    search_rows = scan_searches(runs_dir, "20260701-run")
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
                                  candidate_dir=str(live))],
    )
    from hillclimb.status import write_status
    write_status(search_dir, status)
    store = FileDataStore(search_dir.parents[2])
    row = _search_row(store, store.search(key_for(search_dir)))
    # 240k + 1.0M finished (from make_run_with_search) + 500k in-flight = 1.74M
    assert row.tokens == "1.74M"


def test_scan_searches_detects_crash(tmp_path: Path):
    make_run_with_search(
        tmp_path / "runs",
        "crashed-run",
        SearchStatus(search_id="circle-packing", run_id="crashed-run", state="running", pid=DEAD_PID),
    )
    assert scan_searches(tmp_path / "runs", "crashed-run")[0].state == "crashed"


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


def test_candidate_rows_show_pending_as_running_or_stale(tmp_path: Path):
    from hillclimb.watch import display_status

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    journal.candidate_created(make_candidate("c003", operator="improve", parent_id="c001", status="pending"))
    journal = Journal(search_dir / "journal.jsonl")
    live = {r.candidate_id: r.status for r in candidate_rows(journal, live=True)}
    dead = {r.candidate_id: r.status for r in candidate_rows(journal, live=False)}
    assert live["c003"] == "running" and dead["c003"] == "stale"
    assert live["c001"] == dead["c001"] == "ok"  # only pending is remapped
    assert display_status("buggy", True) == "buggy"


def test_candidate_detail_lines_include_scores_lineage_and_notes(tmp_path: Path):
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    candidate_dir = search_dir / "candidates" / "c001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "notes.md").write_text("tried nearest-neighbor seed\nkept deterministic order\n")
    (candidate_dir / "exec_stdout.log").write_text("val_score: 0.7\n")
    (candidate_dir / "exec_stderr.log").write_text("warning: local search plateau\n")
    (candidate_dir / "agent_stream.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "I will try a constructive heuristic."},
        ]}})
        + "\n"
    )

    detail = "\n".join(
        candidate_detail_lines(_record(search_dir), Journal(search_dir / "journal.jsonl"), "c001")
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
    assert "Operator stream:" in detail
    assert "I will try a constructive heuristic." in detail


def test_candidate_detail_lines_baseline_without_trial(tmp_path: Path):
    """The baseline has no trials; the detail panel must not crash."""
    search_dir = make_run_with_search(tmp_path / "runs", "r")
    detail = "\n".join(
        candidate_detail_lines(_record(search_dir), Journal(search_dir / "journal.jsonl"), "c000")
    )
    assert "Candidate c000 | baseline | ok" in detail
    assert "Trial: (not executed)" in detail


def test_candidate_detail_renderables_are_sectioned(tmp_path: Path):
    from rich.console import Console

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    candidate_dir = search_dir / "candidates" / "c001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "notes.md").write_text("tried nearest-neighbor seed\n")
    (candidate_dir / "exec_stdout.log").write_text("val_score: 0.7\n")
    (candidate_dir / "exec_stderr.log").write_text("warning: local search plateau\n")

    renderables = candidate_detail_renderables(
        _record(search_dir),
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


def test_parse_stream_line_timestamps_kinds_and_noise():
    from hillclimb.watch import parse_stream_line

    stamped = json.dumps({"type": "assistant", "ts": "2026-08-23T05:33:01+00:00", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ]}})
    entry = parse_stream_line(stamped)
    assert entry.kind == "tool" and entry.text == "→ Bash(ls)"
    assert len(entry.ts) == 8 and entry.ts.count(":") == 2  # local HH:MM:SS
    assert render_stream_line(stamped) == f"{entry.ts}  → Bash(ls)"

    # per-turn bookkeeping is noise; only init is shown, with its model
    assert parse_stream_line(json.dumps({"type": "system", "subtype": "thinking_tokens", "session_id": "s"})) is None
    init = parse_stream_line(json.dumps({"type": "system", "subtype": "init", "session_id": "s", "model": "opus"}))
    assert init.kind == "system" and "model=opus" in init.text

    failed = parse_stream_line(json.dumps({"type": "result", "subtype": "error", "is_error": True, "num_turns": 2}))
    assert failed.kind == "error"
    assert parse_stream_line("banner").kind == "raw"

    # the argument that matters, on one line, not Edit's leading boolean flag
    edit = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"replace_all": False, "file_path": "solution.py", "old_string": "x"}},
    ]}})
    assert parse_stream_line(edit).text == "→ Edit(solution.py)"
    multi = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "python3 -c \"\nimport re\n  x = 1\n\""}},
    ]}})
    assert parse_stream_line(multi).text == '→ Bash(python3 -c " import re x = 1 ")'


def test_running_candidate_detail_shows_backend_tokens_and_elapsed(tmp_path: Path):
    from rich.console import Console

    from hillclimb.candidate import BackendInfo, Candidate, utcnow

    search_dir = make_run_with_search(tmp_path / "runs", "r")
    journal = Journal(search_dir / "journal.jsonl")
    candidate_dir = search_dir / "candidates" / "c009"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "agent_stream.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 1500, "output_tokens": 500}, "content": []}}) + "\n"
    )
    journal.candidate_created(
        Candidate(
            candidate_id="c009", operator="draft", status="running", candidate_dir=str(candidate_dir),
            backend=BackendInfo(name="claude-code", model="sonnet"), created_at=utcnow(),
        )
    )
    journal = Journal(search_dir / "journal.jsonl")
    console = Console(record=True, width=120)
    for renderable in candidate_detail_renderables(_record(search_dir), journal, "c009", live=True):
        console.print(renderable)
    rendered = console.export_text()
    assert "claude-code" in rendered and "model=sonnet" in rendered
    assert "2.0k so far" in rendered
    assert "elapsed" in rendered and "0s" in rendered
    assert "path" in rendered and "candidates/c009" in rendered  # the full working dir (may wrap)
    assert "lineage" in rendered


def test_path_link_is_short_label_with_file_uri(tmp_path: Path):
    from hillclimb.watch import _path_label, _path_link

    path = tmp_path / "runs" / "r1" / "searches" / "cp" / "candidates" / "c003"
    assert _path_label(path) == "runs/…/candidates/c003"
    link = _path_link(path)
    assert link.plain == "runs/…/candidates/c003"
    assert f"link {path.resolve().as_uri()}" in str(link.spans[0].style)
    assert not link.style  # no base style: cell padding must not carry the link
    assert _path_label(Path("/odd/place")) == "/odd/place"  # no runs/ layout: full path


def test_open_in_file_manager_uses_the_desktop_opener(tmp_path: Path, monkeypatch):
    from hillclimb.watch import open_in_file_manager

    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: calls.append(cmd))
    open_in_file_manager(tmp_path)
    assert calls[0][-1] == str(tmp_path)
    assert calls[0][0] in ("open", "xdg-open")


def test_scrollbars_are_whole_cell_and_one_wide():
    from textual.scrollbar import ScrollBar

    from hillclimb.theme import WholeCellScrollBarRender

    assert ScrollBar.renderer is WholeCellScrollBarRender
    # a fractional position: the stock renderer would draw ▁/▃ partial cells
    segments = WholeCellScrollBarRender.render_bar(
        size=10, virtual_size=37, window_size=10, position=7.3, thickness=1, vertical=True
    ).segments
    assert {s.text for s in segments} == {" "}
    thumb = [s for s in segments if s.style.reverse]
    assert 1 <= len(thumb) < 10  # a real thumb, not the whole track


@pytest.mark.asyncio
async def test_detail_and_table_scrollbars_are_one_cell(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "bar-run")
    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")
        await pilot.press("o")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#candidates").styles.scrollbar_size_vertical == 1
        assert app.screen.query_one("#candidate-detail").styles.scrollbar_size_vertical == 1


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
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
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
async def test_t_opens_the_tree_panel_and_follows_the_cursor(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "tree-run")
    # a second search in the same run, with a distinct extra candidate
    second = make_run_with_search(
        tmp_path / "runs", "tree-run", SearchStatus(search_id="cp-2", run_id="tree-run", state="done"),
        search_id="cp-2",
    )
    journal2 = Journal(second / "journal.jsonl")
    # scrubber ticks come from finished_at (real searches always stamp it)
    for i, (cid, kwargs) in enumerate([
        ("c000", dict(operator="baseline", status="ok")),
        ("c001", dict(operator="draft", status="ok", val_score=0.7)),
        ("c777", dict(operator="improve", parent_id="c001", status="ok", val_score=0.9)),
    ]):
        candidate = make_candidate(cid, **kwargs)
        candidate.finished_at = f"2026-08-23T10:0{i}:00+00:00"
        journal2.candidate_result(candidate)

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("t")
        await pilot.pause()
        tree = app.screen.query_one("#search-tree")
        assert str(tree.styles.display) != "none"
        # opens as tall as the searches allow (2 searches: fit under both)
        assert app.screen._detail_height == app.screen._fit_detail_height(max_table_rows=8)
        first_ids = {n.id for n in tree._tree.nodes}
        await pilot.press("down")  # cursor to the second search: the tree follows
        await pilot.pause()
        second_ids = {n.id for n in tree._tree.nodes}
        assert first_ids != second_ids and "c777" in second_ids
        # j scrubs back in time: the newest result drops out of the tree
        scrubber = app.screen.query_one("#search-scrubber")
        assert str(scrubber.styles.display) != "none"
        n_live = len(tree._tree.nodes)
        await pilot.press("j")
        await pilot.pause()
        assert scrubber.index is not None
        assert len(tree._tree.nodes) < n_live
        await pilot.press("k")  # forward again
        await pilot.pause()
        assert len(tree._tree.nodes) == n_live

        # selecting a node slides the candidate detail out on the right
        detail = app.screen.query_one("#search-node-detail")
        assert str(detail.styles.display) == "none"
        tree.selected = "c001"
        app.screen._show_node_detail("c001")
        assert str(detail.styles.display) == "block"
        await pilot.pause()
        assert detail.max_scroll_x == 0  # content fits the dock: no horizontal scroll
        # stream entries stay one row each: the long tool line scrolls, not wraps
        cdir = second / "candidates" / "c777"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "agent_stream.jsonl").write_text(json.dumps({
            "type": "assistant", "ts": "2026-08-24T04:00:00+00:00",
            "message": {"content": [{"type": "tool_use", "name": "Bash",
                                     "input": {"command": "x" * 200}}]},
        }) + "\n")
        app.screen._show_node_detail("c777")
        await pilot.pause()
        assert detail.max_scroll_x > 0  # horizontal overflow, no line break

        app.screen._show_node_detail("c001")
        tree.selected = "c001"
        await pilot.pause()
        await pilot.press("escape")  # first escape: deselect/hide the detail, tree stays
        await pilot.pause()
        assert str(detail.styles.display) == "none"
        assert str(tree.styles.display) != "none" and tree.selected is None

        # drilling into a candidate and coming back closes the tree panel
        class _Msg:
            node_id = "c001"

        app.screen.on_tree_plot_widget_node_activated(_Msg())
        await pilot.pause()
        assert app.screen is not app.screen_stack[1]  # candidate screen pushed
        await pilot.press("escape")  # closes the candidate screen's own detail
        await pilot.press("escape")  # pops back to the searches screen
        await pilot.pause()
        searches_screen = app.screen
        assert searches_screen._tree_open is False
        assert str(tree.styles.display) == "none"

        await pilot.press("t")  # reopen, then toggle off
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert str(tree.styles.display) == "none"
        assert str(scrubber.styles.display) == "none"
        assert str(detail.styles.display) == "none"


@pytest.mark.asyncio
async def test_tree_panel_caps_the_searches_table_at_eight_rows(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "cap-run")
    for i in range(11):  # 12 searches in all
        make_run_with_search(
            tmp_path / "runs", "cap-run",
            SearchStatus(search_id=f"cp-{i:02d}", run_id="cap-run", state="done"),
            search_id=f"cp-{i:02d}",
        )
    app = WatchApp(config)
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.press("enter")
        await pilot.press("t")
        await pilot.pause()
        screen = app.screen
        table = screen.query_one("#searches")
        assert table.row_count == 12
        hbar = 1 if table.show_horizontal_scrollbar else 0
        # the tree takes everything past 8 table rows (+ header + chrome)
        assert screen._detail_height == 32 - 4 - (1 + 8 + hbar)


@pytest.mark.asyncio
async def test_hold_column_only_for_holdout_searches(tmp_path: Path):
    from hillclimb.candidate import Candidate, Trial

    search_dir, config = make_demo_search(tmp_path, "hold-run")
    app = WatchApp(config)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("enter")  # open the candidate panel for the search
        await pilot.pause()
        panel = app.screen.query_one("#search-candidates")
        labels = [str(c.label) for c in panel.columns.values()]
        assert "hold" not in labels and "val" in labels  # the demo search has no holdout

        # a holdout score on any candidate brings the column back
        Journal(search_dir / "journal.jsonl").candidate_result(
            Candidate(candidate_id="c009", operator="draft", status="ok",
                      trials=[Trial(val_score=0.9, holdout_score=0.8)])
        )
        app.screen.refresh_data()
        await pilot.pause()
        labels = [str(c.label) for c in panel.columns.values()]
        assert "hold" in labels

        await pilot.press("o")  # full candidate screen follows the same rule
        await pilot.pause()
        labels = [str(c.label) for c in app.screen.query_one("#candidates").columns.values()]
        assert "hold" in labels


@pytest.mark.asyncio
async def test_o_opens_the_selected_candidate_dir(tmp_path: Path, monkeypatch):
    search_dir, config = make_demo_search(tmp_path, "open-run")
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    opened = []
    monkeypatch.setattr("hillclimb.watch.open_in_file_manager", lambda path: opened.append(path))

    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
        await pilot.pause()
        await pilot.press("o")  # on the candidate screen: reveal the selected candidate's dir
        await pilot.pause()
    assert opened == [search_dir / "candidates" / "c000"]


@pytest.mark.asyncio
async def test_candidate_detail_panel_opens_updates_and_closes(tmp_path: Path):
    search_dir, config = make_demo_search(tmp_path, "detail-run")
    (search_dir / "candidates" / "c000").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c001").mkdir(parents=True, exist_ok=True)
    (search_dir / "candidates" / "c000" / "notes.md").write_text("baseline copy\n")
    (search_dir / "candidates" / "c001" / "notes.md").write_text("draft heuristic\n")

    app = WatchApp(config)
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("o")  # -> full candidate screen
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
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
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
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
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
        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
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


def test_search_duration_counts_up_with_the_budget_alongside():
    from hillclimb.watch import _format_budget_total, _format_duration

    assert _format_budget_total(600) == "10m"
    assert _format_budget_total(5400) == "1h 30m"
    assert _format_budget_total(90) == "1m 30s"
    assert _format_duration(247.9, 600) == "4m 07s (budget: 10m)"
    assert _format_duration(600, 600) == "10m 00s (budget: 10m)"
    assert _format_duration(12, 0) == "0m 12s"  # no budget declared
    assert _format_duration(None, 600) == "-"


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


@pytest.mark.asyncio
async def test_searches_screen_inline_candidates_panel(tmp_path: Path):
    from hillclimb.watch import DETAIL_STEP

    _, config = make_demo_search(tmp_path, "panel-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.pause()
        screen = app.screen
        panel = screen.query_one("#search-candidates")
        divider = screen.query_one("#detail-divider")
        assert str(panel.styles.display) == "none" and str(divider.styles.display) == "none"

        await pilot.press("enter")  # open the panel for the highlighted search
        await pilot.pause()
        assert app.screen is screen  # same screen, no push
        assert screen._panel_search_id == "circle-packing"
        assert str(panel.styles.display) == "block"
        assert panel.row_count == 3
        assert [str(panel.get_cell_at((i, 0))).strip() for i in range(3)] == ["c000", "c001", "c002"]

        screen._set_detail_height(DETAIL_MIN_HEIGHT_FOR_TESTS)  # leave room to grow
        await pilot.pause()
        initial = screen._detail_height
        await pilot.press("+")
        assert screen._detail_height == initial + DETAIL_STEP
        await pilot.press("-")
        assert screen._detail_height == initial

        assert await pilot.mouse_down("#detail-divider", offset=(1, 0))
        await pilot.hover(offset=(divider.region.x + 1, divider.region.y - 3))
        await pilot.pause()
        assert screen._detail_height == initial + 3
        await pilot.mouse_up(offset=(divider.region.x + 1, divider.region.y - 3))
        assert not screen._dragging_detail

        await pilot.press("escape")  # closes the panel, stays on the screen
        await pilot.pause()
        assert app.screen is screen and screen._panel_search_id is None
        assert str(panel.styles.display) == "none"
        assert app.focused.id == "searches"

        await pilot.press("enter")  # opens the panel with the cursor in it
        await pilot.pause()
        assert app.focused.id == "search-candidates"
        await pilot.press("down")
        await pilot.press("enter")  # enter on a candidate: full candidates + its details
        await pilot.pause()
        assert app.screen is not screen
        full = app.screen
        assert full.query_one("#candidates").row_count == 3
        assert full._detail_candidate_id == "c001"
        assert full.query_one("#candidates").cursor_row == 1
        assert str(full.query_one("#candidate-detail").styles.display) == "block"


@pytest.mark.asyncio
async def test_searches_panel_cursor_survives_refresh_with_several_searches(tmp_path: Path):
    import asyncio

    runs_dir = tmp_path / "runs"
    make_run_with_search(runs_dir, "multi", search_id="a")  # 3 candidates
    big = make_run_with_search(runs_dir, "multi", search_id="b")
    journal = Journal(big / "journal.jsonl")
    for i in range(3, 8):
        journal.candidate_result(make_candidate(f"c00{i}", operator="improve", parent_id="c001", status="buggy"))
    config = Config()
    config.paths.runs_dir = runs_dir

    app = WatchApp(config)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("enter")  # runs -> searches
        await pilot.press("down")  # highlight "b"
        await pilot.press("enter")  # open its panel
        await pilot.pause()
        screen = app.screen
        assert screen._panel_search_id == "b"
        panel = screen.query_one("#search-candidates")
        screen._set_detail_height(20)
        await pilot.pause()
        rows = [str(panel.get_cell_at((i, 0))).strip() for i in range(panel.row_count)]
        target = rows.index("c007")  # beyond search a's row count
        await pilot.click(offset=(panel.region.x + 3, panel.region.y + 1 + target))
        await pilot.pause()
        assert panel.cursor_row == target
        await asyncio.sleep(1.2)  # a live refresh tick
        await pilot.pause()
        assert screen._panel_search_id == "b"
        assert panel.cursor_row == target


@pytest.mark.asyncio
async def test_m_maximizes_and_restores_the_panel(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "max-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")
        await pilot.press("m")  # nothing open: no-op
        await pilot.pause()
        screen = app.screen
        assert not screen._detail_maximized
        await pilot.press("enter")  # open panel
        await pilot.pause()
        height = screen._detail_height
        table, panel = screen.query_one("#searches"), screen.query_one("#search-candidates")
        await pilot.press("m")
        await pilot.pause()
        assert screen._detail_maximized
        assert str(table.styles.display) == "none" and str(panel.styles.height) == "1fr"
        await pilot.press("m")
        await pilot.pause()
        assert not screen._detail_maximized
        assert str(table.styles.display) == "block" and panel.styles.height.value == height
        await pilot.press("m")
        await pilot.press("escape")  # closing the panel also un-maximizes
        await pilot.pause()
        assert not screen._detail_maximized and str(table.styles.display) == "block"

        await pilot.press("o")  # the candidate screen has the same key for its detail
        await pilot.press("enter")
        await pilot.press("m")
        await pilot.pause()
        cs = app.screen
        assert cs._detail_maximized and str(cs.query_one("#candidates").styles.display) == "none"


@pytest.mark.asyncio
async def test_detail_renders_full_width_on_first_paint(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "width-run")

    app = WatchApp(config)
    async with app.run_test(size=(120, 30)) as pilot:
        await open_candidate_detail(pilot)
        detail = app.screen.query_one("#candidate-detail")
        # the first write happened before the log had a layout width; the
        # rendered lines must still span the screen, not a default 80 columns
        widths = {len(strip.text.rstrip()) for strip in detail.lines if strip.text.strip()}
        assert max(widths) >= 110, widths


@pytest.mark.asyncio
async def test_detail_is_not_rewritten_when_nothing_changed(tmp_path: Path):
    import asyncio

    from textual.widgets import RichLog

    search_dir, config = make_demo_search(tmp_path, "flicker-run")
    clears = []
    original_clear = RichLog.clear

    def counting_clear(self):
        clears.append(self.id)
        return original_clear(self)

    RichLog.clear = counting_clear
    try:
        app = WatchApp(config)
        async with app.run_test(size=(120, 30)) as pilot:
            await open_candidate_detail(pilot)
            detail = app.screen.query_one("#candidate-detail")
            detail.scroll_to(y=0, animate=False)
            baseline = len(clears)
            await asyncio.sleep(2.3)  # two live ticks with identical content
            await pilot.pause()
            assert len(clears) == baseline, "detail was cleared and rewritten on a no-change tick"

            # a change in the journal is still picked up
            journal = Journal(search_dir / "journal.jsonl")
            c000 = journal.candidates["c000"]
            c000.summary = "baseline, now with a new summary"
            journal.candidate_result(c000)
            await asyncio.sleep(1.2)
            await pilot.pause()
            assert len(clears) == baseline + 1
            assert any("new summary" in strip.text for strip in detail.lines)
    finally:
        RichLog.clear = original_clear


@pytest.mark.asyncio
async def test_detail_opens_fitted_under_the_whole_table_and_maximize_hides_divider(tmp_path: Path):
    from hillclimb.watch import DETAIL_CHROME_ROWS

    _, config = make_demo_search(tmp_path, "fit-run")

    app = WatchApp(config)
    async with app.run_test(size=(120, 40)) as pilot:
        await open_candidate_detail(pilot)
        screen = app.screen
        table = screen.query_one("#candidates")
        # 3 rows + header; the detail takes everything else
        expected = 40 - DETAIL_CHROME_ROWS - (1 + 3 + (1 if table.show_horizontal_scrollbar else 0))
        assert screen._detail_height == expected
        assert table.region.height >= 1 + 3  # header and every row visible

        divider = screen.query_one("#detail-divider")
        await pilot.press("m")
        await pilot.pause()
        assert str(divider.styles.display) == "none"
        await asyncio_tick(pilot)  # a live refresh re-renders; divider must stay hidden
        assert str(divider.styles.display) == "none"
        await pilot.press("m")
        await pilot.pause()
        assert str(divider.styles.display) == "block"


async def asyncio_tick(pilot) -> None:
    import asyncio

    await asyncio.sleep(1.2)
    await pilot.pause()


@pytest.mark.asyncio
async def test_header_clock_names_its_zone_and_choice_persists(tmp_path: Path, monkeypatch):
    import re
    from zoneinfo import ZoneInfo

    from textual.widgets import Input

    from hillclimb.header import HillclimbClock, TimezoneChoiceScreen, load_display_timezone

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _, config = make_demo_search(tmp_path, "tz-run")

    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.title == "hillclimb"
        clock = app.screen.query_one(HillclimbClock)
        assert re.fullmatch(r"\d\d:\d\d:\d\d \S+", str(clock.render()))
        # flush with the right edge: no padding past the text
        assert clock.region.x + clock.region.width == app.size.width
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(app.screen, TimezoneChoiceScreen)
        app.screen.query_one("#timezone-filter", Input).value = "tokyo"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.display_timezone == ZoneInfo("Asia/Tokyo")
        assert str(clock.render()).endswith("JST")
    assert load_display_timezone() == ZoneInfo("Asia/Tokyo")


@pytest.mark.asyncio
async def test_searches_runline_says_run(tmp_path: Path):
    _, config = make_demo_search(tmp_path, "runline-run")
    app = WatchApp(config)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert str(app.screen.query_one("#runline").content).startswith("run: Demo")


@pytest.mark.asyncio
async def test_ctrl_c_quits_and_question_mark_lists_every_key(tmp_path: Path):
    from hillclimb.keys import KeysPanel

    _, config = make_demo_search(tmp_path, "keys-run")
    app = WatchApp(config)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("enter")  # searches
        await pilot.pause()
        screen = app.screen
        footer_keys = [b.key for b in screen.BINDINGS if b.show]
        assert footer_keys == ["enter", "escape", "question_mark", "q"]  # lean footer
        assert not screen.query(KeysPanel)
        await pilot.press("question_mark")
        await pilot.pause()
        rows = dict(screen.query_one(KeysPanel).rows())
        assert rows["o"] == "full candidate view" and rows["m"] == "maximize panel"
        assert rows["g"] == "knowledge graph" and rows["esc"] == "back"
        assert rows["drag divider"] == "resize panel"
        assert rows["t"] == "tree panel"  # the screen key shadows the app-level time zone here
        assert screen.query_one("#searches").size.width < 120  # split, not overlay
        await pilot.press("question_mark")
        await pilot.pause()
        assert not screen.query(KeysPanel)

        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code is not None or not app.is_running
