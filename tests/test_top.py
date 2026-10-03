"""`hillclimb top`: the machine-wide live view of engines and their processes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hillclimb.harness.control import read_commands
from hillclimb.harness.orphans import classify, process_table
from hillclimb.harness.status import CurrentCandidate, ScoreRef, SearchStatus
from hillclimb.tui import top
from hillclimb.tui import machine
from hillclimb.tui.machine import (
    ENGINE_SORTS,
    PROC_SORTS,
    SearchReader,
    machine_line,
    parse_etime,
    process_tree,
    scan,
    sort_rows,
)
from tests.test_watch import make_run_with_search

ENGINE = os.getpid()  # the status record's pid must be a live engine's

LISTING = (
    f"{ENGINE} 1 {ENGINE} 0.5 40000 05:00 /venv/bin/python3 -m hillclimb.cli run heilbronn-11 --run-id run-1 --run-name x\n"
    f"901 {ENGINE} 901 1.0 200000 04:00 claude -p --output-format stream-json --model sonnet\n"
    "902 901 901 0.0 40000 03:59 npm exec chrome-devtools-mcp@latest\n"
    "903 902 901 0.0 30000 03:58 node /x/chrome-devtools-mcp/build/index.js\n"
    "904 901 901 0.0 7000 03:59 python3 -m autoharness.stage_skill.server\n"
    "905 901 901 0.0 1000 00:30 /bin/zsh -c source snapshot.sh && python solution.py\n"
    "906 905 901 99.0 30000 00:30 python solution.py\n"
    f"907 {ENGINE} 907 0.0 1000 00:10 /bin/bash /p/heilbronn-11/verifier.sh\n"
    "908 907 907 97.0 17000 00:09 /venv/bin/python solution.py\n"
    "300 1 300 0.0 3000 1-02:00:00 /usr/bin/python3 -m hillclimb.cli watch\n"
)


def test_classify_tells_the_coding_agents_own_runs_from_scored_ones():
    assert classify("claude -p --output-format stream-json", None) == "agent"
    assert classify("sandbox-exec -p (version 1) claude -p --verbose", None) == "agent"
    assert classify("/usr/local/bin/codex exec --json", None) == "agent"
    assert classify("pi -p --mode json", None) == "agent"
    # a coding agent's Bash tool runs through a shell; any other direct
    # child is a server from the user's own Claude config
    assert classify("/bin/zsh -c source snap.sh", "agent") == "tool"
    assert classify("bash -c ls", "agent") == "tool"
    assert classify("python3 -m autoharness.stage_skill.server", "agent") == "mcp"
    assert classify("npm exec chrome-devtools-mcp@latest", "agent") == "mcp"
    assert classify("python solution.py", "tool") == "tool"  # trying its code, not scored
    assert classify("node index.js", "mcp") == "mcp"
    assert classify("bash /p/verifier.sh", None) == "verifier"
    assert classify("C:/py/python.exe /p/verifier.py", None) == "verifier"
    assert classify("python solution.py", "verifier") == "solution"
    assert classify("python worker.py", "solution") == "solution"
    assert classify("python something.py", None) == "child"


def test_parse_etime():
    assert parse_etime("00:09") == 9
    assert parse_etime("05:00") == 300
    assert parse_etime("01:02:03") == 3723
    assert parse_etime("1-02:00:00") == 93600
    assert parse_etime("?") == 0


def test_process_tree_is_preorder_with_roles_and_guides():
    rows = process_tree(ENGINE, process_table(LISTING))
    assert [(r.pid, r.role, r.depth) for r in rows] == [
        (901, "agent", 0),
        (902, "mcp", 1),
        (903, "mcp", 2),
        (904, "mcp", 1),
        (905, "tool", 1),
        (906, "tool", 2),
        (907, "verifier", 0),
        (908, "solution", 1),
    ]
    guides = {r.pid: r.guide for r in rows}
    assert guides[901] == "" and guides[907] == ""
    assert guides[902] == "├─ " and guides[903] == "│  └─ "
    assert guides[905] == "└─ " and guides[906] == "   └─ "


@pytest.fixture
def hillclimb_dir(tmp_path, monkeypatch):
    """A hillclimb dir with one search whose status names ENGINE."""
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hillclimb.yaml").write_text("{}\n")
    make_run_with_search(
        root / "runs",
        "run-1",
        SearchStatus(
            search_id="circle-packing",
            run_id="run-1",
            state="running",
            pid=ENGINE,
            current=[
                CurrentCandidate(candidate_id="c003", operator="improve", phase="agent", candidate_dir="x")
            ],
            best=ScoreRef(candidate_id="c001", val_score=0.7),
        ),
    )
    return root


def _scan(hillclimb_dir, listing=LISTING):
    return scan(
        table=process_table(listing),
        reader=SearchReader(),
        env_dirs={ENGINE: hillclimb_dir},
        agent_slots=8,
    )


def test_scan_reads_each_engines_search_through_its_own_dir(hillclimb_dir):
    machine, engines = _scan(hillclimb_dir)
    assert [e.pid for e in engines] == [ENGINE]  # `watch` is not an engine
    engine = engines[0]
    assert engine.agents == 1 and len(engine.procs) == 8
    assert engine.cpu == pytest.approx(0.5 + 1.0 + 99.0 + 97.0)
    search = engine.search
    assert search.ref == "run-1/circle-packing"
    assert search.best == 0.7
    assert search.in_flight == ["c003 improve (agent)"]
    assert machine.agents == 1 and machine.agent_slots == 8
    line = machine_line(machine, len(engines))
    assert "1 engine " in line and "coding agents 1/8" in line


def test_scan_survives_a_deleted_dir_and_drops_dead_pids(tmp_path):
    gone = tmp_path / "gone"
    env_dirs = {ENGINE: gone, 12345: tmp_path}
    _, engines = scan(table=process_table(LISTING), reader=SearchReader(), env_dirs=env_dirs, agent_slots=8)
    assert engines[0].orphan and engines[0].searches == []
    assert 12345 not in env_dirs  # no such process any more


def test_sorts():
    rows = process_tree(ENGINE, process_table(LISTING))
    cpu = PROC_SORTS.index(next(s for s in PROC_SORTS if s[0] == "cpu"))
    assert [r.pid for r in sort_rows(rows, PROC_SORTS, cpu, False)][:2] == [906, 908]
    assert [r.pid for r in sort_rows(rows, PROC_SORTS, cpu, True)][-2:] == [908, 906]
    assert sort_rows(rows, PROC_SORTS, 0, False) == rows  # tree order
    assert len({label for label, _, _ in ENGINE_SORTS}) == len(ENGINE_SORTS)


@pytest.fixture
def live(monkeypatch, hillclimb_dir):
    monkeypatch.setattr(machine, "process_table", lambda: process_table(LISTING))
    monkeypatch.setattr(machine, "_environ_dir", lambda pid: hillclimb_dir)
    return hillclimb_dir


@pytest.mark.asyncio
async def test_top_is_one_table_with_each_engine_heading_its_tree(live):
    app = top.TopApp(agent_slots=8)
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        assert app.title == "hillclimb top"
        table = app.screen.query_one("#procs")
        assert table.row_count == 9  # the engine + its 8 processes
        assert str(table.get_row_at(0)[1]) == "engine"
        assert "heilbronn-11" in str(table.get_row_at(0)[5])
        await pilot.press("enter")  # nothing to open: top does not jump into watch
        await pilot.pause()
        assert type(app.screen).__name__ == "TopScreen"


@pytest.mark.asyncio
async def test_top_stops_the_rows_engine_through_the_queue(live):
    app = top.TopApp(agent_slots=8)
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        screen = app.screen
        await pilot.press("o")
        assert screen.sort == 1
        await pilot.press("r")
        assert screen.reverse
        await pilot.press("down", "down", "s", "n")  # a process row stops its engine; cancelled
        await pilot.pause()
        search_dir = live / "runs" / "run-1" / "searches" / "circle-packing"
        assert read_commands(search_dir) == []
        await pilot.press("g", "y")
        await pilot.pause()
    commands = read_commands(search_dir)
    assert len(commands) == 1
    assert commands[0][1].action == "stop" and commands[0][1].source == "tui"


@pytest.mark.asyncio
async def test_top_kills_the_highlighted_process_or_engine(live, monkeypatch):
    from hillclimb.harness import orphans

    killed = []
    monkeypatch.setattr(orphans, "kill_process_tree", lambda pid, grace_s=3.0: killed.append(("proc", pid)) or False)
    monkeypatch.setattr(orphans, "kill_engines", lambda engines, grace_s=5.0: killed.append(("engine", engines[0].pid)) or [])
    app = top.TopApp(agent_slots=8)
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        await pilot.press("down", "k", "y")  # row 1: the coding agent
        await pilot.pause(0.5)
        await pilot.press("up", "k", "y")  # row 0: the engine
        await pilot.pause(0.5)
        await app.workers.wait_for_complete()
    assert killed == [("proc", 901), ("engine", ENGINE)]


@pytest.mark.asyncio
async def test_top_app_with_no_engines(monkeypatch):
    monkeypatch.setattr(machine, "process_table", lambda: {})
    app = top.TopApp(agent_slots=8)
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        assert "engines 0" in str(app.screen.query_one("#machine").render())
        await pilot.press("enter", "s", "K")  # nothing selected: no-ops
        await pilot.pause()
        assert type(app.screen).__name__ == "TopScreen"
