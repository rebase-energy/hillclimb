"""`hillclimb top` — the control pane for every hillclimb process on this machine.

`hillclimb ps`, live and with controls: the machine at a glance (engines,
coding agents against the machine's slot cap, cpu, memory) above ONE table
where each engine heads its own process tree — coding agents, the processes
a coding agent starts for itself (its tool shells, its MCP servers),
verifiers and the solution they score. Rows come from `psview.engine_lines`,
so the two views read the same.

Control acts on the highlighted row: `s` / `g` stop the row's engine (now /
gracefully) through its search's command queue, like `hillclimb stop`; `k`
kills the highlighted process with its descendants — on an engine row the
whole engine, like `hillclimb kill`. Search progress is `watch`'s; this view
does not open it. Machine-wide, not tied to the folder it is started in: each
engine's search records are read through its own hillclimb dir
(`tui/machine.py`).
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Static

from hillclimb.tui.header import APP_TITLE, HillclimbHeader, TimezoneMixin
from hillclimb.tui.keys import KEYS_BINDING, QUIT_BINDINGS
from hillclimb.tui.machine import (
    ENGINE_SORTS,
    EngineRow,
    SearchReader,
    format_mem,
    scan,
    sort_rows,
)
from hillclimb.tui.psview import HEAD_ROLES, ROLE_STYLE, Line, counts, engine_lines, summary_line
from hillclimb.terms import ENGINE, role_label
from hillclimb.tui.theme import HILLCLIMB_CSS, apply_theme
from hillclimb.tui.watch import ConfirmScreen, LiveScreen, _restore_table, _snapshot_table

# the engine orders `o` cycles through; a tree cannot be sorted row by row,
# so it is the engines' blocks that move
TOP_SORTS = [sort for sort in ENGINE_SORTS if sort[0] in ("started", "cpu", "memory", "uptime", "folder")]


def _cpu_style(cpu: float) -> str:
    return "bold red" if cpu >= 90 else "yellow" if cpu >= 40 else ""


class TopScreen(LiveScreen):
    """One table: each engine's row, its processes nested under it."""

    BINDINGS = [
        Binding("s", "stop", f"stop {ENGINE}"),
        Binding("g", "stop_graceful", "stop gracefully", show=False),
        Binding("k", "kill", "kill"),
        Binding("o", "cycle_sort", "order"),
        Binding("r", "reverse_sort", "reverse order", show=False),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    DEFAULT_CSS = """
    TopScreen #machine { height: auto; padding: 0 1; }
    TopScreen #procs { height: 1fr; }
    """

    def __init__(self, agent_slots: int | None = None):
        super().__init__()
        self.agent_slots = agent_slots
        self.reader = SearchReader()
        self.env_dirs: dict[int, Path | None] = {}
        self.engines: dict[int, EngineRow] = {}
        # every row's pid -> (its engine, its line)
        self.rows: dict[int, tuple[EngineRow, Line]] = {}
        self.sort = 0
        self.reverse = False

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Static(id="machine")
        yield DataTable(id="procs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#procs", DataTable)
        table.add_columns("pid", "role", "cpu", "mem", "up", "command")
        table.focus()
        self.start_live()

    # --- data ---

    def refresh_data(self) -> None:
        machine, engines = scan(
            reader=self.reader, env_dirs=self.env_dirs, agent_slots=self.agent_slots, root_guides=True
        )
        self.engines = {e.pid: e for e in engines}
        engine_count, jobs = counts(engines)
        summary = summary_line(machine, engine_count, max(40, self.size.width - 2), jobs)
        label, _, _ = TOP_SORTS[self.sort]
        summary.append(f"   order: {label}{' ↑' if self.reverse else ''}", style="dim")
        self.query_one("#machine", Static).update(summary)

        table = self.query_one("#procs", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        self.rows = {}
        # the command column takes what the others leave
        width = max(30, self.size.width - 44)
        for engine in sort_rows(engines, TOP_SORTS, self.sort, self.reverse):
            for line in engine_lines(engine, fold_mcp=False):
                pid = int(line.pid)
                self.rows[pid] = (engine, line)
                command = line.command if len(line.command) <= width else line.command[: width - 1] + "…"
                table.add_row(
                    Text(line.pid, style="bold" if line.role in HEAD_ROLES else ""),
                    Text(role_label(line.role), style=ROLE_STYLE.get(line.role, "")),
                    Text(f"{line.cpu:5.1f}%", style=_cpu_style(line.cpu)),
                    format_mem(line.rss_mb),
                    line.up,
                    # no_wrap: wrapping would strip the tree guide's leading spaces
                    Text(
                        command,
                        style="bold" if line.role in HEAD_ROLES else "dim" if line.role == "mcp" else "",
                        no_wrap=True,
                    ),
                    key=line.pid,
                )
        if not engines:
            table.add_row("", "", "", "", "", Text(f"no hillclimb {ENGINE.pl} running", style="dim"), key="none")
        _restore_table(table, snapshot)

    def _selected(self) -> tuple[EngineRow, Line] | None:
        table = self.query_one("#procs", DataTable)
        if not table.row_count:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        if row_key.value in (None, "none"):
            return None
        return self.rows.get(int(row_key.value))

    # --- control ---

    def _stop(self, graceful: bool) -> None:
        selected = self._selected()
        if selected is None:
            return
        engine, _ = selected
        if engine.kind != "engine":
            self.notify(f"a {engine.kind} in a terminal has no search to stop — k ends it", severity="warning")
            return
        if engine.orphan or engine.hillclimb_dir is None:
            self.notify("its hillclimb dir is gone, so nothing receives a stop — k kills it", severity="warning")
            return
        if not engine.searches:
            self.notify("no search recorded for this process yet — k kills it", severity="warning")
            return
        pair = self.reader.config_store(engine.hillclimb_dir)
        if pair is None:
            self.notify(f"cannot read {engine.hillclimb_dir}", severity="error")
            return
        _, store = pair
        refs = ", ".join(s.ref for s in engine.searches)
        how = "let what is in flight finish, then park" if graceful else "abort what is in flight, then park"

        def go(confirmed: bool | None) -> None:
            if not confirmed:
                return
            from hillclimb.harness.control import request_stop
            from hillclimb.harness.store import key_for

            for search in engine.searches:
                outcome = request_stop(store, key_for(search.search_dir), source="tui", graceful=graceful)
                self.notify(outcome or f"{search.ref}: {ENGINE} is not running", severity="information")

        self.app.push_screen(ConfirmScreen(f"Stop engine {engine.pid} ({refs})? {how}; resumable."), go)

    def action_stop(self) -> None:
        self._stop(graceful=False)

    def action_stop_graceful(self) -> None:
        self._stop(graceful=True)

    def action_kill(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        engine, line = selected
        whole_engine = line.role == "engine"
        if line.role in HEAD_ROLES and not whole_engine:
            question = f"End {role_label(line.role)} {line.pid} and what it started? (SIGTERM, then SIGKILL)"
        elif whole_engine:
            question = (
                f"Kill engine {engine.pid} and its {len(engine.procs)} processes? "
                "SIGTERM, then SIGKILL; the search stays resumable."
            )
        else:
            question = (
                f"Kill {role_label(line.role)} {line.pid} and what it started? "
                f"The {ENGINE} sees it fail like any other crash and carries on."
            )

        def go(confirmed: bool | None) -> None:
            if not confirmed:
                return
            from hillclimb.harness.orphans import kill_engines, kill_process_tree

            # off the UI thread: both wait out the SIGTERM grace
            def work() -> None:
                if whole_engine:
                    forced = bool(kill_engines([engine.as_engine()]))
                else:
                    forced = kill_process_tree(int(line.pid))
                self.app.call_from_thread(
                    self.notify, f"{role_label(line.role)} {line.pid} killed" + (" (needed SIGKILL)" if forced else "")
                )

            self.run_worker(work, thread=True, exclusive=False)

        self.app.push_screen(ConfirmScreen(question), go)

    # --- order ---

    def action_cycle_sort(self) -> None:
        self.sort = (self.sort + 1) % len(TOP_SORTS)
        self.notify(f"{ENGINE.pl} ordered by {TOP_SORTS[self.sort][0]}", timeout=1.5)
        self.refresh_data()

    def action_reverse_sort(self) -> None:
        self.reverse = not self.reverse
        self.refresh_data()


class TopApp(TimezoneMixin, App):
    """`hillclimb top`: machine-wide; stop goes through each search's command
    queue, kill by signal, like the CLI."""

    TITLE = f"{APP_TITLE} top"  # the header names the command that opened it
    ALLOW_SELECT = False
    BINDINGS = [
        Binding("t", "choose_timezone", "time zone", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = HILLCLIMB_CSS

    def __init__(self, agent_slots: int | None = None):
        super().__init__()
        self.agent_slots = agent_slots

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(TopScreen(agent_slots=self.agent_slots))
