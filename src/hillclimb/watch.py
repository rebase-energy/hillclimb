"""`hillclimb watch` — live TUI over the runs directory.

Strictly a *viewer*: all state is read from disk (run.yaml, search.yaml,
status.json, journal.jsonl, agent_stream.jsonl) and the only writes go through
the control-command queue in hillclimb.control, same as the CLI. The pure
data-assembly functions at the top carry the logic so they stay testable
without driving Textual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hillclimb.candidate import Candidate
from hillclimb.config import Config
from hillclimb.control import request_prune, request_stop
from hillclimb.journal import Journal
from hillclimb.run import (
    iter_run_dirs,
    iter_search_dirs,
    load_run_meta,
    load_search_meta,
)
from hillclimb.status import effective_state, read_status

STATE_STYLE = {
    "running": "bold green",
    "parked": "yellow",
    "stopped": "yellow",
    "done": "cyan",
    "failed": "red",
    "crashed": "bold red",
    "unknown": "dim",
}
STATUS_STYLE = {
    "ok": "green",
    "buggy": "red",
    "abandoned": "dim",
    "parked": "yellow",
    "pending": "italic",
}


# --- pure data layer ---


@dataclass
class RunRow:
    run_id: str
    name: str
    state: str
    searches: str
    candidates: str
    selected: str
    budget_left: str
    started: str


@dataclass
class SearchRow:
    search_id: str
    problem: str
    model: str
    state: str
    candidates: str  # "7 (5 ok)"
    best_val: str
    selected: str
    budget_left: str


@dataclass
class CandidateRow:
    candidate_id: str
    label: str  # indent + id
    operator: str
    status: str
    val: str
    hold: str
    marks: str
    summary: str
    style: str


def _fmt(value: float | None) -> str:
    return f"{value:.5g}" if value is not None else "-"


def _format_budget_left(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes = max(0, int(seconds // 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _state_summary(states: list[str]) -> str:
    if not states:
        return "empty"
    for state in ("running", "crashed", "failed", "parked", "stopped"):
        if state in states:
            return state
    if all(state == "done" for state in states):
        return "done"
    return states[0]


def _search_row(search_dir: Path) -> SearchRow:
    meta = load_search_meta(search_dir)
    status = read_status(search_dir)
    journal = Journal(search_dir / "journal.jsonl")
    n_ok = sum(1 for c in journal.candidates.values() if c.status == "ok")
    if status is not None:
        best_val = _fmt(status.best.val_score) if status.best else "-"
        selected = (
            f"{status.selected.candidate_id} "
            f"({_fmt(status.selected.holdout_score or status.selected.val_score)})"
            if status.selected
            else "-"
        )
        budget_left = _format_budget_left(status.budget.remaining_s)
    else:
        best_val, selected, budget_left = "-", "-", "-"
    return SearchRow(
        search_id=search_dir.name,
        problem=meta.problem_id if meta else search_dir.name,
        model=meta.model if meta else "?",
        state=effective_state(search_dir),
        candidates=f"{len(journal.candidates)} ({n_ok} ok)",
        best_val=best_val,
        selected=selected,
        budget_left=budget_left,
    )


def scan_searches(run_dir: Path) -> list[SearchRow]:
    return [_search_row(search_dir) for search_dir in iter_search_dirs(run_dir)]


def _run_row(run_dir: Path) -> RunRow:
    meta = load_run_meta(run_dir)
    search_dirs = iter_search_dirs(run_dir)
    states = [effective_state(search_dir) for search_dir in search_dirs]
    candidate_total = 0
    selected_count = 0
    remaining_s = 0.0
    has_budget = False
    for search_dir in search_dirs:
        journal = Journal(search_dir / "journal.jsonl")
        candidate_total += len(journal.candidates)
        status = read_status(search_dir)
        if status is None:
            continue
        if status.selected is not None:
            selected_count += 1
        remaining_s += status.budget.remaining_s
        has_budget = True
    running = sum(1 for state in states if state == "running")
    searches = f"{len(search_dirs)}" + (f" ({running} running)" if running else "")
    started = meta.started_at if meta else ""
    return RunRow(
        run_id=run_dir.name,
        name=meta.name if meta else run_dir.name,
        state=_state_summary(states),
        searches=searches,
        candidates=str(candidate_total),
        selected=f"{selected_count} selected" if selected_count else "-",
        budget_left=_format_budget_left(remaining_s if has_budget else None),
        started=started[:19].replace("T", " ") if started else "-",
    )


def scan_runs(runs_dir: Path) -> list[RunRow]:
    rows = [_run_row(run_dir) for run_dir in iter_run_dirs(runs_dir)]
    return sorted(rows, key=lambda row: row.started if row.started != "-" else "", reverse=True)


def _tree_order(journal: Journal) -> list[tuple[Candidate, int]]:
    """Depth-first (candidate, depth) pairs so children render under parents."""
    by_parent: dict[str | None, list[Candidate]] = {}
    for candidate in journal.candidates.values():
        by_parent.setdefault(candidate.parent_id, []).append(candidate)
    for children in by_parent.values():
        children.sort(key=lambda c: c.candidate_id)
    ordered: list[tuple[Candidate, int]] = []

    def visit(parent_id: str | None, depth: int) -> None:
        for candidate in by_parent.get(parent_id, []):
            ordered.append((candidate, depth))
            visit(candidate.candidate_id, depth + 1)

    visit(None, 0)
    # orphans (parent vanished from the journal) still get shown
    seen = {c.candidate_id for c, _ in ordered}
    ordered.extend((c, 0) for c in journal.candidates.values() if c.candidate_id not in seen)
    return ordered


def candidate_rows(journal: Journal) -> list[CandidateRow]:
    rows = []
    for candidate, depth in _tree_order(journal):
        marks = []
        if candidate.is_selected and not candidate.pruned:
            marks.append("SELECTED")
        if candidate.is_best:
            marks.append("best-val")
        if candidate.pruned:
            marks.append("PRUNED")
        style = "dim strike" if candidate.pruned else STATUS_STYLE.get(candidate.status, "")
        if candidate.is_selected and not candidate.pruned:
            style = "bold gold1"
        rows.append(
            CandidateRow(
                candidate_id=candidate.candidate_id,
                label="  " * depth + candidate.candidate_id,
                operator=candidate.operator
                + (f"/{candidate.complexity}" if candidate.complexity else ""),
                status=candidate.status,
                val=_fmt(candidate.val_score),
                hold=_fmt(candidate.holdout_score),
                marks=" ".join(marks),
                summary=(candidate.summary or "").strip()[:60],
                style=style,
            )
        )
    return rows


def render_stream_line(raw: str) -> str | None:
    """One compact human line per stream-json message; None = skip."""
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        text = raw.strip()
        return text[:200] if text else None
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if kind == "system":
        return f"[system] {message.get('subtype', '')} session={message.get('session_id', '?')}"
    if kind == "assistant":
        parts = []
        for block in (message.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and block.get("text", "").strip():
                parts.append(block["text"].strip())
            elif block.get("type") == "tool_use":
                args = block.get("input") or {}
                first = next(iter(args.values()), "")
                parts.append(f"→ {block.get('name', 'tool')}({str(first)[:80]})")
        return "\n".join(parts) if parts else None
    if kind == "result":
        cost = message.get("total_cost_usd")
        cost_s = f"${cost:.2f}" if isinstance(cost, (int, float)) else "?"
        return f"[result] {message.get('subtype', '')} turns={message.get('num_turns', '?')} cost={cost_s}"
    return None


def stream_tail(workspace: Path, max_lines: int = 200) -> list[str]:
    path = workspace / "agent_stream.jsonl"
    if not path.exists():
        return []
    rendered = []
    for raw in path.read_text(errors="replace").splitlines()[-max_lines:]:
        line = render_stream_line(raw)
        if line:
            rendered.append(line)
    return rendered


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:].strip()


def _candidate_workspace(search_dir: Path, candidate: Candidate) -> Path:
    if candidate.workspace:
        return Path(candidate.workspace)
    return search_dir / "candidates" / candidate.candidate_id


def _candidate_marks(candidate: Candidate) -> str:
    marks = []
    if candidate.is_selected and not candidate.pruned:
        marks.append("selected")
    if candidate.is_best:
        marks.append("best-val")
    if candidate.pruned:
        marks.append("pruned")
    return ", ".join(marks) if marks else "-"


def _ancestry(journal: Journal, candidate: Candidate) -> list[str]:
    lineage = [candidate.candidate_id]
    seen = {candidate.candidate_id}
    parent_id = candidate.parent_id
    while parent_id and parent_id not in seen and parent_id in journal.candidates:
        lineage.insert(0, parent_id)
        seen.add(parent_id)
        parent_id = journal.candidates[parent_id].parent_id
    if parent_id and parent_id not in journal.candidates:
        lineage.insert(0, f"{parent_id} (missing)")
    return lineage


def _metric_context(search_dir: Path) -> tuple[str, str]:
    meta = load_search_meta(search_dir)
    lower = bool(meta.lower_is_better) if meta else False
    metric = meta.metric if meta else "score"
    direction = "lower is better" if lower else "higher is better"
    return metric, direction


def candidate_detail_lines(search_dir: Path, journal: Journal, candidate_id: str) -> list[str]:
    candidate = journal.candidates.get(candidate_id)
    if candidate is None:
        return [f"Candidate {candidate_id} is no longer in the journal."]

    metric, direction = _metric_context(search_dir)
    operator = candidate.operator + (f"/{candidate.complexity}" if candidate.complexity else "")
    workspace = _candidate_workspace(search_dir, candidate)
    children = journal.children(candidate.candidate_id, include_pruned=True)
    parent = candidate.parent_id if candidate.parent_id else "root"
    trial = candidate.last_trial

    lines = [
        f"Candidate {candidate.candidate_id} | {operator} | {candidate.status}",
        f"Score: val={_fmt(candidate.val_score)}  holdout={_fmt(candidate.holdout_score)}  "
        f"metric={metric} ({direction})",
        f"Marks: {_candidate_marks(candidate)}",
        f"Parent: {parent}  Children: {len(children)}  "
        f"Path: {' -> '.join(_ancestry(journal, candidate))}",
    ]
    if trial is not None:
        duration = f"{trial.duration_s:.5g}s" if trial.duration_s is not None else "-"
        lines.append(
            "Trial: "
            f"returncode={trial.returncode if trial.returncode is not None else '-'}  "
            f"duration={duration}  "
            f"timed_out={trial.timed_out}  "
            f"submission_ok={trial.submission_ok}"
        )
        if trial.holdout_error:
            lines.append(f"Holdout error: {trial.holdout_error}")
    else:
        lines.append("Trial: (not executed)")
    if candidate.backend.name or candidate.backend.session_id or candidate.backend.error_kind:
        cost = f"${candidate.backend.cost_usd:.2f}" if candidate.backend.cost_usd is not None else "-"
        lines.append(
            "Backend: "
            f"{candidate.backend.name or '-'}  "
            f"session={candidate.backend.session_id or '-'}  "
            f"turns={candidate.backend.num_turns if candidate.backend.num_turns is not None else '-'}  "
            f"cost={cost}  "
            f"error={candidate.backend.error_kind or '-'}"
        )

    if children:
        lines += ["", "Children:"]
        for child in children[:12]:
            child_op = child.operator + (f"/{child.complexity}" if child.complexity else "")
            lines.append(
                f"  {child.candidate_id}  {child_op}  {child.status}  val={_fmt(child.val_score)}"
                + ("  PRUNED" if child.pruned else "")
            )
        if len(children) > 12:
            lines.append(f"  ... {len(children) - 12} more")

    notes = _tail_text(workspace / "notes.md", max_chars=3000) or candidate.summary.strip()
    if notes:
        lines += ["", "Notes:", notes]

    stderr = _tail_text(workspace / "exec_stderr.log", max_chars=3000)
    stdout = _tail_text(workspace / "exec_stdout.log", max_chars=3000)
    if stderr:
        lines += ["", "Stderr:", stderr]
    if stdout:
        lines += ["", "Stdout:", stdout]

    stream = stream_tail(workspace, max_lines=80)
    if stream:
        lines += ["", "Agent stream:"]
        lines.extend(stream)

    return lines


def _status_text(status: str):
    from rich.text import Text

    return Text(status, style=STATUS_STYLE.get(status, ""))


def _score_text(value: float | None, *, selected: bool = False, best: bool = False):
    from rich.text import Text

    if value is None:
        return Text("-", style="dim")
    style = "bold gold1" if selected else "bold cyan" if best else "cyan"
    return Text(_fmt(value), style=style)


def _plain_text(value: str, style: str = ""):
    from rich.text import Text

    return Text(value, style=style)


def candidate_detail_renderables(search_dir: Path, journal: Journal, candidate_id: str) -> list[object]:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    candidate = journal.candidates.get(candidate_id)
    if candidate is None:
        return [Panel(Text(f"Candidate {candidate_id} is no longer in the journal.", style="red"))]

    metric, direction = _metric_context(search_dir)
    operator = candidate.operator + (f"/{candidate.complexity}" if candidate.complexity else "")
    workspace = _candidate_workspace(search_dir, candidate)
    children = journal.children(candidate.candidate_id, include_pruned=True)
    parent = candidate.parent_id if candidate.parent_id else "root"
    trial = candidate.last_trial
    title_style = "dim" if candidate.pruned else STATUS_STYLE.get(candidate.status, "")
    border_style = "red" if candidate.status == "buggy" else "yellow" if candidate.pruned else "cyan"

    overview = Table.grid(expand=True)
    overview.add_column("label", style="dim", ratio=1)
    overview.add_column("value", ratio=3)
    overview.add_column("label", style="dim", ratio=1)
    overview.add_column("value", ratio=3)
    overview.add_row(
        "candidate",
        Text(candidate.candidate_id, style=f"bold {title_style}".strip()),
        "operator",
        Text(operator, style="bold"),
    )
    overview.add_row(
        "status", _status_text(candidate.status), "marks", Text(_candidate_marks(candidate), style="gold1")
    )
    overview.add_row("metric", Text(f"{metric} ({direction})"), "parent", Text(parent, style="cyan"))
    overview.add_row(
        "val",
        _score_text(candidate.val_score, selected=candidate.is_selected, best=candidate.is_best),
        "holdout",
        _score_text(candidate.holdout_score, selected=candidate.is_selected),
    )
    overview.add_row(
        "children",
        Text(str(len(children)), style="cyan" if children else "dim"),
        "path",
        Text(" -> ".join(_ancestry(journal, candidate)), style="cyan"),
    )
    if trial is not None:
        duration = f"{trial.duration_s:.5g}s" if trial.duration_s is not None else "-"
        overview.add_row(
            "returncode",
            Text(str(trial.returncode) if trial.returncode is not None else "-", style="dim"),
            "duration",
            Text(duration, style="cyan" if trial.duration_s is not None else "dim"),
        )
        overview.add_row(
            "timed out",
            Text(str(trial.timed_out), style="red" if trial.timed_out else "dim"),
            "submission",
            Text("ok" if trial.submission_ok else "-", style="green" if trial.submission_ok else "dim"),
        )
        if trial.holdout_error:
            overview.add_row("holdout error", Text(trial.holdout_error, style="yellow"), "", "")
    else:
        overview.add_row("trial", Text("(not executed)", style="dim"), "", "")
    if candidate.backend.name or candidate.backend.session_id or candidate.backend.error_kind:
        cost = f"${candidate.backend.cost_usd:.2f}" if candidate.backend.cost_usd is not None else "-"
        backend = (
            f"{candidate.backend.name or '-'}  session={candidate.backend.session_id or '-'}  "
            f"turns={candidate.backend.num_turns if candidate.backend.num_turns is not None else '-'}  "
            f"cost={cost}  error={candidate.backend.error_kind or '-'}"
        )
        overview.add_row("backend", Text(backend), "", "")

    renderables: list[object] = [
        Panel(
            overview,
            title=f"Candidate {candidate.candidate_id}",
            title_align="left",
            border_style=border_style,
        )
    ]

    if children:
        child_table = Table(title="Children", title_style="bold", expand=True)
        child_table.add_column("candidate", style="cyan", no_wrap=True)
        child_table.add_column("operator", no_wrap=True)
        child_table.add_column("status", no_wrap=True)
        child_table.add_column("val", justify="right", no_wrap=True)
        child_table.add_column("marks")
        for child in children[:12]:
            child_op = child.operator + (f"/{child.complexity}" if child.complexity else "")
            child_table.add_row(
                child.candidate_id,
                child_op,
                _status_text(child.status),
                _score_text(child.val_score, selected=child.is_selected, best=child.is_best),
                Text(_candidate_marks(child), style="gold1" if not child.pruned else "dim"),
                style="dim" if child.pruned else "",
            )
        if len(children) > 12:
            child_table.add_row(f"... {len(children) - 12} more", "", "", "", "", style="dim")
        renderables.append(child_table)

    notes = _tail_text(workspace / "notes.md", max_chars=3000) or candidate.summary.strip()
    if notes:
        renderables.append(
            Panel(Text(notes), title="Notes", title_align="left", border_style="green")
        )

    stderr = _tail_text(workspace / "exec_stderr.log", max_chars=3000)
    stdout = _tail_text(workspace / "exec_stdout.log", max_chars=3000)
    if stderr:
        renderables.append(
            Panel(Text(stderr, style="red"), title="Stderr", title_align="left", border_style="red")
        )
    if stdout:
        renderables.append(
            Panel(Text(stdout), title="Stdout", title_align="left", border_style="blue")
        )

    stream = stream_tail(workspace, max_lines=80)
    if stream:
        renderables.append(
            Panel(
                Group(*(_plain_text(line) for line in stream)),
                title="Agent stream",
                title_align="left",
                border_style="magenta",
            )
        )

    return renderables


def _lower_is_better(config: Config, search_dir: Path) -> bool:
    meta = load_search_meta(search_dir)
    return bool(meta.lower_is_better) if meta else False


def _search_ref(search_dir: Path) -> str:
    return f"{search_dir.parents[1].name}/{search_dir.name}"


# --- Textual app ---

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual import events  # noqa: E402
from textual.scrollbar import ScrollBar  # noqa: E402
from textual.screen import ModalScreen, Screen  # noqa: E402
from textual.widgets import DataTable, Footer, Header, Label, RichLog, Static  # noqa: E402

REFRESH_S = 2.0
DETAIL_DEFAULT_HEIGHT = 10
DETAIL_MIN_HEIGHT = 6
DETAIL_STEP = 2
DETAIL_MAX_FRACTION = 0.7


@dataclass(frozen=True)
class ScrollSnapshot:
    x: float
    y: float


@dataclass(frozen=True)
class TableSnapshot(ScrollSnapshot):
    cursor_row: int
    cursor_column: int
    row_key: object | None


def _restore_scroll(widget, snapshot: ScrollSnapshot) -> None:
    """Restore both scroll axes after live refreshes mutate widget content."""

    def apply() -> None:
        widget.scroll_to(
            x=min(max(snapshot.x, 0), widget.max_scroll_x),
            y=min(max(snapshot.y, 0), widget.max_scroll_y),
            animate=False,
            immediate=True,
            force=True,
        )

    apply()
    widget.call_after_refresh(apply)


def _restore_scroll_x(widget, x: float) -> None:
    def apply() -> None:
        widget.scroll_to(
            x=min(max(x, 0), widget.max_scroll_x),
            animate=False,
            immediate=True,
            force=True,
        )

    apply()
    widget.call_after_refresh(apply)


def _snapshot_table(table: DataTable) -> TableSnapshot:
    row_key = None
    if table.row_count:
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:  # Textual may reject stale cursor coordinates mid-refresh.
            row_key = None
    return TableSnapshot(
        x=table.scroll_x,
        y=table.scroll_y,
        cursor_row=table.cursor_row,
        cursor_column=table.cursor_column,
        row_key=row_key,
    )


def _restore_table(table: DataTable, snapshot: TableSnapshot) -> None:
    if table.row_count:
        row = min(max(snapshot.cursor_row, 0), table.row_count - 1)
        if snapshot.row_key is not None:
            try:
                row = table.get_row_index(snapshot.row_key)
            except Exception:
                pass
        column_count = len(table.columns)
        column = min(max(snapshot.cursor_column, 0), column_count - 1) if column_count else None
        table.move_cursor(row=row, column=column, scroll=False)
    _restore_scroll(table, snapshot)


class ConfirmScreen(ModalScreen[bool]):
    """y/n confirmation for stop and prune."""

    BINDINGS = [Binding("y", "confirm", "yes"), Binding("n,escape", "cancel", "no")]

    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Static {
        width: auto; max-width: 80; padding: 1 3;
        background: $surface; border: thick $warning;
    }
    """

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        yield Static(f"{self.question}\n\n[b]y[/b] confirm   [b]n[/b] cancel")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


def _mouse_event_y(event: events.MouseEvent) -> int:
    if getattr(event, "_screen_y", None) is not None:
        return int(event._screen_y)
    if event.screen_y is not None:
        return int(event.screen_y)
    widget = getattr(event, "widget", None)
    if widget is not None:
        return int(widget.region.y + event.y)
    return int(event.y)


def _mouse_event_x(event: events.MouseEvent) -> int:
    if getattr(event, "_screen_x", None) is not None:
        return int(event._screen_x)
    if event.screen_x is not None:
        return int(event.screen_x)
    widget = getattr(event, "widget", None)
    if widget is not None:
        return int(widget.region.x + event.x)
    return int(event.x)


class DetailDivider(Static):
    """Draggable splitter between the candidate table and detail panel."""

    def on_mouse_down(self, event: events.MouseDown) -> None:
        screen = self.screen
        if getattr(screen, "_detail_candidate_id", None) is None:
            return
        screen._begin_detail_drag(_mouse_event_y(event))
        self.capture_mouse()
        event.prevent_default()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        screen = self.screen
        if not getattr(screen, "_dragging_detail", False):
            return
        screen._drag_detail_to(_mouse_event_y(event))
        event.prevent_default()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        screen = self.screen
        if not getattr(screen, "_dragging_detail", False):
            return
        screen._end_detail_drag()
        self.release_mouse()
        event.prevent_default()
        event.stop()


class CandidateHorizontalScrollBar(ScrollBar):
    """Horizontal table scrollbar that doubles as the detail resize handle."""

    DIRECTION_THRESHOLD = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_drag_x = 0
        self._last_drag_y = 0
        self._drag_active = False

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        screen = self.screen
        if getattr(screen, "_detail_candidate_id", None) is None:
            await super()._on_mouse_down(event)
            return
        self._drag_active = True
        self._last_drag_x = _mouse_event_x(event)
        self._last_drag_y = _mouse_event_y(event)
        self.capture_mouse()
        event.prevent_default()
        event.stop()

    async def _on_mouse_move(self, event: events.MouseMove) -> None:
        screen = self.screen
        if not self._drag_active:
            await super()._on_mouse_move(event)
            return
        if getattr(screen, "_detail_candidate_id", None) is None:
            await super()._on_mouse_move(event)
            return
        x = _mouse_event_x(event)
        y = _mouse_event_y(event)
        dx = x - self._last_drag_x
        dy = y - self._last_drag_y
        if max(abs(dx), abs(dy)) < self.DIRECTION_THRESHOLD:
            event.prevent_default()
            event.stop()
            return
        if abs(dx) > abs(dy):
            if isinstance(self.parent, DataTable):
                self.parent.scroll_to(
                    x=max(0, min(self.parent.max_scroll_x, self.parent.scroll_x + dx)),
                    animate=False,
                    immediate=True,
                    force=True,
                )
        else:
            screen._set_detail_height(screen._detail_height - dy)
        self._last_drag_x = x
        self._last_drag_y = y
        event.prevent_default()
        event.stop()

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        if not self._drag_active:
            await super()._on_mouse_up(event)
            return
        self._drag_active = False
        self.release_mouse()
        event.prevent_default()
        event.stop()


class CandidateTable(DataTable):
    """Candidate tree table whose horizontal scrollbar can also resize details."""

    @property
    def horizontal_scrollbar(self) -> ScrollBar:
        if self._horizontal_scrollbar is not None:
            return self._horizontal_scrollbar
        self._horizontal_scrollbar = scroll_bar = CandidateHorizontalScrollBar(
            vertical=False,
            name="horizontal",
            thickness=self.scrollbar_size_horizontal,
        )
        self._horizontal_scrollbar.display = False
        self.app._start_widget(self, scroll_bar)
        return scroll_bar


class CandidateScreen(Screen):
    """One search: candidate tree plus optional selected-candidate details."""

    BINDINGS = [
        Binding("escape", "close_detail_or_back", "back"),
        Binding("enter", "open_detail", "details", priority=True),
        Binding("+", "grow_detail", "larger detail"),
        Binding("-", "shrink_detail", "smaller detail"),
        Binding("s", "stop_search", "stop search"),
        Binding("x", "prune_candidate", "prune candidate"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    CandidateScreen #candidates { height: 1fr; }
    CandidateScreen #detail-divider {
        height: 1;
        background: $surface;
        color: $secondary;
        text-align: center;
    }
    CandidateScreen #candidate-detail {
        min-height: 6;
        padding: 0 1;
    }
    CandidateScreen #searchline { height: 1; padding: 0 1; background: $surface; }
    """

    def __init__(self, config: Config, search_dir: Path):
        super().__init__()
        self.config = config
        self.search_dir = search_dir
        self._detail_candidate_id: str | None = None
        self._detail_height = DETAIL_DEFAULT_HEIGHT
        self._dragging_detail = False
        self._drag_start_y = 0
        self._drag_start_height = DETAIL_DEFAULT_HEIGHT

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(id="searchline")
        yield CandidateTable(id="candidates", cursor_type="row")
        yield DetailDivider(" drag to resize details ", id="detail-divider")
        yield RichLog(id="candidate-detail", wrap=True, markup=False, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#candidates", DataTable)
        table.add_columns("candidate", "operator", "status", "val", "hold", "marks", "summary")
        self._set_detail_visible(False)
        self.refresh_data()
        self.set_interval(REFRESH_S, self.refresh_data)

    def _max_detail_height(self) -> int:
        screen_height = self.size.height or DETAIL_DEFAULT_HEIGHT
        return max(DETAIL_MIN_HEIGHT, int(screen_height * DETAIL_MAX_FRACTION))

    def _clamp_detail_height(self, height: int) -> int:
        return max(DETAIL_MIN_HEIGHT, min(height, self._max_detail_height()))

    def _set_detail_height(self, height: int) -> None:
        self._detail_height = self._clamp_detail_height(height)
        self.query_one("#candidate-detail", RichLog).styles.height = self._detail_height

    def _set_detail_visible(self, visible: bool) -> None:
        display = "block" if visible else "none"
        self.query_one("#detail-divider", Static).styles.display = display
        detail = self.query_one("#candidate-detail", RichLog)
        detail.styles.display = display
        if visible:
            self._set_detail_height(self._detail_height)

    def refresh_data(self) -> None:
        from rich.text import Text

        journal = Journal(self.search_dir / "journal.jsonl")
        status = read_status(self.search_dir)
        state = effective_state(self.search_dir)
        line = f"{_search_ref(self.search_dir)}  [{STATE_STYLE.get(state, '')}]{state}[/]"
        if status is not None:
            minutes = int(status.budget.remaining_s // 60)
            line += f"  budget left: {minutes}m"
            if status.current:
                active = " · ".join(
                    f"{c.candidate_id}({c.operator}/{c.phase})" for c in status.current[:3]
                )
                if len(status.current) > 3:
                    active += f" +{len(status.current) - 3}"
                line += f"  active: {active}"
        self.query_one("#searchline", Label).update(line)

        table = self.query_one("#candidates", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in candidate_rows(journal):
            styled = [Text(v, style=row.style) for v in
                      (row.label, row.operator, row.status, row.val, row.hold, row.marks, row.summary)]
            table.add_row(*styled, key=row.candidate_id)
        _restore_table(table, snapshot)

        if self._detail_candidate_id is not None:
            if self._detail_candidate_id in journal.candidates:
                self._render_detail(journal, preserve_scroll=True)
            else:
                self._close_detail()

    def _render_detail(self, journal: Journal, preserve_scroll: bool = False) -> None:
        if self._detail_candidate_id is None:
            return
        detail = self.query_one("#candidate-detail", RichLog)
        snapshot = ScrollSnapshot(detail.scroll_x, detail.scroll_y)
        self._set_detail_visible(True)
        detail.clear()
        for renderable in candidate_detail_renderables(
            self.search_dir, journal, self._detail_candidate_id
        ):
            detail.write(renderable, expand=True)
        if preserve_scroll:
            _restore_scroll(detail, snapshot)
        else:
            detail.scroll_home(animate=False)

    def _open_detail(self, candidate_id: str | None = None) -> None:
        candidate_id = candidate_id or self._selected_candidate_id()
        if candidate_id is None:
            return
        self._detail_candidate_id = candidate_id
        self._render_detail(Journal(self.search_dir / "journal.jsonl"))

    def _close_detail(self) -> None:
        self._detail_candidate_id = None
        detail = self.query_one("#candidate-detail", RichLog)
        detail.clear()
        self._set_detail_visible(False)

    def _selected_candidate_id(self) -> str | None:
        table = self.query_one("#candidates", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return row_key.value

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "candidates":
            self._open_detail(event.row_key.value)
            event.stop()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "candidates" and self._detail_candidate_id is not None:
            self._detail_candidate_id = event.row_key.value
            self._render_detail(Journal(self.search_dir / "journal.jsonl"))
            event.stop()

    def action_open_detail(self) -> None:
        self._open_detail()

    def action_close_detail_or_back(self) -> None:
        if self._detail_candidate_id is not None:
            self._close_detail()
        else:
            self.app.pop_screen()

    def action_grow_detail(self) -> None:
        if self._detail_candidate_id is not None:
            self._set_detail_height(self._detail_height + DETAIL_STEP)

    def action_shrink_detail(self) -> None:
        if self._detail_candidate_id is not None:
            self._set_detail_height(self._detail_height - DETAIL_STEP)

    def _begin_detail_drag(self, y: int) -> None:
        self._dragging_detail = True
        self._drag_start_y = y
        self._drag_start_height = self._detail_height

    def _drag_detail_to(self, y: int) -> None:
        self._set_detail_height(self._drag_start_height + self._drag_start_y - y)

    def _end_detail_drag(self) -> None:
        self._dragging_detail = False

    def on_resize(self) -> None:
        if self._detail_candidate_id is not None:
            self._set_detail_height(self._detail_height)

    def action_stop_search(self) -> None:
        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(self.search_dir, source="tui")
                self.notify(outcome or "engine is not running", severity="information")

        self.app.push_screen(
            ConfirmScreen(
                f"Stop search {_search_ref(self.search_dir)}? (parks after current operator)"
            ),
            go,
        )

    def action_prune_candidate(self) -> None:
        candidate_id = self._selected_candidate_id()
        if candidate_id is None:
            return

        def go(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                outcome = request_prune(
                    self.search_dir,
                    candidate_id,
                    lower_is_better=_lower_is_better(self.config, self.search_dir),
                    selection_mode=self.config.holdout.selection,
                    source="tui",
                )
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            self.notify(outcome, severity="warning")
            self.refresh_data()

        self.app.push_screen(
            ConfirmScreen(
                f"Prune candidate {candidate_id} and its whole subtree from "
                f"{_search_ref(self.search_dir)}?"
            ),
            go,
        )


class SearchesScreen(Screen):
    """Searches inside one run."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("enter", "open_search", "open", priority=True),
        Binding("s", "stop_search", "stop search"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    SearchesScreen #runline { height: 1; padding: 0 1; background: $surface; }
    """

    def __init__(self, config: Config, run_dir: Path, run_name: str):
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self.run_name = run_name

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(id="runline")
        yield DataTable(id="searches", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#searches", DataTable)
        table.add_columns(
            "problem",
            "search",
            "model",
            "state",
            "candidates",
            "best val",
            "selected",
            "budget left",
        )
        self.refresh_data()
        self.set_interval(REFRESH_S, self.refresh_data)

    def refresh_data(self) -> None:
        from rich.text import Text

        self.query_one("#runline", Label).update(f"{self.run_name}  ({self.run_dir.name})")
        table = self.query_one("#searches", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in scan_searches(self.run_dir):
            state = Text(row.state, style=STATE_STYLE.get(row.state, ""))
            table.add_row(
                row.problem, row.search_id, row.model, state, row.candidates,
                row.best_val, row.selected, row.budget_left, key=row.search_id,
            )
        _restore_table(table, snapshot)

    def _selected_search_dir(self) -> Path | None:
        table = self.query_one("#searches", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return self.run_dir / "searches" / row_key.value

    def action_open_search(self) -> None:
        search_dir = self._selected_search_dir()
        if search_dir is not None:
            self.app.push_screen(CandidateScreen(self.config, search_dir))

    def action_stop_search(self) -> None:
        search_dir = self._selected_search_dir()
        if search_dir is None:
            return

        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(search_dir, source="tui")
                self.notify(outcome or "engine is not running", severity="information")

        self.app.push_screen(
            ConfirmScreen(f"Stop search {_search_ref(search_dir)}? (parks after current operator)"),
            go,
        )


class RunsScreen(Screen):
    """Top-level runs."""

    BINDINGS = [
        Binding("enter", "open_run", "open", priority=True),
        Binding("q", "app.quit", "quit"),
    ]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._names: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="runs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.add_columns(
            "run",
            "state",
            "searches",
            "candidates",
            "selected",
            "budget left",
            "started",
        )
        self.refresh_data()
        self.set_interval(REFRESH_S, self.refresh_data)

    def refresh_data(self) -> None:
        from rich.text import Text

        table = self.query_one("#runs", DataTable)
        snapshot = _snapshot_table(table)
        self._names.clear()
        table.clear()
        for row in scan_runs(self.config.paths.runs_dir):
            state = Text(row.state, style=STATE_STYLE.get(row.state, ""))
            self._names[row.run_id] = row.name
            table.add_row(
                row.name,
                state,
                row.searches,
                row.candidates,
                row.selected,
                row.budget_left,
                row.started,
                key=row.run_id,
            )
        _restore_table(table, snapshot)

    def _selected_run(self) -> tuple[str, str] | None:
        table = self.query_one("#runs", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        run_id = row_key.value
        return run_id, self._names.get(run_id, run_id)

    def action_open_run(self) -> None:
        selected = self._selected_run()
        if selected is None:
            return
        run_id, name = selected
        self.app.push_screen(SearchesScreen(self.config, self.config.paths.runs_dir / run_id, name))


class WatchApp(App):
    """Read-mostly dashboard over runs/; control actions go through the
    same command queue as the CLI."""

    TITLE = "hillclimb watch"

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config.load()

    def on_mount(self) -> None:
        self.push_screen(RunsScreen(self.config))
