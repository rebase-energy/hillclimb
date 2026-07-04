"""`hillclimb watch` — live TUI over the runs directory.

Strictly a *viewer*: all state is read from disk (run.yaml, status.json,
journal.jsonl, agent_stream.jsonl) and the only writes go through the
control-command queue in hillclimb.control, same as the CLI. The pure
data-assembly functions at the top carry the logic so they stay testable
without driving Textual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from hillclimb.config import Config
from hillclimb.control import request_prune, request_stop
from hillclimb.experiment import (
    LEGACY_EXPERIMENT_ID,
    LEGACY_EXPERIMENT_NAME,
    ExperimentMeta,
    load_experiments,
)
from hillclimb.journal import Journal
from hillclimb.node import Node
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
class ExperimentRow:
    experiment_id: str
    name: str
    state: str
    problem_runs: str
    candidates: str
    selected: str
    budget_left: str
    started: str


@dataclass
class ProblemRunRow:
    run_id: str
    problem: str
    model: str
    state: str
    candidates: str  # "7 (5 ok)"
    best_val: str
    selected: str
    budget_left: str


@dataclass
class NodeRow:
    node_id: str
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


def _iter_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        (d for d in runs_dir.iterdir() if (d / "run.yaml").exists()),
        reverse=True,
    )


def _experiment_for_run(meta: dict) -> str:
    return str(meta.get("experiment_id") or LEGACY_EXPERIMENT_ID)


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


def _problem_run_row(run_dir: Path) -> ProblemRunRow:
    meta = _run_meta(run_dir)
    status = read_status(run_dir)
    journal = Journal(run_dir / "journal.jsonl")
    n_ok = sum(1 for n in journal.nodes.values() if n.status == "ok")
    if status is not None:
        best_val = _fmt(status.best.val_score) if status.best else "-"
        selected = (
            f"{status.selected.node_id} ({_fmt(status.selected.holdout_score or status.selected.val_score)})"
            if status.selected
            else "-"
        )
        budget_left = _format_budget_left(status.budget.remaining_s)
    else:
        best_val, selected, budget_left = "-", "-", "-"
    return ProblemRunRow(
        run_id=run_dir.name,
        problem=str(meta.get("problem_id") or meta.get("task") or meta.get("problem", "?")),
        model=str(meta.get("model", "?")),
        state=effective_state(run_dir),
        candidates=f"{len(journal.nodes)} ({n_ok} ok)",
        best_val=best_val,
        selected=selected,
        budget_left=budget_left,
    )


def scan_problem_runs(runs_dir: Path, experiment_id: str) -> list[ProblemRunRow]:
    rows = []
    for run_dir in _iter_run_dirs(runs_dir):
        meta = _run_meta(run_dir)
        if _experiment_for_run(meta) == experiment_id:
            rows.append(_problem_run_row(run_dir))
    return rows


def _experiment_row(
    experiment_id: str,
    name: str,
    started: str,
    run_dirs: list[Path],
) -> ExperimentRow:
    states = [effective_state(run_dir) for run_dir in run_dirs]
    candidate_total = 0
    selected_count = 0
    remaining_s = 0.0
    has_budget = False
    for run_dir in run_dirs:
        journal = Journal(run_dir / "journal.jsonl")
        candidate_total += len(journal.nodes)
        status = read_status(run_dir)
        if status is None:
            continue
        if status.selected is not None:
            selected_count += 1
        remaining_s += status.budget.remaining_s
        has_budget = True
    running = sum(1 for state in states if state == "running")
    problem_runs = f"{len(run_dirs)}" + (f" ({running} running)" if running else "")
    return ExperimentRow(
        experiment_id=experiment_id,
        name=name,
        state=_state_summary(states),
        problem_runs=problem_runs,
        candidates=str(candidate_total),
        selected=f"{selected_count} selected" if selected_count else "-",
        budget_left=_format_budget_left(remaining_s if has_budget else None),
        started=started[:19].replace("T", " ") if started else "-",
    )


def scan_experiments(runs_dir: Path) -> list[ExperimentRow]:
    metas = load_experiments(runs_dir)
    by_experiment: dict[str, list[Path]] = {experiment_id: [] for experiment_id in metas}
    legacy_runs: list[Path] = []
    for run_dir in _iter_run_dirs(runs_dir):
        experiment_id = _experiment_for_run(_run_meta(run_dir))
        if experiment_id == LEGACY_EXPERIMENT_ID:
            legacy_runs.append(run_dir)
        else:
            by_experiment.setdefault(experiment_id, []).append(run_dir)

    rows = []
    for experiment_id, run_dirs in by_experiment.items():
        meta = metas.get(experiment_id)
        if meta is None:
            meta = ExperimentMeta(
                experiment_id=experiment_id,
                name=experiment_id,
                target="",
            )
        rows.append(
            _experiment_row(
                experiment_id,
                meta.name,
                meta.started_at,
                run_dirs,
            )
        )
    if legacy_runs:
        rows.append(
            _experiment_row(
                LEGACY_EXPERIMENT_ID,
                LEGACY_EXPERIMENT_NAME,
                "",
                legacy_runs,
            )
        )
    return sorted(rows, key=lambda row: row.started if row.started != "-" else "", reverse=True)


def _tree_order(journal: Journal) -> list[tuple[Node, int]]:
    """Depth-first (node, depth) pairs so children render under parents."""
    by_parent: dict[str | None, list[Node]] = {}
    for node in journal.nodes.values():
        by_parent.setdefault(node.parent_id, []).append(node)
    for children in by_parent.values():
        children.sort(key=lambda n: n.node_id)
    ordered: list[tuple[Node, int]] = []

    def visit(parent_id: str | None, depth: int) -> None:
        for node in by_parent.get(parent_id, []):
            ordered.append((node, depth))
            visit(node.node_id, depth + 1)

    visit(None, 0)
    # orphans (parent vanished from the journal) still get shown
    seen = {n.node_id for n, _ in ordered}
    ordered.extend((n, 0) for n in journal.nodes.values() if n.node_id not in seen)
    return ordered


def node_rows(journal: Journal) -> list[NodeRow]:
    rows = []
    for node, depth in _tree_order(journal):
        marks = []
        if node.is_selected and not node.pruned:
            marks.append("SELECTED")
        if node.is_best:
            marks.append("best-val")
        if node.pruned:
            marks.append("PRUNED")
        style = "dim strike" if node.pruned else STATUS_STYLE.get(node.status, "")
        if node.is_selected and not node.pruned:
            style = "bold gold1"
        rows.append(
            NodeRow(
                node_id=node.node_id,
                label="  " * depth + node.node_id,
                operator=node.operator + (f"/{node.complexity}" if node.complexity else ""),
                status=node.status,
                val=_fmt(node.val_score),
                hold=_fmt(node.holdout_score),
                marks=" ".join(marks),
                summary=(node.summary or "").strip()[:60],
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


def _node_workspace(run_dir: Path, node: Node) -> Path:
    return Path(node.workspace) if node.workspace else run_dir / "nodes" / node.node_id


def _node_marks(node: Node) -> str:
    marks = []
    if node.is_selected and not node.pruned:
        marks.append("selected")
    if node.is_best:
        marks.append("best-val")
    if node.pruned:
        marks.append("pruned")
    return ", ".join(marks) if marks else "-"


def _ancestry(journal: Journal, node: Node) -> list[str]:
    lineage = [node.node_id]
    seen = {node.node_id}
    parent_id = node.parent_id
    while parent_id and parent_id not in seen and parent_id in journal.nodes:
        lineage.insert(0, parent_id)
        seen.add(parent_id)
        parent_id = journal.nodes[parent_id].parent_id
    if parent_id and parent_id not in journal.nodes:
        lineage.insert(0, f"{parent_id} (missing)")
    return lineage


def candidate_detail_lines(run_dir: Path, journal: Journal, node_id: str) -> list[str]:
    node = journal.nodes.get(node_id)
    if node is None:
        return [f"Candidate {node_id} is no longer in the journal."]

    meta = _run_meta(run_dir)
    lower = bool(meta.get("lower_is_better", False))
    metric = str(meta.get("metric", "score"))
    direction = "lower is better" if lower else "higher is better"
    operator = node.operator + (f"/{node.complexity}" if node.complexity else "")
    workspace = _node_workspace(run_dir, node)
    children = journal.children(node.node_id, include_pruned=True)
    parent = node.parent_id if node.parent_id else "root"
    duration = f"{node.execution.duration_s:.5g}s" if node.execution.duration_s is not None else "-"

    lines = [
        f"Candidate {node.node_id} | {operator} | {node.status}",
        f"Score: val={_fmt(node.val_score)}  holdout={_fmt(node.holdout_score)}  metric={metric} ({direction})",
        f"Marks: {_node_marks(node)}",
        f"Parent: {parent}  Children: {len(children)}  Path: {' -> '.join(_ancestry(journal, node))}",
        (
            "Execution: "
            f"returncode={node.execution.returncode if node.execution.returncode is not None else '-'}  "
            f"duration={duration}  "
            f"timed_out={node.execution.timed_out}  "
            f"submission_ok={node.execution.submission_ok}"
        ),
    ]
    if node.execution.holdout_error:
        lines.append(f"Holdout error: {node.execution.holdout_error}")
    if node.backend.name or node.backend.session_id or node.backend.error_kind:
        cost = f"${node.backend.cost_usd:.2f}" if node.backend.cost_usd is not None else "-"
        lines.append(
            "Backend: "
            f"{node.backend.name or '-'}  "
            f"session={node.backend.session_id or '-'}  "
            f"turns={node.backend.num_turns if node.backend.num_turns is not None else '-'}  "
            f"cost={cost}  "
            f"error={node.backend.error_kind or '-'}"
        )

    if children:
        lines += ["", "Children:"]
        for child in children[:12]:
            child_op = child.operator + (f"/{child.complexity}" if child.complexity else "")
            lines.append(
                f"  {child.node_id}  {child_op}  {child.status}  val={_fmt(child.val_score)}"
                + ("  PRUNED" if child.pruned else "")
            )
        if len(children) > 12:
            lines.append(f"  ... {len(children) - 12} more")

    notes = _tail_text(workspace / "notes.md", max_chars=3000) or node.summary.strip()
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


def candidate_detail_renderables(run_dir: Path, journal: Journal, node_id: str) -> list[object]:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    node = journal.nodes.get(node_id)
    if node is None:
        return [Panel(Text(f"Candidate {node_id} is no longer in the journal.", style="red"))]

    meta = _run_meta(run_dir)
    lower = bool(meta.get("lower_is_better", False))
    metric = str(meta.get("metric", "score"))
    direction = "lower is better" if lower else "higher is better"
    operator = node.operator + (f"/{node.complexity}" if node.complexity else "")
    workspace = _node_workspace(run_dir, node)
    children = journal.children(node.node_id, include_pruned=True)
    parent = node.parent_id if node.parent_id else "root"
    duration = f"{node.execution.duration_s:.5g}s" if node.execution.duration_s is not None else "-"
    title_style = "dim" if node.pruned else STATUS_STYLE.get(node.status, "")
    border_style = "red" if node.status == "buggy" else "yellow" if node.pruned else "cyan"

    overview = Table.grid(expand=True)
    overview.add_column("label", style="dim", ratio=1)
    overview.add_column("value", ratio=3)
    overview.add_column("label", style="dim", ratio=1)
    overview.add_column("value", ratio=3)
    overview.add_row(
        "candidate",
        Text(node.node_id, style=f"bold {title_style}".strip()),
        "operator",
        Text(operator, style="bold"),
    )
    overview.add_row("status", _status_text(node.status), "marks", Text(_node_marks(node), style="gold1"))
    overview.add_row("metric", Text(f"{metric} ({direction})"), "parent", Text(parent, style="cyan"))
    overview.add_row(
        "val",
        _score_text(node.val_score, selected=node.is_selected, best=node.is_best),
        "holdout",
        _score_text(node.holdout_score, selected=node.is_selected),
    )
    overview.add_row(
        "children",
        Text(str(len(children)), style="cyan" if children else "dim"),
        "path",
        Text(" -> ".join(_ancestry(journal, node)), style="cyan"),
    )
    overview.add_row(
        "returncode",
        Text(str(node.execution.returncode) if node.execution.returncode is not None else "-", style="dim"),
        "duration",
        Text(duration, style="cyan" if node.execution.duration_s is not None else "dim"),
    )
    overview.add_row(
        "timed out",
        Text(str(node.execution.timed_out), style="red" if node.execution.timed_out else "dim"),
        "submission",
        Text("ok" if node.execution.submission_ok else "-", style="green" if node.execution.submission_ok else "dim"),
    )
    if node.execution.holdout_error:
        overview.add_row("holdout error", Text(node.execution.holdout_error, style="yellow"), "", "")
    if node.backend.name or node.backend.session_id or node.backend.error_kind:
        cost = f"${node.backend.cost_usd:.2f}" if node.backend.cost_usd is not None else "-"
        backend = (
            f"{node.backend.name or '-'}  session={node.backend.session_id or '-'}  "
            f"turns={node.backend.num_turns if node.backend.num_turns is not None else '-'}  "
            f"cost={cost}  error={node.backend.error_kind or '-'}"
        )
        overview.add_row("backend", Text(backend), "", "")

    renderables: list[object] = [
        Panel(overview, title=f"Candidate {node.node_id}", title_align="left", border_style=border_style)
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
                child.node_id,
                child_op,
                _status_text(child.status),
                _score_text(child.val_score, selected=child.is_selected, best=child.is_best),
                Text(_node_marks(child), style="gold1" if not child.pruned else "dim"),
                style="dim" if child.pruned else "",
            )
        if len(children) > 12:
            child_table.add_row(f"... {len(children) - 12} more", "", "", "", "", style="dim")
        renderables.append(child_table)

    notes = _tail_text(workspace / "notes.md", max_chars=3000) or node.summary.strip()
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


def _run_meta(run_dir: Path) -> dict:
    path = run_dir / "run.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def _lower_is_better(config: Config, run_dir: Path) -> bool:
    return bool(_run_meta(run_dir).get("lower_is_better", False))


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
        if getattr(screen, "_detail_node_id", None) is None:
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
        if getattr(screen, "_detail_node_id", None) is None:
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
        if getattr(screen, "_detail_node_id", None) is None:
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
    """One problem run: candidate tree plus optional selected-candidate details."""

    BINDINGS = [
        Binding("escape", "close_detail_or_back", "back"),
        Binding("enter", "open_detail", "details", priority=True),
        Binding("+", "grow_detail", "larger detail"),
        Binding("-", "shrink_detail", "smaller detail"),
        Binding("s", "stop_run", "stop run"),
        Binding("x", "prune_node", "prune candidate"),
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
    CandidateScreen #runline { height: 1; padding: 0 1; background: $surface; }
    """

    def __init__(self, config: Config, run_dir: Path):
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self._detail_node_id: str | None = None
        self._detail_height = DETAIL_DEFAULT_HEIGHT
        self._dragging_detail = False
        self._drag_start_y = 0
        self._drag_start_height = DETAIL_DEFAULT_HEIGHT

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(id="runline")
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

        journal = Journal(self.run_dir / "journal.jsonl")
        status = read_status(self.run_dir)
        state = effective_state(self.run_dir)
        line = f"{self.run_dir.name}  [{STATE_STYLE.get(state, '')}]{state}[/]"
        if status is not None:
            minutes = int(status.budget.remaining_s // 60)
            line += f"  budget left: {minutes}m"
            if status.current is not None:
                line += (
                    f"  current candidate: {status.current.node_id} "
                    f"({status.current.operator}/{status.current.phase})"
                )
        self.query_one("#runline", Label).update(line)

        table = self.query_one("#candidates", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in node_rows(journal):
            styled = [Text(v, style=row.style) for v in
                      (row.label, row.operator, row.status, row.val, row.hold, row.marks, row.summary)]
            table.add_row(*styled, key=row.node_id)
        _restore_table(table, snapshot)

        if self._detail_node_id is not None:
            if self._detail_node_id in journal.nodes:
                self._render_detail(journal, preserve_scroll=True)
            else:
                self._close_detail()

    def _render_detail(self, journal: Journal, preserve_scroll: bool = False) -> None:
        if self._detail_node_id is None:
            return
        detail = self.query_one("#candidate-detail", RichLog)
        snapshot = ScrollSnapshot(detail.scroll_x, detail.scroll_y)
        self._set_detail_visible(True)
        detail.clear()
        for renderable in candidate_detail_renderables(self.run_dir, journal, self._detail_node_id):
            detail.write(renderable, expand=True)
        if preserve_scroll:
            _restore_scroll(detail, snapshot)
        else:
            detail.scroll_home(animate=False)

    def _open_detail(self, node_id: str | None = None) -> None:
        node_id = node_id or self._selected_node_id()
        if node_id is None:
            return
        self._detail_node_id = node_id
        self._render_detail(Journal(self.run_dir / "journal.jsonl"))

    def _close_detail(self) -> None:
        self._detail_node_id = None
        detail = self.query_one("#candidate-detail", RichLog)
        detail.clear()
        self._set_detail_visible(False)

    def _selected_node_id(self) -> str | None:
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
        if event.data_table.id == "candidates" and self._detail_node_id is not None:
            self._detail_node_id = event.row_key.value
            self._render_detail(Journal(self.run_dir / "journal.jsonl"))
            event.stop()

    def action_open_detail(self) -> None:
        self._open_detail()

    def action_close_detail_or_back(self) -> None:
        if self._detail_node_id is not None:
            self._close_detail()
        else:
            self.app.pop_screen()

    def action_grow_detail(self) -> None:
        if self._detail_node_id is not None:
            self._set_detail_height(self._detail_height + DETAIL_STEP)

    def action_shrink_detail(self) -> None:
        if self._detail_node_id is not None:
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
        if self._detail_node_id is not None:
            self._set_detail_height(self._detail_height)

    def action_stop_run(self) -> None:
        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(self.run_dir, source="tui")
                self.notify(outcome or "engine is not running", severity="information")

        self.app.push_screen(ConfirmScreen(f"Stop run {self.run_dir.name}? (parks after current operator)"), go)

    def action_prune_node(self) -> None:
        node_id = self._selected_node_id()
        if node_id is None:
            return

        def go(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                outcome = request_prune(
                    self.run_dir,
                    node_id,
                    lower_is_better=_lower_is_better(self.config, self.run_dir),
                    selection_mode=self.config.holdout.selection,
                    source="tui",
                )
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            self.notify(outcome, severity="warning")
            self.refresh_data()

        self.app.push_screen(
            ConfirmScreen(f"Prune candidate {node_id} and its whole subtree from {self.run_dir.name}?"), go
        )


class ProblemRunsScreen(Screen):
    """Problem runs inside one experiment."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("enter", "open_run", "open", priority=True),
        Binding("s", "stop_run", "stop run"),
        Binding("q", "app.quit", "quit"),
    ]

    DEFAULT_CSS = """
    ProblemRunsScreen #experimentline { height: 1; padding: 0 1; background: $surface; }
    """

    def __init__(self, config: Config, experiment_id: str, experiment_name: str):
        super().__init__()
        self.config = config
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(id="experimentline")
        yield DataTable(id="problem-runs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#problem-runs", DataTable)
        table.add_columns(
            "problem",
            "run",
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

        self.query_one("#experimentline", Label).update(
            f"{self.experiment_name}  ({self.experiment_id})"
        )
        table = self.query_one("#problem-runs", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in scan_problem_runs(self.config.paths.runs_dir, self.experiment_id):
            state = Text(row.state, style=STATE_STYLE.get(row.state, ""))
            table.add_row(
                row.problem, row.run_id, row.model, state, row.candidates,
                row.best_val, row.selected, row.budget_left, key=row.run_id,
            )
        _restore_table(table, snapshot)

    def _selected_run_dir(self) -> Path | None:
        table = self.query_one("#problem-runs", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return self.config.paths.runs_dir / row_key.value

    def action_open_run(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is not None:
            self.app.push_screen(CandidateScreen(self.config, run_dir))

    def action_stop_run(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None:
            return

        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(run_dir, source="tui")
                self.notify(outcome or "engine is not running", severity="information")

        self.app.push_screen(ConfirmScreen(f"Stop run {run_dir.name}? (parks after current operator)"), go)


class ExperimentsScreen(Screen):
    """Top-level experiments."""

    BINDINGS = [
        Binding("enter", "open_experiment", "open", priority=True),
        Binding("q", "app.quit", "quit"),
    ]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._names: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="experiments", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#experiments", DataTable)
        table.add_columns(
            "experiment",
            "state",
            "problem runs",
            "candidates",
            "selected",
            "budget left",
            "started",
        )
        self.refresh_data()
        self.set_interval(REFRESH_S, self.refresh_data)

    def refresh_data(self) -> None:
        from rich.text import Text

        table = self.query_one("#experiments", DataTable)
        snapshot = _snapshot_table(table)
        self._names.clear()
        table.clear()
        for row in scan_experiments(self.config.paths.runs_dir):
            state = Text(row.state, style=STATE_STYLE.get(row.state, ""))
            self._names[row.experiment_id] = row.name
            table.add_row(
                row.name,
                state,
                row.problem_runs,
                row.candidates,
                row.selected,
                row.budget_left,
                row.started,
                key=row.experiment_id,
            )
        _restore_table(table, snapshot)

    def _selected_experiment(self) -> tuple[str, str] | None:
        table = self.query_one("#experiments", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        experiment_id = row_key.value
        return experiment_id, self._names.get(experiment_id, experiment_id)

    def action_open_experiment(self) -> None:
        selected = self._selected_experiment()
        if selected is None:
            return
        experiment_id, name = selected
        self.app.push_screen(ProblemRunsScreen(self.config, experiment_id, name))


class WatchApp(App):
    """Read-mostly dashboard over runs/; control actions go through the
    same command queue as the CLI."""

    TITLE = "hillclimb watch"

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config.load()

    def on_mount(self) -> None:
        self.push_screen(ExperimentsScreen(self.config))
