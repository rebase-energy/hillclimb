"""`hillclimb watch` — live TUI over the runs directory.

Strictly a *viewer*: all state is read from disk (run.yaml, search.yaml,
status.json, journal.jsonl, agent_stream.jsonl — via the configured DataStore) and the only writes go through
the control-command queue in hillclimb.control, same as the CLI. The pure
data-assembly functions at the top carry the logic so they stay testable
without driving Textual.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hillclimb.backends.claude_code import usage_total_tokens
from hillclimb.candidate import Candidate
from hillclimb.config import Config
from hillclimb.control import request_prune, request_stop
from hillclimb.journal import Journal
from hillclimb.run import RunMeta, SearchMeta, run_display_name
from hillclimb.run import search_ref as _search_ref
from hillclimb.status import live_remaining_s
from hillclimb.store import DataStore, FileDataStore, SearchRecord, key_for, open_store
from hillclimb.theme import HILLCLIMB_CSS, apply_theme

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
    "running": "italic",
    "stale": "dim italic",
}


def display_status(status: str, live: bool) -> str:
    """The journal's `pending` means "an agent is on it" — show that as
    `running` while the engine is alive and `stale` once it is not (the
    engine marks such candidates abandoned on resume)."""
    if status == "pending":
        return "running" if live else "stale"
    return status


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
    tokens: str  # summed agent tokens across the search's candidates
    state: str
    candidates: str  # "7 (5 ok)"
    best_val: str
    selected: str
    duration: str  # time spent so far, ticking while running, with the budget alongside


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


def _fmt_tokens(total: int | None) -> str:
    """Compact token count: 812 -> "812", 24_500 -> "24.5k", 1_240_000 -> "1.24M"."""
    if not total:
        return "-"
    if total < 1000:
        return str(total)
    if total < 1_000_000:
        return f"{total / 1000:.1f}k"
    return f"{total / 1_000_000:.2f}M"


def _format_budget_total(seconds: float) -> str:
    """A budget as it was given: `10m`, `1h 30m`, `1m 30s` — zero parts dropped."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours}h" if hours else "", f"{minutes}m" if minutes else "", f"{secs}s" if secs else ""]
    return " ".join(p for p in parts if p) or "0s"


def _format_duration(spent_s: float | None, total_s: float | None) -> str:
    """`4m 07s (budget: 10m)`: what has been spent, counting up while the
    search runs, next to the budget it was given — so a search that stopped
    early reads as "used 4 of 10 minutes" rather than a countdown stuck
    short of zero."""
    if spent_s is None:
        return "-"
    text = _format_budget_left(spent_s)
    if total_s:
        text += f" (budget: {_format_budget_total(total_s)})"
    return text


def _format_budget_left(seconds: float | None) -> str:
    """`4m 07s`, or `1h 02m 07s` past the hour — seconds always shown, so a
    live search visibly counts down."""
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _stream_tokens(candidate_dir: Path) -> int:
    """Tokens burned so far by an in-flight operator, read from its live
    `agent_stream.jsonl`. Each turn streams twice (partial then final) under
    one message id, so turns are deduped by id, newest kept; a `result`
    message, if the call has just finished, is authoritative. Output is only a
    running estimate mid-call — the stream carries partial output counts — but
    cache tokens (the bulk) reconcile exactly with the final total."""
    path = candidate_dir / "agent_stream.jsonl"
    if not path.exists():
        return 0
    per_turn: dict[str, dict] = {}
    final: dict | None = None
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written trailing line while the agent streams
            if msg.get("type") == "result" and msg.get("usage"):
                final = msg["usage"]
            elif msg.get("type") == "assistant":
                body = msg.get("message") or {}
                usage = body.get("usage")
                if usage and body.get("id"):
                    per_turn[body["id"]] = usage
    except OSError:
        return 0
    if final is not None:
        return usage_total_tokens(final)
    return sum(usage_total_tokens(u) for u in per_turn.values())


def _state_summary(states: list[str]) -> str:
    if not states:
        return "empty"
    for state in ("running", "crashed", "failed", "parked", "stopped"):
        if state in states:
            return state
    if all(state == "done" for state in states):
        return "done"
    return states[0]


def _search_row(store: DataStore, record: SearchRecord) -> SearchRow:
    meta, search_dir, state = record.meta, record.search_dir, record.state
    status = store.read_status(record.key)
    journal = Journal(store.journal(record.key))
    n_ok = sum(1 for c in journal.candidates.values() if c.status == "ok")
    tokens = sum(c.backend.total_tokens or 0 for c in journal.candidates.values())
    if status is not None:
        # in-flight operators are not in the journal yet: read their live
        # streams so the count climbs while the tokens are being burned
        tokens += sum(_stream_tokens(Path(c.candidate_dir)) for c in status.current)
        best_val = _fmt(status.best.val_score) if status.best else "-"
        selected = (
            f"{status.selected.candidate_id} "
            f"({_fmt(status.selected.holdout_score or status.selected.val_score)})"
            if status.selected
            else "-"
        )
        duration = _format_duration(
            status.budget.total_s - live_remaining_s(status, state), status.budget.total_s
        )
    else:
        best_val, selected, duration = "-", "-", "-"
    return SearchRow(
        search_id=search_dir.name,
        problem=meta.problem_id,
        model=meta.model,
        tokens=_fmt_tokens(tokens),
        state=state,
        candidates=f"{len(journal.candidates)} ({n_ok} ok)",
        best_val=best_val,
        selected=selected,
        duration=duration,
    )


def _as_store(store: DataStore | Path) -> DataStore:
    """A runs dir is shorthand for its FileDataStore."""
    return FileDataStore(store) if isinstance(store, Path) else store


def scan_searches(store: DataStore | Path, run_id: str) -> list[SearchRow]:
    store = _as_store(store)
    # by search id, the order the run created them in
    records = sorted(store.searches(run_id=run_id), key=lambda r: r.search_id)
    return [_search_row(store, record) for record in records]


def _run_row(store: DataStore, meta: RunMeta) -> RunRow:
    records = store.searches(run_id=meta.run_id)
    states = [record.state for record in records]
    candidate_total = 0
    selected_count = 0
    remaining_s = 0.0
    has_budget = False
    for record, state in zip(records, states):
        journal = Journal(store.journal(record.key))
        candidate_total += len(journal.candidates)
        status = store.read_status(record.key)
        if status is None:
            continue
        if status.selected is not None:
            selected_count += 1
        remaining_s += live_remaining_s(status, state)
        has_budget = True
    running = sum(1 for state in states if state == "running")
    searches = f"{len(records)}" + (f" ({running} running)" if running else "")
    started = meta.started_at
    return RunRow(
        run_id=meta.run_id,
        name=meta.name or meta.run_id,
        state=_state_summary(states),
        searches=searches,
        candidates=str(candidate_total),
        selected=f"{selected_count} selected" if selected_count else "-",
        budget_left=_format_budget_left(remaining_s if has_budget else None),
        started=started[:19].replace("T", " ") if started else "-",
    )


def scan_runs(store: DataStore | Path) -> list[RunRow]:
    store = _as_store(store)
    rows = [_run_row(store, meta) for meta in store.runs()]
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


CANDIDATE_COLUMNS = ("candidate", "operator", "status", "val", "hold", "marks", "summary")


def _shows_holdout(record: SearchRecord, journal: Journal) -> bool:
    """A hold column only when the search scores a holdout: declared on the
    search, or present on any candidate (older journals predate the flag)."""
    return bool(record.meta.holdout_enabled) or any(
        c.holdout_score is not None for c in journal.candidates.values()
    )


def _candidate_cells(row: CandidateRow, holdout: bool) -> tuple[str, ...]:
    cells = (row.label, row.operator, row.status, row.val, row.hold, row.marks, row.summary)
    return cells if holdout else cells[:4] + cells[5:]


def _set_candidate_columns(table, holdout: bool) -> None:
    """(Re)build a candidate table's columns for this search's layout; a
    no-op when the layout is unchanged, so the cursor and scroll survive."""
    if getattr(table, "_holdout_layout", None) == holdout:
        return
    table.clear(columns=True)
    table.add_columns(*(CANDIDATE_COLUMNS if holdout else CANDIDATE_COLUMNS[:4] + CANDIDATE_COLUMNS[5:]))
    table._holdout_layout = holdout


def candidate_rows(journal: Journal, live: bool = True) -> list[CandidateRow]:
    rows = []
    for candidate, depth in _tree_order(journal):
        marks = []
        if candidate.is_selected and not candidate.pruned:
            marks.append("SELECTED")
        if candidate.is_best:
            marks.append("best-val")
        if candidate.pruned:
            marks.append("PRUNED")
        shown = display_status(candidate.status, live)
        style = "dim strike" if candidate.pruned else STATUS_STYLE.get(shown, "")
        if candidate.is_selected and not candidate.pruned:
            style = "bold gold1"
        rows.append(
            CandidateRow(
                candidate_id=candidate.candidate_id,
                label="  " * depth + candidate.candidate_id,
                operator=candidate.operator
                + (f"/{candidate.complexity}" if candidate.complexity else ""),
                status=shown,
                val=_fmt(candidate.val_score),
                hold=_fmt(candidate.holdout_score),
                marks=" ".join(marks),
                summary=(candidate.summary or "").strip()[:60],
                style=style,
            )
        )
    return rows


@dataclass(frozen=True)
class StreamEntry:
    """One line of the operator stream: when it arrived (local HH:MM:SS, or
    "" for streams written before timestamps), what kind it is — text |
    tool | system | result | error | raw — and the rendered text."""

    ts: str
    kind: str
    text: str


def _local_clock(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


# system subtypes worth a line; the CLI also streams per-turn bookkeeping
# (thinking_tokens and friends) that would drown the transcript
_SYSTEM_SHOWN = {"init"}


# the argument that says what a tool call is about, in preference order;
# otherwise the first string argument (Edit's first key is a boolean flag)
_TOOL_ARG_KEYS = ("command", "file_path", "path", "pattern", "url", "query", "prompt", "description")


def _tool_arg_preview(args: dict, width: int = 80) -> str:
    value = next((args[k] for k in _TOOL_ARG_KEYS if args.get(k)), None)
    if value is None:
        value = next((v for v in args.values() if isinstance(v, str) and v.strip()), "")
    flat = " ".join(str(value).split())  # one line, however the agent indented it
    return flat[:width] + ("…" if len(flat) > width else "")


def parse_stream_line(raw: str) -> StreamEntry | None:
    """One compact entry per stream-json message; None = skip."""
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        text = raw.strip()
        return StreamEntry("", "raw", text[:200]) if text else None
    if not isinstance(message, dict):
        return None
    ts = _local_clock(message.get("ts"))
    kind = message.get("type")
    if kind == "system":
        subtype = message.get("subtype", "")
        if subtype not in _SYSTEM_SHOWN:
            return None
        model = message.get("model")
        text = f"[system] {subtype} session={message.get('session_id', '?')}"
        if model:
            text += f" model={model}"
        return StreamEntry(ts, "system", text)
    if kind == "assistant":
        parts = []
        tool = False
        for block in (message.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and block.get("text", "").strip():
                parts.append(block["text"].strip())
            elif block.get("type") == "tool_use":
                parts.append(f"→ {block.get('name', 'tool')}({_tool_arg_preview(block.get('input') or {})})")
                tool = True
        if not parts:
            return None
        return StreamEntry(ts, "tool" if tool and len(parts) == 1 else "text", "\n".join(parts))
    if kind == "result":
        cost = message.get("total_cost_usd")
        cost_s = f"${cost:.2f}" if isinstance(cost, (int, float)) else "?"
        text = f"[result] {message.get('subtype', '')} turns={message.get('num_turns', '?')} cost={cost_s}"
        return StreamEntry(ts, "error" if message.get("is_error") else "result", text)
    return None


def render_stream_line(raw: str) -> str | None:
    """`parse_stream_line` as one plain line, timestamp first when known."""
    entry = parse_stream_line(raw)
    if entry is None:
        return None
    return f"{entry.ts}  {entry.text}" if entry.ts else entry.text


def stream_entries(candidate_dir: Path, max_lines: int = 200) -> list[StreamEntry]:
    path = candidate_dir / "agent_stream.jsonl"
    if not path.exists():
        return []
    entries = []
    for raw in path.read_text(errors="replace").splitlines()[-max_lines:]:
        entry = parse_stream_line(raw)
        if entry:
            entries.append(entry)
    return entries


def stream_tail(candidate_dir: Path, max_lines: int = 200) -> list[str]:
    return [f"{e.ts}  {e.text}" if e.ts else e.text for e in stream_entries(candidate_dir, max_lines)]


_STREAM_STYLE = {
    "tool": "cyan",
    "system": "dim",
    "result": "bold green",
    "error": "bold red",
    "raw": "yellow",
    "text": "",
}


def _stream_text(entry: StreamEntry):
    """A stream entry as a terminal-log line: dim clock, a kind-coloured
    body; multi-line bodies indent under the clock column."""
    from rich.text import Text

    text = Text()
    if entry.ts:
        text.append(entry.ts, style="dim")
        text.append("  ")
    indent = " " * (len(entry.ts) + 2) if entry.ts else ""
    body = entry.text.replace("\n", "\n" + indent)
    text.append(body, style=_STREAM_STYLE.get(entry.kind, ""))
    return text


def _elapsed_since(iso: str | None) -> str:
    """Wall-clock since an ISO timestamp, as `32s` / `4m 07s`."""
    if not iso:
        return "-"
    try:
        started = datetime.fromisoformat(iso)
    except ValueError:
        return "-"
    seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:].strip()


def _path_label(path: Path) -> str:
    """`runs/…/candidates/<cid>`: the run and search are already on screen,
    so the label keeps only the ends that identify the dir; the full path
    rides along as the link target. Falls back to the full path when the dir
    is not laid out as runs/<run>/searches/<search>/candidates/<cid>."""
    parts = path.parts
    if len(parts) >= 6 and parts[-2] == "candidates" and parts[-4] == "searches" and parts[-6] == "runs":
        return f"runs/…/candidates/{parts[-1]}"
    return str(path)


def _path_link(path: Path):
    """The short label carrying the full path as an OSC 8 `file://` link, so
    a click opens the directory in terminals that support hyperlinks however
    the label is wrapped or cut (the terminal's own path detection would need
    the literal, unbroken path)."""
    from rich.text import Text

    # style the span, not the Text: a Text-level style also covers the
    # padding a table cell adds, and terminals underline the link that far
    text = Text()
    text.append(_path_label(path), style=f"cyan link {path.resolve().as_uri()}")
    return text


def open_in_file_manager(path: Path) -> None:
    """Reveal `path` with the desktop's opener (Finder, the xdg default).
    Detached: the TUI must not wait on it."""
    import subprocess

    opener = ["open"] if sys.platform == "darwin" else ["xdg-open"]
    subprocess.Popen(
        [*opener, str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _candidate_dir(search_dir: Path, candidate: Candidate) -> Path:
    if candidate.candidate_dir:
        return Path(candidate.candidate_dir)
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


def _metric_context(meta: SearchMeta) -> tuple[str, str]:
    higher = bool(meta.higher_is_better)
    metric = meta.metric
    direction = "higher is better" if higher else "lower is better"
    return metric, direction


def candidate_detail_lines(record: SearchRecord, journal: Journal, candidate_id: str) -> list[str]:
    candidate = journal.candidates.get(candidate_id)
    if candidate is None:
        return [f"Candidate {candidate_id} is no longer in the journal."]

    search_dir = record.search_dir
    metric, direction = _metric_context(record.meta)
    operator = candidate.operator + (f"/{candidate.complexity}" if candidate.complexity else "")
    candidate_dir = _candidate_dir(search_dir, candidate)
    children = journal.children(candidate.candidate_id, include_pruned=True)
    parent = candidate.parent_id if candidate.parent_id else "root"
    trial = candidate.last_trial

    lines = [
        f"Candidate {candidate.candidate_id} | {operator} | "
        f"{display_status(candidate.status, record.state == 'running')}",
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
            f"tokens={_fmt_tokens(candidate.backend.total_tokens)}  "
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

    notes = _tail_text(candidate_dir / "notes.md", max_chars=3000) or candidate.summary.strip()
    if notes:
        lines += ["", "Notes:", notes]

    stderr = _tail_text(candidate_dir / "exec_stderr.log", max_chars=3000)
    stdout = _tail_text(candidate_dir / "exec_stdout.log", max_chars=3000)
    if stderr:
        lines += ["", "Stderr:", stderr]
    if stdout:
        lines += ["", "Stdout:", stdout]

    stream = stream_tail(candidate_dir, max_lines=80)
    if stream:
        lines += ["", "Operator stream:"]
        lines.extend(stream)

    return lines


def _status_text(status: str, live: bool = True):
    from rich.text import Text

    shown = display_status(status, live)
    return Text(shown, style=STATUS_STYLE.get(shown, ""))


def _score_text(value: float | None, *, selected: bool = False, best: bool = False):
    from rich.text import Text

    if value is None:
        return Text("-", style="dim")
    style = "bold gold1" if selected else "bold cyan" if best else "cyan"
    return Text(_fmt(value), style=style)


def _plain_text(value: str, style: str = ""):
    from rich.text import Text

    return Text(value, style=style)


def _render_to_text(renderables: list[object], width: int) -> str:
    """Plain-text rendering of rich renderables, as a change fingerprint."""
    import io

    from rich.console import Console

    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, color_system=None, legacy_windows=False)
    for renderable in renderables:
        console.print(renderable)
    return buffer.getvalue()


def candidate_detail_renderables(
    record: SearchRecord, journal: Journal, candidate_id: str, live: bool | None = None
) -> list[object]:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    candidate = journal.candidates.get(candidate_id)
    if candidate is None:
        return [Panel(Text(f"Candidate {candidate_id} is no longer in the journal.", style="red"))]
    search_dir = record.search_dir
    if live is None:
        live = record.state == "running"

    metric, direction = _metric_context(record.meta)
    operator = candidate.operator + (f"/{candidate.complexity}" if candidate.complexity else "")
    candidate_dir = _candidate_dir(search_dir, candidate)
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
        "status", _status_text(candidate.status, live), "marks", Text(_candidate_marks(candidate), style="gold1")
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
        "lineage",
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
    backend = candidate.backend
    if candidate.status == "running" and live:
        # in flight: what is known now, refreshed every tick — the backend
        # and model from the route, tokens from the live stream, a clock
        # counting up since the candidate was created
        model = f"  model={backend.model}" if backend.model else ""
        overview.add_row(
            "backend",
            Text(f"{backend.name or '-'}{model}"),
            "elapsed",
            Text(_elapsed_since(candidate.created_at), style="cyan"),
        )
        overview.add_row(
            "tokens",
            Text(f"{_fmt_tokens(_stream_tokens(candidate_dir))} so far", style="cyan"),
            "",
            "",
        )
    elif backend.name or backend.session_id or backend.error_kind:
        cost = f"${backend.cost_usd:.2f}" if backend.cost_usd is not None else "-"
        agent_s = f"{backend.agent_duration_s:.0f}s" if backend.agent_duration_s is not None else "-"
        model = f"  model={backend.model}" if backend.model else ""
        overview.add_row(
            "backend",
            Text(f"{backend.name or '-'}{model}  session={backend.session_id or '-'}"),
            "agent time",
            Text(agent_s, style="cyan" if backend.agent_duration_s is not None else "dim"),
        )
        overview.add_row(
            "tokens",
            Text(
                f"{_fmt_tokens(backend.total_tokens)}  "
                f"turns={backend.num_turns if backend.num_turns is not None else '-'}  cost={cost}"
            ),
            "error",
            Text(backend.error_kind or "-", style="red" if backend.error_kind else "dim"),
        )

    # where the operator's agent actually ran; a link to the full dir
    overview.add_row("path", _path_link(candidate_dir), "", "")
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
                _status_text(child.status, live),
                _score_text(child.val_score, selected=child.is_selected, best=child.is_best),
                Text(_candidate_marks(child), style="gold1" if not child.pruned else "dim"),
                style="dim" if child.pruned else "",
            )
        if len(children) > 12:
            child_table.add_row(f"... {len(children) - 12} more", "", "", "", "", style="dim")
        renderables.append(child_table)

    notes = _tail_text(candidate_dir / "notes.md", max_chars=3000) or candidate.summary.strip()
    if notes:
        renderables.append(
            Panel(Text(notes), title="Notes", title_align="left", border_style="dim cyan")
        )

    from hillclimb.report import candidate_report, render_report

    report_text = render_report(candidate_report(candidate), metric)
    if report_text:
        renderables.append(
            Panel(
                Text(report_text),
                title="Evaluation breakdown",
                title_align="left",
                border_style="dim cyan",
            )
        )

    stderr = _tail_text(candidate_dir / "exec_stderr.log", max_chars=3000)
    stdout = _tail_text(candidate_dir / "exec_stdout.log", max_chars=3000)
    if stderr:
        renderables.append(
            Panel(Text(stderr, style="red"), title="Stderr", title_align="left", border_style="red")
        )
    if stdout:
        renderables.append(
            Panel(Text(stdout), title="Stdout", title_align="left", border_style="dim cyan")
        )

    stream = stream_entries(candidate_dir, max_lines=80)
    if stream:
        renderables.append(
            Panel(
                Group(*(_stream_text(entry) for entry in stream)),
                title="Operator stream",
                title_align="left",
                border_style="dim cyan",
            )
        )

    return renderables


# --- Textual app ---

from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual import events  # noqa: E402
from textual.scrollbar import ScrollBar  # noqa: E402
from textual.screen import ModalScreen, Screen  # noqa: E402
from textual.widgets import DataTable, Footer, Label, RichLog, Static  # noqa: E402

from hillclimb.header import HillclimbHeader, TimezoneMixin  # noqa: E402
from hillclimb.keys import KEYS_BINDING, QUIT_BINDINGS, KeysMixin  # noqa: E402

# one tick per second: the budget countdown and agent stream should read as live
REFRESH_S = 1.0
DETAIL_DEFAULT_HEIGHT = 10
DETAIL_MIN_HEIGHT = 6
DETAIL_STEP = 2
# rows the detail can never take: header, line above the table, divider,
# footer (4) plus the table's header and two rows so it stays usable
DETAIL_RESERVED_ROWS = 7
DETAIL_CHROME_ROWS = 4


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


class LiveScreen(KeysMixin, Screen):
    """A screen that re-reads disk on a timer — only while it is the one on
    top. Textual keeps timers running on suspended screens, and every screen
    below a pushed one would otherwise keep replaying journals once a second
    for a table nobody can see."""

    def start_live(self) -> None:
        self.refresh_data()
        self._live_timer = self.set_interval(REFRESH_S, self.refresh_data)

    def refresh_data(self) -> None:  # pragma: no cover — subclasses implement
        raise NotImplementedError

    def on_screen_suspend(self) -> None:
        self._live_timer.pause()
        # Kitty-graphics placements are painted by the terminal, not Textual:
        # a pushed screen draws its cells but the image from this screen
        # stays on top of them. Delete the placements while suspended...
        for canvas in self._plot_widgets():
            driver = getattr(self.app, "_driver", None)
            if driver is not None:
                try:
                    from plotui import Plot
                    from plotui.textual import tmux_wrap

                    driver.write(tmux_wrap(Plot.kitty_cleanup()))
                except Exception:
                    pass
                break  # one cleanup deletes every placement

    def on_screen_resume(self) -> None:
        # ...and re-transmit when this screen is on top again.
        for canvas in self._plot_widgets():
            canvas._key = None
            canvas.invalidate()
        self.refresh_data()
        self._live_timer.resume()

    def _plot_widgets(self) -> list:
        try:
            from plotui.textual import PlotWidget
        except ImportError:  # pragma: no cover - plotui is a core dep
            return []
        return list(self.query(PlotWidget))


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


class ResizableDetail:
    """A screen with a table on top and a detail panel below it, resizable
    with +/- and by dragging the divider. Subclasses set DETAIL_WIDGET (the
    panel's `#id`) and implement `_detail_open`."""

    DETAIL_WIDGET = "#candidate-detail"
    TABLE_WIDGET = "#candidates"  # hidden while the detail is maximized

    def _init_detail(self) -> None:
        self._detail_height = DETAIL_DEFAULT_HEIGHT
        self._dragging_detail = False
        self._drag_start_y = 0
        self._drag_start_height = DETAIL_DEFAULT_HEIGHT
        self._detail_maximized = False

    @property
    def _detail_open(self) -> bool:  # pragma: no cover — subclasses implement
        raise NotImplementedError

    def _max_detail_height(self) -> int:
        screen_height = self.size.height or DETAIL_DEFAULT_HEIGHT
        return max(DETAIL_MIN_HEIGHT, screen_height - DETAIL_RESERVED_ROWS)

    def _fit_detail_height(self, max_table_rows: int | None = None) -> int:
        """As tall as the detail can be while the whole table above stays
        visible: its header, every row, and a horizontal scrollbar if shown.
        `max_table_rows` caps how many rows the table keeps — the rest
        scroll behind the panel."""
        table = self.query_one(self.TABLE_WIDGET, DataTable)
        rows = table.row_count if max_table_rows is None else min(table.row_count, max_table_rows)
        table_rows = 1 + rows + (1 if table.show_horizontal_scrollbar else 0)
        screen_height = self.size.height or DETAIL_DEFAULT_HEIGHT
        return self._clamp_detail_height(screen_height - DETAIL_CHROME_ROWS - table_rows)

    def _clamp_detail_height(self, height: int) -> int:
        return max(DETAIL_MIN_HEIGHT, min(height, self._max_detail_height()))

    def _set_detail_height(self, height: int) -> None:
        self._detail_height = self._clamp_detail_height(height)
        if not self._detail_maximized:
            self.query_one(self.DETAIL_WIDGET).styles.height = self._detail_height

    def _set_detail_visible(self, visible: bool) -> None:
        display = "block" if visible else "none"
        # maximized: the divider stays hidden (nothing to resize against)
        divider = "none" if self._detail_maximized else display
        self.query_one("#detail-divider", Static).styles.display = divider
        self.query_one(self.DETAIL_WIDGET).styles.display = display
        if visible:
            self._set_detail_height(self._detail_height)
        elif self._detail_maximized:
            self._set_detail_maximized(False)

    def _set_detail_maximized(self, maximized: bool) -> None:
        """Maximized: the table and divider hide and the detail fills the
        screen. Restoring brings back the previous height."""
        self._detail_maximized = maximized
        self.query_one(self.TABLE_WIDGET).styles.display = "none" if maximized else "block"
        self.query_one("#detail-divider", Static).styles.display = "none" if maximized else "block"
        detail = self.query_one(self.DETAIL_WIDGET)
        if maximized:
            detail.styles.height = "1fr"
        else:
            detail.styles.height = self._detail_height

    def action_toggle_maximize_detail(self) -> None:
        if self._detail_open:
            self._set_detail_maximized(not self._detail_maximized)

    def action_grow_detail(self) -> None:
        if self._detail_open:
            self._set_detail_height(self._detail_height + DETAIL_STEP)

    def action_shrink_detail(self) -> None:
        if self._detail_open:
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
        if self._detail_open:
            self._set_detail_height(self._detail_height)


class DetailDivider(Static):
    """Draggable splitter between a table and the detail panel below it."""

    def on_mouse_down(self, event: events.MouseDown) -> None:
        screen = self.screen
        if not getattr(screen, "_detail_open", False):
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


class CandidateScreen(ResizableDetail, LiveScreen):
    """One search: candidate tree plus optional selected-candidate details."""

    BINDINGS = [
        Binding("enter", "open_detail", "details", priority=True),
        Binding("escape", "close_detail_or_back", "back"),
        Binding("+", "grow_detail", "larger detail", show=False),
        Binding("-", "shrink_detail", "smaller detail", show=False),
        Binding("m", "toggle_maximize_detail", "maximize detail", show=False),
        Binding("s", "stop_search", "stop search", show=False),
        Binding("x", "prune_candidate", "prune candidate", show=False),
        Binding("o", "open_candidate_dir", "open candidate dir", show=False),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]
    POINTER_HELP = [("click", "details"), ("drag divider", "resize detail")]

    DEFAULT_CSS = """
    CandidateScreen #candidates { height: 1fr; }
    CandidateScreen #detail-divider {
        height: 1;
        background: $panel;
        color: $secondary;
        text-align: center;
    }
    CandidateScreen #candidate-detail {
        min-height: 6;
        padding: 0 0 0 1;  /* no right padding: the scrollbar sits flush at the edge, like the table's */
    }
    CandidateScreen #searchline { height: 1; padding: 0 1; background: $surface; }
    """

    def __init__(self, config: Config, search_dir: Path, open_candidate_id: str | None = None):
        super().__init__()
        self.config = config
        self.search_dir = search_dir
        self.store = open_store(config)
        self.key = key_for(search_dir)
        self._detail_candidate_id: str | None = None
        self._detail_fingerprint: tuple | None = None
        self._open_on_mount = open_candidate_id
        self._init_detail()

    @property
    def _detail_open(self) -> bool:
        return self._detail_candidate_id is not None

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="searchline")
        yield CandidateTable(id="candidates", cursor_type="row")
        yield DetailDivider(" drag to resize details ", id="detail-divider")
        yield RichLog(id="candidate-detail", wrap=True, markup=False, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#candidates", DataTable)
        _set_candidate_columns(table, holdout=True)
        self._set_detail_visible(False)
        self.start_live()
        if self._open_on_mount is not None:
            journal = self._journal()
            if self._open_on_mount in journal.candidates:
                table.move_cursor(row=table.get_row_index(self._open_on_mount), scroll=True)
                self._open_detail(self._open_on_mount)

    def _journal(self) -> Journal:
        return Journal(self.store.journal(self.key))

    def _record(self) -> SearchRecord:
        record = self.store.search(self.key)
        if record is None:  # search gone (deleted run): keep the screen coherent
            meta = SearchMeta(
                search_id=self.key[1], run_id=self.key[0], problem=self.key[1], problem_id=self.key[1],
                backend="?", model="?", metric="score",
            )
            record = SearchRecord(meta=meta, run_name=self.key[0], state="unknown", search_dir=self.search_dir)
        return record

    def refresh_data(self) -> None:
        from rich.text import Text

        journal = self._journal()
        record = self._record()
        status = self.store.read_status(self.key)
        state = record.state
        line = f"{record.ref}  [{STATE_STYLE.get(state, '')}]{state}[/]"
        if status is not None:
            line += f"  budget left: {_format_budget_left(live_remaining_s(status, state))}"
            if status.current:
                active = " · ".join(
                    f"{c.candidate_id}({c.operator}/{c.phase})" for c in status.current[:3]
                )
                if len(status.current) > 3:
                    active += f" +{len(status.current) - 3}"
                line += f"  active: {active}"
        self.query_one("#searchline", Label).update(line)

        table = self.query_one("#candidates", DataTable)
        holdout = _shows_holdout(record, journal)
        _set_candidate_columns(table, holdout)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in candidate_rows(journal, live=state == "running"):
            styled = [Text(v, style=row.style) for v in _candidate_cells(row, holdout)]
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
        # Size the renderables from the screen, not the log: the log was just
        # shown and has no (or a stale) layout width, and Textual would render
        # the first paint narrow until the next refresh re-wrote it.
        # 4 = screen margins + the log's `padding: 0 1`.
        width = max(20, self.size.width - 4 - (detail.scrollbar_size_vertical if detail.show_vertical_scrollbar else 0))
        renderables = candidate_detail_renderables(self._record(), journal, self._detail_candidate_id)
        # Only rewrite the log when the rendered content changed. A clear +
        # rewrite every live tick flashes the log's tail for a frame before
        # the scroll position is restored — visible flicker on every refresh.
        fingerprint = (self._detail_candidate_id, width, _render_to_text(renderables, width))
        if preserve_scroll and fingerprint == self._detail_fingerprint:
            return
        self._detail_fingerprint = fingerprint
        detail.clear()
        for renderable in renderables:
            detail.write(renderable, width=width, expand=True)
        if preserve_scroll:
            _restore_scroll(detail, snapshot)
        else:
            detail.scroll_home(animate=False)

    def _open_detail(self, candidate_id: str | None = None) -> None:
        candidate_id = candidate_id or self._selected_candidate_id()
        if candidate_id is None:
            return
        if self._detail_candidate_id is None:  # opening: fit under the full table
            self._detail_height = self._fit_detail_height()
        self._detail_candidate_id = candidate_id
        self._render_detail(self._journal())

    def _close_detail(self) -> None:
        self._detail_candidate_id = None
        self._detail_fingerprint = None
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
        # the detail follows the cursor. Read the cursor now, not the event:
        # each live refresh clears and restores the table, and the stale
        # highlight from that would re-render (and re-scroll) the detail
        if event.data_table.id == "candidates" and self._detail_candidate_id is not None:
            current = self._selected_candidate_id()
            if current is not None and current != self._detail_candidate_id:
                self._detail_candidate_id = current
                self._render_detail(self._journal())
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

    def action_open_candidate_dir(self) -> None:
        """Reveal the selected candidate's working dir in the file manager."""
        candidate_id = self._detail_candidate_id or self._selected_candidate_id()
        if candidate_id is None:
            return
        candidate = self._journal().candidates.get(candidate_id)
        if candidate is None:
            return
        path = _candidate_dir(self.search_dir, candidate)
        if not path.exists():
            self.notify(f"{path} does not exist", severity="warning")
            return
        try:
            open_in_file_manager(path)
        except OSError as exc:
            self.notify(f"could not open {path}: {exc}", severity="error")
            return
        self.notify(f"opened {_path_label(path)}", severity="information")

    def action_stop_search(self) -> None:
        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(self.store, self.key, source="tui")
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
                    self.store,
                    self.key,
                    candidate_id,
                    higher_is_better=bool(self._record().meta.higher_is_better),
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


class SearchesScreen(ResizableDetail, LiveScreen):
    """Searches inside one run, with the highlighted search's candidates in
    a panel underneath (enter opens it; enter again, or `o`, drills into
    the full candidate screen)."""

    DETAIL_WIDGET = "#search-candidates"
    TABLE_WIDGET = "#searches"

    BINDINGS = [
        Binding("enter", "open_search", "candidates", priority=True),
        Binding("escape", "close_panel_or_back", "back"),
        Binding("o", "open_search_screen", "full candidate view", show=False),
        Binding("+", "grow_detail", "larger panel", show=False),
        Binding("-", "shrink_detail", "smaller panel", show=False),
        Binding("m", "toggle_maximize_detail", "maximize panel", show=False),
        Binding("s", "stop_search", "stop search", show=False),
        Binding("g", "open_graph", "knowledge graph", show=False),
        Binding("t", "toggle_tree", "tree panel", show=False),
        Binding("j", "scrub_back", "tree: back in time", show=False),
        Binding("k", "scrub_forward", "tree: forward", show=False),
        Binding("end", "scrub_live", "tree: live", show=False),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]
    POINTER_HELP = [("click", "candidates"), ("drag divider", "resize panel")]

    DEFAULT_CSS = """
    SearchesScreen #runline { height: 1; padding: 0 1; background: $surface; }
    SearchesScreen #searches { height: 1fr; }
    SearchesScreen #detail-divider {
        height: 1;
        background: $panel;
        color: $secondary;
        text-align: center;
    }
    SearchesScreen #search-candidates { min-height: 6; }
    SearchesScreen #search-tree { min-height: 6; }
    SearchesScreen #search-scrubber { height: 2; background: $surface; padding: 0 1; }
    SearchesScreen #search-node-detail { dock: right; width: 48; display: none; padding: 0 1; }
    """

    def __init__(self, config: Config, run_dir: Path, run_name: str):
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self.run_name = run_name
        self.store = open_store(config)
        self._panel_search_id: str | None = None
        self._tree_open = False           # the topology panel, toggled with t
        self._tree_fingerprint: tuple | None = None
        self._init_detail()

    @property
    def _detail_open(self) -> bool:
        return self._panel_search_id is not None or self._tree_open

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
        yield Label(id="runline")
        yield DataTable(id="searches", cursor_type="row")
        yield DetailDivider(" drag to resize candidates ", id="detail-divider")
        yield DataTable(id="search-candidates", cursor_type="row")
        # imported here: treeview imports back into watch (LiveScreen et al)
        from hillclimb.graphview import TimeScrubber
        from hillclimb.treeview import TreePlotWidget

        yield TreePlotWidget(id="search-tree")
        yield TimeScrubber(id="search-scrubber")
        yield RichLog(id="search-node-detail", wrap=True, markup=False, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#searches", DataTable)
        table.add_columns(
            "search",
            "problem",
            "model",
            "tokens",
            "state",
            "candidates",
            "best val",
            "selected",
            "duration",
        )
        panel = self.query_one("#search-candidates", DataTable)
        _set_candidate_columns(panel, holdout=True)
        self.query_one("#search-tree").styles.display = "none"
        self.query_one("#search-scrubber").styles.display = "none"
        self._set_detail_visible(False)
        self.start_live()

    def refresh_data(self) -> None:
        from rich.text import Text

        self.query_one("#runline", Label).update(f"run: {self.run_name}  ({self.run_dir.name})")
        table = self.query_one("#searches", DataTable)
        snapshot = _snapshot_table(table)
        table.clear()
        for row in scan_searches(self.store, self.run_dir.name):
            state = Text(row.state, style=STATE_STYLE.get(row.state, ""))
            table.add_row(
                row.search_id, row.problem, row.model, row.tokens, state, row.candidates,
                row.best_val, row.selected, row.duration, key=row.search_id,
            )
        _restore_table(table, snapshot)
        if self._panel_search_id is not None:
            self._render_panel()
        if self._tree_open:
            self._render_tree()

    def _render_panel(self) -> None:
        from rich.text import Text

        key = (self.run_dir.name, self._panel_search_id)
        panel = self.query_one("#search-candidates", DataTable)
        snapshot = _snapshot_table(panel)
        panel.clear()
        record = self.store.search(key)
        if record is None:
            _restore_table(panel, snapshot)
            return
        journal = Journal(self.store.journal(key))
        live = record.state == "running"
        holdout = _shows_holdout(record, journal)
        _set_candidate_columns(panel, holdout)
        for row in candidate_rows(journal, live=live):
            styled = [Text(v, style=row.style) for v in _candidate_cells(row, holdout)]
            panel.add_row(*styled, key=row.candidate_id)
        _restore_table(panel, snapshot)

    def _selected_search_id(self) -> str | None:
        table = self.query_one("#searches", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return row_key.value

    def _selected_search_dir(self) -> Path | None:
        search_id = self._selected_search_id()
        return None if search_id is None else self.run_dir / "searches" / search_id

    def _open_panel(self, search_id: str, focus: bool = False) -> None:
        if self._panel_search_id is None:  # opening: fit under the full table
            self._detail_height = self._fit_detail_height()
        self._panel_search_id = search_id
        self._set_detail_visible(True)
        self._render_panel()
        if focus:
            self.query_one("#search-candidates", DataTable).focus()

    def _close_panel(self) -> None:
        self._panel_search_id = None
        self.query_one("#search-candidates", DataTable).clear()
        self._set_detail_visible(False)
        self.query_one("#searches", DataTable).focus()

    def action_toggle_tree(self) -> None:
        """`t`: the highlighted search's exploration tree under the table,
        following the cursor; `t` again (or escape) closes it. `j`/`k`
        scrub back/forward over landed results, `end` back to live."""
        if self._tree_open:
            self._close_tree()
            return
        if self._panel_search_id is not None:  # the two share the lower panel
            self._close_panel()
        self.DETAIL_WIDGET = "#search-tree"
        self._tree_open = True
        # all the room the searches leave; past 8 searches the tree wins
        self._detail_height = self._fit_detail_height(max_table_rows=8)
        self._set_detail_visible(True)
        self.query_one("#search-scrubber").styles.display = "block"
        self._render_tree()

    def _close_tree(self) -> None:
        self._tree_open = False
        self._tree_fingerprint = None
        self._show_node_detail(None)
        canvas = self.query_one("#search-tree")
        canvas.selected = None
        self.query_one("#search-scrubber").styles.display = "none"
        self._set_detail_visible(False)
        self.DETAIL_WIDGET = "#search-candidates"
        self.query_one("#searches", DataTable).focus()

    def _render_tree(self) -> None:
        from hillclimb.graphview import TimeScrubber
        from hillclimb.tree import build_tree, candidates_until, tree_events

        search_id = self._selected_search_id()
        if search_id is None:
            return
        record = self.store.search((self.run_dir.name, search_id))
        if record is None:
            return
        journal = Journal(self.store.journal(record.key))
        candidates = list(journal.candidates.values())
        scrubber = self.query_one("#search-scrubber", TimeScrubber)
        canvas = self.query_one("#search-tree")
        switched = self._tree_fingerprint is None or self._tree_fingerprint[0] != search_id
        if switched:  # a different search: its own time, selection and view
            scrubber.set_index(None)
            canvas.selected = None
            canvas.hidden = frozenset()
            self._show_node_detail(None)
        scrubber.set_events(tree_events(candidates))
        fingerprint = (
            search_id,
            scrubber.index,
            tuple((c.candidate_id, c.status, c.val_score, c.pruned, c.finished_at) for c in candidates),
        )
        if fingerprint == self._tree_fingerprint:
            return
        self._tree_fingerprint = fingerprint
        higher = bool(record.meta.higher_is_better)
        live_tree = build_tree(candidates, higher)
        until = None
        if scrubber.index is not None and scrubber.events_list:
            until = scrubber.events_list[scrubber.index]
        shown = build_tree(candidates_until(candidates, until), higher, layout=live_tree)
        canvas.set_tree(shown, frame=live_tree)

    def action_scrub_back(self) -> None:
        if self._tree_open:
            self.query_one("#search-scrubber").step(-1)

    def action_scrub_forward(self) -> None:
        if self._tree_open:
            self.query_one("#search-scrubber").step(1)

    def action_scrub_live(self) -> None:
        if self._tree_open:
            self.query_one("#search-scrubber").set_index(None)
            self._render_tree()

    def on_time_scrubber_time_changed(self, message) -> None:
        self._render_tree()

    def _show_node_detail(self, node_id: str | None) -> None:
        """The clicked node's candidate details slide out on the right, as in
        the full tree view."""
        detail = self.query_one("#search-node-detail", RichLog)
        search_id = self._selected_search_id()
        if node_id is None or search_id is None:
            detail.styles.display = "none"
            return
        record = self.store.search((self.run_dir.name, search_id))
        if record is None:
            return
        journal = Journal(self.store.journal(record.key))
        detail.clear()
        for renderable in candidate_detail_renderables(record, journal, node_id):
            detail.write(renderable)
        detail.styles.display = "block"

    def on_tree_plot_widget_node_selected(self, message) -> None:
        self._show_node_detail(message.node_id)

    def on_tree_plot_widget_node_activated(self, message) -> None:
        """Double-activating a node in the tree drills into that candidate."""
        search_id = self._selected_search_id()
        if search_id is None:
            return
        self.app.push_screen(
            CandidateScreen(
                self.config, self.run_dir / "searches" / search_id, open_candidate_id=message.node_id
            )
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # moving the cursor while the panel is open follows the cursor. Read
        # the cursor now rather than event.row_key: a live refresh clears the
        # table (cursor briefly at row 0) and restores it, and the stale
        # highlight from that clear would flip the panel to another search
        if event.data_table.id == "searches":
            if self._panel_search_id is not None:
                current = self._selected_search_id()
                if current is not None and current != self._panel_search_id:
                    self._open_panel(current)
                event.stop()
            elif self._tree_open:
                self._render_tree()
                event.stop()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "search-candidates":
            self._open_candidate(event.row_key.value)
            event.stop()

    def _panel_candidate_id(self) -> str | None:
        panel = self.query_one("#search-candidates", DataTable)
        if not panel.row_count:
            return None
        row_key, _ = panel.coordinate_to_cell_key(panel.cursor_coordinate)
        return row_key.value

    def _open_candidate(self, candidate_id: str | None) -> None:
        """Enter on a candidate in the panel: the candidates take the whole
        screen with that candidate's details underneath (the full candidate
        screen, opened on it)."""
        if self._panel_search_id is None or candidate_id is None:
            return
        search_dir = self.run_dir / "searches" / self._panel_search_id
        self.app.push_screen(CandidateScreen(self.config, search_dir, open_candidate_id=candidate_id))

    def action_open_search(self) -> None:
        if self.focused is not None and self.focused.id == "search-candidates":
            self._open_candidate(self._panel_candidate_id())
            return
        search_id = self._selected_search_id()
        if search_id is None:
            return
        # enter opens the panel and moves the cursor into it
        self._open_panel(search_id, focus=True)

    def action_open_search_screen(self) -> None:
        search_dir = self._selected_search_dir()
        if search_dir is not None:
            self.app.push_screen(CandidateScreen(self.config, search_dir))

    def action_close_panel_or_back(self) -> None:
        if self._tree_open and self.query_one("#search-tree").selected is not None:
            canvas = self.query_one("#search-tree")
            canvas.selected = None
            canvas.rebuild()
            self._show_node_detail(None)
        elif self._tree_open:
            self._close_tree()
        elif self._panel_search_id is not None:
            self._close_panel()
        else:
            self.app.pop_screen()

    def action_open_graph(self) -> None:
        # lazy so watch never pays for graphview at import time (and the
        # reverse import of the mouse helpers stays cycle-free)
        from hillclimb.graphview import GraphScreen

        self.app.push_screen(GraphScreen(self.config))

    def action_stop_search(self) -> None:
        search_dir = self._selected_search_dir()
        if search_dir is None:
            return

        def go(confirmed: bool | None) -> None:
            if confirmed:
                outcome = request_stop(self.store, key_for(search_dir), source="tui")
                self.notify(outcome or "engine is not running", severity="information")

        self.app.push_screen(
            ConfirmScreen(f"Stop search {_search_ref(search_dir)}? (parks after current operator)"),
            go,
        )


class RunsScreen(LiveScreen):
    """Top-level runs."""

    BINDINGS = [
        Binding("enter", "open_run", "open", priority=True),
        Binding("g", "open_graph", "knowledge graph", show=False),
        KEYS_BINDING,
        *QUIT_BINDINGS,
    ]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.store = open_store(config)
        self._names: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield HillclimbHeader()
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
        self.start_live()

    def refresh_data(self) -> None:
        from rich.text import Text

        table = self.query_one("#runs", DataTable)
        snapshot = _snapshot_table(table)
        self._names.clear()
        table.clear()
        for row in scan_runs(self.store):
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

    def action_open_graph(self) -> None:
        from hillclimb.graphview import GraphScreen

        self.app.push_screen(GraphScreen(self.config))


class WatchApp(TimezoneMixin, App):
    """Read-mostly dashboard over runs/; control actions go through the
    same command queue as the CLI."""

    BINDINGS = [
        Binding("t", "choose_timezone", "time zone", show=False),
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
    ]
    CSS = HILLCLIMB_CSS

    def __init__(self, config: Config | None = None, search_dir: Path | None = None):
        super().__init__()
        self.config = config or Config.load()
        # open straight on this search's candidates, with the run list and
        # its searches underneath so esc walks back the usual way
        self.search_dir = search_dir

    def on_mount(self) -> None:
        apply_theme(self)
        self._init_timezone()
        self.push_screen(RunsScreen(self.config))
        if self.search_dir is not None:
            run_dir = self.search_dir.parents[1]
            self.push_screen(SearchesScreen(self.config, run_dir, run_display_name(run_dir)))
            self.push_screen(CandidateScreen(self.config, self.search_dir))
