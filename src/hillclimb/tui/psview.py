"""`hillclimb ps` drawn like nvidia-smi: one self-contained box sized to the
terminal, about compute only.

The box holds the machine at a glance (engines, coding agents against the
machine's slot cap, cpu, memory) above one process table: each engine
as a row of its own — its problem and hillclimb dir in the command column —
with its coding agents, their tool shells and MCP servers, verifiers and the
solution they score nested under it. Search progress (state, budget, best,
candidates) is `watch`'s, not this view's. Columns that do not fit the width
are dropped, least useful first; commands are shortened
(`machine.short_command`) and then cut with an ellipsis, never wrapped. With
`max_height` (the `--watch` frame) the rows are capped so the whole frame
stays on one screen. Rich only, no Textual: the plain CLI draws it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hillclimb.terms import ENGINE, role_label
from hillclimb.tui.palette import CYAN, MAGENTA, RED, YELLOW
from hillclimb.tui.machine import (
    EngineRow,
    MachineRow,
    engine_target,
    format_mem,
    format_seconds,
    short_command,
    short_dir,
)

BORDER = CYAN
MAX_WIDTH = 150  # past this, a wider box only spreads the same columns thinner

# the head row of a block: an engine, or a hillclimb command computing in a
# terminal (`machine.job_kind`)
HEAD_ROLES = frozenset({"engine", "verify", "grade", "run"})

ROLE_STYLE = {
    "engine": "bold",
    "verify": "bold",
    "grade": "bold",
    "run": "bold",
    "agent": f"bold {CYAN}",
    "tool": CYAN,
    "mcp": "dim",
    "verifier": f"bold {MAGENTA}",
    "solution": MAGENTA,
    "child": "",
}

# (header, width, justify, drop order — higher drops first); `cpu` is per
# process the way `top` counts it, 100% = one core; `command` takes
# whatever the others leave, never less than its width
_COLUMNS = [
    ("pid", 6, "right", 0),
    ("role", 8, "left", 0),
    ("cpu", 6, "right", 0),
    ("mem", 5, "right", 1),
    ("up", 7, "right", 2),
    ("command", 28, "left", 0),
]

# which rows an engine keeps when the screen is short, best first: what is
# doing the work
_KEEP_ORDER = {**{role: -1 for role in HEAD_ROLES}, "agent": 0, "verifier": 1, "solution": 2, "tool": 3, "child": 4, "mcp": 5}


def _cpu_style(cpu: float) -> str:
    return f"bold {RED}" if cpu >= 90 else YELLOW if cpu >= 40 else ""


def _cell(value: str, style: str = "") -> Text:
    text = Text(value, style=style, overflow="ellipsis")
    text.no_wrap = True
    return text


def summary_line(
    machine: MachineRow, engines: int, width: int = 200, jobs: Mapping[str, int] | None = None
) -> Text:
    """The machine at a glance, its items flowing onto a second line rather
    than running past `width` (the inside of the box). `jobs` counts the
    hillclimb commands computing in a terminal by kind (`verify 1`)."""
    items: list[list[tuple[str, str]]] = [
        [(f"{ENGINE.pl} ", "dim"), (str(engines), "bold")],
        *([(f"{kind} ", "dim"), (str(count), "bold")] for kind, count in (jobs or {}).items() if count),
        [
            ("coding agents ", "dim"),
            (f"{machine.agents}/{machine.agent_slots}" if machine.agent_slots else str(machine.agents), "bold"),
        ],
        # in cores, not percent: `ps` counts 100% per core, so "100% of 10
        # cores" read as a full machine when one core was busy
        [
            ("cpu ", "dim"),
            (f"{machine.cpu / 100:.1f}", _cpu_style(machine.cpu / max(1, machine.cores)) or "bold"),
            (f" of {machine.cores} cores busy", "dim"),
        ],
    ]
    mem = [("mem ", "dim"), (format_mem(machine.rss_mb), "bold")]
    if machine.mem_total_mb:
        mem.append((f" / {format_mem(machine.mem_total_mb)}", "dim"))
    items.append(mem)

    text = Text(overflow="ellipsis")
    text.no_wrap = True
    line = 0
    for item in items:
        size = sum(len(part) for part, _ in item)
        if line and line + 5 + size > width:
            text.append("\n")
            line = 0
        elif line:
            text.append("  ·  ", style="dim")
            line += 5
        for part, style in item:
            text.append(part, style=style)
        line += size
    return text


def _fit(width: int) -> list[str]:
    """The headers that fit `width`: each column padded a space a side, a
    one-space divider between columns, the box's border and padding (4)."""
    kept = list(_COLUMNS)

    def need(cols) -> int:
        return sum(c[1] for c in cols) + 3 * len(cols) - 1 + 4

    while need(kept) > width and any(c[3] for c in kept):
        worst = max(c[3] for c in kept)
        kept = [c for c in kept if c[3] != worst]
    return [c[0] for c in kept]


@dataclass
class Line:
    """One row of the table: a process, or folded MCP servers."""

    pid: str
    role: str
    cpu: float
    rss_mb: float
    up: str
    command: str
    folded: int = 0  # MCP server processes this row stands for


def engine_lines(engine: EngineRow, fold_mcp: bool) -> list[Line]:
    """The engine's own row, then its processes in tree order; with
    `fold_mcp`, each run of MCP server processes under a coding agent becomes
    one row (they idle; a crowded screen should spend its rows on what works)."""
    where = short_dir(engine.hillclimb_dir) if engine.hillclimb_dir else "?"
    # the orphan mark ahead of the path, so a long path cannot cut it off
    orphan = "orphan, dir deleted  ·  " if engine.orphan else ""
    # a search's process is named by its search (two parallel searches of
    # one problem would otherwise read the same); a foreground command, or
    # a search not recorded yet, by what its command line says
    name = engine.search.ref.rsplit("/", 1)[-1] if engine.kind == "engine" and engine.search else engine_target(engine.argv)
    head = f"{name}  ·  {orphan}{where}"
    lines = [Line(str(engine.pid), engine.kind, engine.own_cpu, engine.own_rss_mb, format_seconds(engine.up_s), head)]
    for proc in engine.procs:
        if fold_mcp and proc.role == "mcp":
            if lines[-1].folded:
                lines[-1].folded += 1
                lines[-1].cpu += proc.cpu
                lines[-1].rss_mb += proc.rss_mb
            else:
                guide = proc.guide.replace("└─ ", "├─ ")
                lines.append(Line("", "mcp", proc.cpu, proc.rss_mb, "", guide, folded=1))
            continue
        lines.append(
            Line(
                str(proc.pid), proc.role, proc.cpu, proc.rss_mb, format_seconds(proc.up_s),
                proc.guide + short_command(proc.command, proc.role, engine.hillclimb_dir),
            )
        )
    for line in lines:
        if line.folded:
            plural = "es" if line.folded != 1 else ""
            line.command += f"{line.folded} MCP server process{plural} from your Claude config"
    return lines


def _most_useful(lines: list[Line], count: int) -> list[Line]:
    """`count` of `lines`, the most telling roles first (busiest first
    within a role), shown in their tree order."""
    if count <= 0:
        return []
    ranked = sorted(range(len(lines)), key=lambda i: (_KEEP_ORDER.get(lines[i].role, 9), -lines[i].cpu, i))
    return [lines[i] for i in sorted(ranked[:count])]


def _budget(lengths: list[int], max_rows: int) -> list[int]:
    """Rows per engine when `lengths` do not fit `max_rows`: an equal share,
    what one engine does not need going to the others."""
    quota = [0] * len(lengths)
    left = max_rows
    pending = [i for i, n in enumerate(lengths) if n]
    while pending and left > 0:
        share = max(1, left // len(pending))
        for i in list(pending):
            take = min(share, lengths[i] - quota[i], left)
            quota[i] += take
            left -= take
            if quota[i] >= lengths[i]:
                pending.remove(i)
            if left <= 0:
                break
    return quota


def _table(engines: list[EngineRow], width: int, max_rows: int | None) -> Table:
    headers = _fit(width)
    spec = {c[0]: c for c in _COLUMNS}
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=True, header_style="bold dim")
    for header in headers:
        _, size, justify, _ = spec[header]
        stretch = header == "command"
        table.add_column(
            header, justify=justify, no_wrap=True, overflow="ellipsis",
            width=None if stretch else size, min_width=size if stretch else None, ratio=1 if stretch else None,
        )

    per_engine = [engine_lines(e, fold_mcp=False) for e in engines]
    if max_rows is not None and sum(map(len, per_engine)) > max_rows:
        per_engine = [engine_lines(e, fold_mcp=True) for e in engines]
    lengths = [len(lines) for lines in per_engine]
    quotas = _budget(lengths, max_rows) if max_rows is not None and sum(lengths) > max_rows else lengths

    for index, (lines, quota) in enumerate(zip(per_engine, quotas)):
        if index:
            table.add_section()
        cut = quota < len(lines)
        # a cut engine spends one of its rows on the "… n more" line, and
        # always keeps its own row
        visible = _most_useful(lines, max(1, quota - 1)) if cut else lines
        for line in visible:
            values = {
                "pid": _cell(line.pid, "bold" if line.role in HEAD_ROLES else ""),
                "role": _cell(role_label(line.role), ROLE_STYLE.get(line.role, "")),
                "cpu": _cell(f"{line.cpu:.1f}%", _cpu_style(line.cpu)),
                "mem": _cell(format_mem(line.rss_mb)),
                "up": _cell(line.up),
                "command": _cell(
                    line.command,
                    "bold" if line.role in HEAD_ROLES else "dim" if line.role == "mcp" else "",
                ),
            }
            table.add_row(*(values[h] for h in headers))
        if cut:
            values = {h: _cell("") for h in headers}
            values["command"] = _cell(f"… {len(lines) - len(visible)} more · hillclimb top lists all", "dim")
            table.add_row(*(values[h] for h in headers))
    return table


def counts(engines: list[EngineRow]) -> tuple[int, dict[str, int]]:
    """(engines, {other kind: count}) — what the summary line counts."""
    kinds = Counter(e.kind for e in engines)
    return kinds.pop("engine", 0), dict(sorted(kinds.items()))


def render_ps(
    machine: MachineRow,
    engines: list[EngineRow],
    width: int,
    *,
    max_height: int | None = None,
    footer: str | None = None,
    now: datetime | None = None,
) -> RenderableType:
    """The whole `ps` frame at `width` columns (capped at MAX_WIDTH). With
    `max_height`, process rows are capped so the frame fits that many lines."""
    width = max(40, min(width, MAX_WIDTH))
    stamp = (now or datetime.now()).strftime("%a %H:%M:%S")

    def frame(max_rows: int | None) -> Group:
        engine_count, jobs = counts(engines)
        summary = summary_line(machine, engine_count, width - 4, jobs)
        body: RenderableType
        if engines:
            body = Group(summary, Rule(style="dim"), _table(engines, width, max_rows))
        else:
            body = Group(summary, Rule(style="dim"), _cell(f"no hillclimb {ENGINE.pl} running", "dim"))
        parts: list[RenderableType] = [
            Panel(
                body,
                title="[bold]hillclimb ps[/]",
                title_align="left",
                subtitle=f"[dim]{stamp}[/]",
                subtitle_align="right",
                border_style=BORDER,
                box=box.ROUNDED,
                width=width,
                padding=(0, 1),
            )
        ]
        if footer:
            parts.append(Text.from_markup(footer, overflow="ellipsis"))
        return Group(*parts)

    if max_height is None or not engines:
        return frame(None)
    # measure, don't estimate: the summary wraps, sections add gaps, folding
    # changes the row count — shrink the rows until the frame fits
    rows = len(engines) + sum(len(e.procs) for e in engines)
    while rows > len(engines) and _height(frame(rows), width) > max_height:
        rows -= 1
    return frame(rows)


def _height(renderable: RenderableType, width: int) -> int:
    console = Console(width=width, file=StringIO(), color_system=None)
    return len(console.render_lines(renderable, console.options.update(width=width), pad=False))
