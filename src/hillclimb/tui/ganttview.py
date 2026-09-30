"""The operator timeline of one search — a cell-native gantt, live.

One row per agent slot (lanes reconstructed in gantt.py), one bar per
candidate coloured by its operator: the dim stretch is the agent writing
code, the bright stretch the verifier running, `◆` the moment it scored.
Running bars grow toward the right edge every tick and end in `▶`; a slot
parked on rate limits reads hollow (`░`); failed and pruned bars are faded.

Same architecture as graphview.py's scrubber: a pure `render_gantt` up top
(testable outside Textual), a thin `Static` shell below. watch.py toggles
the panel on the searches screen with `a`.
"""

from __future__ import annotations

from itertools import groupby

from rich.style import Style
from rich.text import Text
from textual import events
from textual.widgets import Static

from hillclimb.tui.gantt import GanttLayout, GanttSpan, minute_to_col
from hillclimb.tui.tree import FAILED_STATUSES
from hillclimb.tui.treeview import OPERATOR_RGB, dim_rgb

LANE_LABEL_W = 3  # "a1 " — keeps the track aligned across rows
AGENT_DIM = 0.55  # the agent-phase stretch of a healthy bar
DEAD_DIM = 0.5    # failed/pruned bars and parked fill, matching the tree

GANTT_STYLES = {  # component class -> fallback Rich style (outside Textual)
    "gantt--axis": "dim",
    "gantt--lane-label": "dim",
    "gantt--marker": "bold rgb(255,200,40)",  # the tree's best-label gold
    "gantt--hint": "dim",
    "gantt--track": "bright_black",
}

# Ruler steps in minutes; extended by doubling when a search outgrows them.
_TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 240, 480)


def _rgb_style(rgb: tuple[int, int, int]) -> Style:
    return Style(color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")


def _operator_rgb(operator: str) -> tuple[int, int, int]:
    return OPERATOR_RGB.get(operator, (160, 160, 160))


def _axis_label(minutes: float) -> str:
    m = int(round(minutes))
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h" if m % 60 == 0 else f"{m // 60}h{m % 60:02d}m"


def _tick_step(extent_min: float, track_width: int) -> float:
    """The smallest nice step that keeps ticks ~12 columns apart."""
    for step in _TICK_STEPS:
        if step / extent_min * max(track_width - 1, 1) >= 12:
            return float(step)
    step = float(_TICK_STEPS[-1])
    while step / extent_min * max(track_width - 1, 1) < 12:
        step *= 2
    return step


def _append_cells(text: Text, cells: list[tuple[str, str | Style]]) -> None:
    for style, group in groupby(cells, key=lambda cell: cell[1]):
        text.append("".join(char for char, _ in group), style)


def _lane_row(
    spans: list[GanttSpan], extent_min: float, track_width: int,
    st: dict[str, str | Style],
) -> list[tuple[str, str | Style]]:
    cells: list[tuple[str, str | Style]] = [("·", st["gantt--track"])] * track_width
    for span in sorted(spans, key=lambda s: s.start_min):
        rgb = _operator_rgb(span.operator)
        c0 = minute_to_col(span.start_min, extent_min, track_width)
        c1 = max(minute_to_col(span.end_min, extent_min, track_width), c0)
        if span.status in FAILED_STATUSES or span.pruned:
            fill = [("█", _rgb_style(dim_rgb(rgb, DEAD_DIM)))] * (c1 - c0 + 1)
        elif span.running and span.phase == "waiting-slot":
            fill = [("░", _rgb_style(dim_rgb(rgb, DEAD_DIM)))] * (c1 - c0 + 1)
        else:
            boundary = c1 + 1  # no trial yet: the whole bar is the agent phase
            if span.exec_min is not None:
                boundary = max(minute_to_col(span.exec_min, extent_min, track_width), c0)
            fill = [
                ("█", _rgb_style(dim_rgb(rgb, AGENT_DIM) if col < boundary else rgb))
                for col in range(c0, c1 + 1)
            ]
        if span.running:
            fill[-1] = ("▶", _rgb_style(rgb))
        cells[c0:c1 + 1] = fill
    for span in spans:  # markers last, so they win the cell over any bar
        if span.score_min is not None:
            cells[minute_to_col(span.score_min, extent_min, track_width)] = ("◆", st["gantt--marker"])
    return cells


def _axis_row(
    extent_min: float, track_width: int, live: bool, st: dict[str, str | Style],
) -> list[tuple[str, str | Style]]:
    chars = ["─"] * track_width
    step = _tick_step(extent_min, track_width)
    minute = step
    while minute < extent_min:
        col = minute_to_col(minute, extent_min, track_width)
        chars[col] = "┴"
        for offset, char in enumerate(_axis_label(minute), start=1):
            if col + offset >= track_width:
                break
            chars[col + offset] = char
        minute += step
    tail = "now" if live else _axis_label(extent_min)
    if track_width > len(tail):
        chars[track_width - len(tail):] = list(tail)
    return [(char, st["gantt--axis"]) for char in chars]


def _legend(st: dict[str, str | Style]) -> Text:
    text = Text(no_wrap=True)
    for operator in ("draft", "debug", "improve", "ensemble", "baseline"):
        text.append("█", _rgb_style(_operator_rgb(operator)))
        text.append(f" {operator}  ", st["gantt--hint"])
    text.append("◆", st["gantt--marker"])
    text.append(" scored  ", st["gantt--hint"])
    text.append("░", st["gantt--track"])
    text.append(" waiting", st["gantt--hint"])
    return text


def render_gantt(
    layout: GanttLayout, width: int, height: int,
    styles: dict[str, str | Style] | None = None,
) -> Text:
    """Lane rows, an axis, a legend — fitted to `width` × `height` cells.
    The whole search spans the track: minute 0 at the left edge, "now" (or
    the last finish) at the right."""
    st = {**GANTT_STYLES, **(styles or {})}
    text = Text(no_wrap=True)
    if not layout.spans:
        text.append("no candidates yet", st["gantt--hint"])
        return text
    track_width = max(width - LANE_LABEL_W, 12)
    lane_budget = max(height - 2, 1)  # axis + legend keep their rows
    shown = layout.n_lanes if layout.n_lanes <= lane_budget else max(lane_budget - 1, 1)
    by_lane: dict[int, list[GanttSpan]] = {}
    for span in layout.spans:
        by_lane.setdefault(span.lane, []).append(span)
    for lane in range(shown):
        text.append(f"a{lane + 1}".ljust(LANE_LABEL_W), st["gantt--lane-label"])
        _append_cells(text, _lane_row(by_lane.get(lane, []), layout.extent_min, track_width, st))
        text.append("\n")
    if shown < layout.n_lanes:
        text.append(f"… +{layout.n_lanes - shown} lanes", st["gantt--hint"])
        text.append("\n")
    text.append(" " * LANE_LABEL_W)
    _append_cells(text, _axis_row(layout.extent_min, track_width, layout.live, st))
    text.append("\n")
    legend = _legend(st)
    legend.truncate(max(width, 12))
    text.append_text(legend)
    return text


class GanttPanel(Static):
    """Thin shell: watch.py hands it a GanttLayout, it renders to its size."""

    ALLOW_SELECT = False

    COMPONENT_CLASSES = set(GANTT_STYLES)

    DEFAULT_CSS = """
    GanttPanel > .gantt--axis { color: $foreground 50%; }
    GanttPanel > .gantt--lane-label { color: $text-muted; }
    GanttPanel > .gantt--marker { color: rgb(255,200,40); text-style: bold; }
    GanttPanel > .gantt--hint { color: $text-muted; }
    GanttPanel > .gantt--track { color: $foreground 15%; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._layout: GanttLayout | None = None

    def set_layout(self, layout: GanttLayout | None) -> None:
        self._layout = layout
        self._refresh_gantt()

    def _refresh_gantt(self) -> None:
        if self._layout is None:
            self.update(Text())
            return
        width = self.content_size.width or 60
        height = self.content_size.height or 10
        styles = {name: self.get_component_rich_style(name) for name in GANTT_STYLES}
        self.update(render_gantt(self._layout, width, height, styles))

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_gantt()
