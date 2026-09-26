"""One search's operator schedule — pure data, no Textual.

Each candidate is one interval on a wall-clock axis: from the moment its
operator was dispatched (`created_at`) to the moment it finished (or "now"
while it is still in flight). No slot index is journaled, so lanes are
reconstructed by greedy first-fit interval packing — with `parallel_operators`
slots and no stalls that converges to exactly one lane per slot, and any
extra lanes are themselves signal (bursts). The agent→verifier boundary is
the first trial's `started_at`; the score moment is the last trial's
`finished_at`.

`ganttview.py` renders this with Rich; `watch.py` toggles it as a panel on
the searches screen. Both stay replay-only readers of the journal.
"""

from __future__ import annotations

from dataclasses import dataclass

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.tree import minutes_since

# Two spans whose endpoints meet within this are "back to back": the later
# one reuses the lane the earlier one just freed.
_LANE_EPS = 1e-9


@dataclass(frozen=True)
class GanttSpan:
    candidate_id: str
    lane: int
    operator: str            # baseline | draft | debug | improve | ensemble | seed | …
    status: str
    pruned: bool
    start_min: float         # created_at − origin
    end_min: float           # finished_at − origin; running spans get "now"
    running: bool            # finished_at was None in a live search
    exec_min: float | None   # trials[0].started_at − origin: agent→verifier boundary
    score_min: float | None  # trials[-1].finished_at − origin, only when scored
    score: float | None
    phase: str | None        # live phase (agent | exec | waiting-slot); None if not in flight


@dataclass(frozen=True)
class GanttLayout:
    spans: tuple[GanttSpan, ...]
    n_lanes: int
    extent_min: float        # right edge of the window: "now" live, last finish otherwise
    live: bool


def _clamp(value: float | None, lo: float, hi: float) -> float | None:
    return None if value is None else min(max(value, lo), hi)


def build_gantt(
    candidates: list[Candidate],
    *,
    origin: str | None,
    now: str | None = None,
    phases: dict[str, str] | None = None,
) -> GanttLayout:
    """Pack a journal's candidates into lanes. `origin` is the search's
    start (`SearchStatus.started_at`); an unparseable origin falls back to
    the earliest `created_at`. `now` (`utcnow()`) marks the search live and
    is the right edge in-flight bars grow toward; `phases` maps in-flight
    candidate ids to their `status.current` phase."""
    phases = phases or {}
    if minutes_since(origin, origin) is None:
        origin = min((c.created_at for c in candidates), default=None)

    timed: list[tuple[float, float, bool, Candidate]] = []
    for cand in candidates:
        start = minutes_since(cand.created_at, origin)
        if start is None:  # unparseable stamp: nothing to place
            continue
        running = False
        end = minutes_since(cand.finished_at, origin)
        if end is None:
            live_edge = minutes_since(now, origin)
            if live_edge is not None:
                end, running = live_edge, True
            else:  # crash leftover in a finished search: a degenerate bar
                end = start
        timed.append((start, max(end, start), running, cand))

    now_min = minutes_since(now, origin)
    extent = max(
        [end for _, end, _, _ in timed] + ([now_min] if now_min is not None else []),
        default=0.0,
    )
    extent = max(extent, 1.0)  # a young search still gets a real axis

    timed.sort(key=lambda item: (item[0], item[3].candidate_id))
    lane_free_at: list[float] = []
    spans: list[GanttSpan] = []
    for start, end, running, cand in timed:
        lane = next(
            (i for i, free_at in enumerate(lane_free_at) if free_at <= start + _LANE_EPS),
            len(lane_free_at),
        )
        if lane == len(lane_free_at):
            lane_free_at.append(end)
        else:
            lane_free_at[lane] = end
        exec_min = minutes_since(cand.trials[0].started_at, origin) if cand.trials else None
        score_min = None
        if cand.val_score is not None and cand.trials:
            score_min = minutes_since(cand.trials[-1].finished_at, origin)
        spans.append(GanttSpan(
            candidate_id=cand.candidate_id,
            lane=lane,
            operator=cand.operator,
            status=cand.status,
            pruned=cand.pruned,
            start_min=start,
            end_min=end,
            running=running,
            exec_min=_clamp(exec_min, start, end),
            score_min=_clamp(score_min, start, end),
            score=cand.val_score,
            phase=phases.get(cand.candidate_id),
        ))
    return GanttLayout(
        spans=tuple(spans),
        n_lanes=len(lane_free_at),
        extent_min=extent,
        live=now is not None,
    )


def minute_to_col(minute: float, extent_min: float, width: int) -> int:
    """Linear [0, extent_min] → [0, width-1], clamped at both ends."""
    if width <= 1 or extent_min <= 0:
        return 0
    return min(max(round(minute / extent_min * (width - 1)), 0), width - 1)
