"""Operator timeline: lane packing + time projection (gantt.py) and the Rich
rendering (ganttview.py). Pure — no Textual; the panel-toggle pilot test
lives in test_watch.py."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import random

from hillclimb.candidate import Candidate
from hillclimb.gantt import GanttLayout, GanttSpan, build_gantt, minute_to_col
from hillclimb.ganttview import LANE_LABEL_W, render_gantt

ORIGIN = "2026-08-22T10:00:00+00:00"


def stamp(minute: float) -> str:
    return f"2026-08-22T10:{int(minute):02d}:{int(minute % 1 * 60):02d}+00:00"


def trial(started: float, finished: float, score: float | None = None) -> Trial:
    return mk_trial(val_score=score, started_at=stamp(started), finished_at=stamp(finished))


def cand(
    cid: str, start: float, end: float | None, operator: str = "improve",
    status: str = "ok", trials: list[Trial] | None = None, **kwargs,
) -> Candidate:
    return Candidate(
        candidate_id=cid, operator=operator, status=status,
        trials=trials or [],
        created_at=stamp(start),
        finished_at=None if end is None else stamp(end),
        **kwargs,
    )


class TestLanePacking:
    def test_sequential_candidates_share_lane_zero(self):
        layout = build_gantt(
            [cand("c1", 0, 5), cand("c2", 6, 10)], origin=ORIGIN,
        )
        assert [s.lane for s in layout.spans] == [0, 0]
        assert layout.n_lanes == 1

    def test_back_to_back_reuses_the_lane(self):
        layout = build_gantt(
            [cand("c1", 0, 5), cand("c2", 5, 10)], origin=ORIGIN,
        )
        assert [s.lane for s in layout.spans] == [0, 0]

    def test_overlaps_open_lanes_and_first_fit_reuses_them(self):
        layout = build_gantt(
            [
                cand("c1", 0, 4),
                cand("c2", 1, 8),
                cand("c3", 2, 9),
                cand("c4", 5, 10),  # c1's lane is free again by minute 5
            ],
            origin=ORIGIN,
        )
        lanes = {s.candidate_id: s.lane for s in layout.spans}
        assert lanes == {"c1": 0, "c2": 1, "c3": 2, "c4": 0}
        assert layout.n_lanes == 3

    def test_packing_is_deterministic_under_input_order(self):
        cands = [
            cand("c1", 0, 4), cand("c2", 1, 8), cand("c3", 2, 9), cand("c4", 5, 10),
        ]
        reference = build_gantt(cands, origin=ORIGIN)
        shuffled = list(cands)
        random.Random(7).shuffle(shuffled)
        assert build_gantt(shuffled, origin=ORIGIN) == reference

    def test_simultaneous_starts_break_ties_by_candidate_id(self):
        layout = build_gantt(
            [cand("c2", 0, 5), cand("c1", 0, 5)], origin=ORIGIN,
        )
        lanes = {s.candidate_id: s.lane for s in layout.spans}
        assert lanes == {"c1": 0, "c2": 1}


class TestTimes:
    def test_running_span_extends_to_now_and_blocks_its_lane(self):
        layout = build_gantt(
            [cand("c1", 0, None, status="pending"), cand("c2", 3, 5)],
            origin=ORIGIN, now=stamp(10),
        )
        running = layout.spans[0]
        assert running.running and running.end_min == 10.0
        assert layout.spans[1].lane == 1  # lane 0 stays blocked to "now"
        assert layout.extent_min == 10.0
        assert layout.live

    def test_finished_search_pending_leftover_is_a_degenerate_bar(self):
        layout = build_gantt(
            [cand("c1", 2, None, status="pending")], origin=ORIGIN,
        )
        span = layout.spans[0]
        assert not span.running and span.start_min == span.end_min == 2.0
        assert not layout.live

    def test_origin_falls_back_to_earliest_created_at(self):
        layout = build_gantt(
            [cand("c1", 5, 8), cand("c2", 3, 6)], origin=None,
        )
        starts = {s.candidate_id: s.start_min for s in layout.spans}
        assert starts == {"c1": 2.0, "c2": 0.0}

    def test_exec_and_score_minutes_come_from_the_trials(self):
        layout = build_gantt(
            [cand("c1", 0, 6, trials=[trial(4, 6, score=0.5)])], origin=ORIGIN,
        )
        span = layout.spans[0]
        assert span.exec_min == 4.0
        assert span.score_min == 6.0
        assert span.score == 0.5

    def test_no_score_marker_without_a_score(self):
        layout = build_gantt(
            [cand("c1", 0, 6, status="buggy", trials=[trial(4, 6)])], origin=ORIGIN,
        )
        assert layout.spans[0].score_min is None

    def test_phases_ride_along(self):
        layout = build_gantt(
            [cand("c1", 0, None, status="pending")],
            origin=ORIGIN, now=stamp(2), phases={"c1": "waiting-slot"},
        )
        assert layout.spans[0].phase == "waiting-slot"

    def test_extent_floor_keeps_a_young_search_drawable(self):
        layout = build_gantt(
            [cand("c1", 0, None, status="pending")], origin=ORIGIN, now=stamp(0.1),
        )
        assert layout.extent_min == 1.0

    def test_empty_journal(self):
        layout = build_gantt([], origin=None)
        assert layout == GanttLayout(spans=(), n_lanes=0, extent_min=1.0, live=False)


class TestProjection:
    def test_endpoints_and_clamping(self):
        assert minute_to_col(0.0, 10.0, 60) == 0
        assert minute_to_col(10.0, 10.0, 60) == 59
        assert minute_to_col(5.0, 10.0, 61) == 30
        assert minute_to_col(-1.0, 10.0, 60) == 0
        assert minute_to_col(99.0, 10.0, 60) == 59
        assert minute_to_col(5.0, 10.0, 1) == 0
        assert minute_to_col(5.0, 0.0, 60) == 0


WIDTH = 80
TRACK = WIDTH - LANE_LABEL_W


def lines(text) -> list[str]:
    return text.plain.split("\n")


class TestRender:
    def test_shape_lanes_axis_legend(self):
        layout = build_gantt(
            [cand("c1", 0, 5), cand("c2", 1, 8)], origin=ORIGIN,
        )
        rows = lines(render_gantt(layout, WIDTH, 10))
        assert len(rows) == layout.n_lanes + 2  # lanes + axis + legend
        assert all(len(row) <= WIDTH for row in rows)
        assert rows[0].startswith("a1 ") and rows[1].startswith("a2 ")
        assert "draft" in rows[-1] and "scored" in rows[-1]

    def test_score_marker_lands_on_its_column(self):
        layout = build_gantt(
            [cand("c1", 0, 10, trials=[trial(6, 10, score=0.5)])], origin=ORIGIN,
        )
        row = lines(render_gantt(layout, WIDTH, 10))[0]
        assert row[LANE_LABEL_W + minute_to_col(10.0, layout.extent_min, TRACK)] == "◆"

    def test_running_bar_reaches_now_with_an_arrow_head(self):
        layout = build_gantt(
            [cand("c1", 0, None, status="pending")], origin=ORIGIN, now=stamp(10),
        )
        row = lines(render_gantt(layout, WIDTH, 10))[0]
        assert row[LANE_LABEL_W + TRACK - 1] == "▶"

    def test_waiting_slot_reads_hollow(self):
        layout = build_gantt(
            [cand("c1", 0, None, status="pending")],
            origin=ORIGIN, now=stamp(10), phases={"c1": "waiting-slot"},
        )
        row = lines(render_gantt(layout, WIDTH, 10))[0]
        assert "░" in row

    def test_seed_operator_renders_without_a_palette_entry(self):
        layout = build_gantt(
            [cand("c1", 0, 5, operator="seed")], origin=ORIGIN,
        )
        assert "█" in lines(render_gantt(layout, WIDTH, 10))[0]

    def test_lane_truncation_hint(self):
        layout = build_gantt(
            [cand(f"c{i}", 0, 10) for i in range(6)], origin=ORIGIN,
        )
        rows = lines(render_gantt(layout, WIDTH, 6))  # room for 4 lane rows
        assert any(row.startswith("… +3 lanes") for row in rows)

    def test_now_label_only_when_live(self):
        cands = [cand("c1", 0, 10)]
        live = render_gantt(build_gantt(cands, origin=ORIGIN, now=stamp(10)), WIDTH, 10)
        done = render_gantt(build_gantt(cands, origin=ORIGIN), WIDTH, 10)
        assert lines(live)[-2].endswith("now")
        assert not lines(done)[-2].endswith("now")

    def test_axis_carries_tick_labels(self):
        layout = build_gantt([cand("c1", 0, 10)], origin=ORIGIN)
        axis = lines(render_gantt(layout, WIDTH, 10))[-2]
        assert "2m" in axis  # 10-minute extent on 77 columns ticks every 2m

    def test_legend_is_truncated_to_a_narrow_panel(self):
        layout = build_gantt([cand("c1", 0, 10)], origin=ORIGIN)
        rows = lines(render_gantt(layout, 40, 10))
        assert all(len(row) <= 40 for row in rows)

    def test_empty_layout_renders_a_hint(self):
        text = render_gantt(build_gantt([], origin=None), WIDTH, 10)
        assert text.plain == "no candidates yet"

    def test_two_phase_bar_dims_the_agent_stretch(self):
        layout = build_gantt(
            [cand("c1", 0, 10, operator="draft", trials=[trial(5, 10, score=0.1)])],
            origin=ORIGIN,
        )
        text = render_gantt(layout, WIDTH, 10)
        colors = {
            str(span.style.color) if hasattr(span.style, "color") else str(span.style)
            for span in text.spans
        }
        # both the dimmed and the full draft blue appear
        assert any("rgb(41,77,140)" in c for c in colors)  # 76,141,255 × 0.55
        assert any("rgb(76,141,255)" in c for c in colors)
