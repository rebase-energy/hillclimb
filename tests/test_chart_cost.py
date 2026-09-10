"""The chart's cumulative-cost fold and its y2/y3 overlay.

Pure-function tests: no Textual, no rendering — the plot side goes through a
PlotSpy the way test_tree.py drives build_climb_plot.
"""

from tests.factories import trial as mk_trial
from hillclimb.candidate import BackendInfo, Candidate
from hillclimb.chart import (
    Climb,
    ClimbEvent,
    COST_CPU_LABEL,
    COST_TOKENS_LABEL,
    build_climb_plot,
    climb_from_searches,
    climb_legend,
    cost_series,
)


def cand(
    cid: str,
    *,
    score: float | None = None,
    tokens: int | None = None,
    cpu: float | None = None,
    duration: float | None = None,
    holdout_cpu: float | None = None,
    t: int = 0,
    status: str = "ok",
    pruned: bool = False,
) -> Candidate:
    trials = []
    if score is not None or cpu is not None or duration is not None:
        trials = [mk_trial(
            val_score=score, cpu_s=cpu, duration_s=duration, holdout_cpu_s=holdout_cpu,
        )]
    return Candidate(
        candidate_id=cid, operator="draft", status=status, pruned=pruned,
        backend=BackendInfo(total_tokens=tokens),
        trials=trials,
        created_at=f"2026-08-22T10:{t:02d}:00+00:00",
        finished_at=f"2026-08-22T10:{t + 1:02d}:00+00:00",
    )


class TestCostSeries:
    def test_failures_attach_to_the_next_scored_slot(self):
        candidates = [
            cand("c001", score=0.5, tokens=100, cpu=60.0, t=0),
            cand("c002", tokens=50, cpu=30.0, t=1, status="buggy"),  # never scored
            cand("c003", score=0.7, tokens=100, cpu=30.0, t=2),
        ]
        series = cost_series([("s", candidates, None)])
        # two scored slots; the failure's cost surfaces at slot 2
        assert series.xs == [1.0, 2.0]
        assert series.tokens == [100.0, 250.0]
        assert series.cpu_min == [1.0, 2.0]
        assert series.total_tokens == 250.0
        assert series.total_cpu_min == 2.0

    def test_trailing_failure_lands_at_extent(self):
        candidates = [
            cand("c001", score=0.5, tokens=100, cpu=60.0, t=0),
            cand("c002", tokens=40, cpu=120.0, t=1, status="buggy"),
        ]
        series = cost_series([("s", candidates, None)])
        # a final point at the same x carries the true totals
        assert series.xs == [1.0, 1.0]
        assert series.tokens == [100.0, 140.0]
        assert series.cpu_min == [1.0, 3.0]

    def test_cpu_falls_back_to_duration_and_holdout_adds(self):
        candidates = [
            cand("c001", score=0.5, cpu=None, duration=120.0, t=0),          # pre-feature journal
            cand("c002", score=0.6, cpu=60.0, duration=90.0, holdout_cpu=60.0, t=1),
        ]
        series = cost_series([("s", candidates, None)])
        # c001: wall-clock proxy (2 min); c002: real cpu + holdout (2 min)
        assert series.cpu_min == [2.0, 4.0]
        # tokens None counts as zero, not a crash
        assert series.tokens == [0.0, 0.0]

    def test_xs_align_with_climb_events_under_pruning(self):
        candidates = [
            cand("c001", score=0.5, tokens=10, t=0),
            cand("c002", score=0.9, tokens=10, t=1, pruned=True),  # pruned: no slot
            cand("c003", tokens=10, t=2, status="buggy"),          # unscored: no slot
            cand("c004", score=0.7, tokens=10, t=3),
        ]
        searches = [("s", candidates, None)]
        climb = climb_from_searches(searches)
        series = cost_series(searches)
        assert series.xs == [event.x for event in climb.events] == [1.0, 2.0]
        # every candidate's tokens are counted, slots or not
        assert series.total_tokens == 40.0

    def test_empty_and_all_failed(self):
        assert cost_series([]).xs == []
        series = cost_series([("s", [cand("c001", tokens=25, t=0, status="buggy")], None)])
        assert series.xs == []  # nothing scored: no slots to sample at
        assert series.total_tokens == 25.0  # but the totals still tell the truth


class PlotSpy:
    def __init__(self):
        self.lines = []
        self.scatters = []

    def add_line(self, xs, ys, **kwargs):
        self.lines.append((xs, ys, kwargs))

    def add_scatter(self, *args, **kwargs):
        self.scatters.append((args, kwargs))


def one_step_climb() -> Climb:
    return Climb(events=[ClimbEvent(1.0, 0.6, True, "r", "draft")], extent=1.0)


def sample_cost():
    return cost_series([(
        "s",
        [
            cand("c001", score=0.5, tokens=100, cpu=60.0, t=0),
            cand("c002", score=0.6, tokens=200, cpu=120.0, t=1),
        ],
        None,
    )])


class TestCostOverlay:
    def test_overlay_draws_tokens_on_y2_and_cpu_on_y3(self, monkeypatch):
        plot = PlotSpy()
        monkeypatch.setattr("hillclimb.chart.themed_plot", lambda: plot)
        build_climb_plot(one_step_climb(), cost=sample_cost())
        by_axis = {kw.get("axis"): (xs, ys, kw) for xs, ys, kw in plot.lines if "axis" in kw}
        assert set(by_axis) == {"y2", "y3"}
        assert by_axis["y2"][2]["name"] == COST_TOKENS_LABEL
        assert by_axis["y2"][1] == [100.0, 300.0]
        assert by_axis["y3"][2]["name"] == COST_CPU_LABEL
        assert by_axis["y3"][1] == [1.0, 3.0]

    def test_no_cost_means_no_overlay(self, monkeypatch):
        plot = PlotSpy()
        monkeypatch.setattr("hillclimb.chart.themed_plot", lambda: plot)
        build_climb_plot(one_step_climb())
        assert not any("axis" in kw for _xs, _ys, kw in plot.lines)

    def test_hidden_series_takes_its_axis_with_it(self, monkeypatch):
        plot = PlotSpy()
        monkeypatch.setattr("hillclimb.chart.themed_plot", lambda: plot)
        build_climb_plot(one_step_climb(), cost=sample_cost(), hidden={COST_TOKENS_LABEL})
        axes = [kw["axis"] for _xs, _ys, kw in plot.lines if "axis" in kw]
        assert axes == ["y3"]  # tokens hidden: no y2 line, so no y2 column

    def test_legend_gains_cost_entries(self):
        entries = climb_legend(one_step_climb(), {}, sample_cost())
        assert [entry[0] for entry in entries][-2:] == [COST_TOKENS_LABEL, COST_CPU_LABEL]
        # without the overlay the legend is unchanged
        assert COST_TOKENS_LABEL not in [e[0] for e in climb_legend(one_step_climb(), {})]
