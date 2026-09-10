"""report.py: compaction and rendering of trial evaluation reports."""

from __future__ import annotations

from tests.factories import trial as mk_trial

import json

from hillclimb.candidate import Candidate
from hillclimb.report import (
    candidate_report,
    compact_report,
    render_delta,
    render_report,
)


def make_report(**overrides) -> dict:
    report = {
        "version": 1,
        "split": "validation",
        "objective": "PinballLoss",
        "higher_is_better": False,
        "overall": {"score": 0.02, "n_origins": 100, "n_scored": 2400},
        "zones": [
            {"zone": "z3", "score": 0.03, "n_origins": 50, "n_scored": 1200, "share": 0.6},
            {"zone": "z1", "score": 0.01, "n_origins": 50, "n_scored": 1200, "share": 0.4},
        ],
        "horizons": [
            {"bucket": "1-12h", "score": 0.015, "n": 1200},
            {"bucket": "13-24h", "score": 0.025, "n": 1200},
        ],
        "quantiles": [
            {"q": 0.1, "pinball": 0.005, "coverage": 0.15, "share": 0.2},
            {"q": 0.9, "pinball": 0.01, "coverage": 0.95, "share": 0.5},
        ],
        "worst_origins": [
            {"asof": "2024-06-01T00:00:00", "zone": "z3", "score": 0.09, "n_scored": 24}
        ],
        "residual_bias": {"mean_error": -0.5, "mean_abs_error": 2.0, "mean_actual": 40.0},
        "persistence": {
            "persistence_score": 0.04, "model_score": 0.02,
            "beats_persistence": True, "skill": 0.5,
        },
        "report_error": None,
    }
    report.update(overrides)
    return report


class TestRenderReport:
    def test_full_report_sections(self):
        text = render_report(make_report(), "PinballLoss")
        assert "Attack the largest contributors" in text
        assert "0.02" in text and "100 cases" in text
        # zones worst-first, share rendered as percent
        assert text.index("z3") < text.index("z1")
        assert "60%" in text
        assert "1-12h" in text and "13-24h" in text
        assert "q90 covers 95% (target 90%)" in text
        assert "q90 carries 50% of pinball loss" in text
        assert "2024-06-01" in text
        assert "under-forecasting" in text  # mean_error < 0
        assert "beats last-value persistence" in text
        assert len(text.splitlines()) <= 40

    def test_unusable_reports_render_empty(self):
        assert render_report(None, "m") == ""
        assert render_report({}, "m") == ""
        assert render_report(make_report(version=2), "m") == ""
        assert render_report(make_report(split="holdout"), "m") == ""  # never render holdout
        assert render_report({"version": 1, "report_error": "boom"}, "m") == ""

    def test_partial_report_shows_error_note(self):
        report = make_report(zones=[], horizons=[], quantiles=[], report_error="ValueError('x')")
        text = render_report(report, "m")
        assert "partially unavailable" in text
        assert "ValueError" in text

    def test_positive_bias_reads_over_forecasting(self):
        text = render_report(
            make_report(residual_bias={"mean_error": 0.7, "mean_abs_error": 1.0, "mean_actual": 5.0}),
            "m",
        )
        assert "over-forecasting" in text

    def test_agent_source_labelled_self_reported(self):
        assert "Self-reported" in render_report(make_report(source="agent"), "m")
        assert "Self-reported" not in render_report(make_report(source="evaluator"), "m")
        assert "Self-reported" not in render_report(make_report(), "m")

    def test_segment_label_used_in_table(self):
        text = render_report(make_report(segment_label="store"), "m")
        assert "Per store" in text
        assert "Per zone" not in text


class TestRenderDelta:
    def test_delta_signs_lower_is_better(self):
        parent = make_report()
        child = make_report(
            overall={"score": 0.015, "n_origins": 100, "n_scored": 2400},
            zones=[
                {"zone": "z3", "score": 0.02, "n_origins": 50, "n_scored": 1200},
                {"zone": "z1", "score": 0.012, "n_origins": 50, "n_scored": 1200},
            ],
        )
        text = render_delta(parent, child, higher_is_better=False)
        assert "improved" in text.splitlines()[0]
        assert "improved z3" in text
        assert "regressed z1" in text

    def test_delta_signs_higher_is_better(self):
        parent = make_report(
            higher_is_better=True,
            overall={"score": 0.8, "n_origins": 10, "n_scored": 100},
            zones=[{"zone": "a", "score": 0.7, "n_origins": 5, "n_scored": 50}],
        )
        child = make_report(
            higher_is_better=True,
            overall={"score": 0.7, "n_origins": 10, "n_scored": 100},
            zones=[{"zone": "a", "score": 0.9, "n_origins": 5, "n_scored": 50}],
        )
        text = render_delta(parent, child, higher_is_better=True)
        assert "regressed" in text.splitlines()[0]  # overall dropped
        assert "improved a" in text  # zone rose

    def test_coverage_shifts_reported(self):
        parent = make_report()
        child = make_report(
            quantiles=[{"q": 0.9, "pinball": 0.01, "coverage": 0.85, "share": 0.5}]
        )
        text = render_delta(parent, child, higher_is_better=False)
        assert "q90 -10%" in text

    def test_missing_side_renders_empty(self):
        assert render_delta(None, make_report(), True) == ""
        assert render_delta(make_report(), None, True) == ""
        assert render_delta(make_report(split="holdout"), make_report(), True) == ""


class TestCompact:
    def test_caps_and_size(self):
        zones = [
            {"zone": f"z{i}", "score": 1.0 - i / 100, "n_origins": 5, "n_scored": 50}
            for i in range(14)
        ]
        quantiles = [{"q": i / 100, "pinball": 0.01} for i in range(1, 20)]
        compact = compact_report(make_report(zones=zones, quantiles=quantiles))
        assert len(compact["zones"]) == 9  # worst 8 + best 1
        assert compact["zones"][0]["zone"] == "z0"
        assert compact["zones"][-1]["zone"] == "z13"  # best kept as contrast
        assert len(compact["quantiles"]) == 9
        assert len(json.dumps(compact)) < 2500

    def test_error_and_absent_sections(self):
        compact = compact_report({"version": 1, "split": "validation", "report_error": "x" * 500})
        assert len(compact["report_error"]) == 200
        assert "zones" not in compact and "horizons" not in compact

    def test_source_and_segment_label_survive_compaction(self):
        compact = compact_report(make_report(source="agent", segment_label="store"))
        assert compact["source"] == "agent"
        assert compact["segment_label"] == "store"


class TestCandidateReport:
    def test_best_trials_first_replicate_with_report_wins(self):
        from hillclimb.candidate import Replicate

        report = {"version": 1, "overall": {"score": 1.0}}
        cand = Candidate(
            candidate_id="c1",
            operator="draft",
            trials=[
                mk_trial(val_score=0.2, report={"version": 1}, is_best=False),
                mk_trial(index=1, is_best=True, replicates=[
                    Replicate(val_score=0.5),
                    Replicate(val_score=0.5, report=report),
                    Replicate(val_score=0.5, report={"version": 2}),
                ]),
            ],
        )
        assert candidate_report(cand) == report
        assert candidate_report(Candidate(candidate_id="c2", operator="draft")) is None
        assert candidate_report(None) is None
