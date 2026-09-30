"""BreakdownAnalyzer + build_report in eval_runner.py.

eval_runner is invoked by path in production and its analyzer duck-types the
emflow interface, so these tests load it by path and drive it with fake
SettlementRecord/Result stand-ins — no emflow install required.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

RUNNER_PATH = (
    Path(__file__).parents[1] / "src" / "hillclimb" / "providers" / "emflow" / "eval_runner.py"
)
spec = importlib.util.spec_from_file_location("eval_runner", RUNNER_PATH)
eval_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_runner)


class AbsErrorMetric:
    """Point |error| per timestamp — deterministic elementwise scores."""

    aggregate = "mean"

    def elementwise(self, y_true, y_pred):
        pred = y_pred[0.5] if 0.5 in y_pred.columns else y_pred.iloc[:, 0]
        pred = pred.reindex(y_true.index)
        return np.abs(y_true.to_numpy(float) - pred.to_numpy(float))


class EnvSettledMetric:
    aggregate = "sum"
    settled_by_env = True

    def elementwise(self, y_true, y_pred):
        raise NotImplementedError


def make_record(asof: str, zone: str, score: float, hours: int = 4):
    """Quantile prediction with known errors: q0.1 = y-1, q0.5 = y+1, q0.9 = y+1
    -> per-point pinball 0.1 / 0.5 / 0.1; point |error| = 1 at every timestamp."""
    asof_ts = pd.Timestamp(asof)
    index = pd.date_range(asof_ts + pd.Timedelta("1h"), periods=hours, freq="h")
    actuals = pd.Series(np.linspace(10.0, 13.0, hours), index=index)
    prediction = pd.DataFrame(
        {0.1: actuals - 1.0, 0.5: actuals + 1.0, 0.9: actuals + 1.0}, index=index
    )
    origin = SimpleNamespace(asof=asof_ts, target_index=index, column=zone)
    return SimpleNamespace(origin=origin, prediction=prediction, actuals=actuals, score=score)


def make_analyzer(metric=None):
    analyzer = eval_runner.BreakdownAnalyzer()
    analyzer.setup(
        feed=None,
        target_field="t",
        target_column=None,
        objective=SimpleNamespace(metric=metric or AbsErrorMetric()),
    )
    return analyzer


def make_result(analyzer_records, *, lower=True, aggregate="mean", analysis=None, score=2.0):
    return SimpleNamespace(
        split="validation",
        objective="mae",
        lower_is_better=lower,  # emflow's own field
        score=score,
        n_origins=len(analyzer_records),
        n_scored=sum(int(r.actuals.notna().sum()) for r in analyzer_records),
        aggregate=aggregate,
        analysis=analysis or {},
    )


def build(records, metric=None, **result_kwargs):
    analyzer = make_analyzer(metric)
    for record in records:
        analyzer.on_settlement(record)
    return eval_runner.build_report(make_result(records, **result_kwargs), analyzer), analyzer


class TestBreakdown:
    def test_zones_worst_first_with_shares(self):
        records = [
            make_record("2024-06-01", "a", score=2.0),
            make_record("2024-06-02", "a", score=2.0),
            make_record("2024-06-01", "b", score=4.0),
        ]
        report, analyzer = build(records)
        assert analyzer.error is None
        zones = report["zones"]
        assert [z["zone"] for z in zones] == ["b", "a"]
        assert zones[0]["score"] == 4.0 and zones[1]["score"] == 2.0
        assert zones[0]["n_origins"] == 1 and zones[1]["n_origins"] == 2
        # weighted shares: b = 4*4 / (4*4 + 2*8) = 0.5
        assert zones[0]["share"] == 0.5

    def test_horizons_one_bucket_per_lead(self):
        report, _ = build([make_record("2024-06-01", "a", score=1.0)])
        buckets = report["horizons"]
        assert [b["bucket"] for b in buckets] == ["1h", "2h", "3h", "4h"]
        assert all(b["score"] == 1.0 for b in buckets)  # |error| = 1 everywhere

    def test_horizons_grouped_when_many_leads(self):
        report, _ = build([make_record("2024-06-01", "a", score=1.0, hours=24)])
        buckets = report["horizons"]
        assert len(buckets) == 6
        assert buckets[0]["bucket"] == "1-4h"
        assert buckets[-1]["bucket"] == "21-24h"

    def test_quantile_pinball_and_coverage(self):
        analysis = {"QuantileCalibration": {"coverage_q10": 0.0, "coverage_q50": 1.0, "coverage_q90": 1.0}}
        report, _ = build([make_record("2024-06-01", "a", score=1.0)], analysis=analysis)
        by_q = {entry["q"]: entry for entry in report["quantiles"]}
        assert by_q[0.1]["pinball"] == 0.1
        assert by_q[0.5]["pinball"] == 0.5
        assert by_q[0.9]["pinball"] == 0.1
        assert by_q[0.1]["coverage"] == 0.0
        assert by_q[0.9]["coverage"] == 1.0

    def test_worst_origins_and_bias(self):
        records = [
            make_record("2024-06-01", "a", score=1.0),
            make_record("2024-06-02", "b", score=9.0),
        ]
        report, _ = build(records)
        assert report["worst_origins"][0]["zone"] == "b"
        assert report["worst_origins"][0]["score"] == 9.0
        # median column is y+1 everywhere -> mean_error +1
        assert report["residual_bias"]["mean_error"] == 1.0
        assert report["residual_bias"]["mean_abs_error"] == 1.0

    def test_higher_is_better_orders_reversed(self):
        records = [
            make_record("2024-06-01", "good", score=0.9),
            make_record("2024-06-01", "bad", score=0.2),
        ]
        report, _ = build(records, lower=False)
        assert report["zones"][0]["zone"] == "bad"  # worst first
        assert "share" not in report["zones"][0]  # loss share only for losses
        assert report["worst_origins"][0]["zone"] == "bad"

    def test_env_settled_metric_keeps_zones_only(self):
        report, analyzer = build(
            [make_record("2024-06-01", "a", score=5.0)],
            metric=EnvSettledMetric(),
            aggregate="sum",
        )
        assert analyzer.error is None
        assert report["zones"][0]["score"] == 5.0  # plain sum for sum-aggregate
        assert "horizons" not in report and "quantiles" not in report
        assert "residual_bias" not in report

    def test_analyzer_failure_is_contained(self):
        analyzer = make_analyzer()
        bad = make_record("2024-06-01", "a", score=1.0)
        bad.prediction = "not a frame"
        analyzer.on_settlement(bad)
        assert analyzer.error is not None
        # further records are skipped, no raise
        analyzer.on_settlement(make_record("2024-06-02", "b", score=1.0))
        report = eval_runner.build_report(make_result([]), analyzer)
        assert report["report_error"] == analyzer.error
        assert report["overall"]["score"] == 2.0

    def test_nan_score_is_json_safe(self):
        report, _ = build([make_record("2024-06-01", "a", score=1.0)], score=float("nan"))
        assert report["overall"]["score"] is None

    def test_persistence_copied(self):
        analysis = {"PersistenceSkill": {"persistence_score": 2.0, "model_score": 1.0,
                                         "beats_persistence": True, "skill": 0.5}}
        report, _ = build([make_record("2024-06-01", "a", score=1.0)], analysis=analysis)
        assert report["persistence"]["skill"] == 0.5
        assert report["persistence"]["beats_persistence"] is True

    def test_day_grained_metric_skips_horizons(self):
        class DayMetric(AbsErrorMetric):
            def elementwise(self, y_true, y_pred):
                return np.array([1.0])  # one value per day, not per timestamp

        report, analyzer = build([make_record("2024-06-01", "a", score=1.0)], metric=DayMetric())
        assert analyzer.error is None
        assert "horizons" not in report

    def test_round_helper(self):
        assert eval_runner._round(1.234567) == 1.2346
        assert eval_runner._round(float("nan")) is None
        assert eval_runner._round(float("inf")) is None
        assert eval_runner._round("x") is None
        assert not math.isnan(eval_runner._round(0.0))


class TestInstances:
    def test_one_instance_per_scored_origin(self):
        records = [
            make_record("2010-10-01 00:30", "z1", score=2.0),
            make_record("2010-10-01 00:30", "z2", score=4.0),
            make_record("2010-11-01 00:30", "z1", score=3.0),
        ]
        _, analyzer = build(records)
        instances = eval_runner.build_instances(analyzer)
        assert instances == {
            "2010-10-01T00:30/z1": 2.0,
            "2010-10-01T00:30/z2": 4.0,
            "2010-11-01T00:30/z1": 3.0,
        }

    def test_unscored_origin_is_absent_not_nan(self):
        records = [
            make_record("2010-10-01 00:30", "z1", score=float("nan")),
            make_record("2010-11-01 00:30", "z1", score=1.23456789),
        ]
        _, analyzer = build(records)
        assert eval_runner.build_instances(analyzer) == {"2010-11-01T00:30/z1": 1.2346}  # 5 sig figs

    def test_repeated_origin_gets_a_suffix(self):
        records = [make_record("2010-10-01 00:30", "z1", score=s) for s in (1.0, 2.0, 3.0)]
        _, analyzer = build(records)
        assert list(eval_runner.build_instances(analyzer)) == [
            "2010-10-01T00:30/z1", "2010-10-01T00:30/z1#2", "2010-10-01T00:30/z1#3",
        ]
