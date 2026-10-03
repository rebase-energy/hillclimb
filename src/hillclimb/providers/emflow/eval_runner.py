"""Evaluate a submission module on one split of an emflow problem.

Runs inside the emflow runtime venv (stdlib + emflow only) — invoked BY PATH
by the orchestrator, never imported; hillclimb itself is not installed there.

usage: eval_runner.py SOLUTION_PY --problem NAME [--split validation|holdout]
                      [--result-json eval_result.json] [--no-report]
                      [--verify] [--name NAME] [--metadata-json '{"n_trials": 12}']

Prints `val_score: <float>` as the final stdout line (the orchestrator's
score contract) and writes a result JSON next to the cwd. --verify runs the
official emflow Verifier instead (scorecard + leaderboard row).

Validation runs also embed a `report` block in the result JSON: per-zone /
per-horizon / per-quantile breakdowns accumulated during the eval, the
feedback operators receive in improve prompts. Holdout and verify runs never
get one (their numbers must not reach coding agent prompts), and a report bug can
never fail a scored eval — assembly is best-effort by construction.

Validation runs also emit the reserved `instances` key: one entry per scored
origin (`<asof>/<zone>`, e.g. GEFCom2014's task x zone), the origin's own
score in the metric's direction. These are the per-instance scores engines
with per-instance selection (GEPA's Pareto frontier) compare candidates on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORT_VERSION = 1
MAX_ZONES = 24
MAX_HORIZON_BUCKETS = 8
HORIZON_GROUPS = 6
WORST_ORIGINS_K = 5
# thinning grid for dense quantile sets (e.g. GEFCom's 99 levels)
CANONICAL_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _round(value) -> float | None:
    """5 significant digits; None for NaN/inf/unconvertible (strict-JSON safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(f"{value:.5g}")


def _quantile_columns(prediction) -> list[float]:
    return sorted(c for c in prediction.columns if isinstance(c, float) and 0.0 < c < 1.0)


def _point_column(prediction):
    """The point (or median) column of a canonical prediction frame; None if
    no unambiguous one exists. Mirrors emflow's metrics._point without raising."""
    if "point" in prediction.columns:
        return prediction["point"]
    if 0.5 in prediction.columns:
        return prediction[0.5]
    if prediction.shape[1] == 1:
        return prediction.iloc[:, 0]
    return None


class BreakdownAnalyzer:
    """Duck-typed emflow analyzer (setup / on_settlement / finalize) that
    accumulates the trial-report breakdown while the eval runs — the Result
    alone retains neither actuals nor zone identity, so this is the only
    place the decomposition can happen.

    finalize() returns {} on purpose: nothing lands in Result.analysis (which
    feeds stats()/leaderboard rows); build_report reads the accumulated state
    directly. Any exception disables further accumulation and is surfaced as
    report_error — partial reports still ship, evals never fail.
    """

    name = "BreakdownAnalyzer"

    def __init__(self):
        self.error: str | None = None
        self._elementwise_ok = True
        # zone -> [score*n_scored sum, n_scored, n_origins, raw score sum]
        self._zones: dict[str, list] = {}
        self._leads: dict[float, list] = {}  # lead_hours -> [loss_sum, n]
        self._pinball: dict[float, list] = {}  # q -> [loss_sum, n]
        self._bias_sum = 0.0  # sum(point_pred - actual)
        self._bias_abs = 0.0
        self._actual_sum = 0.0
        self._bias_n = 0
        self._origins: list[tuple] = []  # (score, asof_iso, zone, n_scored)

    def setup(self, feed, target_field, target_column, objective) -> None:
        self.objective = objective
        self.target_column = target_column
        # env-settled objectives (trading revenue) cannot decompose per
        # timestamp from (actuals, prediction) alone — same opt-out
        # PersistenceSkill uses; zone and worst-origin sections still work
        self._elementwise_ok = not getattr(objective.metric, "settled_by_env", False)

    def on_settlement(self, record) -> None:
        if self.error is not None:
            return
        try:
            self._accumulate(record)
        except Exception as exc:  # noqa: BLE001 — a report bug must never fail the eval
            self.error = repr(exc)

    def finalize(self) -> dict:
        return {}

    def _accumulate(self, record) -> None:
        import numpy as np
        import pandas as pd

        actuals = record.actuals
        n_scored = int(actuals.notna().sum())
        zone = str(record.origin.column or self.target_column or "all")
        slot = self._zones.setdefault(zone, [0.0, 0, 0, 0.0])
        slot[2] += 1
        score = record.score
        if n_scored and score == score:
            slot[0] += float(score) * n_scored
            slot[1] += n_scored
            slot[3] += float(score)
            self._origins.append((float(score), str(record.origin.asof), zone, n_scored))
        if not self._elementwise_ok or not n_scored:
            return

        losses = np.asarray(
            self.objective.metric.elementwise(actuals, record.prediction), dtype=float
        )
        leads = (record.origin.target_index - record.origin.asof) / pd.Timedelta(hours=1)
        if len(losses) == len(leads):  # day-grained metrics return fewer rows: skip
            for lead, loss in zip(leads, losses):
                if loss == loss:
                    lead_slot = self._leads.setdefault(round(float(lead), 3), [0.0, 0])
                    lead_slot[0] += float(loss)
                    lead_slot[1] += 1

        y = actuals.to_numpy(float)
        for q in _quantile_columns(record.prediction):
            pred = record.prediction[q].reindex(actuals.index).to_numpy(float)
            err = y - pred
            ok = ~np.isnan(err)
            if ok.any():
                e = err[ok]
                loss = float(np.where(e >= 0, q * e, (q - 1.0) * e).sum())
                q_slot = self._pinball.setdefault(float(q), [0.0, 0])
                q_slot[0] += loss
                q_slot[1] += int(ok.sum())

        point = _point_column(record.prediction)
        if point is not None:
            pred = point.reindex(actuals.index).to_numpy(float)
            ok = ~(np.isnan(pred) | np.isnan(y))
            if ok.any():
                diff = pred[ok] - y[ok]
                self._bias_sum += float(diff.sum())
                self._bias_abs += float(np.abs(diff).sum())
                self._actual_sum += float(y[ok].sum())
                self._bias_n += int(ok.sum())


def _zone_entries(breakdown: BreakdownAnalyzer, aggregate: str, lower: bool) -> list[dict]:
    total = sum(slot[0] for slot in breakdown._zones.values())
    entries = []
    for zone, (weighted, n_scored, n_origins, raw_sum) in breakdown._zones.items():
        if not n_scored:
            continue
        # zone score mirrors Result.score pooling: scored-count-weighted mean
        # for error metrics, plain sum for revenue-type ones
        score = raw_sum if aggregate == "sum" else weighted / n_scored
        entry = {
            "zone": zone,
            "score": _round(score),
            "n_origins": n_origins,
            "n_scored": n_scored,
        }
        if total > 0 and lower:  # loss share only means something for losses
            entry["share"] = _round(weighted / total)
        entries.append(entry)
    entries.sort(key=lambda e: e["score"] if e["score"] is not None else 0, reverse=lower)
    if len(entries) > MAX_ZONES:
        entries = entries[:MAX_ZONES] + [{"zone": "(truncated)", "n_more": len(entries) - MAX_ZONES}]
    return entries


def _lead_label(lead: float) -> str:
    return f"{lead:g}h"


def _horizon_entries(breakdown: BreakdownAnalyzer) -> list[dict]:
    """Bucket per-lead losses. Distinct leads come from the problem's origin
    schedule, so buckets align exactly across candidates of the same search
    (parent/child deltas are a straight label match)."""
    leads = sorted(breakdown._leads)
    if not leads:
        return []
    if len(leads) <= MAX_HORIZON_BUCKETS:
        groups = [[lead] for lead in leads]
    else:
        size, extra = divmod(len(leads), HORIZON_GROUPS)
        groups, start = [], 0
        for i in range(HORIZON_GROUPS):
            end = start + size + (1 if i < extra else 0)
            groups.append(leads[start:end])
            start = end
    entries = []
    for group in groups:
        loss_sum = sum(breakdown._leads[lead][0] for lead in group)
        n = sum(breakdown._leads[lead][1] for lead in group)
        if not n:
            continue
        label = (
            _lead_label(group[0])
            if len(group) == 1
            else f"{group[0]:g}-{group[-1]:g}h"
        )
        entries.append({"bucket": label, "score": _round(loss_sum / n), "n": n})
    return entries


def _quantile_entries(breakdown: BreakdownAnalyzer, coverage: dict) -> list[dict]:
    levels = sorted(breakdown._pinball)
    if not levels:
        return []
    thinned = levels
    if len(levels) > 13:
        canon = {round(q, 4) for q in CANONICAL_QUANTILES}
        thinned = [q for q in levels if round(q, 4) in canon] or levels[:9]
    entries = []
    total = sum(slot[0] for slot in breakdown._pinball.values())
    for q in thinned:
        loss_sum, n = breakdown._pinball[q]
        if not n:
            continue
        entry = {"q": _round(q), "pinball": _round(loss_sum / n)}
        cov = coverage.get(f"coverage_q{int(round(q * 100)):02d}")
        if cov is not None:
            entry["coverage"] = _round(cov)
        if total > 0:
            entry["share"] = _round(loss_sum / total)
        entries.append(entry)
    if len(levels) > len(thinned):
        entries.append({"n_levels": len(levels), "note": "thinned to canonical levels"})
    return entries


def instance_key(asof, zone: str) -> str:
    """`<asof>/<zone>` — minute-resolution ISO stamp so keys read as the task
    they score and stay identical across every candidate of a search."""
    import pandas as pd

    return f"{pd.Timestamp(asof).strftime('%Y-%m-%dT%H:%M')}/{zone}"


def build_instances(breakdown: BreakdownAnalyzer) -> dict[str, float]:
    """The verifier contract's reserved `instances` value: every origin the
    eval settled with a finite score, keyed by `instance_key`. Origins are
    fixed per problem and split, so the key set is the same for every
    candidate that scores them all; an origin a candidate leaves unscored
    (NaN) is simply absent — engines treat a missing key as a failed
    instance, never as a new one. Repeated (asof, zone) pairs (several target
    windows from one origin) get a `#2`, `#3` suffix in settlement order."""
    instances: dict[str, float] = {}
    for score, asof, zone, _n_scored in breakdown._origins:
        value = _round(score)
        if value is None:
            continue
        key = base = instance_key(asof, zone)
        n = 2
        while key in instances:
            key = f"{base}#{n}"
            n += 1
        instances[key] = value
    return instances


def build_report(result, breakdown: BreakdownAnalyzer) -> dict:
    """Assemble the versioned report block from the finished Result and the
    analyzer's accumulated state. Pure dict math — unit-testable without a run."""
    lower = bool(result.lower_is_better)  # legacy-key: emflow's own field
    report = {
        "version": REPORT_VERSION,
        "split": result.split,
        "objective": result.objective,
        "higher_is_better": not lower,
        "source": "evaluator",
        "overall": {
            "score": _round(result.score),
            "n_origins": result.n_origins,
            "n_scored": result.n_scored,
        },
        "report_error": breakdown.error,
    }
    zones = _zone_entries(breakdown, result.aggregate, lower)
    if zones:
        report["zones"] = zones
    horizons = _horizon_entries(breakdown)
    if horizons:
        report["horizons"] = horizons
    quantiles = _quantile_entries(breakdown, result.analysis.get("QuantileCalibration", {}))
    if quantiles:
        report["quantiles"] = quantiles
    origins = sorted(breakdown._origins, reverse=lower)[:WORST_ORIGINS_K]
    if origins:
        report["worst_origins"] = [
            {"asof": asof, "zone": zone, "score": _round(score), "n_scored": n}
            for score, asof, zone, n in origins
        ]
    if breakdown._bias_n:
        report["residual_bias"] = {
            "mean_error": _round(breakdown._bias_sum / breakdown._bias_n),
            "mean_abs_error": _round(breakdown._bias_abs / breakdown._bias_n),
            "mean_actual": _round(breakdown._actual_sum / breakdown._bias_n),
        }
    persistence = result.analysis.get("PersistenceSkill")
    if persistence:
        report["persistence"] = {
            key: (_round(value) if isinstance(value, float) else value)
            for key, value in persistence.items()
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "holdout"])
    ap.add_argument("--result-json", default="eval_result.json")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the validation breakdown report")
    ap.add_argument("--verify", action="store_true",
                    help="official Verifier run (scorecard + leaderboard row)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--metadata-json", default=None)
    args = ap.parse_args()

    solution = Path(args.solution).absolute()
    # ensemble candidates import their inputs as modules (candidate_1.py, ...)
    sys.path.insert(0, str(solution.parent))

    import emflow as ef

    breakdown = None
    if args.verify:
        from emflow.run.verifier import Verifier

        model = ef.load_submission(solution)
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        verifier = Verifier(problem=args.problem, split=args.split)
        result = verifier.verify(model, name=args.name or solution.stem, metadata=metadata)
    else:
        model = ef.load_submission(solution)
        analyzers = "default"
        # validation only: holdout/verify numbers must never reach coding agent prompts
        if args.split == "validation" and not args.no_report:
            from emflow.run.analyzers import default_analyzers

            breakdown = BreakdownAnalyzer()
            # an explicit analyzer list REPLACES the defaults — concatenate or
            # PersistenceSkill/QuantileCalibration silently vanish
            analyzers = default_analyzers(model) + [breakdown]
        result = ef.evaluate(args.problem, model, split=args.split, analyzers=analyzers)

    payload = {
        "problem": args.problem,
        "split": args.split,
        "objective": result.objective,
        "score": result.score,
        "n_origins": result.n_origins,
        "n_scored": result.n_scored,
        "model": result.model,
    }
    if breakdown is not None:
        try:
            payload["report"] = build_report(result, breakdown)
        except Exception as exc:  # noqa: BLE001 — the report must never fail the eval
            payload["report"] = {
                "version": REPORT_VERSION,
                "split": args.split,
                "source": "evaluator",
                "report_error": repr(exc),
            }
        try:
            instances = build_instances(breakdown)
        except Exception as exc:  # noqa: BLE001 — same rule: advisory, never fatal
            print(f"warning: per-instance scores dropped: {exc!r}", file=sys.stderr)
            instances = {}
        if instances:
            payload["instances"] = instances
    Path(args.result_json).write_text(json.dumps(payload, indent=2))
    if result.score != result.score:  # nan: scoring silently found no actuals
        print(
            "error: evaluation scored nan — no actuals matched the predictions "
            "(missing HF credentials for private holdout data?)",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"val_score: {result.score}")


if __name__ == "__main__":
    main()
