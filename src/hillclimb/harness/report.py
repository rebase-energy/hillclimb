"""Trial evaluation reports: compaction and rendering.

eval_runner.py (running in the emflow venv) computes the breakdown during the
eval and owns the schema (`version` field); this module treats reports as
plain versioned dicts and turns them into (a) the compact subset stored on
Trial in the journal — the only state synced off remote machines — and
(b) markdown for improve prompts, `hillclimb show`, and the watch TUI.
The same rendering feeds agent and human, so what you read in `show` is
exactly what the operator saw. Every function degrades to ""/None on
missing, foreign-version, or error-only reports.
"""

from __future__ import annotations

from hillclimb.harness.candidate import Candidate
from hillclimb.harness.direction import legacy_direction_key

KNOWN_VERSION = 1
COMPACT_WORST_ZONES = 8  # plus the single best zone as contrast
COMPACT_QUANTILES = 9
COMPACT_ORIGINS = 5
RENDER_ZONES = 6
RENDER_ORIGINS = 3
COVERAGE_SHIFT_MIN = 0.03


def candidate_report(candidate: Candidate | None) -> dict | None:
    """The candidate's validation report: its best trial's first replicate
    carrying one (r0, matching the artifact-hoist convention)."""
    if candidate is None:
        return None
    return candidate.report


def compact_report(report: dict) -> dict:
    """Journal-bound subset (~<=2 KB): the journal dumps the whole candidate
    on every event, so the stored report keeps only what rendering uses."""
    report = legacy_direction_key(report)
    report = legacy_direction_key(report)
    compact = {
        key: report[key]
        for key in (
            "version", "split", "objective", "higher_is_better", "overall",
            "source", "segment_label",
        )
        if key in report
    }
    if report.get("report_error"):
        compact["report_error"] = str(report["report_error"])[:200]
    zones = [z for z in report.get("zones", []) if "zone" in z and "score" in z]
    if zones:  # worst-first from the builder; keep worst 8 + best 1 as contrast
        if len(zones) > COMPACT_WORST_ZONES + 1:
            zones = zones[:COMPACT_WORST_ZONES] + [zones[-1]]
        compact["zones"] = zones
    if report.get("horizons"):
        compact["horizons"] = report["horizons"]
    quantiles = [q for q in report.get("quantiles", []) if "q" in q]
    if quantiles:
        compact["quantiles"] = quantiles[:COMPACT_QUANTILES]
    if report.get("worst_origins"):
        compact["worst_origins"] = report["worst_origins"][:COMPACT_ORIGINS]
    for key in ("residual_bias", "persistence"):
        if report.get(key):
            compact[key] = report[key]
    return compact


def _usable(report: dict | None) -> bool:
    if not report or report.get("version") != KNOWN_VERSION:
        return False
    if report.get("split") not in (None, "validation"):
        return False  # belt-and-braces: never render a non-validation report
    return isinstance(report.get("overall"), dict)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_report(report: dict | None, metric_name: str = "", max_lines: int = 40) -> str:
    """Markdown breakdown, worst contributors first. "" when unusable."""
    if not _usable(report):
        return ""
    metric = report.get("objective") or metric_name or "score"
    overall = report["overall"]
    provenance = (
        " Self-reported by the solution's own validation (not independently verified)."
        if report.get("source") == "agent"
        else ""
    )
    lines = [
        f"Validation {metric}: **{_fmt(overall.get('score'))}** over "
        f"{overall.get('n_origins', '?')} cases ({overall.get('n_scored', '?')} scored "
        f"points).{provenance} Attack the largest contributors below."
    ]
    segment = report.get("segment_label") or "zone"
    zones = [z for z in report.get("zones", []) if "zone" in z and "score" in z]
    if len(zones) > 1:
        lines.append("")
        lines.append(f"Per {segment} ({metric}, worst first):")
        lines.append(f"| {segment} | score | loss share |")
        lines.append("|---|---|---|")
        for z in zones[:RENDER_ZONES]:
            share = f"{z['share']:.0%}" if isinstance(z.get("share"), float) else "-"
            lines.append(f"| {z['zone']} | {_fmt(z['score'])} | {share} |")
        if len(zones) > RENDER_ZONES:
            lines.append(f"| … {len(zones) - RENDER_ZONES} more | | |")
    horizons = report.get("horizons") or []
    if len(horizons) > 1:
        lines.append("")
        lines.append("Per horizon (lead time from forecast origin):")
        lines.append("| lead | score |")
        lines.append("|---|---|")
        for h in horizons:
            lines.append(f"| {h['bucket']} | {_fmt(h.get('score'))} |")
    quantiles = [q for q in report.get("quantiles", []) if "q" in q]
    coverage_gaps = sorted(
        (q for q in quantiles if isinstance(q.get("coverage"), float)),
        key=lambda q: abs(q["coverage"] - q["q"]),
        reverse=True,
    )[:2]
    if coverage_gaps:
        parts = [
            f"q{int(round(q['q'] * 100))} covers {q['coverage']:.0%} (target {q['q']:.0%})"
            for q in coverage_gaps
        ]
        lines.append("")
        lines.append(f"Calibration, largest gaps: {'; '.join(parts)}.")
    heavy = [q for q in quantiles if isinstance(q.get("share"), float)]
    if heavy:
        worst = max(heavy, key=lambda q: q["share"])
        lines.append(
            f"Heaviest quantile: q{int(round(worst['q'] * 100))} carries "
            f"{worst['share']:.0%} of pinball loss."
        )
    origins = report.get("worst_origins") or []
    if origins:
        parts = [
            f"{o.get('asof', '?')[:16]} ({o.get('zone', '?')}: {_fmt(o.get('score'))})"
            for o in origins[:RENDER_ORIGINS]
        ]
        lines.append("")
        lines.append(f"Worst origins: {', '.join(parts)}.")
    bias = report.get("residual_bias")
    if bias and bias.get("mean_error") is not None:
        direction = "over-forecasting" if bias["mean_error"] > 0 else "under-forecasting"
        lines.append(
            f"Residual bias: mean(pred - actual) = {_fmt(bias['mean_error'])} "
            f"({direction}; mean |error| {_fmt(bias.get('mean_abs_error'))}, "
            f"mean actual {_fmt(bias.get('mean_actual'))})."
        )
    persistence = report.get("persistence")
    if persistence and persistence.get("persistence_score") is not None:
        verdict = "beats" if persistence.get("beats_persistence") else "does NOT beat"
        skill = (
            f", skill {_fmt(persistence['skill'])}" if persistence.get("skill") is not None else ""
        )
        lines.append(
            f"Persistence baseline: model {verdict} last-value persistence "
            f"({_fmt(persistence.get('model_score'))} vs "
            f"{_fmt(persistence.get('persistence_score'))}{skill})."
        )
    if report.get("report_error"):
        lines.append(f"(breakdown partially unavailable: {report['report_error']})")
    return "\n".join(lines[:max_lines])


def _keyed(entries: list[dict], key: str) -> dict:
    return {e[key]: e for e in entries if key in e and e.get("score") is not None}


def render_delta(parent: dict | None, child: dict | None, higher_is_better: bool) -> str:
    """Where the child's score moved relative to its parent. "" unless both
    sides carry a usable report."""
    if not (_usable(parent) and _usable(child)):
        return ""
    lines: list[str] = []
    p_score, c_score = parent["overall"].get("score"), child["overall"].get("score")
    if p_score is not None and c_score is not None:
        delta = c_score - p_score
        improved = delta > 0 if higher_is_better else delta < 0
        word = "improved" if improved else ("regressed" if delta != 0 else "unchanged")
        lines.append(f"Overall: {_fmt(p_score)} -> {_fmt(c_score)} ({word}, {delta:+g}).")
    for section, key, label in (("zones", "zone", "zone"), ("horizons", "bucket", "horizon")):
        p_entries = _keyed(parent.get(section, []), key)
        c_entries = _keyed(child.get(section, []), key)
        deltas = [
            (name, c_entries[name]["score"] - p_entries[name]["score"])
            for name in c_entries.keys() & p_entries.keys()
        ]
        deltas = [(name, d) for name, d in deltas if d != 0]
        if not deltas:
            continue
        sign = 1 if higher_is_better else -1
        improved = sorted((d for d in deltas if sign * d[1] > 0), key=lambda d: abs(d[1]), reverse=True)
        regressed = sorted((d for d in deltas if sign * d[1] < 0), key=lambda d: abs(d[1]), reverse=True)
        parts = []
        if improved:
            parts.append("improved " + ", ".join(f"{n} ({d:+g})" for n, d in improved[:3]))
        if regressed:
            parts.append("regressed " + ", ".join(f"{n} ({d:+g})" for n, d in regressed[:3]))
        lines.append(f"By {label}: {'; '.join(parts)}.")
    p_cov = {q["q"]: q["coverage"] for q in parent.get("quantiles", []) if q.get("coverage") is not None}
    c_cov = {q["q"]: q["coverage"] for q in child.get("quantiles", []) if q.get("coverage") is not None}
    shifts = [
        (q, c_cov[q] - p_cov[q])
        for q in c_cov.keys() & p_cov.keys()
        if abs(c_cov[q] - p_cov[q]) > COVERAGE_SHIFT_MIN
    ]
    if shifts:
        parts = [
            f"q{int(round(q * 100))} {shift:+.0%}"
            for q, shift in sorted(shifts, key=lambda s: abs(s[1]), reverse=True)[:3]
        ]
        lines.append(f"Coverage shifts: {', '.join(parts)}.")
    return "\n".join(lines)
