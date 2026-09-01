"""Official scorer for the circle-packing problem.

Reads ./submission.csv (id,x,y,r; 26 rows), validates the packing, and writes
the score to $HILLCLIMB_RESULT (sum of radii, 0.0 if invalid), together with
a report block — hillclimb's evaluator report
contract — so improve operators see WHERE the packing is weakest (smallest
circles and their growth headroom) instead of just the total.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

N = 26
TOL = 1e-9  # numerical slack on containment/overlap


def write_report(overall_score: float, report_error=None, zones=None, instances=None) -> None:
    report = {
        "version": 1,
        "split": "validation",
        "objective": "sum-radii",
        "higher_is_better": True,
        "source": "evaluator",
        "segment_label": "circle",
        "overall": {"score": overall_score, "n_origins": N, "n_scored": N},
        "report_error": report_error,
    }
    if zones:
        report["zones"] = zones
    payload = {"split": "validation", "score": overall_score, "report": report}
    if instances:
        # reserved key: per-instance breakdown of `score`, same direction —
        # each circle's radius is one instance a search engine can compare
        # across candidates (Pareto frontiers, per-instance views)
        payload["instances"] = instances
    result = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    result.write_text(json.dumps(payload, indent=2))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    write_report(0.0, report_error=reason)
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", "x", "y", "r"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {N} rows with id 0..{N - 1}")
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    r = df["r"].to_numpy(float)
    if not np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(r)):
        fail("non-finite values")
    if np.any(r < -TOL):
        fail("negative radius")
    r = np.clip(r, 0.0, None)
    inside = (x - r >= -TOL) & (x + r <= 1 + TOL) & (y - r >= -TOL) & (y + r <= 1 + TOL)
    if not np.all(inside):
        fail(f"circle(s) {np.where(~inside)[0].tolist()} outside the unit square")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    required = r[:, None] + r[None, :]
    overlap = (dist + TOL < required) & ~np.eye(N, dtype=bool)
    if np.any(overlap):
        i, j = np.argwhere(overlap)[0]
        fail(f"circles {i} and {j} overlap (dist {dist[i, j]:.6f} < {required[i, j]:.6f})")
    # growth headroom per circle: distance to the nearest binding constraint
    # (a wall or a neighbour). Small radius + large slack = wasted space.
    gaps = dist - required + np.where(np.eye(N, dtype=bool), np.inf, 0.0)
    wall = np.minimum.reduce([x - r, 1 - x - r, y - r, 1 - y - r])
    slack = np.minimum(gaps.min(axis=1), wall)
    weakest = np.argsort(r)[:6]  # worst contributors first (higher is better)
    zones = [
        {
            "zone": f"circle-{int(i)}",
            "score": round(float(r[i]), 6),
            "n_scored": 1,
            "slack": round(float(slack[i]), 6),
        }
        for i in weakest
    ]
    instances = {f"circle-{i:02d}": round(float(r[i]), 6) for i in range(N)}
    write_report(round(float(r.sum()), 6), zones=zones, instances=instances)
    print(f"valid packing; sum of radii = {r.sum():.6f}")
    print(f"val_score: {r.sum():.6f}")


if __name__ == "__main__":
    main()
