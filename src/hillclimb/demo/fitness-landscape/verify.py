"""Official scorer for the fitness-landscape demo problem.

Reads ./submission.csv (x,y; one row), validates the point, and writes the
score — the terrain elevation at that point — to $HILLCLIMB_RESULT. The
submitted coordinates ride along as extra numeric keys, so the engine
journals them as Trial.metrics and a 3D chart can place every candidate on
the surface. The report carries the local uphill direction so improve
operators know which way the ground rises.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from landscape import DOMAIN, elevation  # noqa: E402

EPS = 1e-4  # finite-difference step for the report's gradient


def write_result(score: float, x=None, y=None, report_error=None, zones=None) -> None:
    report = {
        "version": 1,
        "split": "validation",
        "objective": "elevation",
        "higher_is_better": True,
        "source": "evaluator",
        "segment_label": "probe",
        "overall": {"score": score, "n_origins": 1, "n_scored": 1},
        "report_error": report_error,
    }
    if zones:
        report["zones"] = zones
    payload = {"split": "validation", "score": score, "report": report}
    if x is not None:
        payload["x"] = x  # journaled as Trial.metrics — the 3D chart reads these
        payload["y"] = y
    result = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    result.write_text(json.dumps(payload, indent=2))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    write_result(0.0, report_error=reason)
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("x", "y"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != 1:
        fail("need exactly one row")
    x = float(df["x"].iloc[0])
    y = float(df["y"].iloc[0])
    if not (np.isfinite(x) and np.isfinite(y)):
        fail("non-finite coordinates")
    lo, hi = DOMAIN
    if not (lo <= x <= hi and lo <= y <= hi):
        fail(f"point ({x}, {y}) outside the domain [{lo}, {hi}]^2")

    score = float(elevation(x, y))
    gx = float((elevation(x + EPS, y) - elevation(x - EPS, y)) / (2 * EPS))
    gy = float((elevation(x, y + EPS) - elevation(x, y - EPS)) / (2 * EPS))
    zones = [
        {
            "zone": "probe",
            "score": round(score, 6),
            "n_scored": 1,
            "uphill_dx": round(gx, 4),
            "uphill_dy": round(gy, 4),
        }
    ]
    write_result(round(score, 6), x=round(x, 6), y=round(y, 6), zones=zones)
    print(f"elevation({x:.4f}, {y:.4f}) = {score:.6f}")
    print(f"val_score: {score:.6f}")


if __name__ == "__main__":
    main()
