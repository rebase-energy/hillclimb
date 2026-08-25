"""Score a 13-point normalized Heilbronn configuration.

The objective matches the Heilbronn Convex benchmark: minimum area over all
point triples, divided by the convex hull area of the complete point set.
"""

from __future__ import annotations

from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

N = 13
N_TRIANGLES = 286


def write_report(overall_score: float, report_error=None, zones=None) -> None:
    report = {
        "version": 1,
        "split": "validation",
        "objective": "normalized-min-triangle-area",
        "higher_is_better": True,
        "source": "evaluator",
        "segment_label": "triangle",
        "overall": {
            "score": overall_score,
            "n_origins": N_TRIANGLES,
            "n_scored": N_TRIANGLES,
        },
        "report_error": report_error,
    }
    if zones:
        report["zones"] = zones
    payload = {
        "split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
        "score": overall_score,
        "report": report,
    }
    result = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    result.write_text(json.dumps(payload, indent=2))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    write_report(0.0, report_error=reason)
    print("val_score: 0.0")
    sys.exit(0)


def cross(origin, a, b) -> float:
    return float(
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return hull vertices counter-clockwise using the monotone chain."""
    ordered = sorted(set(map(tuple, points.tolist())))
    if len(ordered) < 3:
        return np.empty((0, 2), dtype=float)

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    # Translate before applying the shoelace formula for numerical stability.
    shifted = vertices - vertices[0]
    return 0.5 * abs(
        float(
            np.dot(shifted[:, 0], np.roll(shifted[:, 1], -1))
            - np.dot(shifted[:, 1], np.roll(shifted[:, 0], -1))
        )
    )


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot read submission.csv: {exc}")

    for column in ("id", "x", "y"):
        if column not in df.columns:
            fail(f"missing column {column!r}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {N} rows with id 0..{N - 1}")

    try:
        points = df.sort_values("id")[["x", "y"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        fail(f"coordinates must be numbers: {exc}")
    if not np.all(np.isfinite(points)):
        fail("coordinates must be finite")

    # Translation and uniform scaling leave the score unchanged. Normalize
    # internally to make very large or very small valid coordinates safe.
    points = points - points[0]
    scale = float(np.max(np.abs(points)))
    if scale == 0.0:
        fail("degenerate convex hull")
    points = points / scale

    hull_area = polygon_area(convex_hull(points))
    if not np.isfinite(hull_area) or hull_area <= np.finfo(float).eps:
        fail("degenerate convex hull")

    triangles = []
    for i, j, k in combinations(range(N), 3):
        area = 0.5 * abs(cross(points[i], points[j], points[k]))
        triangles.append((area, i, j, k))

    min_area = min(item[0] for item in triangles)
    score = float(min_area / hull_area)
    weakest = sorted(triangles)[:6]
    zones = [
        {
            "zone": f"triangle-{i}-{j}-{k}",
            "score": float(area / hull_area),
            "n_scored": 1,
            "area": float(area),
        }
        for area, i, j, k in weakest
    ]

    write_report(score, zones=zones)
    print(
        "valid configuration; "
        f"minimum triangle area / hull area = {score:.12f}"
    )
    print(f"val_score: {score:.12f}")


if __name__ == "__main__":
    main()
