"""Official scorer for the heilbronn-11 problem.

Reads ./submission.csv (id,x,y; 11 rows), validates, and prints the official
`val_score:` line (minimum triangle area over all triples, 0.0 if invalid).
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd

N = 11
TOL = 1e-9


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", "x", "y"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {N} rows with id 0..{N - 1}")
    pts = df.sort_values("id")[["x", "y"]].to_numpy(float)
    if not np.all(np.isfinite(pts)):
        fail("non-finite coordinates")
    if np.any(pts < -TOL) or np.any(pts > 1 + TOL):
        fail("point(s) outside the unit square")
    min_area = min(
        0.5
        * abs(
            (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
            - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
        )
        for i, j, k in combinations(range(N), 3)
    )
    print(f"valid layout; min triangle area = {min_area:.8f}")
    print(f"val_score: {min_area:.8f}")


if __name__ == "__main__":
    main()
