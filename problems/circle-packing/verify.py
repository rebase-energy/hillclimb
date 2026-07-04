"""Official scorer for the circle-packing problem.

Reads ./submission.csv (id,x,y,r; 26 rows), validates the packing, and prints
the official `val_score:` line (sum of radii, 0.0 if invalid).
"""

import sys

import numpy as np
import pandas as pd

N = 26
TOL = 1e-9  # numerical slack on containment/overlap


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
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
    print(f"valid packing; sum of radii = {r.sum():.6f}")
    print(f"val_score: {r.sum():.6f}")


if __name__ == "__main__":
    main()
