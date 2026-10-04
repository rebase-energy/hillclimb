"""Official scorer for the kissing-11 problem.

Reads ./submission.csv (id,c0..c10; N rows), validates the kissing
condition exactly on integer coordinates, and writes the score to
$HILLCLIMB_RESULT (the number of points N, 0.0 if invalid).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = 11
MAX_POINTS = 2000
MAX_COORD = 100000000
COLS = [f"c{i}" for i in range(D)]


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    emit(0.0)
    print("val_score: 0.0")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", *COLS):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    n = len(df)
    if n < 1 or n > MAX_POINTS:
        fail(f"need between 1 and {MAX_POINTS} rows, got {n}")
    if sorted(df["id"].tolist()) != list(range(n)):
        fail(f"ids must be 0..{n - 1}")
    try:
        raw = df.sort_values("id")[COLS].to_numpy(dtype=float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric coordinates: {e}")
    if not np.all(np.isfinite(raw)):
        fail("non-finite coordinates")
    if np.any(raw != np.round(raw)):
        fail("coordinates must be integers")
    if np.any(np.abs(raw) > MAX_COORD):
        fail(f"coordinates must satisfy |c| <= {MAX_COORD}")
    pts = raw.astype(np.int64)
    norms = np.einsum("ij,ij->i", pts, pts)
    if np.any(norms == 0):
        fail("the origin is not allowed")
    if len(np.unique(pts, axis=0)) != n:
        fail("duplicate points")
    max_norm = int(norms.max())
    if n == 1:
        min_dist = max_norm
    else:
        gram = pts @ pts.T
        dist = norms[:, None] + norms[None, :] - 2 * gram
        np.fill_diagonal(dist, dist.max() + 1)
        min_dist = int(dist.min())
    if min_dist < max_norm:
        fail(f"kissing condition violated: min pairwise |dist|^2 {min_dist} < max |norm|^2 {max_norm}")
    print(f"valid configuration; {n} points, min |dist|^2 {min_dist} >= max |norm|^2 {max_norm}")
    emit(float(n))
    print(f"val_score: {n}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — the verifier never crashes
        fail(f"unexpected error: {e}")
