"""Official scorer for the thomson-100 problem.

Reads ./submission.csv (id,x,y,z; 100 rows), normalizes each row to the unit
sphere, and writes the score to $HILLCLIMB_RESULT (Coulomb energy
sum_{i<j} 1/|p_i - p_j|; 100000.0 penalty if invalid — lower is better).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 100
PENALTY = 100000.0
TOL = 1e-12


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    emit(PENALTY)
    print(f"val_score: {PENALTY}")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", "x", "y", "z"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    ids = pd.to_numeric(df["id"], errors="coerce").to_numpy(float)
    if len(df) != N or not np.all(np.isfinite(ids)) or sorted(ids.tolist()) != list(range(N)):
        fail(f"need exactly {N} rows with id 0..{N - 1}")
    pts = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)[np.argsort(ids)]
    if not np.all(np.isfinite(pts)):
        fail("non-finite or non-numeric coordinates")
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < TOL):
        fail("zero-length vector(s): every row must be a direction")
    pts = pts / norms[:, None]
    i, j = np.triu_indices(N, 1)
    dist = np.linalg.norm(pts[i] - pts[j], axis=1)
    if np.any(dist < TOL):
        fail("coincident charges (infinite energy)")
    energy = float(np.sum(1.0 / dist))
    print(f"valid layout; Coulomb energy = {energy:.9f}")
    emit(energy)
    print(f"val_score: {energy:.9f}")


if __name__ == "__main__":
    main()
