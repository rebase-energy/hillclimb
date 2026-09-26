"""Official scorer for the tsp-200 problem.

Reads ./submission.csv (position,city; a permutation of 0..199), validates it,
and writes the score to $HILLCLIMB_RESULT (closed-tour Euclidean length;
1000.0 penalty if invalid — lower is better).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 200
PENALTY = 1000.0


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    emit(PENALTY)
    print(f"val_score: {PENALTY}")
    sys.exit(0)


def main() -> None:
    cities_path = Path(__file__).parent / "data" / "cities.csv"
    cities = pd.read_csv(cities_path).sort_values("id")[["x", "y"]].to_numpy(float)
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("position", "city"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != N:
        fail(f"need exactly {N} rows")
    tour = df.sort_values("position")["city"].to_numpy()
    if sorted(tour.tolist()) != list(range(N)):
        fail("city column is not a permutation of 0..199")
    ordered = cities[tour]
    diffs = np.diff(np.vstack([ordered, ordered[:1]]), axis=0)
    length = float(np.sqrt((diffs**2).sum(axis=1)).sum())
    print(f"valid tour; length = {length:.6f}")
    emit(length)
    print(f"val_score: {length:.6f}")


if __name__ == "__main__":
    main()
