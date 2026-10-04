"""Official scorer for the labs-40 problem.

Reads ./submission.csv (id,spin; 40 rows of +/-1), validates, and writes the
score to $HILLCLIMB_RESULT (autocorrelation sidelobe energy; a 100000.0
penalty if invalid — lower is better).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 40
PENALTY = 100000.0


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
    for col in ("id", "spin"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != N or sorted(df["id"].tolist()) != list(range(N)):
        fail(f"need exactly {N} rows with id 0..{N - 1}")
    spins = pd.to_numeric(df.sort_values("id")["spin"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(spins)) or not np.all(np.isin(spins, (-1.0, 1.0))):
        fail("spin values must be +1 or -1")
    s = spins.astype(int)
    sidelobes = [int(np.dot(s[: N - k], s[k:])) for k in range(1, N)]
    energy = sum(c * c for c in sidelobes)
    worst = sorted(range(1, N), key=lambda k: -abs(sidelobes[k - 1]))[:5]
    print("largest sidelobes (lag: C_k): " + ", ".join(f"{k}: {sidelobes[k - 1]}" for k in worst))
    print(f"valid sequence; energy = {energy}")
    emit(float(energy))
    print(f"val_score: {energy}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — an invalid submission is the penalty, never a crash
        fail(f"unexpected error: {e}")
