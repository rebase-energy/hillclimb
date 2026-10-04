"""Official scorer for the golomb-20 problem.

Reads ./submission.csv (id,mark; 20 rows), validates the ruler, and writes
the score to $HILLCLIMB_RESULT (the length = largest mark, 1000000 if invalid).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M = 20
PENALTY = 1000000


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (coding-agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}))


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
    emit(float(PENALTY))
    print(f"val_score: {PENALTY}")
    sys.exit(0)


def main() -> None:
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", "mark"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != M or sorted(df["id"].tolist()) != list(range(M)):
        fail(f"need exactly {M} rows with id 0..{M - 1}")
    try:
        raw = df["mark"].to_numpy(dtype=float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric marks: {e}")
    if not np.all(np.isfinite(raw)):
        fail("non-finite marks")
    if np.any(raw != np.round(raw)):
        fail("marks must be integers")
    if np.any(raw < 0) or np.any(raw >= PENALTY):
        fail(f"marks must satisfy 0 <= mark < {PENALTY}")
    marks = np.sort(raw.astype(np.int64))
    if marks[0] != 0:
        fail("mark 0 must be present")
    if len(np.unique(marks)) != M:
        fail("duplicate marks")
    i, j = np.triu_indices(M, k=1)
    diffs = marks[j] - marks[i]
    if len(np.unique(diffs)) != len(diffs):
        fail("two pairs of marks have the same difference")
    length = int(marks[-1])
    print(f"valid Golomb ruler with {M} marks; length = {length}")
    emit(float(length))
    print(f"val_score: {length}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — the verifier never crashes
        fail(f"unexpected error: {e}")
