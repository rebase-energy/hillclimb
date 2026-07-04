"""Official scorer for the labs-60 problem.

Reads ./submission.csv (id,spin; 60 rows of +/-1), validates, and prints the
official `val_score:` line (autocorrelation sidelobe energy; 100000.0 penalty
if invalid — lower is better).
"""

import sys

import numpy as np
import pandas as pd

N = 60
PENALTY = 100000.0


def fail(reason: str) -> None:
    print(f"INVALID: {reason}")
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
    s = df.sort_values("id")["spin"].to_numpy()
    if not np.all(np.isin(s, (-1, 1))):
        fail("spin values must be +1 or -1")
    s = s.astype(int)
    energy = sum(int(np.dot(s[: N - k], s[k:])) ** 2 for k in range(1, N))
    print(f"valid sequence; energy = {energy}")
    print(f"val_score: {energy}")


if __name__ == "__main__":
    main()
