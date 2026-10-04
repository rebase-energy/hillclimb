"""Official scorer for the autocorr-3 problem.

Reads ./submission.csv (id,value; 400 rows), validates, and writes the
score to $HILLCLIMB_RESULT (max |f*f| / (∫f)^2; 10000 penalty if invalid — lower is better).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K = 400
PENALTY = 10000.0
TOL = 1e-9


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


def read_values() -> np.ndarray:
    """The K bin values in id order; any malformed submission is a fail()."""
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {e}")
    for col in ("id", "value"):
        if col not in df.columns:
            fail(f"missing column {col!r}")
    if len(df) != K or sorted(df["id"].tolist()) != list(range(K)):
        fail(f"need exactly {K} rows with id 0..{K - 1}")
    try:
        values = df.sort_values("id")["value"].to_numpy(float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric value(s): {e}")
    if not np.all(np.isfinite(values)):
        fail("non-finite value(s)")
    return values


SIGNED = True


def score_of(a: np.ndarray) -> float:
    """max_t |(f*f)(t)| / (∫f)^2 for the step function f with values `a` on K
    equal bins of [-1/4, 1/4] (bin width h = 1/(2K)) — EXACT: f*f is piecewise
    linear with kinks at the bin edges t_m = -1/2 + m h, where it equals
    h * sum_(i+j=m-1) a[i] a[j], so the maximum over t is the maximum over
    kinks and the h's cancel into 2K."""
    conv = np.convolve(a, a)
    peak = float(np.max(np.abs(conv))) if SIGNED else float(np.max(conv))
    return 2 * K * peak / float(a.sum()) ** 2


def main() -> None:
    a = read_values()
    if not SIGNED:
        if np.any(a < -TOL):
            fail("negative value(s): f must be non-negative")
        a = np.clip(a, 0.0, None)
    if abs(float(a.sum())) <= 1e-12 * float(np.abs(a).sum()):
        fail("the integral of f is zero" + ("" if SIGNED else " (all values are zero)"))
    score = score_of(a)
    print(f"valid step function; {'|f*f|' if SIGNED else 'f*f'} peak / (∫f)^2 = {score:.8f}")
    emit(score)
    print(f"val_score: {score:.8f}")


if __name__ == "__main__":
    main()
