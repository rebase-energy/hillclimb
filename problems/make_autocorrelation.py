#!/usr/bin/env python3
"""Stamp the step-function analysis problems of the AlphaEvolve paper
(arXiv:2506.13131, Appendix B): `problems/<id>/` and the wheel's bundled
copy `src/hillclimb/demo/<id>/` for each of

    autocorr-1     first autocorrelation inequality   (B.1, minimize C1(f))
    autocorr-3     third autocorrelation inequality   (B.3, minimize C3(f))
    erdos-overlap  Erdős' minimum overlap problem     (B.5, minimize the bound)

    uv run python problems/make_autocorrelation.py [autocorr-1 ...]

Every instance is one non-negative (or, for autocorr-3, signed) step function
given as K equal-width bin values, and every verifier is EXACT for step
functions: the autoconvolution f*f of a step function with bin width h is
piecewise linear with kinks at the bin edges t_m = -1/2 + m h, where
(f*f)(t_m) = h · sum_{i+j=m-1} a_i a_j — one `np.convolve` — so the maximum
over t is the maximum over kinks. The same holds for the cross-correlation
in the overlap problem. The generated dirs are committed; rerun this after
editing a template here (`tests/test_autocorrelation_starter.py` checks the
committed dirs match).

Normalizations (the paper's, Appendix B):
    C1(f) = max_t (f*f)(t) / (∫f)^2,  f ≥ 0 on [-1/4, 1/4]           — smaller is better
    C3(f) = max_t |(f*f)(t)| / (∫f)^2, f signed on [-1/4, 1/4]        — smaller is better
    C5(h) = max_k ∫ h(x)(1 - h(x+k)) dx, h: [0,2] → [0,1], ∫h = 1     — smaller is better
Each valid submission f proves the upper bound C ≤ C(f), so lower is better
in all three (a construction can only ever tighten an upper bound).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT.parent / "src" / "hillclimb" / "demo"
PENALTY = 10000.0


@dataclass(frozen=True)
class Instance:
    key: str
    metric: str
    bins: int
    best: float  # best known value
    best_source: str
    previous: float  # the bound the best known improved on
    previous_source: str


INSTANCES: dict[str, Instance] = {
    # AlphaEvolve's own grids: 600 bins for C1, 400 for C3 (Appendix B.1/B.3)
    "autocorr-1": Instance("autocorr-1", "c1-ratio", 600, 1.5053, "AlphaEvolve 2025",
                           1.5098, "Matolcsi & Vinuesa 2010"),
    "autocorr-3": Instance("autocorr-3", "c3-ratio", 400, 1.4557, "AlphaEvolve 2025",
                           1.45810, "Vinuesa 2010"),
    "erdos-overlap": Instance("erdos-overlap", "overlap-bound", 400, 0.380924, "AlphaEvolve 2025",
                              0.380927, "Haugland 2016"),
}
DEFAULT_KEYS = tuple(INSTANCES)

# key -> (best known value, who found it); the reference lines on `hillclimb chart`
BEST_KNOWN: dict[str, tuple[float, str]] = {
    key: (inst.best, inst.best_source) for key, inst in INSTANCES.items()
}


def sample_values(inst: Instance) -> list[float]:
    """The weak valid floor: the indicator of the whole interval for the
    autocorrelation problems (C(f) = 2 exactly: f*f is a tent of height 1/2
    and ∫f = 1/2), the indicator of [0, 1] for the overlap problem (bound 1,
    the trivial one: mean 1/2, and the best lag overlaps h with all of 1-h)."""
    if inst.key == "erdos-overlap":
        return [1.0 if i < inst.bins // 2 else 0.0 for i in range(inst.bins)]
    return [1.0] * inst.bins


def sample_submission(inst: Instance) -> str:
    return "id,value\n" + "".join(f"{i},{v}\n" for i, v in enumerate(sample_values(inst)))


def problem_yaml(inst: Instance) -> str:
    return "\n".join([
        f"problem_id: {inst.key}",
        f"metric: {inst.metric}",
        "higher_is_better: false",
        "description: description.md",
        "# valid-by-construction floor: best/ always holds something",
        "baseline_files: {submission.csv: sample_submission.csv}",
        "chart_baselines:",
        f'  "best known ({inst.best_source})": {inst.best:.6f}',
        f'  "previous best ({inst.previous_source})": {inst.previous:.6f}',
        "time_budget_s: 900",
        "allow_network: false",
        "",
    ])


SUBMISSION_AND_SCORING = """\
## Submission format

Write `submission.csv` in the working directory with the header `id,value` and
{K} rows (`id` = 0..{last}), like `sample_submission.csv` ({sample}). Write
values at full precision (`repr`/`%.17g`), not rounded.

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the submission, computes the score exactly as above and
prints `val_score: <score>` ({penalty:g} if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit. `numpy` and `scipy` are available.
"""


def description_md(inst: Instance) -> str:
    K, last = inst.bins, inst.bins - 1
    if inst.key == "erdos-overlap":
        head = f"""# Erdős' minimum overlap problem: a step function on [0, 2]

Construct a **step function h on [0, 2] with values in [0, 1] and unit mass**
(`∫ h = 1`), given as **K = {K} equal-width bins** (width w = 2/K = {2 / K:g}),
that makes the overlap bound

    C5(h) = max over k of  ∫ h(x) (1 - h(x + k)) dx

**as small as possible** (the integral runs over x with both x and x + k in
[0, 2], i.e. h and 1 - h are extended by zero outside [0, 2]). Haugland
(arXiv:1609.08000) showed that the constant C5 of Erdős' minimum overlap
problem is the infimum of this quantity over such step functions, so every
valid h proves `C5 <= C5(h)`. The known bounds are 0.379005 (White 2023)
<= C5 <= **{inst.best}** ({inst.best_source}, Appendix B.5 of arXiv:2506.13131,
improving Haugland's {inst.previous}).

Constraints (verified programmatically):

- exactly {K} rows; row `id` = i is the value of h on `[i w, (i + 1) w)`
- every value finite and in `[0, 1]`
- unit mass: the **mean of the {K} values is exactly 1/2** (`sum(value) = {K // 2}`,
  tolerance 1e-9 — fix the last bin or project onto the constraint before writing)

The score is exact for step functions: the cross-correlation of h and
g = 1 - h is piecewise linear with kinks at multiples of w, where it equals
`w * sum_i h[i] * g[i + m]`, so
`score = (2 / K) * max(np.correlate(h, 1 - h, mode="full"))`. The sample
(h = 1 on [0, 1], 0 on [1, 2]) scores exactly 1.0 (the trivial bound); the
constant h = 1/2 scores 0.5.

Good approaches: the objective is a max over lags of a bilinear form, so
subgradient or smooth-max (log-sum-exp over lags) descent on h with the box
constraint kept by clipping and the mass constraint by projection; an LP or
iteratively reweighted scheme that pushes down the currently active lags;
restarts from tapered profiles (h rising from 0 at the ends to 1 in the
middle) rather than from indicators. Read the bins that matter off
`np.correlate` and move mass between them.

"""
        sample = "the indicator of [0, 1]"
    else:
        signed = inst.key == "autocorr-3"
        if signed:
            title = "Third autocorrelation inequality: a signed step function on [-1/4, 1/4]"
            what = "**step function f on [-1/4, 1/4], positive and negative values allowed,**"
            ratio = "C3(f) = max over t of |(f*f)(t)|  /  (∫ f)^2"
            para = (f"the constant of the third autocorrelation inequality (Appendix B.3 of\n"
                    f"arXiv:2506.13131). Every f with ∫ f != 0 proves `C3 <= C3(f)`. Since f may\n"
                    f"change sign, cancellations in f*f can push the peak below what any\n"
                    f"non-negative f achieves (C3 <= C1). The best known value is **{inst.best}**\n"
                    f"({inst.best_source}, a {K}-bin step function on this same grid), improving\n"
                    f"{inst.previous} ({inst.previous_source}).")
            domain = "- every value finite; `∫ f` (the sum of the values) must not be zero"
            hint = ("Good approaches: gradient or L-BFGS descent on the values against a\n"
                    "smooth maximum (log-sum-exp) of |f*f| over the lags, re-checking the true\n"
                    "maximum; start from a good non-negative profile (see the first inequality)\n"
                    "and let small negative lobes near the ends cancel the peak; an LP that\n"
                    "pushes down the currently active lags. Restarts from asymmetric, spiky\n"
                    "shapes; smooth bumps are worse than the indicator.")
        else:
            title = "First autocorrelation inequality: a step function on [-1/4, 1/4]"
            what = "**non-negative step function f on [-1/4, 1/4]**"
            ratio = "C1(f) = max over t of (f*f)(t)  /  (∫ f)^2"
            para = (f"the constant of the first autocorrelation inequality (Appendix B.1 of\n"
                    f"arXiv:2506.13131), which arises in additive combinatorics (Sidon sets).\n"
                    f"Every non-negative f proves `C1 <= C1(f)`; it is known that 1.28 <= C1\n"
                    f"(Cloninger & Steinerberger 2017). The best known value is **{inst.best}**\n"
                    f"({inst.best_source}, a {K}-bin step function on this same grid), improving\n"
                    f"{inst.previous} ({inst.previous_source}).")
            domain = "- every value finite and `>= 0`, not all zero"
            hint = ("Good approaches: gradient or L-BFGS descent on log-values (keeps f >= 0)\n"
                    "against a smooth maximum (log-sum-exp) of f*f over the lags, re-checking\n"
                    "the true maximum; an LP or iteratively reweighted scheme that pushes down\n"
                    "the currently active lags; restarts from asymmetric, spiky profiles (the\n"
                    "known good constructions look like x^(-1/2) singularities, not bumps — the\n"
                    "triangle scores 8/3, worse than the indicator's 2).")
        head = f"""# {title}

Construct a {what} given as **K = {K} equal-width bins**
(width h = 1/(2K) = {1 / (2 * K):g}), that makes

    {ratio},     (f*f)(t) = ∫ f(t - x) f(x) dx

**as small as possible**. This is {para}

Constraints (verified programmatically):

- exactly {K} rows; row `id` = i is the value of f on `[-1/4 + i h, -1/4 + (i + 1) h)`
{domain}
- the ratio is scale-invariant, so the overall scale of f is free

The score is exact for step functions: f*f is piecewise linear with kinks at
the bin edges t_m = -1/2 + m h, where `(f*f)(t_m) = h * sum_(i+j=m-1) a[i] a[j]`,
so `score = 2 * K * max({"abs(" if signed else ""}np.convolve(a, a){")" if signed else ""}) / sum(a)**2`.
The sample (all ones, the indicator of [-1/4, 1/4]) scores exactly 2.0: f*f is
a tent of height 1/2 and ∫ f = 1/2.

{hint}

"""
        sample = "all ones"
    return head + SUBMISSION_AND_SCORING.format(K=K, last=last, sample=sample, penalty=PENALTY)


VERIFIER_SH = """#!/usr/bin/env bash
# hillclimb verifier: run the candidate, then score what it produced.
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"   # writes ./submission.csv

# trust boundary: only the scorer may report a score, so anything the
# solution left behind is discarded before the scorer runs
rm -f "$HILLCLIMB_RESULT"
exec "$HILLCLIMB_PYTHON" problem/verify.py
"""


VERIFY_HEAD = '''"""Official scorer for the {key} problem.

Reads ./submission.csv (id,value; {K} rows), validates, and writes the
score to $HILLCLIMB_RESULT ({what}; {penalty:g} penalty if invalid — lower is better).
Generated by problems/make_autocorrelation.py — edit the template there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K = {K}
PENALTY = {penalty}
TOL = 1e-9


def emit(score: float) -> None:
    """hillclimb's verifier contract: the score is written to the result
    file, not scraped from stdout (agent-authored code shares this stream)."""
    path = Path(os.environ.get("HILLCLIMB_RESULT", "eval_result.json"))
    path.write_text(json.dumps({{"split": os.environ.get("HILLCLIMB_SPLIT", "validation"),
                                "score": score}}))


def fail(reason: str) -> None:
    print(f"INVALID: {{reason}}")
    emit(PENALTY)
    print(f"val_score: {{PENALTY}}")
    sys.exit(0)


def read_values() -> np.ndarray:
    """The K bin values in id order; any malformed submission is a fail()."""
    try:
        df = pd.read_csv("submission.csv")
    except Exception as e:  # noqa: BLE001
        fail(f"cannot read submission.csv: {{e}}")
    for col in ("id", "value"):
        if col not in df.columns:
            fail(f"missing column {{col!r}}")
    if len(df) != K or sorted(df["id"].tolist()) != list(range(K)):
        fail(f"need exactly {{K}} rows with id 0..{{K - 1}}")
    try:
        values = df.sort_values("id")["value"].to_numpy(float)
    except Exception as e:  # noqa: BLE001
        fail(f"non-numeric value(s): {{e}}")
    if not np.all(np.isfinite(values)):
        fail("non-finite value(s)")
    return values

'''

VERIFY_AUTOCORR = '''
SIGNED = {signed}


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
    print(f"valid step function; {{'|f*f|' if SIGNED else 'f*f'}} peak / (∫f)^2 = {{score:.8f}}")
    emit(score)
    print(f"val_score: {{score:.8f}}")


if __name__ == "__main__":
    main()
'''

VERIFY_OVERLAP = '''
W = 2.0 / K  # bin width on [0, 2]


def score_of(h: np.ndarray) -> float:
    """max_k ∫ h(x) (1 - h(x + k)) dx for the step function h with values `h`
    on K equal bins of [0, 2] (h and 1 - h extended by zero outside) — EXACT:
    the cross-correlation of two step functions is piecewise linear with kinks
    at multiples of the bin width, where it equals W * sum_i h[i] (1 - h)[i + m],
    so the maximum over k is the maximum over the K*2 - 1 integer lags."""
    return W * float(np.max(np.correlate(h, 1.0 - h, mode="full")))


def main() -> None:
    h = read_values()
    if np.any(h < -TOL) or np.any(h > 1 + TOL):
        fail("value(s) outside [0, 1]")
    h = np.clip(h, 0.0, 1.0)
    if abs(float(h.mean()) - 0.5) > TOL:
        fail(f"unit mass violated: the mean of the values is {{h.mean():.9f}}, not 0.5")
    score = score_of(h)
    print(f"valid step function; overlap bound = {{score:.8f}}")
    emit(score)
    print(f"val_score: {{score:.8f}}")


if __name__ == "__main__":
    main()
'''


def verify_py(inst: Instance) -> str:
    if inst.key == "erdos-overlap":
        what = "max over lags of ∫ h(x)(1 - h(x + k)) dx"
        body = VERIFY_OVERLAP.format()
    else:
        signed = inst.key == "autocorr-3"
        what = ("max |f*f| / (∫f)^2" if signed else "max f*f / (∫f)^2")
        body = VERIFY_AUTOCORR.format(signed=signed)
    return VERIFY_HEAD.format(key=inst.key, K=inst.bins, what=what, penalty=PENALTY) + body


def files_for(inst: Instance) -> dict[str, str]:
    return {
        "problem.yaml": problem_yaml(inst),
        "description.md": description_md(inst),
        "verifier.sh": VERIFIER_SH,
        "verify.py": verify_py(inst),
        "sample_submission.csv": sample_submission(inst),
    }


def stamp(root: Path, key: str) -> Path:
    """Write <root>/<key>/; returns the dir."""
    if key not in INSTANCES:
        raise ValueError(f"unknown instance {key!r} (choose from {', '.join(INSTANCES)})")
    inst = INSTANCES[key]
    problem_dir = root / key
    problem_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files_for(inst).items():
        (problem_dir / name).write_text(text)
    (problem_dir / "verifier.sh").chmod(0o755)
    return problem_dir


def main(argv: list[str]) -> None:
    keys = list(argv) or list(DEFAULT_KEYS)
    for key in keys:
        for root in (ROOT, DEMO_ROOT):
            print(f"wrote {stamp(root, key)}")


if __name__ == "__main__":
    main(sys.argv[1:])
