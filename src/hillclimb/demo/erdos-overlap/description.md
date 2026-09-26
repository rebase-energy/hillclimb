# Erdős' minimum overlap problem: a step function on [0, 2]

Construct a **step function h on [0, 2] with values in [0, 1] and unit mass**
(`∫ h = 1`), given as **K = 400 equal-width bins** (width w = 2/K = 0.005),
that makes the overlap bound

    C5(h) = max over k of  ∫ h(x) (1 - h(x + k)) dx

**as small as possible** (the integral runs over x with both x and x + k in
[0, 2], i.e. h and 1 - h are extended by zero outside [0, 2]). Haugland
(arXiv:1609.08000) showed that the constant C5 of Erdős' minimum overlap
problem is the infimum of this quantity over such step functions, so every
valid h proves `C5 <= C5(h)`. The known bounds are 0.379005 (White 2023)
<= C5 <= **0.380924** (AlphaEvolve 2025, Appendix B.5 of arXiv:2506.13131,
improving Haugland's 0.380927).

Constraints (verified programmatically):

- exactly 400 rows; row `id` = i is the value of h on `[i w, (i + 1) w)`
- every value finite and in `[0, 1]`
- unit mass: the **mean of the 400 values is exactly 1/2** (`sum(value) = 200`,
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

## Submission format

Write `submission.csv` in the working directory with the header `id,value` and
400 rows (`id` = 0..399), like `sample_submission.csv` (the indicator of [0, 1]). Write
values at full precision (`repr`/`%.17g`), not rounded.

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the submission, computes the score exactly as above and
prints `val_score: <score>` (10000 if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit. `numpy` and `scipy` are available.
