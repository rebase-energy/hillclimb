# First autocorrelation inequality: a step function on [-1/4, 1/4]

Construct a **non-negative step function f on [-1/4, 1/4]** given as **K = 600 equal-width bins**
(width h = 1/(2K) = 0.000833333), that makes

    C1(f) = max over t of (f*f)(t)  /  (∫ f)^2,     (f*f)(t) = ∫ f(t - x) f(x) dx

**as small as possible**. This is the constant of the first autocorrelation inequality (Appendix B.1 of
arXiv:2506.13131), which arises in additive combinatorics (Sidon sets).
Every non-negative f proves `C1 <= C1(f)`; it is known that 1.28 <= C1
(Cloninger & Steinerberger 2017). The best known value is **1.5053**
(AlphaEvolve 2025, a 600-bin step function on this same grid), improving
1.5098 (Matolcsi & Vinuesa 2010).

Constraints (verified programmatically):

- exactly 600 rows; row `id` = i is the value of f on `[-1/4 + i h, -1/4 + (i + 1) h)`
- every value finite and `>= 0`, not all zero
- the ratio is scale-invariant, so the overall scale of f is free

The score is exact for step functions: f*f is piecewise linear with kinks at
the bin edges t_m = -1/2 + m h, where `(f*f)(t_m) = h * sum_(i+j=m-1) a[i] a[j]`,
so `score = 2 * K * max(np.convolve(a, a)) / sum(a)**2`.
The sample (all ones, the indicator of [-1/4, 1/4]) scores exactly 2.0: f*f is
a tent of height 1/2 and ∫ f = 1/2.

Good approaches: gradient or L-BFGS descent on log-values (keeps f >= 0)
against a smooth maximum (log-sum-exp) of f*f over the lags, re-checking
the true maximum; an LP or iteratively reweighted scheme that pushes down
the currently active lags; restarts from asymmetric, spiky profiles (the
known good constructions look like x^(-1/2) singularities, not bumps — the
triangle scores 8/3, worse than the indicator's 2).

## Submission format

Write `submission.csv` in the working directory with the header `id,value` and
600 rows (`id` = 0..599), like `sample_submission.csv` (all ones). Write
values at full precision (`repr`/`%.17g`), not rounded.

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the submission, computes the score exactly as above and
prints `val_score: <score>` (10000 if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit. `numpy` and `scipy` are available.
