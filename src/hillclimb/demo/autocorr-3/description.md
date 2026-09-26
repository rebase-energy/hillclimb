# Third autocorrelation inequality: a signed step function on [-1/4, 1/4]

Construct a **step function f on [-1/4, 1/4], positive and negative values allowed,** given as **K = 400 equal-width bins**
(width h = 1/(2K) = 0.00125), that makes

    C3(f) = max over t of |(f*f)(t)|  /  (∫ f)^2,     (f*f)(t) = ∫ f(t - x) f(x) dx

**as small as possible**. This is the constant of the third autocorrelation inequality (Appendix B.3 of
arXiv:2506.13131). Every f with ∫ f != 0 proves `C3 <= C3(f)`. Since f may
change sign, cancellations in f*f can push the peak below what any
non-negative f achieves (C3 <= C1). The best known value is **1.4557**
(AlphaEvolve 2025, a 400-bin step function on this same grid), improving
1.4581 (Vinuesa 2010).

Constraints (verified programmatically):

- exactly 400 rows; row `id` = i is the value of f on `[-1/4 + i h, -1/4 + (i + 1) h)`
- every value finite; `∫ f` (the sum of the values) must not be zero
- the ratio is scale-invariant, so the overall scale of f is free

The score is exact for step functions: f*f is piecewise linear with kinks at
the bin edges t_m = -1/2 + m h, where `(f*f)(t_m) = h * sum_(i+j=m-1) a[i] a[j]`,
so `score = 2 * K * max(abs(np.convolve(a, a))) / sum(a)**2`.
The sample (all ones, the indicator of [-1/4, 1/4]) scores exactly 2.0: f*f is
a tent of height 1/2 and ∫ f = 1/2.

Good approaches: gradient or L-BFGS descent on the values against a
smooth maximum (log-sum-exp) of |f*f| over the lags, re-checking the true
maximum; start from a good non-negative profile (see the first inequality)
and let small negative lobes near the ends cancel the peak; an LP that
pushes down the currently active lags. Restarts from asymmetric, spiky
shapes; smooth bumps are worse than the indicator.

## Submission format

Write `submission.csv` in the working directory with the header `id,value` and
400 rows (`id` = 0..399), like `sample_submission.csv` (all ones). Write
values at full precision (`repr`/`%.17g`), not rounded.

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the submission, computes the score exactly as above and
prints `val_score: <score>` (10000 if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit. `numpy` and `scipy` are available.
