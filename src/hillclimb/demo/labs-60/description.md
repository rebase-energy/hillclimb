# Low-autocorrelation binary sequence (LABS), length 60

Find a binary sequence `s` of **length 60** with entries in `{+1, -1}` that
**minimizes the autocorrelation sidelobe energy**

```
E(s) = sum_{k=1}^{59} C_k(s)^2,   where   C_k(s) = sum_{i=0}^{59-k} s_i * s_{i+k}
```

This is a famously rugged discrete optimization landscape (used in radar and
statistical physics as the "Bernasconi model"). A random sequence has expected
energy **1770**. The optimal energy for N = 60 is **218** (merit factor N^2 / (2E) = 8.257), proven by exhaustive search (Packebusch & Mertens 2016).
Plain hill-climbing stalls quickly — good approaches use tabu search, memetic /
population methods, or self-avoiding walks over the bit-flip neighborhood,
with incremental O(N) energy updates per flip and many restarts (for odd N,
skew-symmetric sequences halve the search space and often contain the
optimum). `numpy` is available.

## Submission format

Write `submission.csv` in the working directory with the header `id,spin` and
60 rows (`id` = 0..59, `spin` = +1 or -1), like `sample_submission.csv` (a
weak valid baseline: runs of length 1, 2, 3, ... with alternating sign).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the sequence and prints `val_score: <energy>` (or a penalty
of 100000.0 if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
