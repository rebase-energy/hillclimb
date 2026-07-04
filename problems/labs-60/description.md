# Low-autocorrelation binary sequence (LABS), length 60

Find a binary sequence `s` of **length 60** with entries in `{+1, -1}` that
**minimizes the autocorrelation sidelobe energy**

```
E(s) = sum_{k=1}^{59} C_k(s)^2,   where   C_k(s) = sum_{i=0}^{59-k} s_i * s_{i+k}
```

This is a famously rugged discrete optimization landscape (used in radar and
statistical physics as the "bernasconi model"). A random sequence has expected
energy **1770**; the best known for length 60 is around **150**. Plain
hill-climbing stalls quickly — good approaches use tabu search, memetic /
population methods, or self-avoiding walks over bit-flip neighborhoods, with
incremental O(N) energy updates per flip and many restarts. `numpy` is
available.

## Submission format

Write `submission.csv` in the working directory with the header `id,spin` and
60 rows (`id` = 0..59, `spin` = +1 or -1), like `sample_submission.csv` (an
alternating baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the sequence and prints `val_score: <energy>` (or a penalty
of 100000.0 if invalid). **Lower is better.**

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
