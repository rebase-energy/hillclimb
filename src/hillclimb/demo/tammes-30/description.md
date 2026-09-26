# Tammes problem: 30 points on the sphere

Place **exactly 30 points on the unit sphere** so that the **smallest angular
separation** between any two of them is **as large as possible** (a max-min
objective over all C(30,2) = 435 pairs — the "spherical code" or
"repelling dictators" problem).

Constraints (verified programmatically):

- exactly 30 rows with finite coordinates
- every row is a non-zero vector; the verifier **normalizes each row to unit
  length**, so you may submit any non-zero direction

This is a classic hard continuous optimization problem. The best known value for n = 30 is **38.5971 degrees** (Hardin, Sloane & Smith 1994).
Latitude rings and lattices are weak. Good approaches: many random or
Fibonacci-spiral starts, a steep repulsion surrogate (maximize the softmin of
the pairwise angles, or minimize `sum 1/d^p` with a large `p`) followed by an
exact polish (SLSQP: maximize `t` subject to every pairwise angle `>= t`),
and restarts that keep the best. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,z` and
30 rows (`id` = 0..29), like `sample_submission.csv` (a weak valid
baseline: points on a few latitude rings).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier normalizes the rows, computes the minimum pairwise angle in degrees
and prints `val_score: <min angle>` (0.0 if invalid). Higher is better.

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
