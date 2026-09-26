# Circle packing: 32 circles, maximize the sum of radii

Place **exactly 32 circles inside the unit square** `[0, 1] x [0, 1]` so that the
**sum of all radii is as large as possible**.

Constraints (all verified programmatically):

- every circle lies entirely inside the unit square: `r <= x <= 1 - r` and
  `r <= y <= 1 - r`
- no two circles overlap: `dist(center_i, center_j) >= r_i + r_j`
- all radii are non-negative
- exactly 32 rows

This is a hard continuous optimization problem. The best known value for n = 32 is about **2.939572** (Georgiev et al. 2025); AlphaEvolve 2025 reported **2.937945**.
Equal circles on a grid are weak (the corners and the gaps between cells are
wasted). Good approaches combine constructive patterns (hexagonal / greedy
layouts, unequal radii — a few large circles surrounded by small ones), local
optimization of the sum under the non-overlap constraints (e.g. SLSQP,
projected gradient, or a physics-style relaxation that pushes overlapping
circles apart while inflating them), and restarts. `numpy` and `scipy` are
available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,r` and
32 rows (`id` = 0..31), like `sample_submission.csv` (a weak valid
baseline: 32 equal circles on a 6 x 6 grid).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the constraints (with a numerical tolerance of 1e-9) and
prints `val_score: <sum of radii>` (or a score of 0.0 with a reason if the
packing is invalid). Higher is better.

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
