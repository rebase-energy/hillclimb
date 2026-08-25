# Circle packing: maximize the sum of radii

Place **exactly 26 circles inside the unit square** `[0, 1] x [0, 1]` so that the
**sum of all radii is as large as possible**.

Constraints (all verified programmatically):

- every circle lies entirely inside the unit square: `r <= x <= 1 - r` and
  `r <= y <= 1 - r`
- no two circles overlap: `dist(center_i, center_j) >= r_i + r_j`
- all radii are non-negative
- exactly 26 rows

This is a hard continuous optimization problem. The comparison targets are
**2.6359773947566274** from a reported OpenEvolve run and
**2.6359830849176067** from AlphaEvolve's current published construction.
Good approaches combine constructive patterns (hexagonal/greedy
layouts, unequal radii), local optimization (e.g. SLSQP / projected gradient /
physics-style relaxation), and restarts. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,r` and
26 rows (`id` = 0..25), like `sample_submission.csv` (which is a weak valid
baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the constraints and prints `val_score: <sum of radii>` (or a
score of 0.0 with a reason if the packing is invalid). Higher is better.

There is no train/test data; this is a pure optimization problem. Your script
should finish well within the execution time limit — budget your optimization
loop accordingly (e.g. a few minutes of solver time at most).
