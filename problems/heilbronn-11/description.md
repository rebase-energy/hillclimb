# Heilbronn triangle problem: 11 points

Place **exactly 11 points inside the unit square** `[0, 1] x [0, 1]` so that the
**smallest triangle** formed by any 3 of the points has **as large an area as
possible** (a max-min objective over all C(11,3) = 165 triangles).

Constraints (verified programmatically):

- all points inside the unit square (`0 <= x, y <= 1`)
- exactly 11 rows
- no 3 points may be exactly collinear (that makes the minimum area 0)

This is a classic hard continuous optimization problem. The best known value for n = 11 is about **0.0370** (Goldberg 1972).
Regular grids are terrible (many nearly collinear triples). Good approaches:
random/structured starts + local optimization of the max-min objective (e.g.
maximize the smallest area with SLSQP or a soft-min surrogate), simulated
annealing on point positions, and restarts. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y` and 11
rows (`id` = 0..10), like `sample_submission.csv` (a weak valid baseline:
points on a parabola).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the layout and prints `val_score: <min triangle area>` (0.0
if invalid). Higher is better.

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
