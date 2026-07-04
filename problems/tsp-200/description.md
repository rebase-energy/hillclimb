# Traveling salesman: 200 cities

`data/cities.csv` contains **200 cities** (`id,x,y`, coordinates in the unit
square, fixed instance). Find a **closed tour visiting every city exactly once**
with the **shortest possible total Euclidean length** (the tour returns from the
last city to the first).

This instance is fixed and seeded — the score is fully reproducible. A random
order is around 100; nearest-neighbor lands near 12; good local search (2-opt /
Or-opt, candidate lists, restarts or simulated annealing) pushes toward ~10.
`numpy`, `scipy` and `pandas` are available.

## Submission format

Write `submission.csv` in the working directory with the header `position,city`:
200 rows where `position` = 0..199 and `city` is a permutation of the city ids
0..199, in visiting order. `sample_submission.csv` is the identity order (a weak
valid baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the tour and prints `val_score: <tour length>` (or a large
penalty score of 1000.0 if the tour is invalid). **Lower is better.**

There is no train/test split; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
