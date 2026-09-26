# Thomson problem: 100 charges on the sphere

Place **exactly 100 unit point charges on the unit sphere** so that their
**Coulomb energy** `U = sum over the C(100,2) = 4950 pairs of 1 / |p_i - p_j|`
(units `e = 1`, `k_e = 1`) is **as small as possible**.

Constraints (verified programmatically):

- exactly 100 rows with finite coordinates
- every row is a non-zero vector; the verifier **normalizes each row to unit
  length**, so you may submit any non-zero direction
- no two charges may coincide (that makes the energy infinite)

This is a classic hard continuous optimization problem. The best known energy for n = 100 is **4448.350634** (Wikipedia Thomson table).
The landscape has many local minima that differ only in the 3rd–4th decimal,
so a precise local optimizer matters as much as the start. Good approaches: a
Fibonacci-spiral or random start, then gradient descent / L-BFGS with the
gradient projected onto the tangent plane and the points renormalized after
every step, basin hopping or simulated annealing over the local minima, and
restarts that keep the best. `numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,x,y,z` and
100 rows (`id` = 0..99), like `sample_submission.csv` (a weak valid
baseline: points on a few latitude rings).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier normalizes the rows, computes the Coulomb energy and prints
`val_score: <energy>` (100000.0 if invalid). Lower is better.

There is no train/test data; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
