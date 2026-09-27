# Multidimensional knapsack: 100 items, 5 constraints

`data/items.csv` lists **100 items** (`id,value,w1..w5`): each has an integer
value and 5 integer weights, one per constraint. `data/capacities.csv` gives
the 5 capacities (`constraint,capacity`). **Choose a subset of the items of
the largest total value** such that, for every constraint, the sum of the
chosen items' weights stays within its capacity.

This is instance **5.100-00** of Chu & Beasley's benchmark set (OR-Library file
`mknapcb1.txt`, tightness ratio 0.25: each capacity is a quarter of the total
weight on that constraint), fixed, so the score is fully reproducible. The
single-constraint knapsack falls to dynamic programming; with 5 capacities
at once the problem is NP-hard in a way that matters at this size. The LP
relaxation is **24585.9**, an upper bound no subset reaches; the best known
feasible value is **24381** (Chu & Beasley 1998). A greedy by value per unit
of surrogate weight lands a few percent below it; tabu search, genetic
algorithms with a repair operator, and add/drop/swap local search with
restarts close most of the gap. `numpy` and `pandas` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,take`:
100 rows, `id` = 0..99, `take` = 1 for a chosen item and 0 otherwise.
`sample_submission.csv` takes nothing (a valid, worthless baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier checks every constraint and prints `val_score: <total value>`, with
the slack left on each constraint so an improve step sees where the room
is (or a score of 0.0 if any constraint is exceeded or the file is
malformed). **Higher is better.**

There is no train/test split; this is a pure optimization problem. Keep total
runtime well within the execution time limit.
