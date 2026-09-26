# Kissing configuration in dimension 11

Find **as many nonzero integer points as possible** in `Z^11` such that every
pairwise distance is at least the largest norm among the points:

    min over i != j of ||p_i - p_j||  >=  max over i of ||p_i||

Unit spheres centred at `2 p_i / ||p_i||` then all touch the unit sphere at the
origin without overlapping, so the number of points N is a lower bound on the
kissing number in dimension 11. The score is N.

Constraints (verified programmatically, exactly, on squared integer norms):

- every coordinate is an integer with `|c| <= 100000000`
- no point is the origin, no two points are equal
- `min_(i != j) ||p_i - p_j||^2 >= max_i ||p_i||^2`
- at most 2000 rows

The best known configuration in dimension 11 has **593 points** (AlphaEvolve 2025, arXiv:2506.13131 Appendix B.11; the previous record was 592, Ganzhinov 2022).
The score is an integer count, so most edits are plateaus: a candidate that
keeps N is a tie, not a loss, and progress comes in unit steps. Good approaches:
start from lattice shells (the 220 points `±e_i ± e_j` of `D_11` already
satisfy the condition), scale a configuration up so there is integer room to
insert extra points, and then search for insertions/replacements that keep
the min-distance inequality, repairing the worst pair after each move. The
check is cheap (a Gram matrix), so many local moves fit in the time budget.
`numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10`
and one row per point (`id` = 0..N-1, any N >= 1, integer coordinates), like
`sample_submission.csv` (a weak valid baseline: the 22 points `±e_i`).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the configuration and prints `val_score: <N>` (0 if
invalid). Higher is better.

There is no train/test data; this is a pure construction problem. Keep total
runtime well within the execution time limit.
