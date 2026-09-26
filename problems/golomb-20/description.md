# Golomb ruler with 20 marks

Find **20 distinct non-negative integers** (the marks of a ruler), the first
of them 0, such that all C(20,2) = 190 **pairwise differences are distinct**,
and make the ruler **as short as possible**: the score is its length, the
largest mark.

Constraints (verified programmatically):

- exactly 20 rows, integer marks with `0 <= mark < 1000000`
- mark 0 is present, no mark repeats
- no two pairs of marks have the same difference

The optimal length for m = 20 is **283** (Garry, Vanderschel et al. 1997; proven optimal, table of optimal Golomb rulers on Wikipedia), so no ruler can score below it.
The score is an integer, so most edits are plateaus: a candidate that keeps
the length is a tie, not a loss, and progress comes in whole units. Good
approaches: fix a target length L and search for a placement of the inner
marks (constraint propagation / backtracking over the difference table,
simulated annealing or tabu search on mark positions with the number of
repeated differences as the cost), start from affine or projective-plane
constructions (Singer, Bose–Chowla) that give near-optimal rulers directly
and then shrink, and exploit the mirror symmetry (`L - mark` is a ruler too).
`numpy` and `scipy` are available.

## Submission format

Write `submission.csv` in the working directory with the header `id,mark` and
20 rows (`id` = 0..19, one integer mark each, any order), like
`sample_submission.csv` (a weak valid baseline: the greedy ruler).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. The
verifier validates the ruler and prints `val_score: <length>` (1000000 if
invalid). Lower is better.

There is no train/test data; this is a pure construction problem. Keep total
runtime well within the execution time limit.
