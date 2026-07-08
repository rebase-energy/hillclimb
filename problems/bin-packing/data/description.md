# One-dimensional bin packing

Pack items into as few unit-capacity bins as possible.

Your solution is a Python module defining a `pack` function (see the solution
contract). The evaluator runs it on **50 fixed instances** — 120 items each,
sizes drawn uniformly from [0.05, 0.7], capacity 1.0 — and scores the **mean
number of bins** across instances (lower is better). A packing is valid only
if every bin's total fits the capacity and the bins exactly partition the
input items; any invalid packing fails the whole evaluation.

Selection additionally uses a hidden holdout instance set (same generator,
different seed), so overfitting the visible instances doesn't pay.

The theoretical floor for each instance is `ceil(sum(items) / capacity)`;
the evaluator's report shows the instances furthest above their floor —
that's where a better algorithm gains the most. First-fit decreasing is the
shipped baseline; beating it takes smarter placement (best-fit variants,
lookahead, local search on the worst instances, ...).
