# 0/1 knapsack: maximize the packed value

Solve a deterministic batch of **24 zero-one knapsack instances**. Each item
has an integer weight and value; an item may be selected once or not at all.
The total selected weight must not exceed the instance's capacity.

The instances contain 160 items and mix ordinary random cases with structured
"density traps." A value-per-weight greedy solver is a strong start, but on the
structured cases a locally attractive item can block a much better combination.
Dynamic programming, branch-and-bound, meet-in-the-middle techniques, local
replacement, and hybrids are all useful directions.

## Scoring

For each instance, the selected value is divided by its fractional-knapsack
upper bound. The final score is the mean percentage across all 24 instances,
so **100 is an upper bound and higher is better**. The evaluator report shows
the six weakest instances to make targeted improvement possible.

The search climbs on one fixed validation batch. A second deterministic batch
is used as a holdout when Hillclimb selects the final candidate. Both batches
contain the same mixture of instance families.

Your submission is a Python module implementing the function in `contract.md`.
The evaluator imports it and calls it once per instance. There are no data
files and no network access.
