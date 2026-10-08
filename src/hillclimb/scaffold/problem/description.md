# Number partitioning: split each list into two equal halves

Each instance is a list of 60 large positive integers. Split it into two
groups whose sums are as close as possible. Every number goes into exactly
one group.

## Scoring

For each instance the imbalance is the absolute difference between the two
group sums. The score is the mean of `log10(1 + imbalance)` over the
instances, so **lower is better** and 0 is a perfect split of every
instance. Each step of 1 is a tenfold smaller imbalance.

The baseline puts each number, largest first, into the group with the
smaller sum. The differencing method (Karmarkar–Karp), complete
anytime search, and local improvement by swapping numbers between the
groups all do much better.

The instances are fixed: the same 20 lists on every run, built from a seed
in `instances.py`.
