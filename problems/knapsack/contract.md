`solution.py` must define this function:

```python
def select_items(items: list[tuple[int, int]], capacity: int) -> list[int]:
    """Return the indices of the items selected for this knapsack."""
```

Each item is `(weight, value)`, and both numbers are positive integers.

Requirements:

- return a list of unique integer indices into `items`
- the sum of the selected weights must be at most `capacity`
- each item is zero-one: it may be selected at most once
- do not modify files, use the network, or rely on state between calls
- be deterministic for a given `(items, capacity)` input

The evaluator calls `select_items` 24 times per split with 160 items per call.
Keep the complete batch comfortably inside the problem's 60-second time limit.
