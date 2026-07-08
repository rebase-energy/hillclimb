`solution.py` must define exactly this function:

```python
def pack(items: list[float], capacity: float) -> list[list[float]]:
    """Partition `items` into bins; each bin's sum must be <= capacity."""
```

Requirements:
- The returned bins must contain exactly the input items (a permutation into
  groups — no dropping, duplicating, or altering values; tolerance 1e-9).
- Every bin's total must be <= capacity (tolerance 1e-9).
- Pure computation: no file or network I/O, deterministic for a given input.
- The evaluator calls `pack` 50 times (120 items each); the total run must
  stay well inside the time limit, so keep per-call complexity sane.
