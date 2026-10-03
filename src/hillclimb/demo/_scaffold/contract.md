`solution.py` must define this function:

```python
def partition(numbers: list[int]) -> list[int]:
    """Return the indices of the numbers that go in the first group."""
```

Requirements:

- return a list of unique integer indices into `numbers`; every other number
  is in the second group
- do not modify files, use the network, or rely on state between calls
- be deterministic for a given input

The runner calls `partition` 20 times with 60 numbers each. The whole batch
must finish within the problem's 60-second time limit.
