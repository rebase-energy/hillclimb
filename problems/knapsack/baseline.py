"""Baseline: greedy selection by value density."""


def select_items(items: list[tuple[int, int]], capacity: int) -> list[int]:
    order = sorted(
        range(len(items)),
        key=lambda index: (items[index][1] / items[index][0], items[index][1]),
        reverse=True,
    )
    selected: list[int] = []
    remaining = capacity
    total_value = 0
    for index in order:
        weight, value = items[index]
        if weight <= remaining:
            selected.append(index)
            remaining -= weight
            total_value += value

    # The usual guard for the classic greedy approximation: one large item
    # can be worth more than the whole density-ordered packing.
    feasible = [index for index, (weight, _) in enumerate(items) if weight <= capacity]
    if feasible:
        best_single = max(feasible, key=lambda index: items[index][1])
        if items[best_single][1] > total_value:
            return [best_single]
    return selected
