"""Baseline: first-fit decreasing."""


def pack(items: list[float], capacity: float) -> list[list[float]]:
    bins: list[list[float]] = []
    totals: list[float] = []
    for item in sorted(items, reverse=True):
        for index, total in enumerate(totals):
            if total + item <= capacity + 1e-12:
                bins[index].append(item)
                totals[index] = total + item
                break
        else:
            bins.append([item])
            totals.append(item)
    return bins
