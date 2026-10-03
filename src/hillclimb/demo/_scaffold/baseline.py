"""Baseline: largest first, each number into the group with the smaller sum."""


def partition(numbers: list[int]) -> list[int]:
    first, first_sum, second_sum = [], 0, 0
    for index in sorted(range(len(numbers)), key=lambda i: numbers[i], reverse=True):
        if first_sum <= second_sum:
            first.append(index)
            first_sum += numbers[index]
        else:
            second_sum += numbers[index]
    return first
