"""Score direction is `higher_is_better` everywhere (hillclimb climbs). Files
written before the rename — problem.yaml, search.json, knowledge cards,
skill metadata, eval_result.json reports — carry the inverse
(lower-is-better) key; `legacy_direction_key` maps it on load so old data keeps working."""

from __future__ import annotations


def better(a: float, b: float, higher_is_better: bool) -> bool:
    """Is score `a` an improvement over `b` in the metric's direction?"""
    return a > b if higher_is_better else a < b


def legacy_direction_key(data):
    """Return `data` with any legacy-key (the inverse) folded into
    `higher_is_better` (inverted). Non-dicts pass through untouched; an
    explicit `higher_is_better` wins when both are present."""
    if not isinstance(data, dict) or "lower_is_better" not in data:  # legacy-key
        return data
    data = dict(data)
    lower = data.pop("lower_is_better")  # legacy-key
    data.setdefault("higher_is_better", not bool(lower))
    return data
