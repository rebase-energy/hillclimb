"""Representations through the machine cache, and the pairwise matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hillclimb.similarity_scores.base import SimilarityScore, Solution


def cache_root() -> Path:
    return Path.home() / ".cache" / "hillclimb" / "similarity"


class DiskCache:
    """JSON values under `cache_root()/<namespace>/<key[:2]>/<key>.json`.
    Unreadable or corrupt entries are misses; write failures are ignored
    (a cache must never fail the computation it speeds up)."""

    def __init__(self, namespace: str):
        self.dir = cache_root() / namespace.replace("/", "_")

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any:
        try:
            return json.loads(self._path(key).read_text())["value"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"value": value}))
            tmp.replace(path)
        except (OSError, TypeError, ValueError):
            pass


def representations(score: SimilarityScore, solutions: Sequence[Solution]) -> list[Any]:
    """`score.represent_many` over the solutions the cache does not already
    hold; one entry per solution, None where it could not be represented."""
    out: list[Any] = [None] * len(solutions)
    cache = DiskCache(score.name or type(score).__name__) if score.cache else None
    keys: list[str | None] = [None] * len(solutions)
    misses: list[int] = []
    for i, solution in enumerate(solutions):
        if cache is not None:
            ident = score.cache_key(solution)
            if ident is not None:
                keys[i] = DiskCache.key(score.name, score.version, score.params, ident)
                hit = cache.get(keys[i])
                if hit is not None:
                    out[i] = hit
                    continue
        misses.append(i)
    if misses:
        fresh = score.represent_many([solutions[i] for i in misses])
        if len(fresh) != len(misses):
            raise ValueError(
                f"{type(score).__name__}.represent_many returned {len(fresh)} values for {len(misses)} solutions"
            )
        for i, value in zip(misses, fresh):
            out[i] = value
            if cache is not None and keys[i] is not None and value is not None:
                cache.put(keys[i], value)
    return out


@dataclass(frozen=True)
class SimilarityMatrix:
    score: str
    ids: tuple[str, ...]
    values: np.ndarray               # (n, n); NaN in rows/cols of unrepresented solutions
    unrepresented: tuple[str, ...]

    def get(self, a: str, b: str) -> float:
        return float(self.values[self.ids.index(a), self.ids.index(b)])

    def distances(self) -> np.ndarray:
        """1 - similarity, floored at 0 — what a layout or clustering wants."""
        return np.maximum(1.0 - self.values, 0.0)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "ids": list(self.ids),
            "similarity": [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in self.values],
            "unrepresented": list(self.unrepresented),
        }


def similarity_matrix(score: SimilarityScore, solutions: Sequence[Solution]) -> SimilarityMatrix:
    reps = representations(score, solutions)
    n = len(solutions)
    values = np.full((n, n), np.nan)
    for i in range(n):
        if reps[i] is None:
            continue
        values[i, i] = 1.0
        for j in range(i + 1, n):
            if reps[j] is not None:
                values[i, j] = values[j, i] = float(score.compare(reps[i], reps[j]))
    return SimilarityMatrix(
        score=score.name or type(score).__name__,
        ids=tuple(s.id for s in solutions),
        values=values,
        unrepresented=tuple(s.id for s, r in zip(solutions, reps) if r is None),
    )
