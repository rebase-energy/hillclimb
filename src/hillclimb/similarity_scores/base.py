"""The similarity-score contract: what one implementation must provide.

A score answers "how alike are these two solutions?" in two steps, so the
expensive part runs once per solution rather than once per pair:

1. `represent(solution)` turns ONE solution into something comparable — a
   vector, a sparse `{feature: weight}` dict, a set, or anything the score's
   own `compare` understands. `None` means "cannot represent this one"
   (unparsable source, missing file); that solution simply has no row.
2. `compare(a, b)` turns two representations into a similarity: higher is
   more alike, 1.0 is "the same". The default handles the three common
   shapes (cosine for vectors and sparse dicts, Jaccard for sets), so most
   scores only write `represent`.

`represent_many` exists for batch APIs (one embeddings request for many
texts); the default loops over `represent`. A score that raises
`SimilarityUnavailable` is unusable as a whole (no API key, missing extra)
— any other exception is that one solution's problem and leaves it
unrepresented.

With `cache = True`, representations persist in the machine cache
(`~/.cache/hillclimb/similarity/`), keyed by the score's name, `version`,
params and `cache_key(solution)` (the solution file's bytes by default) —
bump `version` when `represent` changes meaning. Cached values must be
JSON-serializable. Nothing is ever written into a run: similarities are
derived views, like the rest of `hillclimb similarity`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from hillclimb.candidate import Candidate

SOLUTION_FILE = "solution.py"


class SimilarityUnavailable(RuntimeError):
    """The score cannot run at all here (missing key, extra, or setup)."""


@dataclass(eq=False)
class Solution:
    """One thing to compare: a directory holding a solution file, and the
    journaled candidate when it came from a search."""

    id: str
    dir: Path
    file: str = SOLUTION_FILE
    candidate: Candidate | None = None
    _bytes: bytes | None = field(default=None, repr=False)

    @classmethod
    def from_file(cls, path: Path, id: str | None = None) -> Solution:
        path = Path(path)
        return cls(id=id or str(path), dir=path.parent, file=path.name)

    @property
    def path(self) -> Path:
        return self.dir / self.file

    def read_bytes(self) -> bytes:
        if self._bytes is None:
            self._bytes = self.path.read_bytes()
        return self._bytes

    @property
    def source(self) -> str:
        return self.read_bytes().decode(errors="replace")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.read_bytes()).hexdigest()


class SimilarityScore:
    """Subclass, set `name`, implement `represent` (and `compare` when the
    representation is not a vector, dict or set)."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    cache: ClassVar[bool] = False
    defaults: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, params: Mapping[str, Any] | None = None):
        unknown = set(params or {}) - set(self.defaults)
        if self.defaults and unknown:
            raise ValueError(
                f"similarity score {self.name!r}: unknown params {sorted(unknown)} "
                f"(accepted: {sorted(self.defaults)})"
            )
        self.params: dict[str, Any] = {**self.defaults, **(params or {})}

    def represent(self, solution: Solution) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement represent(solution)")

    def represent_many(self, solutions: Sequence[Solution]) -> list[Any]:
        out = []
        for solution in solutions:
            try:
                out.append(self.represent(solution))
            except SimilarityUnavailable:
                raise
            except Exception:  # noqa: BLE001 — one solution's failure, not the score's
                out.append(None)
        return out

    def compare(self, a: Any, b: Any) -> float:
        return default_compare(a, b)

    def cache_key(self, solution: Solution) -> str | None:
        """What identifies a solution's representation besides the score
        and its params; None = never cache this one."""
        try:
            return solution.digest
        except OSError:
            return None

    def explain(self, solution: Solution) -> str | None:
        """Optional human-readable view of a representation (e.g. the card
        an LLM wrote); None when the score has nothing to show."""
        return None


# ---------------------------------------------------------------------------
# default comparisons


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"cosine over vectors of different length ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def sparse_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    dot = sum(value * b[key] for key, value in a.items() if key in b)
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def jaccard(a: frozenset | set, b: frozenset | set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def default_compare(a: Any, b: Any) -> float:
    """Cosine for vectors and `{feature: weight}` dicts, Jaccard for sets."""
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return jaccard(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return sparse_cosine(a, b)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
        return cosine([float(x) for x in a], [float(x) for x in b])
    try:  # numpy arrays and other array-likes
        return cosine(list(map(float, a)), list(map(float, b)))
    except TypeError as exc:
        raise TypeError(
            f"no default comparison for {type(a).__name__}; override compare(a, b)"
        ) from exc
