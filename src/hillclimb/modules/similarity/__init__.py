"""Similarity scores: pluggable "how alike are these two solutions?" measures.

A score is a `SimilarityScore` subclass (`base.py` holds the contract:
`represent` one solution, `compare` two representations). Scores are named
like policies are:

- a registry name: `solution-card` (LLM method card → embedding → cosine),
  `api-calls` (imports + library calls, rename-invariant), `code-tokens`
  (token-set Jaccard, the similarity map's `structural` distance);
- a Python file, any name ending in `.py` (relative to the folder holding
  the hillclimb dir, like `search.policy`): the file sets
  `SIMILARITY_SCORE = <class>` or defines exactly one `SimilarityScore`
  subclass;
- `package.module:ClassName` for scores shipped in an installed package;
- or `register_score(cls)` from code that embeds hillclimb.

`similarity.scores` in config.yaml maps the names to their params; the
`hillclimb similarity scores` command computes the pairwise matrices.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hillclimb.modules import refs
from hillclimb.modules.similarity.base import (
    SimilarityScore,
    SimilarityUnavailable,
    Solution,
    cosine,
    default_compare,
    jaccard,
    sparse_cosine,
)
from hillclimb.modules.similarity.builtin import ApiCalls, CodeTokens
from hillclimb.modules.similarity.compute import SimilarityMatrix, representations, similarity_matrix
from hillclimb.modules.similarity.solution_card import SolutionCard

__all__ = [
    "SimilarityMatrix",
    "SimilarityScore",
    "SimilarityUnavailable",
    "Solution",
    "cosine",
    "default_compare",
    "get_score",
    "jaccard",
    "load_score_file",
    "register_score",
    "registered_scores",
    "representations",
    "similarity_matrix",
    "sparse_cosine",
]

KIND = "similarity"
SCORE_ATTR = refs.KINDS[KIND].attr  # SIMILARITY_SCORE

# the registry itself lives with every other kind's, in modules/refs.py
_SCORES: dict[str, type[SimilarityScore]] = refs.KINDS[KIND].registry


def register_score(cls: type[SimilarityScore], name: str | None = None) -> type[SimilarityScore]:
    """Add a score class to the registry (usable as a decorator)."""
    key = name or cls.name
    if not key:
        raise ValueError(f"{cls.__name__} has no name; set `name = ...` or pass one")
    refs.register(KIND, key, cls)
    return cls


def registered_scores() -> dict[str, type[SimilarityScore]]:
    return refs.registered(KIND)


for _builtin in (SolutionCard, ApiCalls, CodeTokens):
    register_score(_builtin)


def load_score_file(path: Path) -> type[SimilarityScore]:
    """The score class a `.py` file exposes. Import errors and a missing or
    ambiguous class surface as ValueError naming the file."""
    return refs.resolve_ref(str(path), KIND).target


def get_score(
    name: str, params: Mapping[str, Any] | None = None, *, base_dir: Path | None = None,
) -> SimilarityScore:
    """A score instance by registry name, `.py` path (`base_dir` anchors a
    relative one — pass `climber.climber_base_dir(config)`), or
    `module:Class`. A class without a `name` is named after the file stem
    or class, so matrices and caches stay labelled."""
    resolved = refs.resolve_ref(name, KIND, base_dir=base_dir)
    score = resolved.target(dict(params or {}))
    if not score.name:
        score.name = resolved.label  # type: ignore[misc] — instance attribute shadows the ClassVar
    return score
