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

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hillclimb._moved import modernize
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

SCORE_FILE_SUFFIX = ".py"
SCORE_ATTR = "SIMILARITY_SCORE"

_SCORES: dict[str, type[SimilarityScore]] = {}


def register_score(cls: type[SimilarityScore], name: str | None = None) -> type[SimilarityScore]:
    """Add a score class to the registry (usable as a decorator)."""
    key = name or cls.name
    if not key:
        raise ValueError(f"{cls.__name__} has no name; set `name = ...` or pass one")
    _SCORES[key] = cls
    return cls


def registered_scores() -> dict[str, type[SimilarityScore]]:
    return dict(_SCORES)


for _builtin in (SolutionCard, ApiCalls, CodeTokens):
    register_score(_builtin)


def _is_score_class(obj: Any, module_name: str) -> bool:
    return (
        inspect.isclass(obj)
        and issubclass(obj, SimilarityScore)
        and obj is not SimilarityScore
        and obj.__module__ == module_name
    )


def load_score_file(path: Path) -> type[SimilarityScore]:
    """The score class a `.py` file exposes. Import errors and a missing or
    ambiguous class surface as ValueError naming the file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"similarity score file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    module_name = f"hillclimb_similarity_{path.stem}_{digest}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import similarity score file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — user code
            del sys.modules[module_name]
            raise ValueError(f"similarity score file {path} failed to import: {type(exc).__name__}: {exc}") from exc
    target = getattr(module, SCORE_ATTR, None)
    if target is None:
        classes = [obj for obj in vars(module).values() if _is_score_class(obj, module_name)]
        if len(classes) != 1:
            raise ValueError(
                f"similarity score file {path} must define exactly one SimilarityScore subclass "
                f"(found {[c.__name__ for c in classes]}) or set {SCORE_ATTR} = <class>"
            )
        target = classes[0]
    if not (inspect.isclass(target) and issubclass(target, SimilarityScore)):
        raise ValueError(f"{SCORE_ATTR} in {path} is not a SimilarityScore subclass: {target!r}")
    return target


def _load_import_path(spec: str) -> type[SimilarityScore]:
    module_name, _, attr = modernize(spec).partition(":")  # a ref written before a move
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import similarity score {spec!r}: {exc}") from exc
    if not (inspect.isclass(target) and issubclass(target, SimilarityScore)):
        raise ValueError(f"{spec!r} is not a SimilarityScore subclass")
    return target


def get_score(
    name: str, params: Mapping[str, Any] | None = None, *, base_dir: Path | None = None,
) -> SimilarityScore:
    """A score instance by registry name, `.py` path (`base_dir` anchors a
    relative one — pass `policies.policy_base_dir(config)`), or
    `module:Class`. A class without a `name` is named after the file stem
    or class, so matrices and caches stay labelled."""
    if name.endswith(SCORE_FILE_SUFFIX):
        path = Path(name).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        cls = load_score_file(path)
        fallback = path.stem
    elif ":" in name:
        cls = _load_import_path(name)
        fallback = cls.__name__
    elif name in _SCORES:
        cls = _SCORES[name]
        fallback = name
    else:
        raise ValueError(
            f"unknown similarity score {name!r} (available: {', '.join(sorted(_SCORES))}, "
            "a path to a .py file, or module:Class)"
        )
    score = cls(dict(params or {}))
    if not score.name:
        score.name = fallback  # type: ignore[misc] — instance attribute shadows the ClassVar
    return score
