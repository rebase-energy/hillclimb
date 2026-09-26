"""`hillclimb similarity map` data: candidates embedded by pairwise distance.

Where `similarity.py`'s reference cube puts every candidate at its distance
from ONE candidate, the map compares every candidate with every other and
lays them out so that screen distance approximates pairwise distance:
two dots close together really are similar solutions. Same inputs — the
behavioral prints (problem fingerprint, submission, or evaluator report,
in the mode the search's origin settles), solution.py tokens, and the
exploration tree — so the two views agree on what "behavioral" means and
read through the same caches; nothing is stored.

Three pairwise matrices, one per distance, and a fourth that blends them:

- **behavioral**: RMS between prints in units of the origin's own spread
  (fingerprint and submission modes: one global scale, the origin's MAD —
  the same unit the cube's behavioral axis has; report mode: per-key
  relative to the origin's value, clipped, over the keys a pair shares).
- **structural**: Jaccard distance between solution token sets.
- **lineage**: hops through the lowest common ancestor; disjoint trees
  connect through a virtual super-root, so the run scope's searches sit
  two hops apart at the root and never further.
- **blend**: the mean of the three, each normalized by its 95th
  percentile, so no one unit dominates.

The layout is classical multidimensional scaling (Torgerson): the top
three eigenpairs of the double-centred squared-distance matrix. The
`stress` it reports is the relative residual — how much of the pairwise
structure the three axes fail to show. Eigenvector signs are pinned
(largest-magnitude entry positive) so a rebuild from the same data is the
same picture, and a caller can pass the previous layout to align the new
one onto it by orthogonal Procrustes over the ids both share: a live
search grows in place instead of flipping on every new candidate.

`similarity_mapview.py` renders this; both stay replay-only readers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hillclimb.candidate import Candidate
from hillclimb.tui.similarity import (
    CLIP,
    DEFAULT_ARTIFACTS,
    EPS,
    MIN_SHARED_FRAC,
    MIN_SHARED_KEYS,
    N_BINS,
    FingerprintError,
    Prints,
    Reference,
    SearchInput,
    _live_paths,
    _load_fingerprinter,
    _prune_caches,
    _reference_prints,
    _resolve_reference,
    _run_champion,
    candidate_prints,
    shared_seed_references,
    spread,
    structural_distance,
    submission_vector,
)
from hillclimb.tui.tree import accepted_lineage, build_tree

METRICS = ("behavioral", "structural", "blend")
DIM = 3
AXIS_P95 = 95.0


@dataclass(frozen=True)
class MapNode:
    id: str
    x: float
    y: float
    z: float
    score: float | None
    bin: int | None          # 0 (worst) .. N_BINS-1 (best) over scored nodes; None = unscored
    best: bool               # the current champion
    origin: bool             # the seed/baseline the search grew from
    fate: str                # tree.py vocabulary
    on_path: bool
    operator: str
    parent_id: str | None    # namespaced like `id` in run scope
    search_id: str = ""
    arm: str | None = None


@dataclass(frozen=True)
class MapView:
    nodes: tuple[MapNode, ...]
    edges: tuple[tuple[int, int], ...]   # (parent index, child index) into `nodes`
    trail: tuple[int, ...]               # node indices of the best-so-far sequence, in accept order
    metric: str                          # METRICS
    mode: str                            # "fingerprint" | "submission" | "report"
    stress: float                        # relative MDS residual, 0 = the picture is exact
    scales: tuple[float, float, float]   # p95 of the behavioral / structural / lineage matrices
    n_unpositioned: int
    unavailable: str | None = None
    behavioral: np.ndarray | None = None  # the pairwise matrices, node order — for readouts
    structural: np.ndarray | None = None
    lineage: np.ndarray | None = None
    scope: str = "search"                 # "search" | "run"
    arms: tuple[str, ...] = ()
    n_searches: int = 1
    problem_key: str = ""

    @classmethod
    def none(cls, metric: str, reason: str, scope: str = "search") -> MapView:
        return cls(
            nodes=(), edges=(), trail=(), metric=metric, mode="", stress=0.0,
            scales=(0.0, 0.0, 0.0), n_unpositioned=0, unavailable=reason, scope=scope,
        )

    def positions(self) -> dict[str, tuple[float, float, float]]:
        """The layout as a caller passes it back in as `previous`."""
        return {n.id: (n.x, n.y, n.z) for n in self.nodes}

    def index(self, node_id: str) -> int | None:
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                return i
        return None


# ---------------------------------------------------------------------------
# pairwise matrices


def _vector_matrix(vectors: Sequence[np.ndarray], scale: float) -> np.ndarray:
    """Pairwise RMS between equal-length vectors, in units of `scale`. Row
    by row so a duplicate print lands at exactly zero (the Gram-matrix
    shortcut would not), and so only one row's worth of differences is
    ever in memory."""
    n = len(vectors)
    out = np.zeros((n, n))
    if n == 0:
        return out
    stack = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
    for i in range(n - 1):
        diff = (stack[i + 1:] - stack[i]) / scale
        out[i, i + 1:] = np.sqrt(np.mean(np.square(diff, dtype=np.float64), axis=1))
    return out + out.T


def _report_matrix(reports: Sequence[Mapping[str, float]], origin: Mapping[str, float]) -> np.ndarray:
    """Pairwise report distance over the origin's keys: each key scaled by
    the origin's value (so the matrix is symmetric), clipped, averaged over
    the keys a pair shares. A pair sharing too few keys is imputed at the
    largest finite distance rather than dropped."""
    keys = sorted(origin)
    denom = np.asarray([max(abs(origin[k]), EPS) for k in keys])
    table = np.full((len(reports), len(keys)), np.nan)
    for i, report in enumerate(reports):
        for j, key in enumerate(keys):
            if key in report:
                table[i, j] = report[key] / denom[j]
    n = len(reports)
    out = np.zeros((n, n))
    for i in range(n - 1):
        diff = np.minimum(np.abs(table[i + 1:] - table[i]), CLIP)
        shared = np.sum(~np.isnan(diff), axis=1)
        with np.errstate(invalid="ignore"):
            rms = np.sqrt(np.nanmean(np.square(diff), axis=1))
        rms[shared < MIN_SHARED_KEYS] = np.nan
        out[i, i + 1:] = rms
    out = out + out.T
    if np.isnan(out).any():
        finite = out[np.isfinite(out)]
        out = np.where(np.isnan(out), float(finite.max()) if finite.size else 0.0, out)
    return out


def _report_usable(report: Mapping[str, float], origin: Mapping[str, float]) -> bool:
    shared = len(set(report) & set(origin))
    return shared >= MIN_SHARED_KEYS and shared >= MIN_SHARED_FRAC * len(origin)


def behavioral_matrix(
    prints: Sequence[Prints], origin: Reference,
) -> tuple[np.ndarray, list[int]]:
    """The pairwise behavioral matrix over the prints that are comparable
    with the origin, and which input indices made it in (the rest have no
    position)."""
    mode = origin.mode
    if mode == "fingerprint":
        ref = np.asarray(origin.behavioral, dtype=np.float64)
        kept = [i for i, p in enumerate(prints) if np.asarray(p.behavioral).shape == ref.shape]
        vectors = [np.asarray(prints[i].behavioral) for i in kept]
        return _vector_matrix(vectors, spread(ref)), kept
    if mode == "submission":
        ref = submission_vector(origin.behavioral, origin.behavioral)
        assert ref is not None
        aligned = [(i, submission_vector(origin.behavioral, p.behavioral)) for i, p in enumerate(prints)]
        kept = [i for i, v in aligned if v is not None]
        vectors = [v for _, v in aligned if v is not None]
        return _vector_matrix(vectors, spread(ref)), kept
    origin_report = origin.behavioral  # type: ignore[assignment]
    kept = [i for i, p in enumerate(prints) if _report_usable(p.behavioral, origin_report)]  # type: ignore[arg-type]
    return _report_matrix([prints[i].behavioral for i in kept], origin_report), kept  # type: ignore[misc]


def structural_matrix(tokens: Sequence[frozenset[str]]) -> np.ndarray:
    n = len(tokens)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = structural_distance(tokens[i], tokens[j])
    return out


def lineage_matrix(ids: Sequence[str], parents: Mapping[str, str | None]) -> np.ndarray:
    """Pairwise hops through the lowest common ancestor, `parents` mapping
    every id to its parent (None at a root; an unknown parent is a root
    too). Disjoint trees meet at a virtual super-root, +2 — the rule
    `similarity.lineage_distance` uses."""
    chains: list[dict[str, int]] = []
    for cid in ids:
        chain: dict[str, int] = {}
        node: str | None = cid
        while node is not None and node not in chain:
            chain[node] = len(chain)
            node = parents.get(node)
        chains.append(chain)
    n = len(ids)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chains[i], chains[j]
            hops = None
            for cid, depth in a.items():
                if cid in b:
                    hops = depth + b[cid]
                    break
            if hops is None:
                hops = len(a) - 1 + len(b) - 1 + 2
            out[i, j] = out[j, i] = float(hops)
    return out


def _p95(matrix: np.ndarray) -> float:
    n = len(matrix)
    if n < 2:
        return EPS
    values = matrix[np.triu_indices(n, k=1)]
    return max(float(np.percentile(values, AXIS_P95)), EPS)


def blend_matrix(matrices: Sequence[np.ndarray], scales: Sequence[float]) -> np.ndarray:
    return sum(m / s for m, s in zip(matrices, scales)) / len(matrices)


# ---------------------------------------------------------------------------
# the layout


def classical_mds(distances: np.ndarray, dim: int = DIM) -> tuple[np.ndarray, float]:
    """Torgerson scaling: coordinates (n × dim) whose Euclidean distances
    best reproduce `distances` in the least-squares sense, plus the
    relative stress of that reproduction. Fewer than `dim` positive
    eigenvalues pad with zero columns; eigenvector signs are pinned so the
    same matrix always yields the same coordinates."""
    n = len(distances)
    if n == 0:
        return np.zeros((0, dim)), 0.0
    d2 = np.square(np.asarray(distances, dtype=np.float64))
    centering = np.eye(n) - np.full((n, n), 1.0 / n)
    b = -0.5 * centering @ d2 @ centering
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:dim]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    coords = vectors * np.sqrt(values)
    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((n, dim - coords.shape[1]))])
    for k in range(dim):
        column = coords[:, k]
        if column.size and column[np.argmax(np.abs(column))] < 0:
            coords[:, k] = -column
    total = float(np.sum(d2))
    if total <= 0.0:
        return coords, 0.0
    fitted = np.sqrt(np.maximum(
        np.sum(np.square(coords[:, None, :] - coords[None, :, :]), axis=2), 0.0,
    ))
    stress = float(np.sqrt(np.sum(np.square(fitted - distances)) / total))
    return coords, stress


def procrustes_align(
    coords: np.ndarray, ids: Sequence[str], previous: Mapping[str, tuple[float, float, float]],
) -> np.ndarray:
    """Rotate/reflect (never scale — distances mean something) and shift
    `coords` so the ids present in `previous` land as close as possible to
    where they were. With nothing shared the coordinates come back as
    they are."""
    shared = [i for i, cid in enumerate(ids) if cid in previous]
    if not shared:
        return coords
    source = coords[shared]
    target = np.asarray([previous[ids[i]] for i in shared], dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vt
    return (coords - source_mean) @ rotation + target_mean


# ---------------------------------------------------------------------------
# the view


@dataclass(frozen=True)
class _Member:
    """One candidate as the assembler carries it: which search, its
    namespaced ids, and its prints."""
    candidate: Candidate
    search_id: str
    arm: str | None
    node_id: str
    parent_id: str | None
    prints: Prints


def _assemble(
    members: list[_Member], origin: Reference, higher_is_better: bool, metric: str,
    previous: Mapping[str, tuple[float, float, float]] | None, *,
    best: set[str], origins: set[str], fates: Mapping[str, str], on_path: set[str],
    trail_ids: Sequence[str], n_missing: int, **view_fields,
) -> MapView:
    """Matrices → layout → nodes/edges/trail. `fates`, `on_path`, `best`,
    `origins`, `trail_ids` are keyed by node id."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    behavioral, kept = behavioral_matrix([m.prints for m in members], origin)
    n_missing += len(members) - len(kept)
    members = [members[i] for i in kept]
    structural = structural_matrix([m.prints.tokens for m in members])
    ids = [m.node_id for m in members]
    parents = {m.node_id: m.parent_id for m in members}
    lineage = lineage_matrix(ids, parents)
    scales = (_p95(behavioral), _p95(structural), _p95(lineage))
    if metric == "behavioral":
        matrix = behavioral
    elif metric == "structural":
        matrix = structural
    else:
        matrix = blend_matrix((behavioral, structural, lineage), scales)
    coords, stress = classical_mds(matrix)
    if previous:
        coords = procrustes_align(coords, ids, previous)

    scored = [
        m for m in members if m.candidate.val_score is not None and m.candidate.status == "passing"
    ]
    scored.sort(key=lambda m: m.candidate.val_score, reverse=not higher_is_better)  # type: ignore[arg-type,return-value]
    bins = {m.node_id: i * N_BINS // len(scored) for i, m in enumerate(scored)} if scored else {}
    nodes = tuple(
        MapNode(
            id=m.node_id, x=float(coords[i, 0]), y=float(coords[i, 1]), z=float(coords[i, 2]),
            score=m.candidate.val_score, bin=bins.get(m.node_id),
            best=m.node_id in best, origin=m.node_id in origins,
            fate=fates.get(m.node_id, "pending"), on_path=m.node_id in on_path,
            operator=m.candidate.operator, parent_id=m.parent_id,
            search_id=m.search_id, arm=m.arm,
        )
        for i, m in enumerate(members)
    )
    index = {node_id: i for i, node_id in enumerate(ids)}
    edges = tuple(
        (index[m.parent_id], index[m.node_id])
        for m in members if m.parent_id is not None and m.parent_id in index
    )
    trail = tuple(index[t] for t in trail_ids if t in index)
    return MapView(
        nodes=nodes, edges=edges, trail=trail, metric=metric, mode=origin.mode, stress=stress,
        scales=scales, n_unpositioned=n_missing,
        behavioral=behavioral, structural=structural, lineage=lineage, **view_fields,
    )


def _members(search: SearchInput, ref: Reference, fingerprinter, namespaced: bool) -> tuple[list[_Member], int]:
    prints = candidate_prints(search, ref, fingerprinter)

    def node_id(cid: str) -> str:
        return f"{search.search_id}/{cid}" if namespaced else cid

    members = [
        _Member(
            candidate=c, search_id=search.search_id, arm=search.arm,
            node_id=node_id(c.candidate_id),
            parent_id=node_id(c.parent_id) if c.parent_id else None,
            prints=prints[c.candidate_id],
        )
        for c in search.candidates if c.candidate_id in prints
    ]
    return members, len(search.candidates) - len(members)


def build_map(
    candidates: list[Candidate],
    search_dir: Path,
    higher_is_better: bool,
    metric: str = "behavioral",
    output_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
    fingerprint_path: Path | None = None,
    previous: Mapping[str, tuple[float, float, float]] | None = None,
) -> MapView:
    """One search's candidates embedded by pairwise `metric` distance. The
    origin (seed, else baseline, else earliest) settles the behavioral
    mode and the unit; `previous` (a prior `MapView.positions()`) keeps
    the layout in place across rebuilds."""
    if not candidates:
        return MapView.none(metric, "no candidates yet")
    try:
        fingerprinter = _load_fingerprinter(fingerprint_path)
    except FingerprintError as exc:
        return MapView.none(metric, str(exc))
    accepted = accepted_lineage(candidates, higher_is_better)
    resolved = _resolve_reference(candidates, accepted, "baseline")
    if resolved is None:
        return MapView.none(metric, "no baseline candidate")
    origin_candidate, label = resolved
    search = SearchInput(search_id="", search_dir=search_dir, candidates=candidates)
    artifacts = tuple(output_artifacts) or DEFAULT_ARTIFACTS
    _prune_caches(_live_paths([search], artifacts))
    origin = _reference_prints(origin_candidate, label, "", search_dir, artifacts, fingerprinter)
    if isinstance(origin, str):
        return MapView.none(metric, origin)
    members, missing = _members(search, origin, fingerprinter, namespaced=False)
    tree = build_tree(candidates, higher_is_better)
    return _assemble(
        members, origin, higher_is_better, metric, previous,
        best={accepted[-1]} if accepted else set(),
        origins={origin_candidate.candidate_id},
        fates={n.id: n.fate for n in tree.nodes}, on_path=set(tree.accepted),
        trail_ids=accepted, n_missing=missing,
    )


def build_run_map(
    searches: Sequence[SearchInput],
    higher_is_better: bool,
    metric: str = "behavioral",
    output_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
    fingerprint_path: Path | None = None,
    previous: Mapping[str, tuple[float, float, float]] | None = None,
    problem_key: str = "",
) -> MapView:
    """Every search of one problem in an experiment run in one map. The
    searches must share one seed (the same rule as the run cube), which
    settles the mode and unit; ids are `<search-id>/<candidate-id>`;
    lineage edges never cross searches, and the trail is the best-so-far
    sequence of the search holding the run's champion."""
    scope = "run"
    if not searches:
        return MapView.none(metric, f"no searches for {problem_key or 'this problem'} in the run", scope)
    try:
        fingerprinter = _load_fingerprinter(fingerprint_path)
    except FingerprintError as exc:
        return MapView.none(metric, str(exc), scope)
    artifacts = tuple(output_artifacts) or DEFAULT_ARTIFACTS
    _prune_caches(_live_paths(searches, artifacts))
    anchored = shared_seed_references(searches, artifacts, fingerprinter)
    if isinstance(anchored, str):
        return MapView.none(metric, anchored, scope)
    seeds, refs = anchored
    origin = refs[searches[0].search_id]

    members: list[_Member] = []
    missing = 0
    fates: dict[str, str] = {}
    on_path: set[str] = set()
    for search in searches:
        rows, skipped = _members(search, origin, fingerprinter, namespaced=True)
        members.extend(rows)
        missing += skipped
        tree = build_tree(search.candidates, higher_is_better)
        fates.update({f"{search.search_id}/{n.id}": n.fate for n in tree.nodes})
        on_path.update(f"{search.search_id}/{cid}" for cid in tree.accepted)
    champion = _run_champion(searches, higher_is_better)
    best: set[str] = set()
    trail_ids: list[str] = []
    if champion is not None:
        champion_search, champion_candidate = champion
        best = {f"{champion_search.search_id}/{champion_candidate.candidate_id}"}
        trail_ids = [
            f"{champion_search.search_id}/{cid}"
            for cid in accepted_lineage(champion_search.candidates, higher_is_better)
        ]
    arms = tuple(dict.fromkeys(s.arm for s in searches if s.arm))
    return _assemble(
        members, origin, higher_is_better, metric, previous,
        best=best, origins={f"{sid}/{seed.candidate_id}" for sid, seed in seeds.items()},
        fates=fates, on_path=on_path, trail_ids=trail_ids, n_missing=missing,
        scope=scope, arms=arms, n_searches=len(searches), problem_key=problem_key,
    )
