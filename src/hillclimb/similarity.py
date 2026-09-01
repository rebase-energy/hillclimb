"""`hillclimb similarity` data: candidates as distances from a reference.

Every candidate of one search is placed at three distances from a
**reference candidate** (the baseline by default, the current champion on
toggle): behavioral — outputs on the fixed validation inputs; structural —
solution.py source tokens; lineage — hops through the exploration tree.
Everything is derived at build time from artifacts that already exist (the
submission file or the evaluator report, the solution source, the journal);
distances are never stored, so the reference point is purely a view-time
choice and the append-only journal is untouched.

Behavioral fingerprints come in two modes, decided per search by the
reference candidate and never mixed (the two RMS spaces aren't comparable):

- **submission mode**: the reference dir has one of the problem's declared
  `output_artifacts` (`submission.csv` for tabular problems,
  `submission.json` for JSON-native ones). A CSV fingerprint is its numeric
  columns flattened in column order; a JSON fingerprint is its numeric
  leaves keyed by JSON path (`circles[0][2]`), which is the same contract
  one value per column. Either way candidates align by key names + count
  (order within a key is already the scoring contract). Scale is one scalar
  — the MAD of the reference's own vector — so distance 1 means "differs
  from the reference by as much as the reference's own predictions vary".
- **report mode** (emflow-style problems with no submission file):
  fingerprints are labeled dicts from trial-0's evaluator report —
  `horizon:` buckets (aligned across candidates by construction), `q:`
  pinball entries, journaled metrics, and `zone:` entries last (top-N-only,
  so they contribute through the key intersection or not at all). Scale is
  per-dim relative to the reference, clipped so one exploded bucket can't
  own the distance.

Fingerprints (not distances) are cached per file path + mtime, so a live
refresh with one new candidate costs that candidate's file reads plus an
O(N) numpy pass, and toggling the reference invalidates nothing.

`similarityview.py` renders this; both stay replay-only readers.
"""

from __future__ import annotations

import io
import json
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hillclimb.candidate import Candidate
from hillclimb.tree import build_tree

EPS = 1e-8
CLIP = 10.0            # report mode: max scaled deviation per dimension
MIN_SHARED_KEYS = 3    # report mode: minimum key overlap with the reference
MIN_SHARED_FRAC = 0.5  # ... and at least this fraction of the reference's keys
MAX_VECTOR = 65_536    # submission vectors longer than this are stride-subsampled
AXIS_P95 = 95.0        # each axis is normalized to [0, 1] by its 95th percentile
N_BINS = 6             # score rank bins (the colour ramp in similarityview)

DEFAULT_ARTIFACTS = ("submission.csv",)  # problems that declare nothing
SOLUTION_FILE = "solution.py"

_STRUCTURAL_TOKENS = frozenset(
    {tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING}
)


@dataclass(frozen=True)
class SimilarityNode:
    id: str
    x: float            # behavioral distance, normalized to [0, 1]
    y: float            # structural distance, normalized
    z: float            # lineage distance, normalized
    raw: tuple[float, float, float]  # the unnormalized distances
    score: float | None
    bin: int | None     # 0 (worst) .. N_BINS-1 (best) over scored nodes; None = unscored
    best: bool          # the current champion
    fate: str           # tree.py vocabulary
    on_path: bool
    operator: str


@dataclass(frozen=True)
class SimilarityView:
    nodes: tuple[SimilarityNode, ...]
    reference_id: str
    reference: str                       # "baseline" | "champion"
    mode: str                            # "submission" | "report"
    scales: tuple[float, float, float]   # raw p95 per axis — what the cube edge means
    n_unpositioned: int
    unavailable: str | None = None       # set = nothing to draw, here is why

    @classmethod
    def none(cls, reference: str, reason: str) -> SimilarityView:
        return cls(
            nodes=(), reference_id="", reference=reference, mode="",
            scales=(0.0, 0.0, 0.0), n_unpositioned=0, unavailable=reason,
        )


# ---------------------------------------------------------------------------
# candidate artifacts

def dir_for(search_dir: Path, candidate: Candidate) -> Path:
    """The candidate's directory — same rule as watch's `_candidate_dir`,
    plus a fallback for runs moved from another machine (candidate_dir is
    stored absolute)."""
    if candidate.candidate_dir:
        path = Path(candidate.candidate_dir)
        if path.exists():
            return path
    return search_dir / "candidates" / candidate.candidate_id


# fingerprint caches: path -> (mtime, size, value). Candidates are immutable
# once finished, so entries never churn; `_prune_caches` drops paths that
# left the current candidate set and `clear_caches` resets on search switch.
_SUBMISSIONS: dict[Path, tuple[float, int, tuple[tuple[str, ...], int, dict[str, np.ndarray]]]] = {}
_TOKENS: dict[Path, tuple[float, int, frozenset[str]]] = {}


def clear_caches() -> None:
    _SUBMISSIONS.clear()
    _TOKENS.clear()


def _prune_caches(live: set[Path]) -> None:
    for cache in (_SUBMISSIONS, _TOKENS):
        for path in [p for p in cache if p not in live]:
            del cache[path]


def _cached(cache: dict, path: Path, parse):
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (stat.st_mtime, stat.st_size)
    hit = cache.get(path)
    if hit is not None and hit[:2] == key:
        return hit[2]
    value = parse(path)
    if value is not None:
        cache[path] = (*key, value)
    return value


def _flatten_json_numbers(node, prefix: str, out: dict[str, float]) -> None:
    """Numeric leaves of a JSON document, keyed by path. Bools are flags, not
    measurements; non-finite values would poison the RMS, so both are skipped."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        value = float(node)
        if value == value and abs(value) != float("inf"):
            out[prefix or "$"] = value
        return
    if isinstance(node, list):
        for i, item in enumerate(node):
            _flatten_json_numbers(item, f"{prefix}[{i}]", out)
        return
    if isinstance(node, dict):
        for key, item in node.items():
            _flatten_json_numbers(item, f"{prefix}.{key}" if prefix else str(key), out)


def _parse_json_submission(path: Path):
    """Same triple as the CSV parser, one numeric leaf per column: candidates
    that lay out the same solution shape align, ones that don't are reported
    unalignable rather than compared across mismatched keys."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):  # buggy candidate output
        return None
    flat: dict[str, float] = {}
    _flatten_json_numbers(data, "", flat)
    if not flat:
        return None
    names = tuple(sorted(flat))
    stride = max(1, (len(names) + MAX_VECTOR - 1) // MAX_VECTOR)
    names = names[::stride]
    return (names, len(names), {n: np.asarray([flat[n]], dtype=np.float32) for n in names})


def _parse_submission(path: Path):
    """(numeric column names, original row count, {column: float32 array,
    stride-subsampled to the memory cap}) — or None if unreadable."""
    if path.suffix.lower() == ".json":
        return _parse_json_submission(path)
    import pandas as pd

    try:
        frame = pd.read_csv(path).select_dtypes(include="number")
    except Exception:  # noqa: BLE001 — buggy candidate output
        return None
    if frame.empty:
        return None
    nrows = len(frame)
    stride = max(1, (nrows * len(frame.columns) + MAX_VECTOR - 1) // MAX_VECTOR)
    columns = {
        str(name): frame[name].to_numpy(dtype=np.float32)[::stride]
        for name in frame.columns
    }
    return (tuple(str(c) for c in frame.columns), nrows, columns)


def submission_fingerprint(search_dir: Path, candidate: Candidate, artifact: str):
    """The parsed submission of a candidate whose trial-0 produced a usable
    result — the free `submission_ok` gate skips buggy output unparsed. The
    artifact name is fixed by the reference for the whole view: two artifacts
    are two RMS spaces, and mixing them would compare nothing."""
    if not candidate.trials or not candidate.trials[0].submission_ok:
        return None
    return _cached(_SUBMISSIONS, dir_for(search_dir, candidate) / artifact, _parse_submission)


def report_fingerprint(candidate: Candidate) -> dict[str, float] | None:
    """Labeled behavioral dims from trial-0's evaluator report + journaled
    metrics. Horizon buckets first (aligned across candidates by
    construction), zones last (top-N-only; the intersection rule absorbs
    their instability)."""
    if not candidate.trials:
        return None
    report = candidate.trials[0].report or {}
    dims: dict[str, float] = {}
    for entry in report.get("horizons") or []:
        if isinstance(entry.get("score"), (int, float)):
            dims[f"horizon:{entry.get('bucket')}"] = float(entry["score"])
    for entry in report.get("quantiles") or []:
        if isinstance(entry.get("pinball"), (int, float)):
            dims[f"q:{entry.get('q')}"] = float(entry["pinball"])
    for key, value in candidate.metrics.items():
        dims[f"metric:{key}"] = float(value)
    for entry in report.get("zones") or []:
        if isinstance(entry.get("score"), (int, float)):
            dims[f"zone:{entry.get('zone')}"] = float(entry["score"])
    return dims or None


def _tokenize_source(path: Path) -> frozenset[str] | None:
    try:
        source = path.read_text(errors="replace")
    except OSError:
        return None
    try:
        tokens = {
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type in _STRUCTURAL_TOKENS
        }
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = set(source.split())  # half-written file: degrade, don't drop
    return frozenset(tokens) or None


def token_set(search_dir: Path, candidate: Candidate) -> frozenset[str] | None:
    return _cached(_TOKENS, dir_for(search_dir, candidate) / SOLUTION_FILE, _tokenize_source)


# ---------------------------------------------------------------------------
# the three distances

def submission_distance(ref, cand) -> float | None:
    """Scaled RMS between aligned submissions; None when unalignable."""
    ref_cols, ref_rows, ref_data = ref
    cols, rows, data = cand
    if rows != ref_rows or any(c not in data for c in ref_cols):
        return None
    r = np.concatenate([ref_data[c] for c in ref_cols])
    v = np.concatenate([data[c][: len(ref_data[c])] for c in ref_cols])
    if len(v) != len(r):
        return None
    scale = float(np.median(np.abs(r - np.median(r))))
    if scale < EPS:
        scale = max(float(np.std(r)), EPS)
    return float(np.sqrt(np.mean(((v - r) / scale) ** 2)))


def report_distance(ref: dict[str, float], cand: dict[str, float]) -> float | None:
    shared = sorted(set(ref) & set(cand))
    if len(shared) < MIN_SHARED_KEYS or len(shared) < MIN_SHARED_FRAC * len(ref):
        return None
    deltas = [
        min(abs(cand[k] - ref[k]) / max(abs(ref[k]), EPS), CLIP) for k in shared
    ]
    return float(np.sqrt(np.mean(np.square(deltas))))


def structural_distance(ref: frozenset[str], cand: frozenset[str]) -> float:
    union = ref | cand
    if not union:
        return 0.0
    return 1.0 - len(ref & cand) / len(union)


def _ancestor_chain(candidates: dict[str, Candidate], cid: str) -> list[str]:
    chain = [cid]
    seen = {cid}
    parent = candidates[cid].parent_id
    while parent and parent not in seen and parent in candidates:
        chain.append(parent)
        seen.add(parent)
        parent = candidates[parent].parent_id
    return chain  # self first, root last

def lineage_distance(candidates: dict[str, Candidate], a: str, b: str) -> float:
    """Hops through the lowest common ancestor; disjoint components connect
    through a virtual super-root (+2) rather than dropping the point."""
    chain_a = _ancestor_chain(candidates, a)
    chain_b = _ancestor_chain(candidates, b)
    depth_b = {cid: i for i, cid in enumerate(chain_b)}
    for depth_a, cid in enumerate(chain_a):
        if cid in depth_b:
            return float(depth_a + depth_b[cid])
    return float(len(chain_a) - 1 + len(chain_b) - 1 + 2)


# ---------------------------------------------------------------------------
# the view

def _resolve_reference(candidates: list[Candidate], accepted: tuple[str, ...], reference: str) -> Candidate | None:
    if reference == "champion":
        by_id = {c.candidate_id: c for c in candidates}
        return by_id.get(accepted[-1]) if accepted else None
    baselines = [c for c in candidates if c.operator == "baseline"]
    if baselines:
        return baselines[0]
    return min(candidates, key=lambda c: c.created_at, default=None)


def _p95(values: list[float]) -> float:
    return max(float(np.percentile(values, AXIS_P95)), EPS) if values else EPS


def build_similarity(
    candidates: list[Candidate],
    search_dir: Path,
    higher_is_better: bool,
    reference: str = "baseline",
    output_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
) -> SimilarityView:
    if not candidates:
        return SimilarityView.none(reference, "no candidates yet")
    tree = build_tree(candidates, higher_is_better)
    fates = {n.id: n.fate for n in tree.nodes}
    on_path = set(tree.accepted)

    ref = _resolve_reference(candidates, tree.accepted, reference)
    if ref is None:
        return SimilarityView.none(
            reference,
            "no champion yet — press c for the baseline" if reference == "champion"
            else "no baseline candidate",
        )

    artifacts = tuple(output_artifacts) or DEFAULT_ARTIFACTS
    _prune_caches(
        {dir_for(search_dir, c) / name for c in candidates for name in (*artifacts, SOLUTION_FILE)}
    )

    ref_tokens = token_set(search_dir, ref)
    if ref_tokens is None:
        return SimilarityView.none(
            reference, f"{reference} {ref.candidate_id} has no readable solution.py"
        )
    # The reference settles which declared artifact this view measures in;
    # every other candidate is then read through that same file.
    submission_file = artifacts[0]
    ref_submission = None
    for name in artifacts:
        ref_submission = submission_fingerprint(search_dir, ref, name)
        if ref_submission is not None:
            submission_file = name
            break
    mode = "submission" if ref_submission is not None else "report"
    ref_report = report_fingerprint(ref) if mode == "report" else None
    if mode == "report" and ref_report is None:
        return SimilarityView.none(
            reference,
            f"{reference} {ref.candidate_id} has no submission and no evaluator "
            "report — nothing behavioral to measure against"
            + (" — press c for the champion" if reference == "baseline" else ""),
        )

    by_id = {c.candidate_id: c for c in candidates}
    placed: list[tuple[Candidate, float, float, float]] = []
    skipped = 0
    for cand in candidates:
        if cand.candidate_id == ref.candidate_id:
            behavioral: float | None = 0.0
            structural = 0.0
        else:
            if mode == "submission":
                fp = submission_fingerprint(search_dir, cand, submission_file)
                behavioral = submission_distance(ref_submission, fp) if fp is not None else None
            else:
                fp = report_fingerprint(cand)
                behavioral = report_distance(ref_report, fp) if fp is not None else None
            tokens = token_set(search_dir, cand)
            if behavioral is None or tokens is None:
                skipped += 1
                continue
            structural = structural_distance(ref_tokens, tokens)
        lineage = lineage_distance(by_id, cand.candidate_id, ref.candidate_id)
        placed.append((cand, behavioral, structural, lineage))

    scales = (
        _p95([b for _, b, _, _ in placed]),
        _p95([s for _, _, s, _ in placed]),
        _p95([l for _, _, _, l in placed]),
    )
    # score rank bins over the scored placed candidates, worst first
    scored = [c for c, _, _, _ in placed if c.val_score is not None and c.status == "ok"]
    scored.sort(key=lambda c: c.val_score, reverse=not higher_is_better)  # type: ignore[arg-type,return-value]
    bins = {c.candidate_id: i * N_BINS // len(scored) for i, c in enumerate(scored)} if scored else {}
    best_id = tree.best_id

    nodes = tuple(
        SimilarityNode(
            id=cand.candidate_id,
            x=min(b / scales[0], 1.0),
            y=min(s / scales[1], 1.0),
            z=min(l / scales[2], 1.0),
            raw=(b, s, l),
            score=cand.val_score,
            bin=bins.get(cand.candidate_id),
            best=cand.candidate_id == best_id,
            fate=fates.get(cand.candidate_id, "pending"),
            on_path=cand.candidate_id in on_path,
            operator=cand.operator,
        )
        for cand, b, s, l in placed
    )
    return SimilarityView(
        nodes=nodes,
        reference_id=ref.candidate_id,
        reference=reference,
        mode=mode,
        scales=scales,
        n_unpositioned=skipped,
    )
