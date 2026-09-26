"""`hillclimb similarity` data: candidates as distances from a reference.

Every candidate of one search is placed at three distances from a
**reference candidate** (the baseline by default, the current champion on
toggle): behavioral — outputs on the fixed validation inputs; structural —
solution.py source tokens; lineage — hops through the exploration tree.
Everything is derived at build time from artifacts that already exist (the
submission file or the evaluator report, the solution source, the journal);
distances are never stored, so the reference point is purely a view-time
choice and the append-only journal is untouched.

Behavioral fingerprints come in three modes, decided by the reference
candidate and never mixed (the RMS spaces aren't comparable):

- **fingerprint mode**: the problem ships `fingerprint.py` (picked up by
  default like `landscape.py`) whose `fingerprint(candidate_dir)` returns a
  numeric vector — the problem's own notion of what an output *is*,
  invariant to whatever it considers equivalent (heilbronn: sorted triangle
  areas, blind to point order and the square's symmetries). Scale is the
  MAD of the reference's vector, as in submission mode. Problem-authored
  code, run in the viewer process: per-candidate failures leave that
  candidate unpositioned, an unimportable module makes the view unavailable.
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

**Run scope** (`build_run_similarity`): every search of one problem in an
experiment run in one cube — each search measured from its own seed
candidate (the arms share one seed file, so the origin is the same point),
lineage counted within its own tree, axes and rank bins normalized over the
union, node ids namespaced `<search-id>/<candidate-id>`. The `champion`
reference there is the run's best candidate: behavior and structure are
measured to it, lineage stays hops-from-own-seed (trees never join).

`similarityview.py` renders this; both stay replay-only readers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tokenize
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.tree import build_tree

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
    search_id: str = ""      # run scope: which search the candidate belongs to
    arm: str | None = None   # run scope: that search's experiment arm


@dataclass(frozen=True)
class SimilarityView:
    nodes: tuple[SimilarityNode, ...]
    reference_id: str
    reference: str                       # "baseline" | "champion" | "seed" (run scope)
    mode: str                            # "fingerprint" | "submission" | "report"
    scales: tuple[float, float, float]   # raw p95 per axis — what the cube edge means
    n_unpositioned: int
    unavailable: str | None = None       # set = nothing to draw, here is why
    reference_ids: tuple[str, ...] = ()  # every node at the origin (run scope: one seed per search)
    reference_label: str = ""            # what the reference resolved to: seed | baseline | earliest | champion
    scope: str = "search"                # "search" | "run"
    arms: tuple[str, ...] = ()           # run scope: experiment arms, first-seen order
    n_searches: int = 1
    problem_key: str = ""
    lineage_note: str = ""               # run scope, champion reference: "lineage from own seed"

    @classmethod
    def none(cls, reference: str, reason: str, scope: str = "search") -> SimilarityView:
        return cls(
            nodes=(), reference_id="", reference=reference, mode="",
            scales=(0.0, 0.0, 0.0), n_unpositioned=0, unavailable=reason, scope=scope,
        )


@dataclass(frozen=True)
class SearchInput:
    """One search as the run-scope builder needs it: its candidates, where
    their artifacts live, and the experiment tags the view groups on."""
    search_id: str
    search_dir: Path
    candidates: list[Candidate]
    arm: str | None = None
    seed_from: str | None = None
    seed_sha256: str | None = None


Fingerprinter = Callable[[Path], "Sequence[float] | None"]


class FingerprintError(Exception):
    """The problem's fingerprint.py cannot be used (unreadable, failed to
    import, or defines no `fingerprint`)."""


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
# problem-fingerprint vectors, keyed by the candidate's first declared
# artifact (its write stamp); dropped wholesale when fingerprint.py reloads
_FINGERPRINTS: dict[Path, tuple[float, int, np.ndarray]] = {}
# loaded fingerprint modules: (resolved path, mtime) -> the function; only
# one problem is ever on screen, so a new key evicts the old
_FINGERPRINTERS: dict[tuple[Path, float], Fingerprinter] = {}


def clear_caches() -> None:
    _SUBMISSIONS.clear()
    _TOKENS.clear()
    _FINGERPRINTS.clear()


def _prune_caches(live: set[Path]) -> None:
    for cache in (_SUBMISSIONS, _TOKENS, _FINGERPRINTS):
        for path in [p for p in cache if p not in live]:
            del cache[path]


def load_fingerprinter(path: Path) -> Fingerprinter:
    """Import a problem's fingerprint.py (same rules as surface's
    landscape loader): reloaded when the file changes, which also drops
    every cached vector, since they came from the old code."""
    path = Path(path).resolve()
    try:
        key = (path, path.stat().st_mtime)
    except OSError as exc:
        raise FingerprintError(f"fingerprint module unreadable: {path}") from exc
    if key in _FINGERPRINTERS:
        return _FINGERPRINTERS[key]
    spec = importlib.util.spec_from_file_location(f"hillclimb_fingerprint_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise FingerprintError(f"fingerprint module cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — problem-authored code
        raise FingerprintError(f"fingerprint module failed to import ({exc}): {path}") from exc
    function = getattr(module, "fingerprint", None)
    if not callable(function):
        raise FingerprintError(f"fingerprint module must define fingerprint(candidate_dir): {path}")
    _FINGERPRINTERS.clear()
    _FINGERPRINTS.clear()
    _FINGERPRINTERS[key] = function
    return function


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
    best = candidate.best_trial
    if best is None or not best.submission_ok:
        return None
    return _cached(_SUBMISSIONS, dir_for(search_dir, candidate) / artifact, _parse_submission)


def fingerprint_vector(
    search_dir: Path, candidate: Candidate, artifact: str, fingerprinter: Fingerprinter
) -> np.ndarray | None:
    """The problem's own fingerprint of a candidate whose trial-0 produced
    a usable result, as a finite float vector; None when the module declines
    (or raises — a buggy candidate is not the view's problem). Cached by the
    declared artifact's write stamp, like the submission parse."""
    best = candidate.best_trial
    if best is None or not best.submission_ok:
        return None
    candidate_dir = dir_for(search_dir, candidate)

    def compute(_path: Path) -> np.ndarray | None:
        try:
            raw = fingerprinter(candidate_dir)
        except Exception:  # noqa: BLE001 — problem-authored code on one candidate
            return None
        if raw is None:
            return None
        vector = np.asarray(list(raw), dtype=np.float64).ravel()
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        return vector

    return _cached(_FINGERPRINTS, candidate_dir / artifact, compute)


def report_fingerprint(candidate: Candidate) -> dict[str, float] | None:
    """Labeled behavioral dims from trial-0's evaluator report + journaled
    metrics. Horizon buckets first (aligned across candidates by
    construction), zones last (top-N-only; the intersection rule absorbs
    their instability)."""
    if not candidate.trials:
        return None
    report = candidate.report or {}
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

def submission_vector(ref, cand) -> np.ndarray | None:
    """A candidate's submission laid out in the reference's column order and
    lengths — the vector both views compare — or None when unalignable
    (different row count, or a reference column it never wrote)."""
    ref_cols, ref_rows, ref_data = ref
    _cols, rows, data = cand
    if rows != ref_rows or any(c not in data for c in ref_cols):
        return None
    v = np.concatenate([data[c][: len(ref_data[c])] for c in ref_cols])
    if len(v) != sum(len(ref_data[c]) for c in ref_cols):
        return None
    return v


def submission_distance(ref, cand) -> float | None:
    """Scaled RMS between aligned submissions; None when unalignable."""
    v = submission_vector(ref, cand)
    if v is None:
        return None
    return _scaled_rms(submission_vector(ref, ref), v)  # type: ignore[arg-type]


def spread(r: np.ndarray) -> float:
    """A vector's own spread — its MAD, std when the MAD collapses — the
    unit both similarity views measure behavioral distance in."""
    scale = float(np.median(np.abs(r - np.median(r))))
    if scale < EPS:
        scale = max(float(np.std(r)), EPS)
    return scale


def _scaled_rms(r: np.ndarray, v: np.ndarray) -> float:
    """RMS of (v - r) in units of the reference's own spread: distance 1 =
    "differs from the reference by as much as the reference's own values
    vary"."""
    return float(np.sqrt(np.mean(((v - r) / spread(r)) ** 2)))


def fingerprint_distance(ref: np.ndarray, cand: np.ndarray) -> float | None:
    """Scaled RMS between two problem fingerprints; None when their
    lengths differ (the problem declined to make them comparable)."""
    if cand.shape != ref.shape:
        return None
    return _scaled_rms(ref, cand)


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

def _resolve_reference(
    candidates: list[Candidate], accepted: Sequence[str], reference: str
) -> tuple[Candidate, str] | None:
    """(candidate, what it is). `champion` is the last accepted candidate;
    anything else is the origin the search grew from: its seed when it has
    one (the executable every experiment arm started from — the declared
    baseline of a `baseline_files` problem ships no source at all), else the
    baseline, else the earliest candidate."""
    if reference == "champion":
        by_id = {c.candidate_id: c for c in candidates}
        champion = by_id.get(accepted[-1]) if accepted else None
        return (champion, "champion") if champion is not None else None
    for operator in ("seed", "baseline"):
        found = [c for c in candidates if c.operator == operator]
        if found:
            return found[0], operator
    earliest = min(candidates, key=lambda c: c.created_at, default=None)
    return (earliest, "earliest") if earliest is not None else None


def _p95(values: list[float]) -> float:
    return max(float(np.percentile(values, AXIS_P95)), EPS) if values else EPS


@dataclass(frozen=True)
class Reference:
    """A resolved anchor with its fingerprints: the mode it settled and the
    prints every other candidate is measured against."""
    candidate: Candidate
    search_id: str
    search_dir: Path
    label: str
    mode: str                 # "fingerprint" | "submission" | "report"
    artifact: str | None      # the declared file read (None in report mode)
    behavioral: object        # np.ndarray | submission triple | dict[str, float]
    tokens: frozenset[str]


@dataclass(frozen=True)
class Placed:
    candidate: Candidate
    search_id: str
    arm: str | None
    behavioral: float
    structural: float
    lineage: float
    fate: str
    on_path: bool


def _live_paths(searches: Sequence[SearchInput], artifacts: tuple[str, ...]) -> set[Path]:
    return {
        dir_for(s.search_dir, c) / name
        for s in searches for c in s.candidates for name in (*artifacts, SOLUTION_FILE)
    }


def _load_fingerprinter(fingerprint_path: Path | None) -> Fingerprinter | None:
    return load_fingerprinter(fingerprint_path) if fingerprint_path is not None else None


def _reference_prints(
    candidate: Candidate, label: str, search_id: str, search_dir: Path,
    artifacts: tuple[str, ...], fingerprinter: Fingerprinter | None,
) -> Reference | str:
    """The reference's prints, or the reason it cannot anchor a view. The
    reference settles the mode: the problem's fingerprint when it yields
    one, else the first declared artifact it produced, else its report."""
    tokens = token_set(search_dir, candidate)
    if tokens is None:
        return f"{label} {candidate.candidate_id} has no readable solution.py"
    common = dict(candidate=candidate, search_id=search_id, search_dir=search_dir, label=label, tokens=tokens)
    if fingerprinter is not None:
        vector = fingerprint_vector(search_dir, candidate, artifacts[0], fingerprinter)
        if vector is not None:
            return Reference(mode="fingerprint", artifact=artifacts[0], behavioral=vector, **common)
    for name in artifacts:
        submission = submission_fingerprint(search_dir, candidate, name)
        if submission is not None:
            return Reference(mode="submission", artifact=name, behavioral=submission, **common)
    report = report_fingerprint(candidate)
    if report is not None:
        return Reference(mode="report", artifact=None, behavioral=report, **common)
    return (
        f"{label} {candidate.candidate_id} has no submission and no evaluator "
        "report — nothing behavioral to measure against"
    )


def _behavioral(
    candidate: Candidate, search_dir: Path, ref: Reference, fingerprinter: Fingerprinter | None
) -> float | None:
    if ref.mode == "fingerprint":
        vector = fingerprint_vector(search_dir, candidate, ref.artifact, fingerprinter)  # type: ignore[arg-type]
        return fingerprint_distance(ref.behavioral, vector) if vector is not None else None  # type: ignore[arg-type]
    if ref.mode == "submission":
        submission = submission_fingerprint(search_dir, candidate, ref.artifact)  # type: ignore[arg-type]
        return submission_distance(ref.behavioral, submission) if submission is not None else None
    report = report_fingerprint(candidate)
    return report_distance(ref.behavioral, report) if report is not None else None  # type: ignore[arg-type]


def _prints(
    candidate: Candidate, search_dir: Path, ref: Reference, fingerprinter: Fingerprinter | None,
) -> object | None:
    """A candidate's behavioral print in the reference's mode: a vector
    (fingerprint mode), a submission triple, or a report dict; None when
    it produced nothing usable."""
    if ref.mode == "fingerprint":
        return fingerprint_vector(search_dir, candidate, ref.artifact, fingerprinter)  # type: ignore[arg-type]
    if ref.mode == "submission":
        return submission_fingerprint(search_dir, candidate, ref.artifact)  # type: ignore[arg-type]
    return report_fingerprint(candidate)


@dataclass(frozen=True)
class Prints:
    """What one candidate looks like, for pairwise comparison: its
    behavioral print (in the reference's mode) and its solution tokens."""
    behavioral: object
    tokens: frozenset[str]


def candidate_prints(
    search: SearchInput, ref: Reference, fingerprinter: Fingerprinter | None,
) -> dict[str, Prints]:
    """Every candidate of one search that has both prints, by candidate id
    — the input `similarity_map` builds its pairwise matrices from. Reads
    through the same caches as `_place`, so the cube and the map never
    parse a file twice between them."""
    out: dict[str, Prints] = {}
    for cand in search.candidates:
        behavioral = _prints(cand, search.search_dir, ref, fingerprinter)
        tokens = token_set(search.search_dir, cand)
        if behavioral is None or tokens is None:
            continue
        out[cand.candidate_id] = Prints(behavioral, tokens)
    return out


def _place(
    search: SearchInput, ref: Reference, lineage_ref_id: str, higher_is_better: bool,
    fingerprinter: Fingerprinter | None,
) -> tuple[list[Placed], int]:
    """Every candidate of one search at its three distances: behavior and
    structure to `ref` (wherever it lives), lineage to `lineage_ref_id`
    within this search's own tree. Returns (placed, unpositioned count)."""
    tree = build_tree(search.candidates, higher_is_better)
    fates = {n.id: n.fate for n in tree.nodes}
    on_path = set(tree.accepted)
    by_id = {c.candidate_id: c for c in search.candidates}
    placed: list[Placed] = []
    skipped = 0
    for cand in search.candidates:
        if search.search_id == ref.search_id and cand.candidate_id == ref.candidate.candidate_id:
            behavioral: float | None = 0.0
            structural = 0.0
        else:
            behavioral = _behavioral(cand, search.search_dir, ref, fingerprinter)
            tokens = token_set(search.search_dir, cand)
            if behavioral is None or tokens is None:
                skipped += 1
                continue
            structural = structural_distance(ref.tokens, tokens)
        lineage = lineage_distance(by_id, cand.candidate_id, lineage_ref_id)
        placed.append(Placed(
            cand, search.search_id, search.arm, behavioral, structural, lineage,
            fates.get(cand.candidate_id, "pending"), cand.candidate_id in on_path,
        ))
    return placed, skipped


def _finish(
    placed: list[Placed], higher_is_better: bool, *, best: set[tuple[str, str]],
    reference_ids: tuple[str, ...], namespaced: bool, **view_fields,
) -> SimilarityView:
    """Normalize, bin and wrap: axes by their p95 over everything placed,
    rank bins over every scored placed candidate, ids namespaced by search
    when several searches share the cube."""
    scales = (
        _p95([p.behavioral for p in placed]),
        _p95([p.structural for p in placed]),
        _p95([p.lineage for p in placed]),
    )
    scored = [
        p for p in placed
        if p.candidate.val_score is not None and p.candidate.status == "passing"
    ]
    scored.sort(key=lambda p: p.candidate.val_score, reverse=not higher_is_better)  # type: ignore[arg-type,return-value]
    bins = {
        (p.search_id, p.candidate.candidate_id): i * N_BINS // len(scored) for i, p in enumerate(scored)
    } if scored else {}

    def node_id(p: Placed) -> str:
        return f"{p.search_id}/{p.candidate.candidate_id}" if namespaced else p.candidate.candidate_id

    nodes = tuple(
        SimilarityNode(
            id=node_id(p),
            x=min(p.behavioral / scales[0], 1.0),
            y=min(p.structural / scales[1], 1.0),
            z=min(p.lineage / scales[2], 1.0),
            raw=(p.behavioral, p.structural, p.lineage),
            score=p.candidate.val_score,
            bin=bins.get((p.search_id, p.candidate.candidate_id)),
            best=(p.search_id, p.candidate.candidate_id) in best,
            fate=p.fate,
            on_path=p.on_path,
            operator=p.candidate.operator,
            search_id=p.search_id,
            arm=p.arm,
        )
        for p in placed
    )
    return SimilarityView(
        nodes=nodes, reference_id=reference_ids[0] if reference_ids else "",
        reference_ids=reference_ids, scales=scales, **view_fields,
    )


def build_similarity(
    candidates: list[Candidate],
    search_dir: Path,
    higher_is_better: bool,
    reference: str = "baseline",
    output_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
    fingerprint_path: Path | None = None,
) -> SimilarityView:
    """One search's candidates as distances from its reference — the origin
    it grew from (`baseline`: seed, else baseline, else earliest) or its
    `champion`. `fingerprint_path` is the problem's fingerprint.py, if any."""
    from hillclimb.tui.tree import accepted_lineage

    if not candidates:
        return SimilarityView.none(reference, "no candidates yet")
    try:
        fingerprinter = _load_fingerprinter(fingerprint_path)
    except FingerprintError as exc:
        return SimilarityView.none(reference, str(exc))
    accepted = accepted_lineage(candidates, higher_is_better)
    resolved = _resolve_reference(candidates, accepted, reference)
    if resolved is None:
        return SimilarityView.none(
            reference,
            "no champion yet — press c for the baseline" if reference == "champion"
            else "no baseline candidate",
        )
    ref_candidate, label = resolved
    search = SearchInput(search_id="", search_dir=search_dir, candidates=candidates)
    artifacts = tuple(output_artifacts) or DEFAULT_ARTIFACTS
    _prune_caches(_live_paths([search], artifacts))
    ref = _reference_prints(ref_candidate, label, "", search_dir, artifacts, fingerprinter)
    if isinstance(ref, str):
        return SimilarityView.none(
            reference, ref + (" — press c for the champion" if reference == "baseline" else ""),
        )
    placed, skipped = _place(search, ref, ref_candidate.candidate_id, higher_is_better, fingerprinter)
    best = {("", accepted[-1])} if accepted else set()
    return _finish(
        placed, higher_is_better, best=best, reference_ids=(ref_candidate.candidate_id,),
        namespaced=False, reference=reference, reference_label=label, mode=ref.mode,
        n_unpositioned=skipped,
    )


def _seed_digest(search: SearchInput, seed: Candidate) -> str | None:
    """What identifies a search's seed: the bytes of the seed candidate's
    solution.py (what was actually measured), else the hash the search
    recorded at start, else the seed path."""
    try:
        return hashlib.sha256((dir_for(search.search_dir, seed) / SOLUTION_FILE).read_bytes()).hexdigest()
    except OSError:
        return search.seed_sha256 or search.seed_from


def shared_seed_references(
    searches: Sequence[SearchInput], artifacts: tuple[str, ...], fingerprinter: Fingerprinter | None,
) -> tuple[dict[str, Candidate], dict[str, Reference]] | str:
    """Run scope's anchor: every search's seed candidate and its prints,
    or the reason the run cannot share one origin (a search without a
    seed, seeds that differ, seeds whose prints settle different modes)."""
    seeds: dict[str, Candidate] = {}
    digests: dict[str, str | None] = {}
    for search in searches:
        seed = next((c for c in search.candidates if c.operator == "seed"), None)
        if seed is None:
            return (
                f"{search.search_id} has no seed candidate — the run view anchors "
                "every search on the shared seed"
            )
        seeds[search.search_id] = seed
        digests[search.search_id] = _seed_digest(search, seed)
    first = searches[0].search_id
    for search in searches[1:]:
        if digests[search.search_id] is None or digests[search.search_id] != digests[first]:
            return f"seeds differ: {first} vs {search.search_id} — the run view needs one shared seed"

    refs: dict[str, Reference] = {}
    for search in searches:
        ref = _reference_prints(
            seeds[search.search_id], "seed", search.search_id, search.search_dir, artifacts, fingerprinter,
        )
        if isinstance(ref, str):
            return f"{search.search_id}: {ref}"
        refs[search.search_id] = ref
    modes = {r.mode for r in refs.values()}
    if len(modes) > 1:
        detail = ", ".join(f"{sid}={r.mode}" for sid, r in refs.items())
        return f"seed fingerprint modes differ ({detail}) — the cube would mix RMS spaces"
    return seeds, refs


def _run_champion(searches: Sequence[SearchInput], higher_is_better: bool) -> tuple[SearchInput, Candidate] | None:
    pool = [
        (s, c) for s in searches for c in s.candidates
        if c.val_score is not None and c.status == "passing" and not c.pruned
    ]
    if not pool:
        return None
    pick = max if higher_is_better else min
    top = pick(c.val_score for _, c in pool)  # type: ignore[type-var]
    tied = [item for item in pool if item[1].val_score == top]
    return min(tied, key=lambda item: item[1].finished_at or item[1].created_at or "")  # earliest to land


def build_run_similarity(
    searches: Sequence[SearchInput],
    higher_is_better: bool,
    reference: str = "seed",
    output_artifacts: Sequence[str] = DEFAULT_ARTIFACTS,
    fingerprint_path: Path | None = None,
    problem_key: str = "",
) -> SimilarityView:
    """Every search of one problem in one cube. `seed` (default) measures
    each search from its own seed candidate — all searches must have one
    and they must be the same file, so the origin is one point; `champion`
    measures behavior and structure to the run's best candidate. Lineage is
    always hops from the search's own seed."""
    scope = "run"
    if not searches:
        return SimilarityView.none(reference, f"no searches for {problem_key or 'this problem'} in the run", scope)
    try:
        fingerprinter = _load_fingerprinter(fingerprint_path)
    except FingerprintError as exc:
        return SimilarityView.none(reference, str(exc), scope)
    artifacts = tuple(output_artifacts) or DEFAULT_ARTIFACTS
    _prune_caches(_live_paths(searches, artifacts))

    anchored = shared_seed_references(searches, artifacts, fingerprinter)
    if isinstance(anchored, str):
        return SimilarityView.none(reference, anchored, scope)
    seeds, refs = anchored
    modes = {r.mode for r in refs.values()}

    champion = _run_champion(searches, higher_is_better)
    best = {(champion[0].search_id, champion[1].candidate_id)} if champion else set()
    lineage_note = ""
    if reference == "champion":
        if champion is None:
            return SimilarityView.none(reference, "no scored candidate yet — press c for the seed", scope)
        champion_ref = _reference_prints(
            champion[1], "champion", champion[0].search_id, champion[0].search_dir, artifacts, fingerprinter,
        )
        if isinstance(champion_ref, str):
            return SimilarityView.none(reference, f"{champion[0].search_id}: {champion_ref}", scope)
        if champion_ref.mode not in modes:
            return SimilarityView.none(
                reference, f"champion measures in {champion_ref.mode} mode, the seeds in "
                f"{modes.pop()} — press c for the seed", scope,
            )
        refs = {sid: champion_ref for sid in refs}
        reference_ids: tuple[str, ...] = (f"{champion[0].search_id}/{champion[1].candidate_id}",)
        label = "champion"
        lineage_note = "lineage from own seed"
    else:
        reference_ids = tuple(f"{sid}/{seed.candidate_id}" for sid, seed in seeds.items())
        label = "seed"

    placed: list[Placed] = []
    skipped = 0
    for search in searches:
        rows, missed = _place(
            search, refs[search.search_id], seeds[search.search_id].candidate_id, higher_is_better, fingerprinter,
        )
        placed.extend(rows)
        skipped += missed
    arms = tuple(dict.fromkeys(s.arm for s in searches if s.arm))
    return _finish(
        placed, higher_is_better, best=best, reference_ids=reference_ids, namespaced=True,
        reference=reference, reference_label=label, mode=next(iter(modes)), n_unpositioned=skipped,
        scope=scope, arms=arms, n_searches=len(searches), problem_key=problem_key,
        lineage_note=lineage_note,
    )
