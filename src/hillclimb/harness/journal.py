from __future__ import annotations

import json
import threading
from pathlib import Path
from statistics import median
from typing import Protocol

from hillclimb.harness.candidate import Candidate, utcnow


def _ranks(values: list[float], higher_is_better: bool) -> list[float]:
    """Average-tie ranks, 1 = best."""
    order = sorted(values, reverse=higher_is_better)
    return [
        (order.index(v) + 1 + len(order) - 1 - order[::-1].index(v) + 1) / 2 for v in values
    ]


class JournalBackend(Protocol):
    """Where a search's journal records live (store.py hands one out per
    search). `records()` returns every event in append order — candidate
    events and `control` audit lines alike — and `append` adds one."""

    def records(self) -> list[dict]: ...
    def append(self, record: dict) -> None: ...


class FileJournal:
    """The JSONL file: one JSON object per line, only ever appended to."""

    def __init__(self, path: Path):
        self.path = path

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class Journal:
    """Append-only journal of search candidates over a JournalBackend.

    Events: `candidate_created` (candidate enters the tree, status=pending) and
    `candidate_result` (terminal state for the candidate). The in-memory view
    is the replay of all events in append order; records are never rewritten,
    which is what makes `resume` and `status` safe against crashes mid-run.

    A `Path` is accepted as shorthand for `FileJournal(path)`.
    """

    def __init__(self, backend: JournalBackend | Path):
        self.backend: JournalBackend = FileJournal(backend) if isinstance(backend, Path) else backend
        self.candidates: dict[str, Candidate] = {}
        # Writes stay single-threaded (the strategy's scheduler); the lock
        # lets the host's evaluator take a consistent read (the holdout
        # top-k gate) from a worker thread while the scheduler appends.
        self.lock = threading.RLock()
        self._replay()

    CANDIDATE_EVENTS = ("candidate_created", "candidate_result")

    @property
    def path(self) -> Path | None:
        """The JSONL file when the backend is one (tests and viewers peek)."""
        return self.backend.path if isinstance(self.backend, FileJournal) else None

    def _replay(self) -> None:
        for record in self.backend.records():
            record = dict(record)
            # only candidate events carry tree state; other events (control
            # audit lines, future kinds) are skipped so old code tolerates new ones
            if record.pop("event", None) not in self.CANDIDATE_EVENTS:
                continue
            candidate = Candidate.model_validate(record)
            self.candidates[candidate.candidate_id] = candidate

    def _append_line(self, record: dict) -> None:
        self.backend.append(record)

    def _append(self, event: str, candidate: Candidate) -> None:
        with self.lock:
            self._append_line({"event": event, **candidate.model_dump()})
            self.candidates[candidate.candidate_id] = candidate.model_copy(deep=True)

    def candidate_created(self, candidate: Candidate) -> None:
        self._append("candidate_created", candidate)

    def candidate_result(self, candidate: Candidate) -> None:
        self._append("candidate_result", candidate)

    def control_event(self, action: str, **payload) -> None:
        """Audit line for a user control action (stop/prune). Carries no tree
        state; replay skips it."""
        self._append_line({"event": "control", "action": action, "applied_at": utcnow(), **payload})

    def audit_event(self, event: str, **payload) -> None:
        """Generic append-only audit line (e.g. an engine's failed-proposal
        cost record). Carries no tree state; replay skips unknown events, so
        old readers tolerate new kinds. Consumers re-read backend.records()."""
        self._append_line({"event": event, "recorded_at": utcnow(), **payload})

    # --- queries ---

    def get(self, candidate_id: str) -> Candidate:
        return self.candidates[candidate_id]

    def next_candidate_id(self) -> str:
        # max-based, not count-based: a journal with gaps (crash recovery,
        # pruned history) must never reissue an existing id
        highest = -1
        for candidate_id in self.candidates:
            suffix = candidate_id.removeprefix("c")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return f"c{highest + 1:03d}"

    def children(self, candidate_id: str, include_pruned: bool = False) -> list[Candidate]:
        return [
            c
            for c in self.candidates.values()
            if c.parent_id == candidate_id and (include_pruned or not c.pruned)
        ]

    def descendants(self, candidate_id: str) -> list[Candidate]:
        """Full subtree below a candidate (pruned included), for subtree walks."""
        found, frontier = [], [candidate_id]
        while frontier:
            batch = self.children(frontier.pop(), include_pruned=True)
            found.extend(batch)
            frontier.extend(c.candidate_id for c in batch)
        return found

    def drafts(self) -> list[Candidate]:
        """Every live candidate made from the problem alone (role `create`)."""
        return [c for c in self.candidates.values() if c.role == "create" and not c.pruned]

    def noise_floor(self) -> float | None:
        """How much identical code and params move between identical
        evaluations, measured from the search's own repeated replicates: the
        median of the per-trial replicate spreads. None when no trial has
        been evaluated twice (n_replicates = 1), which is the honest answer —
        the search has no evidence about its own noise."""
        spreads = [
            spread
            for candidate in self.candidates.values()
            for spread in candidate.replicate_spreads
        ]
        return median(spreads) if spreads else None

    def scored_candidates(self) -> list[Candidate]:
        with self.lock:
            return [c for c in self.candidates.values() if c.is_scored and not c.pruned]

    def best_candidate(self, higher_is_better: bool) -> Candidate | None:
        scored = self.scored_candidates()
        if not scored:
            return None
        return min(scored, key=lambda c: -c.val_score if higher_is_better else c.val_score)

    def selected_candidate(self, higher_is_better: bool, mode: str = "rank-blend") -> Candidate | None:
        """Candidate whose submission ships. Falls back to val_score when no
        candidate has a holdout score (holdout disabled).

        rank-blend (default): min(val_rank + holdout_rank) — robust when either
        signal is unreliable: an overfit val score is vetoed by its holdout
        rank, a noisy holdout outlier is vetoed by its val rank. `holdout` and
        `val` select by a single signal.
        """
        ranked = self.ranked_candidates(higher_is_better, mode)
        return ranked[0] if ranked else None

    def ranked_candidates(self, higher_is_better: bool, mode: str = "rank-blend") -> list[Candidate]:
        """Scored candidates ordered best-first by the selection rule."""
        direction = -1 if higher_is_better else 1
        with_holdout = [c for c in self.scored_candidates() if c.holdout_score is not None]
        if not with_holdout or mode == "val":
            return sorted(self.scored_candidates(), key=lambda c: direction * c.val_score)
        if mode == "holdout":
            return sorted(with_holdout, key=lambda c: direction * c.holdout_score)
        val_rank = _ranks([c.val_score for c in with_holdout], higher_is_better)
        hold_rank = _ranks([c.holdout_score for c in with_holdout], higher_is_better)
        return [
            t[0]
            for t in sorted(
                zip(with_holdout, val_rank, hold_rank),
                key=lambda t: (
                    t[1] + t[2],
                    direction * t[0].holdout_score,
                    direction * t[0].val_score,
                ),
            )
        ]

    def pending_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates.values() if c.status == "pending" and not c.pruned]

    def debug_chain(self, candidate_id: str) -> list[Candidate]:
        """The failed candidate being repaired plus every debug attempt so far,
        oldest first (the context a DEBUG operator needs)."""
        chain = [self.get(candidate_id)]
        while chain[0].role == "repair" and chain[0].parent_id:
            chain.insert(0, self.get(chain[0].parent_id))
        return chain

    def siblings(self, candidate_id: str) -> list[Candidate]:
        candidate = self.get(candidate_id)
        return [
            c
            for c in self.candidates.values()
            if c.parent_id == candidate.parent_id and c.candidate_id != candidate_id
        ]


class PolicyJournal(Journal):
    """What a search policy sees of a journal: every query, no holdout, no
    writes. The candidates are `holdout_blind()` copies taken at construction,
    so `ranked_candidates`/`selected_candidate` degrade to val order whatever
    mode they are asked for — the hidden split stays the host's. Snapshot
    semantics: a view never follows later appends; build one per policy call
    (`PolicyInput` does)."""

    def __init__(self, source: Journal):
        if isinstance(source, PolicyJournal):
            source = source.source
        self.source = source
        self.lock = source.lock
        with source.lock:
            self.candidates = {
                cid: candidate.holdout_blind() for cid, candidate in source.candidates.items()
            }

    @property
    def path(self) -> Path | None:
        # the file would hand back what the view strips
        return None

    def _append_line(self, record: dict) -> None:
        raise TypeError("a policy's journal view is read-only: the engine is the single writer")
