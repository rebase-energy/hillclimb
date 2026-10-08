"""What a search left behind, read in Python instead of out of `runs/`.

    outcome = hc.run("fitness-landscape", agent="toy", max_evaluations=20)
    outcome.best.val_score          # the best candidate by validation score
    outcome.history                 # every time the best-so-far moved
    outcome.spend                   # evaluations, tokens, cost, seconds
    outcome.solution                # the source that ships
    outcome.to_frame()              # one row per candidate

    hc.open_search("run-id/search-id")    # any earlier search (None: the latest)

A `SearchOutcome` reads through the store, like every other viewer: it never
writes, and the journal it hands out refuses to. A finished search is read
once; a running one is read again on every access, so the same object follows
a search another process is still climbing.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from hillclimb.harness.budget import Spend, journal_spend
from hillclimb.harness.candidate import Candidate, read_solution
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import search_ref, search_selection

if TYPE_CHECKING:
    from hillclimb.config import Config
    from hillclimb.harness.run import SearchMeta
    from hillclimb.harness.status import SearchStatus


class HistoryStep(NamedTuple):
    """One rise of the best-so-far: when it landed, who, and its score."""

    minutes: float | None
    candidate_id: str
    score: float


class _ReadOnlyJournal:
    """The backend of a result's journal: the records as read, and nothing
    else. The engine is the single writer of a search."""

    def __init__(self, records: list[dict]):
        self._records = records

    def records(self) -> list[dict]:
        return self._records

    def append(self, record: dict) -> None:
        raise TypeError("a result's journal is read-only: the engine is the single writer of a search")


@dataclass
class _Records:
    journal: Journal
    meta: SearchMeta | None
    status: SearchStatus | None
    state: str


@dataclass
class SearchOutcome:
    run_dir: Path
    search_dir: Path
    selected: Candidate | None
    state: str  # done | parked | stopped — and, for a search opened by ref, any state it is in
    error: str | None = None
    config: Config | None = field(default=None, repr=False, compare=False)
    _records: _Records | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def ref(self) -> str:
        return search_ref(self.search_dir)

    # --- the records ---

    def _read(self) -> _Records:
        """The search's records. Cached once the search is over; a running
        search is read again, and `state` / `selected` / `error` follow it."""
        if self._records is not None and self._records.state != "running":
            return self._records
        from hillclimb.harness.store import FileDataStore, key_for, open_store

        # no config: the folder layout itself says where the files store is
        store = open_store(self.config) if self.config is not None else FileDataStore(self.search_dir.parents[2])
        with closing(store):
            key = key_for(self.search_dir)
            record = store.search(key)
            journal = Journal(_ReadOnlyJournal(store.journal(key).records()))
            status = store.read_status(key)
        live = self._records is not None or self.state == "running"
        records = _Records(
            journal=journal,
            meta=record.meta if record is not None else None,
            status=status,
            state=record.state if record is not None and live else self.state,
        )
        if live:
            self.state = records.state
            self.error = status.last_error if status is not None else self.error
            self.selected = journal.selected_candidate(self.higher_is_better_from(records), search_selection(records.meta))
        self._records = records
        return records

    @staticmethod
    def higher_is_better_from(records: _Records) -> bool:
        return records.meta.higher_is_better if records.meta is not None else True

    @property
    def journal(self) -> Journal:
        """The search's journal, holdout scores included. Read-only."""
        return self._read().journal

    @property
    def meta(self) -> SearchMeta | None:
        """The search's record: problem, metric, climber block, budget, tags."""
        return self._read().meta

    @property
    def status(self) -> SearchStatus | None:
        """The last status record the engine wrote."""
        return self._read().status

    @property
    def higher_is_better(self) -> bool:
        return self.higher_is_better_from(self._read())

    # --- candidates ---

    @property
    def candidates(self) -> list[Candidate]:
        """Every candidate, in the order the search created them."""
        return list(self.journal.candidates.values())

    @property
    def best(self) -> Candidate | None:
        """The best candidate by validation score (`selected` is the one that
        ships: the same candidate unless a holdout split says otherwise)."""
        return self.journal.best_candidate(self.higher_is_better)

    @property
    def history(self) -> list[HistoryStep]:
        """Every time the best-so-far moved, oldest first — the staircase
        `hillclimb chart` draws."""
        from hillclimb.tui.tree import accepted_lineage, minutes_since

        records = self._read()
        started = records.meta.started_at if records.meta is not None else None
        by_id = records.journal.candidates
        return [
            HistoryStep(
                minutes_since(by_id[cid].finished_at or by_id[cid].created_at, started),
                cid,
                by_id[cid].val_score,
            )
            for cid in accepted_lineage(list(by_id.values()), self.higher_is_better_from(records))
        ]

    @property
    def spend(self) -> Spend:
        """Evaluations, tokens and cost from the journal; seconds from the clock."""
        from hillclimb.harness.status import live_spent_s

        records = self._read()
        seconds = live_spent_s(records.status, records.state) if records.status is not None else None
        return replace(journal_spend(records.journal), seconds=seconds)

    # --- what ships ---

    def source(self, candidate_id: str) -> str | None:
        """A candidate's `solution.py` (None when it wrote none, or its
        candidate dir is not on this machine)."""
        candidate = self.journal.candidates.get(candidate_id)
        return read_solution(candidate) if candidate is not None else None

    @property
    def solution(self) -> str | None:
        """The source of the selected candidate: `best/solution.py`."""
        self._read()
        shipped = self.search_dir / "best" / "solution.py"
        try:
            return shipped.read_text(errors="replace")
        except OSError:
            return read_solution(self.selected) if self.selected is not None else None

    @property
    def params(self) -> dict[str, Any]:
        """The parameter values the selected candidate's best trial ran with
        ({} when it declares none)."""
        self._read()
        trial = self.selected.best_trial if self.selected is not None else None
        return dict(trial.params) if trial is not None and trial.params else {}

    # --- a table ---

    def to_frame(self):
        """One row per candidate, as a pandas DataFrame indexed by candidate id:
        lineage, verdict, scores, what it cost, when it landed, and the numbers
        its verifier reported (`metrics`)."""
        import pandas as pd

        from hillclimb.tui.tree import minutes_since

        records = self._read()
        started = records.meta.started_at if records.meta is not None else None
        best = self.best
        rows = []
        for cand in records.journal.candidates.values():
            row = {
                "candidate_id": cand.candidate_id,
                "parent_id": cand.parent_id,
                "operator": cand.operator,
                "kind": cand.kind,
                "status": cand.status,
                "val_score": cand.val_score,
                "holdout_score": cand.holdout_score,
                "best": best is not None and cand.candidate_id == best.candidate_id,
                "selected": self.selected is not None and cand.candidate_id == self.selected.candidate_id,
                "pruned": cand.pruned,
                "tunable": cand.tunable,
                "n_trials": len(cand.trials),
                "model": cand.agent.model,
                "tokens": cand.agent.total_tokens,
                "cost_usd": cand.agent.cost_usd,
                "agent_s": cand.agent.agent_duration_s,
                "created_at": cand.created_at,
                "finished_at": cand.finished_at,
                "minutes": minutes_since(cand.finished_at, started),
                "summary": cand.summary,
            }
            for name, value in (cand.metrics or {}).items():
                row[f"metric_{name}" if name in row else name] = value
            rows.append(row)
        frame = pd.DataFrame(rows)
        return frame.set_index("candidate_id") if rows else frame


def open_search(ref: str | None = None, *, config: Config | None = None) -> SearchOutcome:
    """A search that already exists, to read: `"<run-id>/<search-id>"`, a run
    id with one search, or None for the most recently active one. Raises
    LookupError when `ref` names no search. Opening never touches the search;
    a running one is followed live."""
    from hillclimb.config import Config
    from hillclimb.harness.store import open_store, resolve_search

    config = config if config is not None else Config.load()
    with closing(open_store(config)) as store:
        record = resolve_search(store, ref)
        journal = Journal(_ReadOnlyJournal(store.journal(record.key).records()))
        status = store.read_status(record.key)
    outcome = SearchOutcome(
        run_dir=record.search_dir.parents[1],
        search_dir=record.search_dir,
        selected=journal.selected_candidate(record.meta.higher_is_better, search_selection(record.meta)),
        state=record.state,
        error=status.last_error if status is not None else None,
        config=config,
    )
    outcome._records = _Records(journal=journal, meta=record.meta, status=status, state=record.state)
    return outcome
