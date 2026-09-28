"""The DataStore: where a search's records live, and where every view reads
them back.

hillclimb keeps four kinds of record per search — the run/search metadata,
the append-only candidate journal, the status heartbeat, and the stop/prune
command queue. A *DataStore* is the backend that holds all four:

- `FileDataStore` — the hillclimb folder, the zero-setup default:
  `runs/<run>/run.yaml`, `searches/<search>/{search.yaml, journal.jsonl,
  status.json, control/}`. Byte-compatible with every run dir written so far.
- `SqliteDataStore` — one database file (`store.backend: sqlite`). Safe for
  N concurrent engines of a parallel run (WAL + busy timeout) and the engine's
  worker threads (one connection behind a lock).

The engine is the single writer of a search's records whatever the backend;
viewers and control commands go through the same store. Candidate working
directories, `best/`, agent streams and logs, problems, knowledge YAML and
agent slots stay on the local filesystem in every backend: agents and
verifiers need real files. Search ids are still minted by atomic `mkdir`
(`dirs.allocate_search_dir`) because the directory must exist regardless.

`sync_store(src, dst)` imports one store into another (the way a database
backend picks up history written as files before it was configured).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from hillclimb.harness.candidate import utcnow
from hillclimb.config import Config
from hillclimb.harness.control import (
    ControlCommand,
    clear_stale_stops_dir,
    drain_commands_dir,
    write_command,
)
from hillclimb.harness.journal import FileJournal, JournalBackend
from hillclimb.harness.run import (
    SEARCHES_DIRNAME,
    RunMeta,
    SearchMeta,
    iter_run_dirs,
    iter_search_dirs,
    load_run_meta,
    load_search_meta,
    write_run_meta,
    write_search_meta,
)
from hillclimb.harness.status import SearchStatus, derive_state
from hillclimb.harness.status import read_status as read_status_file
from hillclimb.harness.status import write_status as write_status_file

SQLITE_SCHEMA_VERSION = 2

SearchKey = tuple[str, str]  # (run_id, search_id)


def key_for(search_dir: Path) -> SearchKey:
    """The store key of a search dir: `runs/<run>/searches/<search>`."""
    return (search_dir.parents[1].name, search_dir.name)


class SearchRecord(BaseModel):
    """One search as the views see it: its metadata, the run's display name,
    its state derived from the last status record, and its directory (where
    candidates live — always `runs_dir/<run>/searches/<search>`)."""

    meta: SearchMeta
    run_name: str
    state: str  # running | done | parked | stopped | failed | crashed | unknown
    search_dir: Path
    activity_at: str = ""  # last heartbeat, else started_at: what `latest` ranks on

    @property
    def key(self) -> SearchKey:
        return (self.meta.run_id, self.meta.search_id)

    @property
    def run_id(self) -> str:
        return self.meta.run_id

    @property
    def search_id(self) -> str:
        return self.meta.search_id

    @property
    def ref(self) -> str:
        return f"{self.meta.run_id}/{self.meta.search_id}"


@runtime_checkable
class DataStore(Protocol):
    """Records are upserts for runs/searches/status, appends for the journal,
    and a consume-once queue for commands. Ordering rules every backend
    honours: `runs()`/`searches()` by start time (ties by id), journal records
    in append order, commands in enqueue order."""

    runs_dir: Path

    def record_run(self, meta: RunMeta) -> None: ...
    def record_search(self, meta: SearchMeta) -> None: ...
    def runs(self) -> list[RunMeta]: ...
    def searches(
        self, problem_key: str | None = None, run_id: str | None = None
    ) -> list[SearchRecord]: ...
    def search(self, key: SearchKey) -> SearchRecord | None: ...
    def journal(self, key: SearchKey) -> JournalBackend: ...
    def write_status(self, key: SearchKey, status: SearchStatus) -> None: ...
    def read_status(self, key: SearchKey) -> SearchStatus | None: ...
    def enqueue_command(self, key: SearchKey, cmd: ControlCommand) -> None: ...
    def drain_commands(self, key: SearchKey) -> list[ControlCommand]: ...
    def clear_stale_stops(self, key: SearchKey) -> None: ...
    def close(self) -> None: ...


def _sorted_searches(records: list[SearchRecord]) -> list[SearchRecord]:
    return sorted(records, key=lambda r: (r.meta.started_at, r.meta.run_id, r.meta.search_id))


def _record(meta: SearchMeta, run_name: str, status: SearchStatus | None, search_dir: Path) -> SearchRecord:
    return SearchRecord(
        meta=meta,
        run_name=run_name,
        state=derive_state(status),
        search_dir=search_dir,
        activity_at=max(status.updated_at if status else "", meta.started_at),
    )


class FileDataStore:
    """The hillclimb folder."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def search_dir(self, key: SearchKey) -> Path:
        run_id, search_id = key
        return self.runs_dir / run_id / SEARCHES_DIRNAME / search_id

    # --- runs / searches ---

    def record_run(self, meta: RunMeta) -> None:
        write_run_meta(self.runs_dir / meta.run_id, meta)

    def record_search(self, meta: SearchMeta) -> None:
        write_search_meta(self.search_dir((meta.run_id, meta.search_id)), meta)

    def runs(self) -> list[RunMeta]:
        found = [load_run_meta(d) for d in iter_run_dirs(self.runs_dir)]
        return sorted((m for m in found if m is not None), key=lambda m: (m.started_at, m.run_id))

    def searches(
        self, problem_key: str | None = None, run_id: str | None = None
    ) -> list[SearchRecord]:
        records = []
        run_dirs = [self.runs_dir / run_id] if run_id else iter_run_dirs(self.runs_dir)
        for run_dir in run_dirs:
            run_meta = load_run_meta(run_dir)
            if run_meta is None:
                continue
            for search_dir in iter_search_dirs(run_dir):
                meta = load_search_meta(search_dir)
                if meta is None or (problem_key and meta.problem_key != problem_key):
                    continue
                records.append(
                    _record(meta, run_meta.name or run_dir.name, read_status_file(search_dir), search_dir)
                )
        return _sorted_searches(records)

    def search(self, key: SearchKey) -> SearchRecord | None:
        search_dir = self.search_dir(key)
        meta = load_search_meta(search_dir)
        if meta is None:
            return None
        run_meta = load_run_meta(self.runs_dir / key[0])
        run_name = run_meta.name if run_meta and run_meta.name else key[0]
        return _record(meta, run_name, read_status_file(search_dir), search_dir)

    # --- journal / status / commands ---

    def journal(self, key: SearchKey) -> JournalBackend:
        return FileJournal(self.search_dir(key) / "journal.jsonl")

    def write_status(self, key: SearchKey, status: SearchStatus) -> None:
        write_status_file(self.search_dir(key), status)

    def read_status(self, key: SearchKey) -> SearchStatus | None:
        return read_status_file(self.search_dir(key))

    def enqueue_command(self, key: SearchKey, cmd: ControlCommand) -> None:
        write_command(self.search_dir(key), cmd)

    def drain_commands(self, key: SearchKey) -> list[ControlCommand]:
        return drain_commands_dir(self.search_dir(key))

    def clear_stale_stops(self, key: SearchKey) -> None:
        clear_stale_stops_dir(self.search_dir(key))

    def close(self) -> None:
        pass


_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    record      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS searches (
    run_id       TEXT NOT NULL,
    search_id    TEXT NOT NULL,
    search_uid   TEXT NOT NULL,
    problem_key  TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    record       TEXT NOT NULL,
    PRIMARY KEY (run_id, search_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS searches_uid ON searches (search_uid);
CREATE INDEX IF NOT EXISTS searches_problem_key ON searches (problem_key, started_at);
CREATE TABLE IF NOT EXISTS journal (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    search_id    TEXT NOT NULL,
    event        TEXT NOT NULL,
    candidate_id TEXT,
    record       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS journal_search ON journal (run_id, search_id, seq);
CREATE TABLE IF NOT EXISTS status (
    run_id     TEXT NOT NULL,
    search_id  TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    record     TEXT NOT NULL,
    PRIMARY KEY (run_id, search_id)
);
CREATE TABLE IF NOT EXISTS commands (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    search_id    TEXT NOT NULL,
    action       TEXT NOT NULL,
    record       TEXT NOT NULL,
    consumed_at  TEXT
);
CREATE INDEX IF NOT EXISTS commands_pending ON commands (run_id, search_id, consumed_at);
"""


class _SqliteJournal:
    """JournalBackend over the store's `journal` table; append order = seq."""

    def __init__(self, store: SqliteDataStore, key: SearchKey):
        self.store = store
        self.key = key

    def records(self) -> list[dict]:
        import json

        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT record FROM journal WHERE run_id = ? AND search_id = ? ORDER BY seq",
                self.key,
            ).fetchall()
        return [json.loads(row["record"]) for row in rows]

    def append(self, record: dict) -> None:
        import json

        with self.store._lock, self.store._conn:
            self.store._conn.execute(
                "INSERT INTO journal (run_id, search_id, event, candidate_id, record) VALUES (?, ?, ?, ?, ?)",
                (*self.key, record.get("event", ""), record.get("candidate_id"), json.dumps(record)),
            )


class SqliteDataStore:
    """One SQLite file. Every row keeps the full record as JSON next to the
    columns the views filter and sort on, so the schema can grow columns
    without losing anything. Writes no yaml: the search dir holds only what
    must be files (candidates/, best/)."""

    def __init__(self, path: Path, runs_dir: Path):
        self.path = path
        self.runs_dir = runs_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SQLITE_DDL)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SQLITE_SCHEMA_VERSION),),
            )

    def search_dir(self, key: SearchKey) -> Path:
        run_id, search_id = key
        return self.runs_dir / run_id / SEARCHES_DIRNAME / search_id

    # --- runs / searches ---

    def record_run(self, meta: RunMeta) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO runs (run_id, name, started_at, record) VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET name=excluded.name,
                   started_at=excluded.started_at, record=excluded.record""",
                (meta.run_id, meta.name, meta.started_at, meta.model_dump_json()),
            )

    def record_search(self, meta: SearchMeta) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO searches (run_id, search_id, search_uid, problem_key, started_at, record)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, search_id) DO UPDATE SET search_uid=excluded.search_uid,
                   problem_key=excluded.problem_key, started_at=excluded.started_at,
                   record=excluded.record""",
                (
                    meta.run_id, meta.search_id, meta.search_uid, meta.problem_key,
                    meta.started_at, meta.model_dump_json(),
                ),
            )

    def runs(self) -> list[RunMeta]:
        with self._lock:
            rows = self._conn.execute("SELECT record FROM runs ORDER BY started_at, run_id").fetchall()
        return [RunMeta.model_validate_json(row["record"]) for row in rows]

    def _search_rows(self, where: str, params: tuple) -> list[SearchRecord]:
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT s.record AS record, r.name AS run_name, st.record AS status
                    FROM searches s
                    LEFT JOIN runs r ON r.run_id = s.run_id
                    LEFT JOIN status st ON st.run_id = s.run_id AND st.search_id = s.search_id
                    {where} ORDER BY s.started_at, s.run_id, s.search_id""",
                params,
            ).fetchall()
        records = []
        for row in rows:
            meta = SearchMeta.model_validate_json(row["record"])
            status = SearchStatus.model_validate_json(row["status"]) if row["status"] else None
            records.append(
                _record(meta, row["run_name"] or meta.run_id, status, self.search_dir((meta.run_id, meta.search_id)))
            )
        return _sorted_searches(records)

    def searches(
        self, problem_key: str | None = None, run_id: str | None = None
    ) -> list[SearchRecord]:
        clauses, params = [], []
        if problem_key:
            clauses.append("s.problem_key = ?")
            params.append(problem_key)
        if run_id:
            clauses.append("s.run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._search_rows(where, tuple(params))

    def search(self, key: SearchKey) -> SearchRecord | None:
        rows = self._search_rows("WHERE s.run_id = ? AND s.search_id = ?", key)
        return rows[0] if rows else None

    # --- journal / status / commands ---

    def journal(self, key: SearchKey) -> JournalBackend:
        return _SqliteJournal(self, key)

    def write_status(self, key: SearchKey, status: SearchStatus) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO status (run_id, search_id, updated_at, record) VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id, search_id) DO UPDATE SET updated_at=excluded.updated_at,
                   record=excluded.record""",
                (*key, status.updated_at, status.model_dump_json()),
            )

    def read_status(self, key: SearchKey) -> SearchStatus | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM status WHERE run_id = ? AND search_id = ?", key
            ).fetchone()
        return SearchStatus.model_validate_json(row["record"]) if row else None

    def enqueue_command(self, key: SearchKey, cmd: ControlCommand) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO commands (run_id, search_id, action, record) VALUES (?, ?, ?, ?)",
                (*key, cmd.action, cmd.model_dump_json()),
            )

    def drain_commands(self, key: SearchKey) -> list[ControlCommand]:
        """Pending commands in enqueue order, marked consumed in the same
        statement — consumed before applied, like the file backend's unlink."""
        with self._lock, self._conn:
            rows = self._conn.execute(
                """UPDATE commands SET consumed_at = ?
                   WHERE run_id = ? AND search_id = ? AND consumed_at IS NULL
                   RETURNING id, record""",
                (utcnow(), *key),
            ).fetchall()
        rows.sort(key=lambda row: row["id"])
        return [ControlCommand.model_validate_json(row["record"]) for row in rows]

    def clear_stale_stops(self, key: SearchKey) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE commands SET consumed_at = ?
                   WHERE run_id = ? AND search_id = ? AND consumed_at IS NULL AND action = 'stop'""",
                (utcnow(), *key),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_store(config: Config) -> DataStore:
    """The store the hillclimb dir is configured for (`store.backend`)."""
    backend = config.store.backend
    if backend == "files":
        return FileDataStore(config.paths.runs_dir)
    if backend == "sqlite":
        return SqliteDataStore(config.store.sqlite_path, runs_dir=config.paths.runs_dir)
    raise ValueError(f"unknown store backend {backend!r} (expected 'files' or 'sqlite')")


def sync_store(source: DataStore, target: DataStore) -> dict[str, int]:
    """Import every search `source` holds that `target` does not yet: run and
    search metadata, the journal in append order, the latest status. Searches
    already in `target` are left untouched (their engine owns them), so the
    import is idempotent. Pending commands are not carried over."""
    counts = {"runs": 0, "searches": 0, "records": 0}
    known_runs = {run.run_id for run in target.runs()}
    for run in source.runs():
        if run.run_id in known_runs:
            continue
        target.record_run(run)
        counts["runs"] += 1
    for record in source.searches():
        if target.search(record.key) is not None:
            continue
        target.record_search(record.meta)
        counts["searches"] += 1
        sink = target.journal(record.key)
        for entry in source.journal(record.key).records():
            sink.append(entry)
            counts["records"] += 1
        status = source.read_status(record.key)
        if status is not None:
            target.write_status(record.key, status)
    return counts


def latest_search(store: DataStore) -> SearchRecord | None:
    """The most recently active search: newest heartbeat, else newest start."""
    records = store.searches()
    return max(records, key=lambda r: (r.activity_at, r.ref)) if records else None


def running_searches(store: DataStore) -> list[SearchRecord]:
    """Every search whose engine is live right now."""
    return [r for r in store.searches() if r.state == "running"]


def resolve_search(store: DataStore, ref: str | None) -> SearchRecord:
    """Resolve a search reference, raising LookupError with a user-facing
    message when it does not name exactly one search:

    - `latest` (or empty) — the most recently active search
    - `<run-id>/<search-id>` — exact address
    - `<run-id>` — the run's only search; error listing choices if several
    """
    if not ref or ref == "latest":
        latest = latest_search(store)
        if latest is None:
            raise LookupError(f"No searches found in {store.runs_dir}")
        return latest
    if "/" in ref:
        run_id, _, search_id = ref.partition("/")
        record = store.search((run_id, search_id))
        if record is None:
            raise LookupError(f"No search at {store.search_dir((run_id, search_id))}")
        return record
    if not any(run.run_id == ref for run in store.runs()):
        raise LookupError(f"No run named {ref!r} in {store.runs_dir}")
    records = store.searches(run_id=ref)
    if not records:
        raise LookupError(f"Run {ref} has no searches")
    if len(records) > 1:
        choices = "\n".join(f"  {r.ref}" for r in records)
        raise LookupError(f"Run {ref} has {len(records)} searches; pick one:\n{choices}")
    return records[0]


__all__ = [
    "DataStore",
    "FileDataStore",
    "SearchKey",
    "SearchRecord",
    "SqliteDataStore",
    "key_for",
    "latest_search",
    "open_store",
    "resolve_search",
    "running_searches",
    "sync_store",
]
