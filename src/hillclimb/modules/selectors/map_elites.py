"""`map-elites`: OpenEvolve's MAP-Elites program database as a selector.

Its feature grid, islands, migration and elite archive — and its
parent/inspiration sampler — decide WHICH candidate to expand and what to
show beside it. Everything else stays hillclimb's: the policy's schedule, the
mutation itself (a coding agent call rather than a one-shot LLM diff), evaluation
(the verifier) and journaling.

What is outsourced: parent selection (exploration / exploitation / weighted,
per island), inspiration sampling, feature binning, island migration.

Replay contract: the database is a function of the journal alone (`sync`),
and every random draw is seeded from (random_seed, journal position) inside a
saved/restored global-RNG window — OpenEvolve samples through the `random`
module — so `resume` and a fresh process make the same picks from the same
journal.

Params (`selector_params`, all optional):
  num_islands, population_size, archive_size, feature_dimensions,
  feature_bins, exploration_ratio, exploitation_ratio, elite_selection_ratio,
  migration_interval, migration_rate, random_seed
      → passed straight to openevolve's DatabaseConfig
  num_inspirations (2)   inspirations copied in as candidate_<i>.py

Feature dimensions: the built-ins `complexity` (code length), `diversity`
(edit distance to a reference set) and `score` need nothing from the
problem; any other name must be a numeric key the verifier writes next to
`score` in `$HILLCLIMB_RESULT` (journaled as `Trial.metrics`).

Needs the optional extra: pip install 'hillclimb[openevolve]'
"""

from __future__ import annotations

import random
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from hillclimb.sdk import Candidate, SearchState, Selection, SelectorPolicy, improvable

BUILTIN_FEATURES = ("complexity", "diversity", "score")
DEFAULT_SEED = 42
# the selector's own knobs; everything else is a DatabaseConfig field
OWN_PARAMS = ("num_inspirations", "random_seed")


def _require_openevolve():
    try:
        from openevolve.config import DatabaseConfig
        from openevolve.database import Program, ProgramDatabase
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the map-elites selector needs the `openevolve` package: "
            "pip install 'hillclimb[openevolve]'"
        ) from exc
    return DatabaseConfig, Program, ProgramDatabase


def _timestamp(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def known_params() -> tuple[str, ...]:
    """Every setting the selector takes: the schedule every selector has,
    its own, and the database's."""
    DatabaseConfig, _, _ = _require_openevolve()
    return (*SelectorPolicy.defaults(), *OWN_PARAMS, *sorted(f.name for f in dataclass_fields(DatabaseConfig)))


class MapElites(SelectorPolicy):
    """A population kept diverse over feature dimensions, on islands."""

    name = "map-elites"
    DEFAULTS = {"num_inspirations": 2, "random_seed": DEFAULT_SEED}

    def __init__(self, params: dict | None = None, **knobs):
        super().__init__(params, **knobs)
        DatabaseConfig, self._Program, self._ProgramDatabase = _require_openevolve()
        db_fields = {f.name for f in dataclass_fields(DatabaseConfig)}
        unknown = sorted(set(self.params) - db_fields - set(OWN_PARAMS) - set(SelectorPolicy.defaults()))
        if unknown:
            raise ValueError(
                f"map-elites has no setting {unknown} (it takes the schedule's "
                f"{', '.join(sorted(SelectorPolicy.defaults()))}, num_inspirations, and openevolve's "
                f"DatabaseConfig fields: {', '.join(sorted(db_fields))})"
            )
        self.num_inspirations = int(self.param("num_inspirations"))
        self.seed = int(self.param("random_seed"))
        db_kwargs = {k: v for k, v in self.params.items() if k in db_fields}
        db_kwargs.setdefault("random_seed", self.seed)
        db_kwargs.setdefault("log_prompts", False)
        db_kwargs["in_memory"] = True
        db_kwargs["db_path"] = None
        self.db_config = DatabaseConfig(**db_kwargs)
        self.feature_dimensions = list(self.db_config.feature_dimensions)
        self._reset()

    def _reset(self) -> None:
        with self._seeded(0):  # ProgramDatabase.__init__ seeds the global RNG
            self.db = self._ProgramDatabase(self.db_config)
        self._added = 0  # programs added, drives island round-robin for drafts
        # what the database was built from, in journal order: (id, score)
        self._synced: list[tuple[str, float]] = []

    # --- Selector ---

    def sync(self, state: SearchState) -> None:
        """Bin every scored candidate into the grid. The database is a
        function of the journal alone — the scored candidates in JOURNAL
        order, each added at its own position — never of the order results
        landed in or of when this was called, so a resumed search rebuilds
        exactly the database the live one had. A result that lands out of
        order, or a tune trial that moves a score already binned, rebuilds
        it. Buggy/abandoned candidates are not programs (OpenEvolve drops
        failed evaluations too); the policy's debug step recovers them."""
        wanted = [
            (position, candidate)
            for position, candidate in enumerate(state.journal.candidates.values(), 1)
            if self._is_program(candidate)
        ]
        signature = [(c.candidate_id, float(c.val_score)) for _, c in wanted]
        if signature[: len(self._synced)] != self._synced:
            self._reset()
        for position, candidate in wanted[len(self._synced):]:
            self._add(state, candidate, position)
            self._synced.append((candidate.candidate_id, float(candidate.val_score)))

    def select(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        pool = self._parent_pool(state)
        if not pool:
            return None
        iteration = len(state.journal.candidates)
        island = iteration % self.db_config.num_islands
        with self._seeded(iteration):
            parent, inspirations = self.db.sample_from_island(island, self.num_inspirations)
            if parent.id not in pool:
                # the sampler may return a code-less floor or a pruned lineage;
                # fall back to the island's best improvable program
                parent = self._best_in(pool, island) or parent
            cell = self._cell(parent)  # inside the window: the grid lookup may draw
        inspirations = [p for p in inspirations if p.id != parent.id and p.id in pool]
        island = int(parent.metadata.get("island", island))
        return Selection(
            target_id=parent.id,
            inspiration_ids=tuple(p.id for p in inspirations),
            prompt_context=self._render_context(state, parent, inspirations, island, cell),
            meta={
                "island": island,
                "cell": cell,
                "parent_fitness": self._fitness(parent),
                "inspirations": [p.id for p in inspirations],
            },
        )

    def creation_meta(self, state: SearchState) -> dict:
        self.sync(state)
        return {"island": self._added % self.db_config.num_islands}

    # --- introspection shared with the TUI/status surfaces ---

    def island_stats(self) -> list[dict]:
        return self.db.get_island_stats()

    # --- the database ---

    @staticmethod
    def _is_program(candidate: Candidate) -> bool:
        if candidate.val_score is None:
            return False
        # a declared floor has no code to evolve from
        return not (candidate.operator == "baseline" and not improvable(candidate))

    def _add(self, state: SearchState, candidate: Candidate, position: int) -> None:
        metrics = self._metrics(state, candidate)
        missing = [
            dim for dim in self.feature_dimensions
            if dim not in BUILTIN_FEATURES and dim not in metrics
        ]
        if missing:
            raise ValueError(
                f"map-elites: feature_dimensions {missing} are not in the "
                f"verifier's result metrics {sorted(metrics)} — write them next to "
                f"`score` in $HILLCLIMB_RESULT or drop them from selector_params"
            )
        parent = (
            self.db.programs.get(candidate.parent_id) if candidate.parent_id else None
        )
        program = self._Program(
            id=candidate.candidate_id,
            code=self._code(candidate),
            parent_id=parent.id if parent is not None else None,
            generation=parent.generation + 1 if parent is not None else 0,
            timestamp=_timestamp(candidate.finished_at or candidate.created_at),
            iteration_found=position,
            metrics=metrics,
            metadata={"candidate_id": candidate.candidate_id, "operator": candidate.operator},
        )
        island = candidate.climber_meta.get("island")
        if island is None:
            island = (
                parent.metadata.get("island")
                if parent is not None and "island" in parent.metadata
                else self._added % self.db_config.num_islands
            )
        island = int(island) % self.db_config.num_islands
        with self._seeded(position):
            self.db.add(program, iteration=program.iteration_found, target_island=island)
            self.db.increment_island_generation(island_idx=island)
            if self.db.should_migrate():
                self.db.migrate_programs()
        self._added += 1

    def _parent_pool(self, state: SearchState) -> dict[str, Candidate]:
        """Programs in the database that can still be expanded: scored,
        unpruned, with a solution.py on disk."""
        pool = {}
        for pid in self.db.programs:
            candidate = state.journal.candidates.get(pid)
            if candidate is None or candidate.pruned or not improvable(candidate):
                continue
            if not (Path(candidate.candidate_dir) / "solution.py").exists():
                continue
            pool[pid] = candidate
        return pool

    def _best_in(self, pool: dict[str, Candidate], island: int):
        members = [
            self.db.programs[pid]
            for pid in pool
            if self.db.programs[pid].metadata.get("island") == island
        ] or [self.db.programs[pid] for pid in pool]
        if not members:
            return None
        return max(members, key=self._fitness)

    def _metrics(self, state: SearchState, candidate: Candidate) -> dict[str, float]:
        score = float(candidate.val_score)
        fitness = score if state.higher_is_better else -score  # OpenEvolve maximizes
        return {"combined_score": fitness, **candidate.metrics}

    def _fitness(self, program) -> float:
        return float(program.metrics.get("combined_score", 0.0))

    def _code(self, candidate: Candidate) -> str:
        path = Path(candidate.candidate_dir) / "solution.py"
        try:
            return path.read_text()
        except OSError:
            return ""

    def _cell(self, program) -> list[int]:
        try:
            return [int(c) for c in self.db._calculate_feature_coords(program)]
        except Exception:  # noqa: BLE001 - a missing dim was already rejected in sync
            return []

    def _render_context(self, state, parent, inspirations, island: int, cell: list[int]) -> str:
        direction = "higher" if state.higher_is_better else "lower"
        dims = ", ".join(
            f"{dim}={bin_}" for dim, bin_ in zip(self.feature_dimensions, cell)
        ) or "n/a"
        lines = [
            "This search runs OpenEvolve's MAP-Elites strategy over hillclimb's "
            "operators: a population of solutions kept diverse across feature "
            f"dimensions ({', '.join(self.feature_dimensions)}) on "
            f"{self.db_config.num_islands} islands.",
            f"Parent: {parent.id} on island {island}, score {self._score(state, parent)} "
            f"({direction} is better), grid cell [{dims}], "
            f"{len(parent.code)} chars of code.",
        ]
        if inspirations:
            lines.append(
                "Inspirations from the same island were copied into this directory "
                "— read them for ideas, never resubmit one verbatim:"
            )
            for i, program in enumerate(inspirations, 1):
                lines.append(
                    f"- candidate_{i}.py: {program.id}, score {self._score(state, program)}, "
                    f"{len(program.code)} chars"
                )
        lines.append(
            "Make one deliberate change to solution.py that could move the score "
            "OR land in an unexplored region of the feature space (a different "
            "approach or complexity level) — both are wins for this strategy."
        )
        return "\n".join(lines)

    def _score(self, state, program) -> str:
        fitness = self._fitness(program)
        return f"{fitness if state.higher_is_better else -fitness:.6g}"

    def _seeded(self, step: int) -> "_SeededRNG":
        return _SeededRNG(self.seed, step)


class _SeededRNG:
    def __init__(self, seed: int, step: int):
        self.seed = seed
        self.step = step
        self._saved = None

    def __enter__(self):
        self._saved = random.getstate()
        random.seed(f"hillclimb-openevolve:{self.seed}:{self.step}")
        return self

    def __exit__(self, *exc):
        random.setstate(self._saved)
        return False
