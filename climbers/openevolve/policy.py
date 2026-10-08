"""The bundled `openevolve` climber, as one file: OpenEvolve's MAP-Elites
program database as the selector policy (π_sel, `MapElites`) and the same
operator policy the `greedy` climber runs (π_op, `Greedy`), with tuning off.
Everything that decides is written out here — there is no hidden schedule in
a base class — so `hillclimb climber get openevolve` copies this file as is.

The selector's feature grid, islands, migration and elite archive — and its
parent/inspiration sampler — decide WHICH candidate to expand and what to
show beside it. Everything else stays hillclimb's: the schedule around that
choice (a failing tip first, the ensemble window — off here by default —
`num_drafts` roots before anything is built on), the mutation itself (a
coding agent call rather than a one-shot LLM diff), evaluation (the verifier)
and journaling. What is outsourced: parent selection (exploration /
exploitation / weighted, per island), inspiration sampling, feature binning,
island migration.

Replay contract: the database is a function of the journal alone (`sync`),
and every random draw is seeded from (random_seed, journal position) inside a
saved/restored global-RNG window — OpenEvolve samples through the `random`
module — so `resume` and a fresh process make the same picks from the same
journal.

Selector knobs (`selector_params`, default in brackets):
  num_drafts (3)                     root candidates before anything is built on
  debug (True)                       repair failing tips at all
  max_debug_depth (3)                failed fixes per failing chain
  ensemble (False)                   combine the top candidates in the final window
  ensemble_reserve_fraction (0.2)    final slice of the budget reserved for it
  ensemble_top_k (3)                 candidates combined
  ensemble_max_attempts (2)          combinations tried in the window
  num_inspirations (2)               inspirations copied in as candidate_<i>.py
  random_seed (42)                   the seed every draw derives from
  num_islands, population_size, archive_size, feature_dimensions,
  feature_bins, exploration_ratio, exploitation_ratio, elite_selection_ratio,
  migration_interval, migration_rate
      → passed straight to openevolve's DatabaseConfig

Feature dimensions: the built-ins `complexity` (code length), `diversity`
(edit distance to a reference set) and `score` need nothing from the
problem; any other name must be a numeric key the verifier writes next to
`score` in `$HILLCLIMB_RESULT` (journaled as `Trial.metrics`).

Operator policy knobs (`params`, default in brackets):
  complexity_start (0)  offset of the draft-complexity cue (memory may have learned one)
  tune_budget (0)       extra trials per candidate beyond its defaults trial; 0 = off
  tune_gate ("band")    which chosen candidates get tuned: "band" = within the
                        accept band of the current best (best included), "best" =
                        the best only, "always" = every scored one
  tune_parallel (1)     tune jobs in flight per candidate (>1 engages the
                        tuner's constant liar)
  tune_burst (2)        tune trials released between coding agent proposals, so tuning
                        interleaves with improving instead of starving it

Needs the optional extra: pip install 'hillclimb[openevolve]'
"""

from __future__ import annotations

import random
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from hillclimb.sdk import (
    TUNE_ACTION,
    Action,
    Candidate,
    OperatorPolicy,
    SearchState,
    Selection,
    SelectorPolicy,
    improvable,
    improves,
    top_distinct,
)

BUILTIN_FEATURES = ("complexity", "diversity", "score")
DEFAULT_SEED = 42
# the selector's own knobs beyond the schedule; everything else is a DatabaseConfig field
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
    """Every setting the selector takes: the schedule, its own, and the
    database's."""
    DatabaseConfig, _, _ = _require_openevolve()
    return (*MapElites.defaults(), *sorted(f.name for f in dataclass_fields(DatabaseConfig)))


class MapElites(SelectorPolicy):
    """A population kept diverse over feature dimensions, on islands."""

    name = "map-elites"
    DEFAULTS = {
        "num_drafts": 3,
        "debug": True,
        "max_debug_depth": 3,
        "ensemble": False,
        "ensemble_reserve_fraction": 0.2,
        "ensemble_top_k": 3,
        "ensemble_max_attempts": 2,
        "num_inspirations": 2,
        "random_seed": DEFAULT_SEED,
    }

    def __init__(self, params: dict | None = None, **knobs):
        super().__init__(params, **knobs)
        DatabaseConfig, self._Program, self._ProgramDatabase = _require_openevolve()
        db_fields = {f.name for f in dataclass_fields(DatabaseConfig)}
        unknown = sorted(set(self.params) - db_fields - set(self.defaults()))
        if unknown:
            raise ValueError(
                f"map-elites has no setting {unknown} (it takes the schedule's "
                f"{', '.join(sorted(k for k in self.defaults() if k not in OWN_PARAMS))}, "
                f"{', '.join(OWN_PARAMS)}, and openevolve's "
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

    # --- the schedule: which node(s) the next attempt starts from ---

    def schedule(self, state: SearchState, *, busy: frozenset[str] | set[str] = frozenset()) -> Selection | None:
        """In order: a failing tip (repair comes first), the top candidates
        once the final window is open (`combine=True`), None while fewer than
        `num_drafts` roots exist (a root step), else whatever `select`
        chooses. `busy` holds the candidates an in-flight attempt is already
        building on."""
        self.sync(state)
        tip = self.debuggable_tip(state)
        if tip is not None:
            return Selection(tip.candidate_id)
        if self.should_combine(state):
            picks = self.combine_candidates(state)
            return Selection(
                picks[0].candidate_id, inspiration_ids=tuple(c.candidate_id for c in picks), combine=True,
            )
        if self.prospective_branches(state) < int(self.param("num_drafts")):
            return None
        return self.select(state, busy=busy)

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

    # --- the schedule's questions ---

    def debuggable_tip(self, state: SearchState) -> Candidate | None:
        """Newest failing/buggy candidate with no active child and chain depth
        under `max_debug_depth`. In serial history this is exactly the serial
        debug rule."""
        if not bool(self.param("debug")):
            return None
        journal = state.journal
        for candidate in reversed(list(journal.candidates.values())):
            if candidate.status not in ("failing", "buggy") or candidate.pruned:
                continue
            children = journal.children(candidate.candidate_id, include_pruned=True)
            if any(c.status in ("pending", "passing", "failing", "buggy") for c in children):
                continue
            chain = journal.debug_chain(candidate.candidate_id)
            depth = sum(1 for c in chain if c.operator == "debug")
            if depth < int(self.param("max_debug_depth")):
                return candidate
        return None

    def prospective_branches(self, state: SearchState) -> int:
        """Draft branches whose subtree holds a scored OR pending candidate —
        in-flight work counts toward a number-of-drafts target."""
        journal = state.journal
        count = 0
        for draft in journal.drafts():
            frontier = [draft]
            while frontier:
                candidate = frontier.pop()
                if candidate.is_scored or candidate.status == "pending":
                    count += 1
                    break
                frontier.extend(journal.children(candidate.candidate_id))
        return count

    def in_ensemble_window(self, state: SearchState) -> bool:
        # window sits ABOVE the stop margin, else margin swallows it: with a
        # 45m budget, reserve(540s) - margin(300s) left a 240s slot that one
        # improve cycle stepped over entirely
        budget = state.budget
        reserve = budget.total_s * float(self.param("ensemble_reserve_fraction"))
        return budget.remaining_s <= reserve + budget.stop_margin_s

    def should_combine(self, state: SearchState) -> bool:
        if not bool(self.param("ensemble")) or not self.in_ensemble_window(state):
            return False
        attempts = sum(1 for c in state.journal.candidates.values() if c.kind == "combine")
        if attempts >= int(self.param("ensemble_max_attempts")) or self.combine_succeeded(state):
            return False
        return len(self.combine_candidates(state)) >= 2

    def combine_succeeded(self, state: SearchState) -> bool:
        for candidate in state.journal.candidates.values():
            if candidate.status != "passing":
                continue
            root = state.journal.debug_chain(candidate.candidate_id)[0]
            if root.kind == "combine":
                return True
        return False

    def combine_candidates(self, state: SearchState) -> list[Candidate]:
        """Top-k scored candidates by val score that are not combinations
        themselves, deduped by script content so near-identical improves
        don't fill the slots."""
        return top_distinct(state, int(self.param("ensemble_top_k")), skip_kind="combine")

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


class Greedy(OperatorPolicy):
    """Draft a root, debug a failing tip, ensemble the chosen set, and tune the chosen candidate before improving it."""

    name = "greedy"
    DEFAULTS = {
        "complexity_start": 0,
        "tune_budget": 0,
        "tune_gate": "band",
        "tune_parallel": 1,
        "tune_burst": 2,
    }

    # --- the contract ---

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        """The operator for what the selector chose, as an Action; None =
        hold (keep the slot empty until an in-flight result lands). No node:
        draft; the nodes of a combination: ensemble, once nothing is in
        flight; a failing node: debug; a scored node: a tune trial while it
        has budget for one, else improve."""
        if selection is None:
            return self.draft_action(state)
        if selection.combine:
            if state.inflight:
                return None  # drain: the inputs of a combination snapshot at launch
            return Action(
                operator="ensemble",
                target_id=selection.target_id,
                inspiration_ids=tuple(selection.inspiration_ids),
                extra_prompt_context=selection.prompt_context,
                climber_meta=dict(selection.meta),
            )
        node = state.journal.candidates[selection.target_id]
        if node.status in ("failing", "buggy"):
            return Action(operator="debug", target_id=node.candidate_id)
        if node.is_scored and self.tune_now(state, node):
            return Action(operator=TUNE_ACTION, target_id=node.candidate_id)
        return self.expand_action(state, selection)

    def action_for(self, state: SearchState, operator: str, target_id: str | None) -> Action:
        """Fully-populated Action for an explicitly requested operator — the
        run_operator path, where the harness names the move and the policy
        fills in its decision-time details."""
        selector = self.selector
        if operator == "draft":
            return self.draft_action(state)
        if operator == "ensemble":
            picks = selector.combine_candidates(state) if selector is not None else []
            return Action(
                operator="ensemble",
                target_id=target_id or (picks[0].candidate_id if picks else None),
                inspiration_ids=tuple(c.candidate_id for c in picks),
            )
        if operator == "improve" and target_id is None and selector is not None:
            selector.sync(state)
            chosen = selector.select(state)
            if chosen is not None:
                return self.expand_action(state, chosen)
        return Action(operator=operator, target_id=target_id)

    # --- the moves ---

    def draft_action(self, state: SearchState) -> Action:
        """A root step: a fresh draft, with the complexity cue and whatever
        the selector tags new roots with."""
        selector = self.selector
        return Action(
            operator="draft",
            args={"complexity": self.draft_complexity(state)},
            climber_meta=selector.creation_meta(state) if selector is not None else {},
        )

    def expand_action(self, state: SearchState, selection: Selection, operator: str = "improve") -> Action:
        """Build on the chosen node with a `refine` operator (`improve`)."""
        return Action(
            operator=operator,
            target_id=selection.target_id,
            inspiration_ids=tuple(selection.inspiration_ids),
            extra_prompt_context=selection.prompt_context,
            climber_meta=dict(selection.meta),
        )

    def draft_complexity(self, state: SearchState) -> str:
        """The complexity cue for the next draft: it escalates per draft."""
        index = len(state.journal.drafts()) + int(self.param("complexity_start"))
        return "minimal" if index == 0 else "moderate" if index == 1 else "advanced"

    # --- tune ---

    def tune_param(self, name: str):
        return self.param(name)

    def tune_now(self, state: SearchState, candidate: Candidate) -> bool:
        """Spend the next tune trial on the chosen candidate? Derived from
        the journal + in-flight refs only: `spent` on a candidate is its
        trial count beyond the defaults trial plus its in-flight tune jobs, so
        a resumed search continues exactly where a killed one stopped."""
        budget = int(self.tune_param("tune_budget"))
        if budget <= 0 or not candidate.tunable:
            return False
        journal = state.journal
        best = journal.best_candidate(state.higher_is_better)
        if not self._tune_gate_passes(
            candidate, best, state.accept_band, str(self.tune_param("tune_gate")), state.higher_is_better
        ):
            return False
        headroom = state.budget.remaining_s - state.budget.stop_margin_s
        if headroom < _trial_cost_s(candidate):
            return False  # a trial of this code could not finish before the wall
        inflight = sum(
            1 for ref in state.inflight
            if ref.operator == TUNE_ACTION and ref.candidate_id == candidate.candidate_id
        )
        spent = (len(candidate.trials) - 1) + inflight
        if spent >= budget or inflight >= int(self.tune_param("tune_parallel")):
            return False
        # interleave: allow `burst` tune trials per coding agent proposal made
        # since this candidate landed (later ids + in-flight coding agent jobs)
        agent_jobs = sum(1 for ref in state.inflight if ref.operator != TUNE_ACTION)
        agents_since = sum(1 for cid in journal.candidates if cid > candidate.candidate_id)
        return spent < int(self.tune_param("tune_burst")) * (1 + agents_since + agent_jobs)

    @staticmethod
    def _tune_gate_passes(
        candidate: Candidate, best: Candidate | None, band: float, gate: str, higher_is_better: bool
    ) -> bool:
        if gate == "always":
            return True
        if best is None or candidate.candidate_id == best.candidate_id:
            return True
        if gate == "best":
            return False
        # "band": the best does not beat this candidate by more than the band
        return not improves(
            best.val_score, candidate.val_score, higher_is_better=higher_is_better, band=band
        )


MIN_TRIAL_COST_S = 30.0


def _trial_cost_s(candidate: Candidate) -> float:
    """How long a trial of this candidate takes, from its own record: the
    best trial's replicate wall-clock (parallel replicates overlap, so the
    longest one), with a floor for near-instant verifiers."""
    best = candidate.best_trial
    if best is None:
        return MIN_TRIAL_COST_S
    durations = [r.duration_s for r in best.replicates if r.duration_s is not None]
    return max([MIN_TRIAL_COST_S, *durations]) * 1.5


# --- the climber -------------------------------------------------------------
# This file IS the climber (`climber: climbers/openevolve/policy.py` in hillclimb.yaml,
# `--climber climbers/openevolve/policy.py`): the two policies above, the operators that
# make an attempt (each renders its template under prompts/ beside this file — a
# climber file's prompts dir), the tuner and the memory. Edit anything here or a
# template and the next run climbs with the change; a started search keeps the
# copy it snapshotted.
from hillclimb import Climber
from hillclimb.memory import FilesMemory
from hillclimb.operators import Debug, Draft, Ensemble, Improve
from hillclimb.tuners import RandomSearch

climber = Climber(
    selector_policy=MapElites(),  # which candidate the next attempt starts from
    operator_policy=Greedy(),  # which operator to apply to it
    operators=[Draft(), Debug(), Improve(), Ensemble()],
    tuner=RandomSearch(),  # which parameter values a tunable candidate tries
    memory=FilesMemory(),  # what a search knows from earlier ones, and leaves for the next
    name='openevolve',
)
