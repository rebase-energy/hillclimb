"""OpenEvolve as a SearchPolicy: its MAP-Elites program database (feature
grid, islands, migration, elite archive) and its parent/inspiration sampler
decide WHAT to try next; hillclimb's harness, verifier and agentic operators
do everything else.

What is outsourced: parent selection (exploration / exploitation / weighted,
per island), inspiration sampling, feature binning, island migration. What is
not: the mutation itself (an `improve` agent call rather than a one-shot LLM
diff), evaluation (the verifier), journaling, and hillclimb's `debug` step,
which OpenEvolve has no equivalent of and which this policy keeps.

Replay contract (see policy.py): the database is rebuilt by `observe()` in
journal order on construction, and every random draw is seeded from
(random_seed, journal size) inside a saved/restored global-RNG window —
OpenEvolve samples through the `random` module — so `resume` and a fresh
process make the same proposals from the same journal.

Params (config `search.policy_params`, all optional):
  num_islands, population_size, archive_size, feature_dimensions,
  feature_bins, exploration_ratio, exploitation_ratio, elite_selection_ratio,
  migration_interval, migration_rate, random_seed
      → passed straight to openevolve's DatabaseConfig
  num_inspirations (2)   inspirations copied in as candidate_<i>.py
  debug (true)           keep hillclimb's debug-the-buggy-tip rule
  num_drafts             seed population size; default config.search.num_drafts

Feature dimensions: the built-ins `complexity` (code length), `diversity`
(edit distance to a reference set) and `score` need nothing from the
problem; any other name must be a numeric key the verifier writes next to
`score` in `$HILLCLIMB_RESULT` (journaled as `Trial.metrics`).
"""

from __future__ import annotations

import random
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from hillclimb.candidate import Candidate
from hillclimb.policies.greedy import GreedyPolicy, _improvable
from hillclimb.policy import Action, SearchView

BUILTIN_FEATURES = ("complexity", "diversity", "score")
DEFAULT_SEED = 42


def _require_openevolve():
    try:
        from openevolve.config import DatabaseConfig
        from openevolve.database import Program, ProgramDatabase
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the openevolve policy needs the `openevolve` package: "
            "pip install 'hillclimb[openevolve]'"
        ) from exc
    return DatabaseConfig, Program, ProgramDatabase


def _timestamp(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


class OpenEvolvePolicy:
    name = "openevolve"

    def __init__(self, params: dict | None = None, complexity_start: int = 0):
        DatabaseConfig, self._Program, ProgramDatabase = _require_openevolve()
        self.params = dict(params or {})
        self.num_inspirations = int(self.params.get("num_inspirations", 2))
        self.debug_enabled = bool(self.params.get("debug", True))
        self.seed = int(self.params.get("random_seed", DEFAULT_SEED))
        self._greedy = GreedyPolicy(complexity_start=complexity_start)  # draft/debug rules
        db_fields = {f.name for f in dataclass_fields(DatabaseConfig)}
        db_kwargs = {k: v for k, v in self.params.items() if k in db_fields}
        db_kwargs.setdefault("random_seed", self.seed)
        db_kwargs.setdefault("log_prompts", False)
        db_kwargs["in_memory"] = True
        db_kwargs["db_path"] = None
        self.db_config = DatabaseConfig(**db_kwargs)
        self.feature_dimensions = list(self.db_config.feature_dimensions)
        with self._seeded(0):  # ProgramDatabase.__init__ seeds the global RNG
            self.db = ProgramDatabase(self.db_config)
        self._added = 0  # programs added, drives island round-robin for drafts

    # --- SearchPolicy protocol ---

    def propose(self, view: SearchView) -> Action | None:
        if self.debug_enabled:
            tip = self._greedy.debuggable_tip(view)
            if tip is not None:
                return Action(operator="debug", target_id=tip.candidate_id)
        num_drafts = int(self.params.get("num_drafts", view.config.search.num_drafts))
        if self._greedy.prospective_branches(view) < num_drafts or not self._parent_pool(view):
            return self._draft_action(view)
        return self._evolve_action(view)

    def observe(self, view: SearchView, candidate: Candidate) -> None:
        """Bin every scored candidate into the grid. Buggy/abandoned ones are
        not programs (OpenEvolve drops failed evaluations too); the debug
        chain is hillclimb's way of recovering them."""
        if candidate.val_score is None or candidate.candidate_id in self.db.programs:
            return
        if candidate.operator == "baseline" and not _improvable(candidate):
            return  # a declared floor has no code to evolve from
        metrics = self._metrics(view, candidate)
        missing = [
            dim for dim in self.feature_dimensions
            if dim not in BUILTIN_FEATURES and dim not in metrics
        ]
        if missing:
            raise ValueError(
                f"openevolve policy: feature_dimensions {missing} are not in the "
                f"verifier's result metrics {sorted(metrics)} — write them next to "
                f"`score` in $HILLCLIMB_RESULT or drop them from policy_params"
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
            iteration_found=len(view.journal.candidates),
            metrics=metrics,
            metadata={"candidate_id": candidate.candidate_id, "operator": candidate.operator},
        )
        island = candidate.policy_meta.get("island")
        if island is None:
            island = (
                parent.metadata.get("island")
                if parent is not None and "island" in parent.metadata
                else self._added % self.db_config.num_islands
            )
        island = int(island) % self.db_config.num_islands
        with self._seeded(len(view.journal.candidates)):
            self.db.add(program, iteration=program.iteration_found, target_island=island)
            self.db.increment_island_generation(island_idx=island)
            if self.db.should_migrate():
                self.db.migrate_programs()
        self._added += 1

    def action_for(self, view: SearchView, operator: str, target_id: str | None) -> Action:
        """Explicitly requested operator (run_operator/smoke): fill in the
        decision-time details the harness cannot know."""
        if operator == "draft":
            return self._draft_action(view)
        if operator == "improve" and target_id is None:
            return self._evolve_action(view)
        return self._greedy.action_for(view, operator, target_id)

    # --- introspection shared with the TUI/status surfaces ---

    def debuggable_tip(self, view: SearchView) -> Candidate | None:
        return self._greedy.debuggable_tip(view) if self.debug_enabled else None

    def prospective_branches(self, view: SearchView) -> int:
        return self._greedy.prospective_branches(view)

    def draft_complexity(self, view: SearchView) -> str:
        return self._greedy.draft_complexity(view)

    def island_stats(self) -> list[dict]:
        return self.db.get_island_stats()

    # --- decisions ---

    def _draft_action(self, view: SearchView) -> Action:
        return Action(
            operator="draft",
            complexity=self._greedy.draft_complexity(view),
            policy_meta={"island": self._added % self.db_config.num_islands},
        )

    def _evolve_action(self, view: SearchView) -> Action:
        iteration = len(view.journal.candidates)
        island = iteration % self.db_config.num_islands
        pool = self._parent_pool(view)
        with self._seeded(iteration):
            parent, inspirations = self.db.sample_from_island(island, self.num_inspirations)
            if parent.id not in pool:
                # the sampler may return a code-less floor or a pruned lineage;
                # fall back to the island's best improvable program
                parent = self._best_in(pool, island) or parent
        inspirations = [p for p in inspirations if p.id != parent.id and p.id in pool]
        island = int(parent.metadata.get("island", island))
        cell = self._cell(parent)
        context = self._render_context(view, parent, inspirations, island, cell)
        return Action(
            operator="improve",
            target_id=parent.id,
            inspiration_ids=tuple(p.id for p in inspirations),
            extra_prompt_context=context,
            policy_meta={
                "island": island,
                "cell": cell,
                "parent_fitness": self._fitness(parent),
                "inspirations": [p.id for p in inspirations],
            },
        )

    # --- helpers ---

    def _parent_pool(self, view: SearchView) -> dict[str, Candidate]:
        """Programs in the database that can still be expanded: scored,
        unpruned, with a solution.py on disk."""
        pool = {}
        for pid in self.db.programs:
            candidate = view.journal.candidates.get(pid)
            if candidate is None or candidate.pruned or not _improvable(candidate):
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

    def _metrics(self, view: SearchView, candidate: Candidate) -> dict[str, float]:
        score = float(candidate.val_score)
        fitness = score if view.higher_is_better else -score  # OpenEvolve maximizes
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
        except Exception:  # noqa: BLE001 - a missing dim was already rejected in observe
            return []

    def _render_context(self, view, parent, inspirations, island: int, cell: list[int]) -> str:
        direction = "higher" if view.higher_is_better else "lower"
        dims = ", ".join(
            f"{dim}={bin_}" for dim, bin_ in zip(self.feature_dimensions, cell)
        ) or "n/a"
        lines = [
            "This search runs OpenEvolve's MAP-Elites strategy over hillclimb's "
            "operators: a population of solutions kept diverse across feature "
            f"dimensions ({', '.join(self.feature_dimensions)}) on "
            f"{self.db_config.num_islands} islands.",
            f"Parent: {parent.id} on island {island}, score {self._score(view, parent)} "
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
                    f"- candidate_{i}.py: {program.id}, score {self._score(view, program)}, "
                    f"{len(program.code)} chars"
                )
        lines.append(
            "Make one deliberate change to solution.py that could move the score "
            "OR land in an unexplored region of the feature space (a different "
            "approach or complexity level) — both are wins for this strategy."
        )
        return "\n".join(lines)

    def _score(self, view, program) -> str:
        fitness = self._fitness(program)
        return f"{fitness if view.higher_is_better else -fitness:.6g}"

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
