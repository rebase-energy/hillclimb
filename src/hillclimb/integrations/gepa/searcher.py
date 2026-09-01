"""GEPASearcher: the SearchRunner that hands the loop to gepa while
hillclimb stays the outer authority — canonical candidate dirs, the
append-only journal, budgets, control commands, and a post-completion
private holdout gepa never sees."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Protocol

from hillclimb.backends.base import OperatorBackend
from hillclimb.baseline import write_baseline
from hillclimb.budget import BudgetManager
from hillclimb.candidate import BackendInfo, Candidate, utcnow
from hillclimb.config import Config
from hillclimb.control import ControlCommand, apply_prune, drain_commands_dir, resync_best
from hillclimb.evaluation import CandidateEvaluator
from hillclimb.executor import Executor, HoldoutScorer
from hillclimb.integrations.gepa.config import GEPAParams, validate_gepa_search_config
from hillclimb.integrations.gepa.evaluator import GEPAEvaluatorBridge
from hillclimb.integrations.gepa.proposer import (
    COMPONENT,
    GEPAProposer,
    ProposalBridge,
    ProposerError,
    source_hash,
)
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec
from hillclimb.routing import BackendPool, Router
from hillclimb.search_runner import ParkedSearch, StopRequested
from hillclimb.slots import MachineSlots
from hillclimb.status import CandidateCounts, CurrentCandidate, ScoreRef, StatusWriter

SEED_ERROR = (
    "GEPA needs an executable seed: pass --seed-from PATH "
    "(or ship an executable baseline solution in the problem)"
)


class GEPADriver(Protocol):
    """The piece that actually calls gepa.optimize — injectable so the test
    suite drives GEPASearcher without the optional dependency."""

    def run(
        self,
        *,
        seed_source: str,
        bridge: GEPAEvaluatorBridge,
        proposer: GEPAProposer,
        params: GEPAParams,
        run_dir: Path,
        instance_keys: list[str],
        should_stop: Callable[[], bool],
    ) -> dict: ...


class GEPASearcher:
    def __init__(
        self,
        *,
        problem: ProblemSpec,
        config: Config,
        journal: Journal,
        backend: OperatorBackend,
        executor: Executor,
        budget: BudgetManager,
        search_dir: Path,
        log=print,
        holdout_scorer: HoldoutScorer | None = None,
        status: StatusWriter | None = None,
        slots: MachineSlots | None = None,
        abort: threading.Event | None = None,
        seed_solution: Path | None = None,
        knowledge_context: str | None = None,
        reference_solution: Path | None = None,  # unused by GEPA (greedy-only feature)
        reference_note: str = "",
        complexity_start: int = 0,  # unused by GEPA
        router: Router | None = None,
        backends: BackendPool | None = None,
        drain_commands: Callable[[], list[ControlCommand]] | None = None,
        driver: GEPADriver | None = None,
    ):
        self.params = validate_gepa_search_config(config)
        self.problem = problem
        self.config = config
        self.journal = journal
        self.budget = budget
        self.search_dir = search_dir
        self.log = log
        self.holdout_scorer = holdout_scorer
        self.status = status
        self.abort = abort or threading.Event()
        self.seed_solution = seed_solution
        self.drain_commands = drain_commands or (lambda: drain_commands_dir(search_dir))
        self.driver = driver
        self._pending: Exception | None = None
        self._selection_id: str | None = None
        self._consecutive_failures = 0
        if router is None:
            router = Router(config)
        if backends is None:
            backends = BackendPool(abort=self.abort)
            backends.seed(config.backend, config.backend_auth, backend)
        evaluator = CandidateEvaluator(executor=executor, problem=problem, config=config)
        proposal_bridge = ProposalBridge()
        self.bridge = GEPAEvaluatorBridge(
            journal=journal,
            evaluator=evaluator,
            problem=problem,
            config=config,
            params=self.params,
            search_dir=search_dir,
            bridge=proposal_bridge,
            checkpoint=self.checkpoint,
            on_phase=self._on_phase,
            on_result=self._on_result,
            log=log,
        )
        self.proposer = GEPAProposer(
            problem=problem,
            config=config,
            router=router,
            backends=backends,
            budget=budget,
            search_dir=search_dir,
            bridge=proposal_bridge,
            checkpoint=self.checkpoint,
            abort=self.abort,
            slots=slots,
            knowledge_context=knowledge_context,
            reference_note=reference_note,
            on_phase=self._on_proposer_phase,
            on_failure=self._on_proposer_failure,
            log=log,
        )
        # crash recovery + warm cache: same stale-pending rule as greedy,
        # then every terminal candidate preloads the hash cache so gepa's
        # checkpoint replay costs nothing
        for stale in journal.pending_candidates():
            stale.status = "abandoned"
            stale.summary = stale.summary or "orchestrator died mid-operator (crash recovery)"
            stale.finished_at = utcnow()
            journal.candidate_result(stale)
            log(f"  recovered stale pending candidate {stale.candidate_id} -> abandoned")
        for candidate in journal.candidates.values():
            if candidate.operator == "seed" or candidate.policy_meta.get("optimizer") == "gepa":
                self.bridge.preload(candidate)

    # --- SearchRunner ---

    def run(self) -> Candidate | None:
        if not self.journal.candidates:
            baseline = write_baseline(
                self.problem,
                self.search_dir,
                executor=self.bridge.evaluator.executor,
                holdout_scorer=self.holdout_scorer,
                timeout_s=self.config.budget.exec_timeout_s,
            )
            self.journal.candidate_result(baseline)
            self.log(f"baseline written: {baseline.summary}")
        seed_source = self._seed_source()
        self._verify_identity(seed_source)
        self.log("seeding GEPA with the incumbent solution")
        seed_result = self.bridge.eval_source(seed_source, operator="seed")
        if not seed_result.valid:
            raise RuntimeError(
                f"the GEPA seed does not pass the verifier (candidate "
                f"{seed_result.candidate_id}); fix the seed before searching"
            )
        self._raise_pending()
        instance_keys = self._instance_keys(seed_result.instance_scores)
        run_dir = self.search_dir / "gepa" / "state"
        run_dir.mkdir(parents=True, exist_ok=True)
        driver = self.driver
        if driver is None:
            from hillclimb.integrations.gepa.driver import build_driver

            driver = build_driver()
        try:
            driver.run(
                seed_source=seed_source,
                bridge=self.bridge,
                proposer=self.proposer,
                params=self.params,
                run_dir=run_dir,
                instance_keys=instance_keys,
                should_stop=self._should_stop,
            )
        except (ParkedSearch, StopRequested):
            raise  # escaped through the evaluator: correct control path
        except ProposerError as exc:
            raise ParkedSearch(f"GEPA proposer failed: {exc}") from exc
        self._raise_pending()
        self.finalize_holdout()
        self._selection_id = resync_best(
            self.search_dir, self.journal, self.problem.higher_is_better,
            self.config.holdout.selection, self.problem.output_artifacts,
        )
        self._status()
        return self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )

    def total_cost_usd(self) -> float:
        return self.bridge.total_cost_usd()

    # --- control plane ---

    def _should_stop(self, _gepa_state=None) -> bool:
        """gepa stop callback AND the poll inside checkpoint(): drains user
        commands, applies prunes, records (not raises) park/stop, checks the
        wall clock and cost ceiling."""
        for cmd in sorted(self.drain_commands(), key=lambda c: c.action != "prune"):
            if cmd.action == "stop":
                self.journal.control_event("stop", reason=cmd.reason, source=cmd.source)
                self._pending = StopRequested(f"stop requested by {cmd.source}")
            elif cmd.action == "prune" and cmd.candidate_id:
                try:
                    pruned = apply_prune(self.journal, cmd.candidate_id, cmd.reason, cmd.source)
                except ValueError as exc:
                    self.log(f"  prune {cmd.candidate_id} rejected: {exc}")
                    continue
                if pruned:
                    # display/selection only: gepa's own population is not
                    # touched (documented MVP concession)
                    self.log(f"  pruned {', '.join(pruned)} (by {cmd.source})")
        ceiling = self.config.budget.max_cost_usd
        if self._pending is None and ceiling > 0 and self.total_cost_usd() >= ceiling:
            self._pending = ParkedSearch(
                f"cost ceiling reached (${self.total_cost_usd():.2f} >= ${ceiling:.2f})"
            )
        return (
            self._pending is not None
            or self.budget.should_stop()
            or self.abort.is_set()
        )

    def checkpoint(self) -> None:
        """Called by the proposer and evaluator before spending anything.
        Raising here is safe on both paths: the evaluator propagates out of
        gepa.optimize (verified), the proposer's raise aborts one proposal
        and the stop callback then ends the loop. Plain wall-clock exhaustion
        deliberately does NOT raise — the stop callback ends the loop at the
        iteration boundary and the search completes normally (holdout runs),
        matching greedy's in-flight-work-finishes semantics."""
        if self._should_stop():
            self._raise_pending()
            if self.abort.is_set():
                raise StopRequested("abort")

    def _raise_pending(self) -> None:
        if self._pending is not None:
            raise self._pending

    # --- seed + identity ---

    def _seed_source(self) -> str:
        if self.seed_solution is not None:
            seed = self.seed_solution.absolute()
            if not seed.exists():
                raise FileNotFoundError(f"seed solution not found: {seed}")
            return seed.read_text()
        if self.problem.baseline_text:
            return self.problem.baseline_text
        raise RuntimeError(SEED_ERROR)

    def _identity(self, seed_source: str) -> dict:
        return {
            "seed_source_hash": source_hash(seed_source),
            "metric": self.problem.metric_name,
            "higher_is_better": self.problem.higher_is_better,
            "components": [COMPONENT],
            "frontier_type": self.params.frontier_type,
            "candidate_selection_strategy": self.params.candidate_selection_strategy,
            "gepa_seed": self.params.seed,
        }

    def _verify_identity(self, seed_source: str) -> None:
        """A resumed search must be the same search: any drift in the fields
        that shape the optimization or its privacy is a hard error."""
        path = self.search_dir / "gepa" / "identity.json"
        identity = self._identity(seed_source)
        if path.exists():
            stored = json.loads(path.read_text())
            for field, value in identity.items():
                if stored.get(field) != value:
                    raise RuntimeError(
                        f"resume mismatch: gepa identity field {field!r} changed "
                        f"({stored.get(field)!r} -> {value!r}); refusing to resume "
                        "a different optimization over this journal"
                    )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, indent=2))

    def _instance_keys(self, seed_instances: dict[str, float]) -> list[str]:
        if self.params.frontier_type == "instance" and seed_instances:
            return sorted(seed_instances)
        if self.params.frontier_type == "instance":
            self.log(
                "  verifier emits no per-instance scores; the instance frontier "
                "degenerates to the aggregate score"
            )
        return [self.problem.problem_id]

    # --- holdout: post-completion only ---

    def finalize_holdout(self) -> None:
        """Score the hidden split for the top-k validation candidates, only
        after the optimizer is done — gepa never sees these values, and
        nothing under gepa/ is written after this starts."""
        if self.holdout_scorer is None:
            return
        ranked = [
            c
            for c in sorted(
                self.journal.scored_candidates(),
                key=lambda c: (
                    -c.val_score if self.problem.higher_is_better else c.val_score
                ),
            )
            if c.trials
        ]
        top_k = self.config.holdout.top_k
        wanted = len(ranked) if top_k <= 0 else top_k
        scored = 0
        for candidate in ranked:
            if scored >= wanted:
                break
            if candidate.holdout_score is not None:
                scored += 1
                continue
            self._on_phase(candidate.candidate_id, "holdout")
            try:
                score, error, cpu_s = self.holdout_scorer.score(Path(candidate.candidate_dir))
            finally:
                if self.status is not None:
                    self.status.remove_current(candidate.candidate_id)
            last = candidate.trials[-1]
            last.holdout_cpu_s = cpu_s
            if error is not None:
                last.holdout_error = error
                self.log(f"  holdout failed for {candidate.candidate_id}: {error}")
            else:
                last.holdout_score = score
                scored += 1
            self.journal.candidate_result(candidate)  # replay keeps the last event

    # --- status plumbing ---

    def _on_phase(self, candidate_id: str, phase: str) -> None:
        if self.status is None:
            return
        candidate = self.journal.candidates.get(candidate_id)
        current = CurrentCandidate(
            candidate_id=candidate_id,
            operator=candidate.operator if candidate else "improve",
            phase=phase,
            candidate_dir=candidate.candidate_dir if candidate else "",
        )
        self.status.add_current(current)

    def _on_proposer_phase(self, phase: str) -> None:
        # proposals have no candidate id yet; refresh counts/cost so `watch`
        # shows the search is working
        self._status()

    def _on_proposer_failure(self, exc: "ProposerError") -> None:
        """gepa swallows proposer exceptions and keeps looping, so the
        3-consecutive-failures park rule (greedy's) lives here: journal the
        failed call's burn, count, and pend a ParkedSearch that the stop
        callback turns into loop exit."""
        result = exc.result
        if result is not None:
            self.bridge.record_failed_proposal(
                BackendInfo(
                    model=getattr(result, "model_id", None),
                    cost_usd=result.cost_usd,
                    total_tokens=result.total_tokens,
                ),
                str(exc),
            )
        self._consecutive_failures += 1
        self.log(f"  proposal failed ({self._consecutive_failures} in a row): {exc}")
        if self._consecutive_failures >= 3 and self._pending is None:
            self._pending = ParkedSearch(f"3 consecutive proposal failures; last: {exc}")

    def _on_result(self, candidate: Candidate) -> None:
        self._consecutive_failures = 0  # an evaluated candidate = a good round
        if self.status is not None:
            self.status.remove_current(candidate.candidate_id)
        self._status()

    def _status(self, **fields) -> None:
        if self.status is None:
            return
        candidates = self.journal.candidates.values()
        fields.setdefault("cost_usd", round(self.total_cost_usd(), 6))
        fields.setdefault(
            "candidates",
            CandidateCounts(
                total=len(self.journal.candidates),
                ok=sum(1 for c in candidates if c.status == "ok"),
                buggy=sum(1 for c in candidates if c.status == "buggy"),
                pruned=sum(1 for c in candidates if c.pruned),
            ),
        )
        best = self.journal.best_candidate(self.problem.higher_is_better)
        if best is not None:
            fields.setdefault(
                "best", ScoreRef(candidate_id=best.candidate_id, val_score=best.val_score)
            )
        selected = self.journal.selected_candidate(
            self.problem.higher_is_better, self.config.holdout.selection
        )
        if selected is not None:
            fields.setdefault(
                "selected",
                ScoreRef(
                    candidate_id=selected.candidate_id,
                    val_score=selected.val_score,
                    holdout_score=selected.holdout_score,
                ),
            )
        self.status.update(**fields)
