"""The GEPA evaluator bridge: candidate source text in, canonical journaled
hillclimb candidate out, EvalResult + maximizing fitness + bounded reflective
feedback (ASI) back to gepa.

Privacy: this bridge never holds a holdout scorer — holdout runs only after
the optimizer finishes (searcher.finalize_holdout). The ASI is built by
allow-list from EvalResult fields; holdout values are structurally absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from hillclimb import evaluation
from hillclimb.candidate import BackendInfo, Candidate, utcnow
from hillclimb.config import Config
from hillclimb.dirs import create_candidate_dir
from hillclimb.evaluation import CandidateEvaluator, EvalResult, eval_result_for
from hillclimb.integrations.gepa.config import GEPAParams
from hillclimb.integrations.gepa.proposer import COMPONENT, ProposalBridge, source_hash
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec

ASI_TAIL_CAP = 2000
ASI_TOTAL_CAP = 16_384
FAILED_PROPOSAL_EVENT = "gepa_proposal_failed"


class InstanceKeyMismatch(Exception):
    """The verifier changed its per-instance key set mid-search — a broken
    verifier signal, never something to average over silently."""


class GEPAEvaluatorBridge:
    def __init__(
        self,
        *,
        journal: Journal,
        evaluator: CandidateEvaluator,
        problem: ProblemSpec,
        config: Config,
        params: GEPAParams,
        search_dir: Path,
        bridge: ProposalBridge,
        checkpoint: Callable[[], None] | None = None,
        on_phase: Callable[[str, str], None] | None = None,
        on_result: Callable[[Candidate], None] | None = None,
        log=print,
    ):
        self.journal = journal
        self.evaluator = evaluator
        self.problem = problem
        self.config = config
        self.params = params
        self.search_dir = search_dir
        self.bridge = bridge
        self.checkpoint = checkpoint or (lambda: None)
        self.on_phase = on_phase or (lambda cid, phase: None)
        self.on_result = on_result or (lambda candidate: None)
        self.log = log
        self.results: dict[str, EvalResult] = {}  # source hash -> cached result
        self.cid_by_hash: dict[str, str] = {}
        self.instance_keys: tuple[str, ...] | None = None  # fixed by first non-empty vector
        self._min_valid_fitness: float | None = None

    # --- resume ---

    def preload(self, candidate: Candidate) -> None:
        """Warm the cache from a journaled candidate (resume): identical
        source re-proposed by gepa's replay resolves to the existing id with
        no verifier call."""
        meta_hash = candidate.policy_meta.get("source_hash")
        solution = Path(candidate.candidate_dir) / COMPONENT
        actual = source_hash(solution.read_text()) if solution.exists() else None
        if meta_hash and actual and meta_hash != actual:
            raise RuntimeError(
                f"resume mismatch: {candidate.candidate_id} journals source_hash "
                f"{meta_hash[:12]} but {COMPONENT} hashes to {actual[:12]} — "
                "the candidate dir was modified; refusing to reattach lineage"
            )
        key = meta_hash or actual
        if key is None:
            return
        result = eval_result_for(candidate)
        self.results[key] = result
        self.cid_by_hash[key] = candidate.candidate_id
        self._observe(result)

    # --- the service ---

    def eval_source(self, source: str, *, operator: str = "improve") -> EvalResult:
        key = source_hash(source)
        cached = self.results.get(key)
        if cached is not None:
            return cached
        self.checkpoint()
        candidate_id = self.journal.next_candidate_id()
        candidate_dir = create_candidate_dir(
            self.search_dir, candidate_id, self.problem.data_dir, self.problem.problem_dir
        )
        (candidate_dir / COMPONENT).write_text(source)
        record = self.bridge.pop(key)
        parent_id = None
        if record is not None:
            parent_id = self.cid_by_hash.get(record.parent_hash)
            if parent_id is None:
                raise RuntimeError(
                    f"GEPA proposal parent {record.parent_hash[:12]} has no journaled "
                    "candidate — lineage cannot be resolved; refusing to guess a parent"
                )
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_id=parent_id,
            operator=operator,
            candidate_dir=str(candidate_dir),
            backend=record.backend if record is not None else BackendInfo(),
            policy_meta={
                "optimizer": "gepa",
                "source_hash": key,
                "parent_source_hash": record.parent_hash if record else None,
            },
        )
        self.journal.candidate_created(candidate)
        self.on_phase(candidate_id, "exec")
        try:
            exec_timeout = self.config.budget.exec_timeout_s
            all_ok = self.evaluator.run_trials(
                candidate, candidate_dir / COMPONENT, candidate_dir, exec_timeout
            )
            self._commit(candidate, all_ok)  # fitness from PRIOR candidates' floor
        finally:
            self.on_result(candidate)
        result = self.results[key] = eval_result_for(candidate)
        self.cid_by_hash[key] = candidate_id
        self._observe(result)
        self._check_instance_keys(result)  # after journaling: hard error, consistent journal
        return result

    def _commit(self, candidate: Candidate, all_ok: bool) -> None:
        candidate.status = "ok" if all_ok else "buggy"
        candidate.finished_at = utcnow()
        if candidate.status == "ok":
            previous_best = self.journal.best_candidate(self.problem.higher_is_better)
            band = evaluation.accept_band(self.config, self.journal)
            if previous_best is None or evaluation.improves(
                candidate.val_score,
                previous_best.val_score,
                higher_is_better=self.problem.higher_is_better,
                band=band,
            ):
                candidate.is_best = True
        candidate.policy_meta["gepa_fitness"] = self.fitness(eval_result_for(candidate))
        self.journal.candidate_result(candidate)

    def _observe(self, result: EvalResult) -> None:
        if result.valid and result.score is not None:
            fitness = self._direction(result.score)
            if self._min_valid_fitness is None or fitness < self._min_valid_fitness:
                self._min_valid_fitness = fitness

    def _check_instance_keys(self, result: EvalResult) -> None:
        keys = tuple(sorted(result.instance_scores))
        if not keys:
            return  # buggy candidates and instance-less verifiers never trip this
        if self.instance_keys is None:
            self.instance_keys = keys
            return
        if keys != self.instance_keys:
            missing = set(self.instance_keys) - set(keys)
            extra = set(keys) - set(self.instance_keys)
            raise InstanceKeyMismatch(
                "the verifier changed its per-instance keys mid-search "
                f"(missing: {sorted(missing)}, new: {sorted(extra)}); fix the verifier "
                "or set search.policy_params.frontier_type=objective"
            )

    # --- fitness (the only place direction is transformed) ---

    def _direction(self, score: float) -> float:
        return score if self.problem.higher_is_better else -score

    def _failure_fitness(self) -> float:
        floor = self.params.failure_fitness
        if self._min_valid_fitness is not None:
            floor = min(floor, self._min_valid_fitness - 1.0)
        return floor

    def fitness(self, result: EvalResult) -> float:
        if not result.valid or result.score is None:
            return self._failure_fitness()
        return self._direction(result.score)

    def instance_fitness(self, result: EvalResult, key: str) -> float:
        if not result.valid or result.score is None:
            return self._failure_fitness()
        if key in result.instance_scores:
            return self._direction(result.instance_scores[key])
        return self._direction(result.score)  # degenerate single-instance fallback

    # --- reflective feedback (allow-list only; no holdout, ever) ---

    def asi(self, result: EvalResult) -> dict:
        trials = [
            {
                "seed": t.seed,
                "val_score": t.val_score,
                "returncode": t.returncode,
                "timed_out": t.timed_out,
                "duration_s": t.duration_s,
                "stdout_tail": (t.stdout_tail or "")[-ASI_TAIL_CAP:],
            }
            for t in result.trials
        ]
        report = None
        candidate = self.journal.candidates.get(result.candidate_id)
        if candidate is not None and candidate.trials and candidate.trials[0].report:
            report = candidate.trials[0].report
        payload = {
            "candidate_id": result.candidate_id,
            "valid": result.valid,
            "val_score": result.score,
            "metric": self.problem.metric_name,
            "higher_is_better": self.problem.higher_is_better,
            "fitness": self.fitness(result),
            "instance_scores": result.instance_scores,
            "metrics": result.features,
            "report": report,
            "trials": trials,
            "advice": (
                "propose one concrete change to solution.py that addresses the "
                "weakest part of this evaluation"
            ),
        }
        text = json.dumps(payload, default=str)
        while len(text) > ASI_TOTAL_CAP and trials:
            longest = max(trials, key=lambda t: len(t.get("stdout_tail") or ""))
            longest["stdout_tail"] = (longest["stdout_tail"] or "")[: max(0, len(longest["stdout_tail"]) // 2)]
            if not longest["stdout_tail"]:
                trials.remove(longest)
            text = json.dumps(payload, default=str)
        return json.loads(text)

    # --- cost accounting ---

    def record_failed_proposal(self, backend: BackendInfo, error: str) -> None:
        """A proposal that produced no candidate still burned tokens — an
        append-only audit line keeps the ledger honest across resume."""
        self.journal.audit_event(
            FAILED_PROPOSAL_EVENT,
            cost_usd=backend.cost_usd,
            total_tokens=backend.total_tokens,
            model=backend.model,
            error=error[:300],
        )

    def total_cost_usd(self) -> float:
        journaled = sum(
            c.backend.cost_usd or 0.0
            for c in self.journal.candidates.values()
            if c.backend is not None
        )
        failed = sum(
            record.get("cost_usd") or 0.0
            for record in self.journal.backend.records()
            if record.get("event") == FAILED_PROPOSAL_EVENT
        )
        return journaled + failed
