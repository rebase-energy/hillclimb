"""GEPA's scoring view of a hillclimb search: cached `EvalResult`s by source
hash, the maximizing fitness gepa optimizes, and the bounded reflective
feedback (ASI) its proposer is shown.

Privacy: everything here is built from `Outcome.result` and the loop's
`view()` — holdout-blind by construction — and the ASI is an allow-list on
top of that. Holdout values are structurally absent.
"""

from __future__ import annotations

import json
from typing import Callable

from hillclimb.climbers.gepa.config import GEPAParams
from hillclimb.sdk import Candidate, EvalResult

ASI_TAIL_CAP = 2000
ASI_TOTAL_CAP = 16_384


class InstanceKeyMismatch(Exception):
    """The verifier changed its per-instance key set mid-search — a broken
    verifier signal, never something to average over silently."""


class GepaScoring:
    def __init__(
        self,
        *,
        params: GEPAParams,
        metric_name: str,
        higher_is_better: bool,
        evaluate: Callable[[str], EvalResult],
        candidate_of: Callable[[str], Candidate | None],
    ):
        self.params = params
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self._evaluate = evaluate  # source text -> EvalResult (the loop's inject path)
        self._candidate_of = candidate_of  # candidate id -> holdout-blind candidate
        self.results: dict[str, EvalResult] = {}  # source hash -> cached result
        self.cid_by_hash: dict[str, str] = {}
        self.instance_keys: tuple[str, ...] | None = None  # fixed by first non-empty vector
        self._min_valid_fitness: float | None = None

    # --- the cache ---

    def remember(self, key: str, result: EvalResult) -> EvalResult:
        self.results[key] = result
        self.cid_by_hash[key] = result.candidate_id
        if result.valid and result.score is not None:
            fitness = self._direction(result.score)
            if self._min_valid_fitness is None or fitness < self._min_valid_fitness:
                self._min_valid_fitness = fitness
        return result

    def eval_source(self, source: str) -> EvalResult:
        """What gepa's adapter calls: a text it proposed is a cache hit (the
        proposal was scored when it was made); any other text is injected."""
        from hillclimb.sdk import source_hash

        result = self.results.get(source_hash(source))
        if result is None:
            result = self._evaluate(source)
        # checked HERE, on gepa's evaluate path, where an exception propagates
        # out of optimize() — one raised inside the proposer would be swallowed
        self.check_instance_keys(result)
        return result

    def check_instance_keys(self, result: EvalResult) -> None:
        """The instance vocabulary is fixed by the first candidate that
        reports one. A later candidate may MISS instances — a model that
        leaves an origin unscored (NaN) simply has no entry for it, and
        `instance_fitness` scores that as a failure — but it may never
        introduce a key nobody else was scored on: that is a verifier bug."""
        keys = tuple(sorted(result.instance_scores))
        if not keys:
            return  # buggy candidates and instance-less verifiers never trip this
        if self.instance_keys is None:
            self.instance_keys = keys
            return
        extra = set(keys) - set(self.instance_keys)
        if extra:
            raise InstanceKeyMismatch(
                "the verifier introduced per-instance keys mid-search "
                f"(new: {sorted(extra)}, expected: {list(self.instance_keys)}); fix the verifier "
                "so every candidate reports keys from the same instance set"
            )

    # --- fitness (the only place direction is transformed) ---

    def _direction(self, score: float) -> float:
        return score if self.higher_is_better else -score

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
        if result.instance_scores:
            return self._failure_fitness()  # scored the others, not this one: failed it
        return self._direction(result.score)  # degenerate single-instance fallback

    # --- reflective feedback (allow-list only; no holdout, ever) ---

    def asi(self, result: EvalResult) -> dict:
        # one entry per replicate, flattened (GEPA's reflection prompt has no
        # notion of parameter sets; the key stays "trials" for its payload)
        trials = [
            {
                "trial": t.index,
                "params": t.params,
                "seed": r.seed,
                "val_score": r.val_score,
                "returncode": r.returncode,
                "timed_out": r.timed_out,
                "duration_s": r.duration_s,
                "stdout_tail": (r.stdout_tail or "")[-ASI_TAIL_CAP:],
            }
            for t in result.trials
            for r in t.replicates
        ]
        candidate = self._candidate_of(result.candidate_id)
        payload = {
            "candidate_id": result.candidate_id,
            "valid": result.valid,
            "val_score": result.score,
            "metric": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "fitness": self.fitness(result),
            "instance_scores": result.instance_scores,
            "metrics": result.features,
            "report": candidate.report if candidate is not None else None,
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
