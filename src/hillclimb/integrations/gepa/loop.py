"""GepaLoop: gepa drives the iteration, the harness does everything that
costs or counts.

gepa owns proposal order, Pareto selection and its own checkpoint; every
agent call and every verifier run goes through `harness.run(...)`:

- a reflective mutation is one `gepa-reflect` attempt (an ordinary journaled,
  scored candidate whose parent is the candidate gepa mutated);
- a text gepa produced some other way (the problem's baseline as seed, a
  merge) is an `inject`;
- results are cached by `source_hash`, so gepa's later `evaluate(text)` of
  its own proposal is a lookup, never a second verifier run.

The harness stays the authority: budgets, the cost ceiling, stop and park
close it, gepa's stop callback reads `harness.open`, and a swallowed exception
cannot spend anything. Holdout is scored after the loop returns
(`holdout.timing: after`) and is absent from everything gepa sees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Protocol

from hillclimb.integrations.gepa.config import GEPAParams
from hillclimb.integrations.gepa.evaluator import GepaScoring
from hillclimb.integrations.gepa.operator import OPERATOR_NAME
from hillclimb.integrations.gepa.proposer import COMPONENT, ProposerError, feedback_json
from hillclimb.sdk import (
    INJECT_ACTION,
    Action,
    EvalResult,
    Harness,
    HarnessClosed,
    ParkedSearch,
    SearchLoop,
    eval_result_for,
    source_hash,
)

SEED_ERROR = (
    "GEPA needs an executable seed: pass --seed-from PATH "
    "(or ship an executable baseline solution in the problem)"
)
# outcomes that mean "the agent had its turn and produced nothing usable"
FAILED_ROUNDS = ("agent_failed", "no_solution", "unchanged", "crashed")
SCORED_STATUSES = ("passing", "failing", "buggy")


class GEPADriver(Protocol):
    """The piece that actually calls gepa.optimize — injectable so the test
    suite drives GepaLoop without the optional dependency."""

    def run(
        self,
        *,
        seed_source: str,
        bridge: GepaScoring,
        proposer: Callable[..., dict[str, str]],
        params: GEPAParams,
        run_dir: Path,
        instance_keys: list[str],
        should_stop: Callable[[], bool],
    ) -> dict: ...


class _Halt(Exception):
    """The harness would not evaluate (closed, out of budget, cut off at the
    wall): unwind out of gepa.optimize — evaluate-path exceptions propagate."""


class GepaLoop(SearchLoop):
    name = "gepa"

    def __init__(self, params: GEPAParams | Mapping | None = None, driver: GEPADriver | None = None,
                 log=print, parallelism: int = 1):
        if parallelism > 1:
            raise ValueError("the GEPA climber is serial: set concurrency.parallel_agents=1")
        if not isinstance(params, GEPAParams):
            params = GEPAParams.model_validate(dict(params or {}))  # a typo fails before any spend
        self.params = params
        self.driver = driver
        self.log = log
        self.scoring: GepaScoring | None = None
        self._harness: Harness | None = None
        self._misses = 0
        self._give_up: Exception | None = None

    # --- SearchLoop ---

    def run(self, harness: Harness) -> None:
        self._harness = harness
        info = harness.info
        self.scoring = GepaScoring(
            params=self.params,
            metric_name=info.metric_name,
            higher_is_better=info.higher_is_better,
            evaluate=self._inject,
            candidate_of=lambda cid: harness.view().journal.candidates.get(cid),
        )
        self._warm_cache()
        seed_source = self._seed_source()
        self._verify_identity(seed_source)
        self.log("seeding GEPA with the incumbent solution")
        try:
            seed_result = self.scoring.eval_source(seed_source)
        except (_Halt, HarnessClosed):
            return  # closed before the first evaluation: the harness ends the search
        if not seed_result.valid:
            raise RuntimeError(
                f"the GEPA seed does not pass the verifier (candidate "
                f"{seed_result.candidate_id}); fix the seed before searching"
            )
        run_dir = harness.state_dir / "state"
        run_dir.mkdir(parents=True, exist_ok=True)
        driver = self.driver
        if driver is None:
            from hillclimb.integrations.gepa.driver import build_driver

            driver = build_driver()
        try:
            driver.run(
                seed_source=seed_source,
                bridge=self.scoring,
                proposer=self.propose,
                params=self.params,
                run_dir=run_dir,
                instance_keys=self._instance_keys(seed_result.instance_scores),
                should_stop=self._should_stop,
            )
        except (_Halt, HarnessClosed):
            pass  # the harness closed under gepa: it maps the terminal state
        if self._give_up is not None:
            raise self._give_up

    def _should_stop(self, _gepa_state=None) -> bool:
        return self._give_up is not None or not self._harness.open

    # --- gepa's custom_candidate_proposer ---

    def propose(self, candidate, reflective_dataset, components_to_update) -> dict[str, str]:
        if set(components_to_update) != {COMPONENT} or set(candidate) != {COMPONENT}:
            raise ProposerError(
                f"GEPA MVP mutates only {COMPONENT}; got components={list(components_to_update)}"
            )
        parent_hash = source_hash(candidate[COMPONENT])
        parent_id = self.scoring.cid_by_hash.get(parent_hash)
        if parent_id is None:
            raise ProposerError(
                f"GEPA proposal parent {parent_hash[:12]} has no journaled candidate — "
                "lineage cannot be resolved; refusing to guess a parent"
            )
        try:
            outcome = self._harness.run(
                Action(
                    operator=OPERATOR_NAME,
                    target_id=parent_id,
                    payload={"feedback": feedback_json(reflective_dataset)},
                    policy_meta={"optimizer": "gepa", "parent_source_hash": parent_hash},
                )
            )
        except HarnessClosed as exc:
            raise ProposerError(f"the harness is closed: {exc}") from exc
        if outcome.kind != "evaluated":
            self._did_not_propose(outcome.kind, outcome.ticket.rejected or "")
        self._misses = 0
        source = self._harness.source(outcome.candidate.candidate_id)
        self.scoring.remember(source_hash(source), outcome.result)
        return {COMPONENT: source}

    def _did_not_propose(self, kind: str, detail: str) -> None:
        """gepa swallows proposer exceptions and keeps looping, so giving up
        after three empty-handed rounds lives here (agent failures also count
        in the harness, which parks on its own third)."""
        if kind in FAILED_ROUNDS:
            self._misses += 1
            self.log(f"  proposal failed ({self._misses} in a row): {kind}")
            if self._misses >= 3 and self._give_up is None:
                self._give_up = ParkedSearch(f"3 consecutive proposal failures; last: {kind}")
        raise ProposerError(f"did not propose ({kind}{': ' + detail if detail else ''})")

    # --- evaluating a text gepa did not get from a proposal ---

    def _inject(self, source: str) -> EvalResult:
        outcome = self._harness.run(
            Action(
                operator=INJECT_ACTION,
                payload={"source": source},
                policy_meta={"optimizer": "gepa"},
            )
        )
        if outcome.kind != "evaluated":
            # refused (out of budget) or cut off at the wall: never cached as
            # a failure — the text was not shown to be bad
            raise _Halt(outcome.ticket.rejected or outcome.kind)
        return self.scoring.remember(source_hash(source), outcome.result)

    # --- seed, identity, resume ---

    def _warm_cache(self) -> None:
        """Resume: every scored candidate's text resolves to its id with no
        verifier call, so gepa's checkpoint replay costs nothing."""
        for candidate in self._harness.view().journal.candidates.values():
            if candidate.status not in SCORED_STATUSES:
                continue
            source = self._harness.source(candidate.candidate_id)
            if source is None:
                continue
            key = source_hash(source)
            if candidate.solution_sha256 and candidate.solution_sha256 != key:
                raise RuntimeError(
                    f"resume mismatch: {candidate.candidate_id} was scored as "
                    f"{candidate.solution_sha256[:12]} but {COMPONENT} now hashes to {key[:12]} — "
                    "the candidate dir was modified; refusing to reattach lineage"
                )
            self.scoring.remember(key, eval_result_for(candidate))

    def _seed_source(self) -> str:
        journal = self._harness.view().journal
        for candidate in journal.candidates.values():
            if candidate.operator == "seed":  # the harness scored --seed-from already
                source = self._harness.source(candidate.candidate_id)
                if source is not None:
                    return source
        if self._harness.info.baseline_source:
            return self._harness.info.baseline_source
        raise RuntimeError(SEED_ERROR)

    def _identity(self, seed_source: str) -> dict:
        info = self._harness.info
        return {
            "seed_source_hash": source_hash(seed_source),
            "metric": info.metric_name,
            "higher_is_better": info.higher_is_better,
            "components": [COMPONENT],
            "frontier_type": self.params.frontier_type,
            "candidate_selection_strategy": self.params.candidate_selection_strategy,
            "gepa_seed": self.params.seed,
        }

    def _verify_identity(self, seed_source: str) -> None:
        """A resumed search must be the same search: any drift in the fields
        that shape the optimization or its privacy is a hard error."""
        path = self._harness.state_dir / "identity.json"
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
        path.write_text(json.dumps(identity, indent=2))

    def _instance_keys(self, seed_instances: dict[str, float]) -> list[str]:
        if self.params.frontier_type == "instance" and seed_instances:
            return sorted(seed_instances)
        if self.params.frontier_type == "instance":
            self.log(
                "  verifier emits no per-instance scores; the instance frontier "
                "degenerates to the aggregate score"
            )
        return [self._harness.info.problem_id]
