"""The only module that imports the gepa library: the core gepa.optimize
loop wired to hillclimb's evaluator bridge and agentic proposer.

Verified against gepa 0.1.4:
- custom_candidate_proposer(candidate, reflective_dataset,
  components_to_update) -> dict[str, str]; reflection_lm=None is accepted
  alongside it (no provider call happens).
- skip_perfect_score=False is MANDATORY: the default True with
  perfect_score=1.0 silently disables mutation for unbounded raw scores.
- stop_callbacks entries are callables (gepa_state) -> bool, polled at
  iteration boundaries.
- run_dir checkpoints (gepa_state.bin) and resumes across processes.
- Exceptions raised in the adapter's evaluate PROPAGATE out of optimize();
  exceptions in the proposer are swallowed (the iteration logs "did not
  propose") — control flow rides the stop callback there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from hillclimb.climbers.gepa.config import GEPAParams
from hillclimb.climbers.gepa.evaluator import GepaScoring
from hillclimb.climbers.gepa.proposer import COMPONENT

MISSING_EXTRA = "the GEPA optimizer needs the `gepa` package: pip install 'hillclimb[gepa]'"


def build_driver() -> "CoreOptimizeDriver":
    try:
        import gepa  # noqa: F401
    except ImportError as exc:
        raise ImportError(MISSING_EXTRA) from exc
    return CoreOptimizeDriver()


class CoreOptimizeDriver:
    """gepa.optimize with a thin GEPAAdapter over the evaluator bridge: one
    verifier evaluation per unique candidate source (the bridge caches by
    hash), per-instance score slices for the Pareto frontier, and the
    bridge's allow-list ASI as the reflective dataset."""

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
    ) -> dict:
        import gepa
        from gepa import EvaluationBatch, GEPAAdapter

        class HillclimbAdapter(GEPAAdapter):
            def evaluate(self, batch, candidate, capture_traces=False):
                result = bridge.eval_source(candidate[COMPONENT])
                scores = [bridge.instance_fitness(result, key) for key in batch]
                trajectories = None
                if capture_traces:
                    asi = bridge.asi(result)
                    trajectories = [{"instance": key, "asi": asi} for key in batch]
                return EvaluationBatch(
                    outputs=[result.candidate_id] * len(batch),
                    scores=scores,
                    trajectories=trajectories,
                )

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                records = [
                    trajectory["asi"] | {"instance": trajectory["instance"]}
                    for trajectory in (eval_batch.trajectories or [])
                ]
                return {component: records for component in components_to_update}

        result = gepa.optimize(
            seed_candidate={COMPONENT: seed_source},
            trainset=list(instance_keys),
            valset=list(instance_keys),
            adapter=HillclimbAdapter(),
            custom_candidate_proposer=proposer,
            reflection_lm=None,
            module_selector="all",
            candidate_selection_strategy=params.candidate_selection_strategy,
            frontier_type=params.frontier_type,
            use_merge=False,
            max_metric_calls=params.max_metric_calls,
            reflection_minibatch_size=params.reflection_minibatch_size,
            seed=params.seed,
            run_dir=str(run_dir),
            stop_callbacks=[lambda gepa_state: should_stop()],
            skip_perfect_score=False,
            cache_evaluation=params.cache_evaluation,
            display_progress_bar=False,
            track_best_outputs=False,
        )
        return {
            "best_candidate": dict(result.best_candidate),
            "num_candidates": len(result.candidates),
        }
