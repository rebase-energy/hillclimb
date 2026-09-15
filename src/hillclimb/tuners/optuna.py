"""Optuna's TPE sampler as a hillclimb tuner (`search.tuner: optuna`,
extra `hillclimb[optuna]`).

Every `ask` rebuilds an in-memory study from the candidate's trial history
— completed sets as COMPLETE trials, failed sets as FAIL, in-flight sets as
RUNNING so the constant liar spreads concurrent asks — then asks once. No
study object outlives the call: the journal is the only state, and the
same (seed, history) always yields the same proposal. Raw journal-direction
scores go in unchanged; direction is the study's. No pruners: a verifier
run is atomic.

`tuner_params` pass through to `TPESampler` (`n_startup_trials`,
`multivariate`, ...); `seed` is the base of the per-ask seed.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Sequence

from hillclimb.params import ParamSpace, coerce
from hillclimb.tuner import Observation

MISSING_EXTRA = "the optuna tuner needs the `optuna` package: pip install 'hillclimb[optuna]'"
DEFAULT_STARTUP_TRIALS = 3  # TPE's own default (10) would make a small tune budget pure random


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(MISSING_EXTRA) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # constant_liar is flagged experimental in every 3.x/4.x release; the
    # interface is stable enough for one keyword
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


def distributions_for(optuna, space: ParamSpace) -> dict:
    dists = {}
    for name, spec in space.items():
        if spec.type == "categorical":
            dists[name] = optuna.distributions.CategoricalDistribution(list(spec.choices))
        elif spec.type == "int":
            dists[name] = optuna.distributions.IntDistribution(
                int(spec.low), int(spec.high), log=spec.log, step=int(spec.step) if spec.step else 1
            )
        else:
            dists[name] = optuna.distributions.FloatDistribution(
                float(spec.low), float(spec.high), log=spec.log, step=spec.step
            )
    return dists


class OptunaTuner:
    name = "optuna"

    def __init__(self, params: dict | None = None):
        self.optuna = _require_optuna()
        self.params = dict(params or {})
        accepted = set(inspect.signature(self.optuna.samplers.TPESampler.__init__).parameters)
        self.sampler_kwargs = {
            k: v for k, v in self.params.items()
            if k in accepted and k not in ("seed", "constant_liar")
        }
        self.sampler_kwargs.setdefault("n_startup_trials", DEFAULT_STARTUP_TRIALS)

    def ask(
        self,
        space: ParamSpace,
        history: Sequence[Observation],
        *,
        higher_is_better: bool,
        seed: int,
    ) -> dict:
        optuna = self.optuna
        dists = distributions_for(optuna, space)
        pending = [h for h in history if h.pending]
        sampler = optuna.samplers.TPESampler(
            seed=seed, constant_liar=bool(pending), **self.sampler_kwargs
        )
        study = optuna.create_study(
            direction="maximize" if higher_is_better else "minimize", sampler=sampler
        )
        for h in history:
            if h.pending:
                continue
            values = coerce(space, h.params)
            if h.score is None:
                study.add_trial(optuna.trial.create_trial(
                    params=values, distributions=dists, state=optuna.trial.TrialState.FAIL
                ))
            else:
                study.add_trial(optuna.trial.create_trial(
                    params=values, distributions=dists, value=float(h.score)
                ))
        for h in pending:  # RUNNING trials with fixed params: what the liar reads
            study.enqueue_trial(coerce(space, h.params))
            study.ask(dists)
        trial = study.ask(dists)
        return coerce(space, trial.params)
