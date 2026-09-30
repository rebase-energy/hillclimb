"""The tuner seam: ask() is a pure function of (space, history, seed)."""

from __future__ import annotations

import pytest

from hillclimb.harness.params import parse_space
from hillclimb.modules.tuners.base import Observation, history_for, tune_seed
from hillclimb.modules.tuners import get_tuner
from tests.factories import candidate, trial

SPACE = parse_space({
    "restarts": {"type": "int", "low": 1, "high": 64, "log": True, "default": 8},
    "step": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.25, "default": 0.5},
    "init": {"type": "categorical", "choices": ["grid", "random"], "default": "grid"},
})
HISTORY = [
    Observation({"restarts": 8, "step": 0.5, "init": "grid"}, 0.5),
    Observation({"restarts": 2, "step": 0.25, "init": "random"}, None),  # failed set
    Observation({"restarts": 16, "step": 0.75, "init": "grid"}, 0.7, pending=True),
]


def in_domain(values: dict) -> bool:
    return (
        set(values) == set(SPACE)
        and isinstance(values["restarts"], int) and 1 <= values["restarts"] <= 64
        and isinstance(values["step"], float) and values["step"] in (0.0, 0.25, 0.5, 0.75, 1.0)
        and values["init"] in ("grid", "random")
    )


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown tuner 'grid' \\(available: optuna, random"):
        get_tuner("grid")


def test_tune_seed_is_deterministic_and_history_sensitive():
    assert tune_seed(0, "c001", 1) == tune_seed(0, "c001", 1)
    assert tune_seed(0, "c001", 1) != tune_seed(0, "c001", 2)
    assert tune_seed(0, "c001", 1) != tune_seed(1, "c001", 1)


def test_history_for_orders_trials_and_appends_pending():
    cand = candidate("c1", trials=[
        trial(0.7, params={"k": 2}, index=1), trial(0.5, params={"k": 1}, index=0),
        trial(None, params={"k": 3}, index=2, is_best=False),
    ])
    history = history_for(cand, pending=[{"k": 4}])
    assert [h.params["k"] for h in history] == [1, 2, 3, 4]
    assert [h.score for h in history] == [0.5, 0.7, None, None]
    assert [h.pending for h in history] == [False, False, False, True]


@pytest.mark.parametrize("name", ["random", pytest.param("optuna", marks=pytest.mark.optuna)])
def test_ask_is_deterministic_and_in_domain(name):
    if name == "optuna":
        pytest.importorskip("optuna")
    tuner = get_tuner(name, {"seed": 3})
    first = tuner.ask(SPACE, HISTORY, higher_is_better=True, seed=11)
    again = tuner.ask(SPACE, HISTORY, higher_is_better=True, seed=11)
    other = tuner.ask(SPACE, HISTORY, higher_is_better=True, seed=12)
    assert first == again
    assert in_domain(first) and in_domain(other)
    lower = tuner.ask(SPACE, HISTORY, higher_is_better=False, seed=11)
    assert in_domain(lower)


def test_random_avoids_repeating_history():
    tuner = get_tuner("random")
    proposal = tuner.ask(SPACE, HISTORY, higher_is_better=True, seed=1)
    assert proposal not in [h.params for h in HISTORY]


def test_optuna_accepts_json_round_tripped_values():
    pytest.importorskip("optuna")
    tuner = get_tuner("optuna")
    history = [Observation({"restarts": 8.0, "step": 0.5, "init": "grid"}, 0.5)]  # int came back as float
    assert in_domain(tuner.ask(SPACE, history, higher_is_better=True, seed=1))


def test_optuna_constant_liar_only_with_pending(monkeypatch):
    optuna = pytest.importorskip("optuna")
    seen = []
    original = optuna.samplers.TPESampler

    class Spy(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("constant_liar"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(optuna.samplers, "TPESampler", Spy)
    tuner = get_tuner("optuna")
    tuner.ask(SPACE, HISTORY[:2], higher_is_better=True, seed=1)
    tuner.ask(SPACE, HISTORY, higher_is_better=True, seed=1)
    assert seen == [False, True]
