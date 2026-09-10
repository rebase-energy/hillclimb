"""GEPAParams validation and the engine's config gate."""

from __future__ import annotations

import pytest

from hillclimb.integrations.gepa.config import GEPAParams, validate_gepa_search_config


def test_defaults_are_the_documented_mvp():
    params = GEPAParams()
    assert params.max_metric_calls == 50
    assert params.candidate_selection_strategy == "pareto"
    assert params.frontier_type == "instance"
    assert params.use_merge is False
    assert params.cache_evaluation is True


def test_documented_fields_accepted():
    params = GEPAParams.model_validate(
        {
            "max_metric_calls": 100,
            "reflection_minibatch_size": 2,
            "candidate_selection_strategy": "current_best",
            "frontier_type": "instance",
            "cache_evaluation": False,
            "use_merge": False,
            "failure_fitness": -1e9,
            "seed": 42,
        }
    )
    assert params.max_metric_calls == 100
    assert params.frontier_type == "instance"


def test_objective_frontier_is_rejected_up_front():
    # upstream demands objective_scores from the evaluator for it, which the
    # bridge never produces: the run would die at the seed evaluation
    with pytest.raises(ValueError, match="frontier_type"):
        GEPAParams.model_validate({"frontier_type": "objective"})


@pytest.mark.parametrize(
    "bad",
    [
        {"max_metric_calls": 0},
        {"unknown_knob": 1},
        {"use_merge": True},
        {"failure_fitness": float("-inf")},
        {"failure_fitness": float("nan")},
        {"frontier_type": "hybrid"},
        {"frontier_type": "cartesian"},
        {"candidate_selection_strategy": "epsilon_greedy_typo"},
    ],
)
def test_invalid_params_rejected(bad):
    with pytest.raises(Exception):
        GEPAParams.model_validate(bad)


def test_parallel_operators_rejected_before_spend(config):
    config.search.policy = "gepa"
    config.search.parallel_operators = 2
    with pytest.raises(ValueError, match="serial in the MVP"):
        validate_gepa_search_config(config)


def test_valid_config_parses_policy_params(config):
    config.search.policy = "gepa"
    config.search.policy_params = {"max_metric_calls": 7}
    assert validate_gepa_search_config(config).max_metric_calls == 7


def test_missing_extra_message():
    import builtins

    from hillclimb.integrations.gepa import driver as driver_mod

    real_import = builtins.__import__

    def no_gepa(name, *args, **kwargs):
        if name == "gepa":
            raise ImportError("No module named 'gepa'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_gepa
    try:
        with pytest.raises(ImportError, match=r"pip install 'hillclimb\[gepa\]'"):
            driver_mod.build_driver()
    finally:
        builtins.__import__ = real_import
