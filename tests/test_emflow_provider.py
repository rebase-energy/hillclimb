"""emflow problem provider: target resolution + problem materialization.

Uses swedish-temperatures:ar (data ships inside the emflow package — no HF,
no token, no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("emflow")

from hillclimb.problem import load_problem, resolve_target  # noqa: E402


@pytest.fixture
def econfig(config, tmp_path):
    config.paths.runs_dir = tmp_path / "runs"
    return config


def test_load_emflow_problem(econfig):
    spec = load_problem("emflow://swedish-temperatures:ar", econfig)
    assert spec.kind == "emflow"
    assert spec.problem_id == "swedish-temperatures-ar"
    assert spec.emflow_problem == "swedish-temperatures:ar"
    assert spec.metric_name == "MeanAbsoluteError"
    assert spec.lower_is_better is True
    assert spec.sample_submission is None
    assert spec.holdout_mode == "evaluator"
    # materialized problem dir feeds prompts and workspace symlinks
    assert spec.problem_dir.is_dir()
    assert "MeanAbsoluteError" in spec.description
    assert "Holdout split" in spec.description


def test_resolve_single_variant_is_problem(econfig):
    resolved = resolve_target("emflow://swedish-temperatures:ar", econfig)
    assert resolved.kind == "problem"
    assert resolved.problem.kind == "emflow"


def test_bare_package_resolves_to_virtual_suite(econfig):
    # registry listing only — no dataset load, works without the HF cache
    resolved = resolve_target("emflow://gefcom2014", econfig)
    assert resolved.kind == "suite"
    assert resolved.suite.suite_id == "gefcom2014"
    assert resolved.suite.problems == [
        "emflow://gefcom2014:load",
        "emflow://gefcom2014:price",
        "emflow://gefcom2014:solar",
        "emflow://gefcom2014:wind",
    ]


def test_suite_targets_pass_scheme_entries_through(econfig):
    from hillclimb.problem import SuiteSpec, suite_problem_targets

    suite = SuiteSpec(
        suite_id="mixed", suite_path=econfig.paths.runs_dir / "s.yaml",
        problems=["emflow://gefcom2014:solar"],
    )
    assert suite_problem_targets(suite, econfig) == ["emflow://gefcom2014:solar"]


def test_unknown_emflow_problem_raises(econfig):
    with pytest.raises(KeyError):
        load_problem("emflow://no-such-problem", econfig)


def test_baseline_discovery():
    from hillclimb.integrations.emflow.provider import _find_baseline

    assert _find_baseline("gefcom2014:solar") == "emflow.benchmarks.gefcom2014.baseline"
    assert _find_baseline("swedish-temperatures:ar") is None


def test_csv_path_unaffected(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_problem("does-not-exist", config)
