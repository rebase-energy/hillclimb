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
    assert spec.problem_id == "swedish-temperatures-ar"
    assert spec.emflow_problem == "swedish-temperatures:ar"
    assert spec.metric_name == "MeanAbsoluteError"
    assert spec.higher_is_better is False
    # the provider supplies the verifier: the emflow evaluator, one command
    # per split
    assert spec.verifier_cmd[1].endswith("eval_runner.py")
    assert spec.verifier_cmd[-3:] == ["--split", "validation", "--result-json"] or (
        "validation" in spec.verifier_cmd
    )
    assert "holdout" in spec.holdout_cmd
    assert spec.runtime == "emflow"
    assert spec.holdout_needs_credentials
    assert spec.contract_template == "contract_emflow"
    # materialized problem dir feeds prompts and candidate_dir symlinks
    assert spec.problem_dir.is_dir()
    assert "MeanAbsoluteError" in spec.description
    assert "Holdout split" in spec.description


def test_resolve_single_variant_is_problem(econfig):
    resolved = resolve_target("emflow://swedish-temperatures:ar", econfig)
    assert resolved.kind == "problem"
    assert resolved.problem.emflow_problem == "swedish-temperatures:ar"


def test_bare_package_resolves_to_virtual_suite(econfig):
    # registry listing only — no dataset load, works without the HF cache
    resolved = resolve_target("emflow://gefcom2014", econfig)
    assert resolved.kind == "suite"
    assert resolved.suite.suite_id == "gefcom2014"
    assert [entry.target for entry in resolved.suite.problems] == [
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


def test_emflow_contract_prompt(econfig):
    """The emflow contract teaches the Predictor API and never mentions the
    CSV artifacts; the quantile note reflects the problem."""
    import sys
    from pathlib import Path

    from hillclimb.backends.fake import FakeBackend
    from hillclimb.budget import BudgetManager
    from tests.conftest import local_executor
    from hillclimb.journal import Journal
    from hillclimb.search import GreedySearcher

    spec = load_problem("emflow://swedish-temperatures:ar", econfig)
    search_dir = econfig.paths.runs_dir / "prompt-test"
    (search_dir / "candidates").mkdir(parents=True)
    searcher = GreedySearcher(
        problem=spec, config=econfig, journal=Journal(search_dir / "journal.jsonl"),
        backend=FakeBackend(), executor=local_executor(),
        budget=BudgetManager(600, stop_margin_s=1), search_dir=search_dir,
        log=lambda *_: None,
    )
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "get_model()" in prompt
    assert "swedish-temperatures:ar" in prompt
    assert "FeaturePredictor" in prompt
    assert 'output_kind = "point"' in prompt  # MAE problem: point forecasts
    assert "do NOT write\n   `submission.csv`" in prompt.replace("\r", "") or "do NOT write" in prompt
    assert "matching `./problem/sample_submission.csv`" not in prompt  # CSV contract absent
    assert "writes `./submission.csv`" not in prompt
    assert "{{" not in prompt  # all tokens rendered


def test_quantile_note_literal():
    from hillclimb.problem import ProblemSpec
    from hillclimb.search import GreedySearcher

    note = GreedySearcher._quantile_note
    spec = ProblemSpec(
        problem_id="q", problem_dir=Path("."), data_dir=Path("."),
        description="", metric_name="pinball", higher_is_better=False,
        time_budget_s=600, emflow_problem="x", verifier_cmd=["eval"],
        emflow_quantiles=[i / 100 for i in range(1, 100)],
    )
    class Stub:  # noqa: N801 — minimal receiver for the unbound method
        problem = spec
    rendered = note(Stub())
    assert "tuple(i / 100 for i in range(1, 100))" in rendered
    spec2 = spec.model_copy(update={"emflow_quantiles": [0.1, 0.5, 0.9]})
    class Stub2:
        problem = spec2
    assert "(0.1, 0.5, 0.9)" in note(Stub2())


from pathlib import Path  # noqa: E402
