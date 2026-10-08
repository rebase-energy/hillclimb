"""How a problem's score is measured (`evaluation:`) and reported (`report:`)
is the problem's: its problem.yaml says it, a search records it, and no
config file or run overrides it."""

from __future__ import annotations

import pytest

from hillclimb.config import Config, ConfigError
from hillclimb.problem import ProblemError, load_problem
from hillclimb.project import RUNS_CONFIG, scaffold_hillclimb_dir
from hillclimb.scaffold import scaffold_problem
from tests.catalog_fixture import pin


@pytest.fixture
def problem_yaml(tmp_path):
    return scaffold_problem(tmp_path / "problems", "p") / "problem.yaml"


def test_a_problem_says_how_its_score_is_measured(problem_yaml):
    problem_yaml.write_text(problem_yaml.read_text() + "evaluation: {n_replicates: 3, noise_k: 2}\nreport: {enabled: false}\n")
    spec = load_problem(problem_yaml, Config())
    assert spec.evaluation.n_replicates == 3 and spec.evaluation.noise_k == 2
    assert spec.evaluation.model_fields_set == {"n_replicates", "noise_k"}
    assert spec.report.enabled is False


def test_an_unknown_evaluation_key_is_a_problem_error(problem_yaml):
    problem_yaml.write_text(problem_yaml.read_text() + "evaluation: {n_replicate: 3}\n")
    with pytest.raises(ProblemError, match="unknown `evaluation.n_replicate`"):
        load_problem(problem_yaml, Config())


def test_a_search_measures_as_its_problem_says_and_records_it(problem_yaml, tmp_path):
    from hillclimb.api import create_search
    from hillclimb.harness.run import load_search_meta

    problem_yaml.write_text(problem_yaml.read_text() + "evaluation: {n_replicates: 3}\n")
    config = Config()
    pin(config)
    config.paths.runs_dir = tmp_path / "runs"
    config.evaluation.noise_k = 1.5  # not the problem's to say here: it stays
    problem = load_problem(problem_yaml, config)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    search_dir = create_search(config, problem, run_dir, "r1", 60)
    assert config.evaluation.n_replicates == 3 and config.evaluation.noise_k == 1.5
    meta = load_search_meta(search_dir)
    assert meta.evaluation["n_replicates"] == 3 and meta.report["enabled"] is True


@pytest.mark.parametrize("where", ["hillclimb.yaml", f"runs/{RUNS_CONFIG}"])
def test_no_config_file_sets_the_problems_measurement(tmp_path, where):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / where).write_text("evaluation: {n_replicates: 3}\n")
    with pytest.raises(ConfigError, match="evaluation is the problem's"):
        Config.load(start=folder)


def test_no_run_overrides_it_either(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    with pytest.raises(ConfigError, match="report is the problem's"):
        Config.load(start=folder, **{"report.enabled": False})
