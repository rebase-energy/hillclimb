"""A problem folder with a mistake in it fails with one line that says what
to fix, and problem.yaml cannot silently climb the wrong way."""

from __future__ import annotations

import pytest

from hillclimb import cli
from hillclimb.config import Config
from hillclimb.scaffold import scaffold_problem
from hillclimb.problem import ProblemError, load_problem


@pytest.fixture
def problem_yaml(tmp_path):
    return scaffold_problem(tmp_path / "problems", "p") / "problem.yaml"


def _edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("higher_is_better: false", 'higher_is_better: "false"', "must be true or false, unquoted"),
        ("higher_is_better: false", "higher_is_better: maybe", "must be true or false, unquoted"),
        ("higher_is_better: false", "", "`higher_is_better:` is required"),
        ("metric: mean-log10-imbalance", "", "`metric:` is required"),
    ],
)
def test_bad_keys_are_problem_errors(problem_yaml, old, new, message):
    _edit(problem_yaml, old, new)
    with pytest.raises(ProblemError, match=message):
        load_problem(problem_yaml, Config())


def test_a_misspelt_key_is_warned_about_with_the_right_name(problem_yaml, capsys):
    _edit(problem_yaml, "time_budget_s:", "time_budjet_s:")
    load_problem(problem_yaml, Config())
    assert "did you mean `time_budget_s:`?" in capsys.readouterr().err


def test_legacy_lower_is_better_still_loads(problem_yaml):
    _edit(problem_yaml, "higher_is_better: false", "lower_is_better: true")
    assert load_problem(problem_yaml, Config()).higher_is_better is False


def test_a_missing_problem_says_how_to_get_it(tmp_path):
    (tmp_path / "hillclimb.yaml").write_text("")
    config = Config.load(start=tmp_path)
    with pytest.raises(ProblemError, match="hillclimb problem get heilbronn-11"):
        load_problem("heilbronn-11", config)
    with pytest.raises(ProblemError, match="hillclimb problem new my-own"):
        load_problem("my-own", config)


def test_the_cli_prints_one_line_not_a_traceback(tmp_path, monkeypatch, capsys):
    (tmp_path / "hillclimb.yaml").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["verify", "heilbronn-11", "--skip-intro"])
    assert exit_info.value.code == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "hillclimb problem get heilbronn-11" in text and "Traceback" not in text


def test_a_missing_agent_cli_is_found_before_anything_detaches(tmp_path, monkeypatch):
    """No claude on PATH: run_fleet refuses before it makes a run dir or
    spawns an engine, naming the install command."""
    import hillclimb.api as api
    from hillclimb.agents import AgentCLIMissing

    monkeypatch.setattr("shutil.which", lambda name: None)
    (tmp_path / "hillclimb.yaml").write_text("")
    config = Config.load(start=tmp_path)
    with pytest.raises(AgentCLIMissing, match="npm install -g @anthropic-ai/claude-code"):
        api.run_fleet("anything", config=config, agent="claude-code")
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "no score written"),
        ("", "is empty"),
        ("great", "neither JSON nor a number"),
        ('{"score": "great"}', "the score is not a number: 'great'"),
        ('{"score": NaN}', "the score is NaN"),
        ('{"value": 1}', "no `score` key"),
    ],
)
def test_a_result_without_a_score_says_why(tmp_path, text, message):
    from hillclimb.harness.executor import read_result, result_problem

    path = tmp_path / "result.json"
    if text is not None:
        path.write_text(text)
    assert read_result(path)[0] is None
    assert message in result_problem(path)


def test_the_python_api_makes_a_fresh_folder_a_hillclimb_dir(tmp_path, monkeypatch, capsys):
    from hillclimb.api import sdk_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    config = sdk_config()
    assert (tmp_path / "hillclimb.yaml").is_file() and config.hillclimb_dir.resolve() == tmp_path.resolve()
    assert "a hillclimb dir" in capsys.readouterr().out
