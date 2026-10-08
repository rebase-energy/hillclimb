"""A run never climbs on a budget nobody chose: the flag, else the folder's
run defaults (runs/config.yaml), else a yes to the default at a terminal —
and with no one to ask, it refuses before anything is written."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hillclimb.cli import common
from hillclimb.config import DEFAULT_BUDGET_S, Config, ConfigError
from hillclimb.harness.budget import NoBudget, resolve_budget
from hillclimb.project import RUNS_CONFIG, scaffold_hillclimb_dir


def test_the_flag_beats_the_run_defaults_which_beat_the_default():
    config = Config()
    config.budget.total_s = 600
    assert resolve_budget("2m", config) == 120
    assert resolve_budget(None, config) == 600
    config.budget.total_s = None
    assert resolve_budget(None, config, ask=lambda seconds: True) == DEFAULT_BUDGET_S


def test_no_budget_and_no_yes_is_an_error():
    config = Config()
    asked = []
    with pytest.raises(NoBudget, match="runs/config.yaml"):
        resolve_budget(None, config)  # nobody to ask: a script, an agent
    with pytest.raises(NoBudget, match="--budget"):
        resolve_budget(None, config, ask=lambda seconds: asked.append(seconds) or False)
    assert asked == [DEFAULT_BUDGET_S]


def test_the_prompt_defaults_to_no(monkeypatch):
    seen = {}
    monkeypatch.setattr(common, "is_interactive", lambda: True)
    monkeypatch.setattr(common.typer, "confirm", lambda text, default: seen.update(text=text, default=default) or default)
    with pytest.raises(NoBudget):
        common.resolve_run_budget(None, Config())
    assert seen["default"] is False and "7200s" in seen["text"]


def test_init_writes_run_defaults_with_the_budget_left_unset(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    assert (folder / "runs" / RUNS_CONFIG).is_file()
    assert Config.load(start=folder).budget.total_s is None


def test_the_run_defaults_are_read_above_hillclimb_yaml(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / "runs" / RUNS_CONFIG).write_text(yaml.safe_dump({"budget": {"total_s": 900}}))
    assert Config.load(start=folder).budget.total_s == 900


def test_a_key_that_is_not_a_run_default_is_refused(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / "runs" / RUNS_CONFIG).write_text(yaml.safe_dump({"problems": ["circle-packing"]}))
    with pytest.raises(ConfigError, match="problems cannot go in the run defaults"):
        Config.load(start=folder)


def test_run_without_a_budget_refuses_before_writing_anything(config, monkeypatch, tmp_path):
    from hillclimb import cli

    config.budget.total_s = None
    launched = []
    monkeypatch.setattr(common, "load_config", lambda agent=None, model=None: config)
    monkeypatch.setattr("hillclimb.cli.run.run_fleet", lambda *a, **k: launched.append(a))
    result = CliRunner().invoke(cli.app, ["run", "circle-packing", "--agent", "dummy"])
    assert isinstance(result.exception, NoBudget)
    assert not launched and not Path(config.paths.runs_dir).exists()


def test_run_hands_the_run_defaults_budget_to_its_engine(config, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from hillclimb import cli

    config.budget.total_s = 900
    calls = []

    def fake_run_fleet(target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(startup_failures=lambda **_: [], procs=[], run_id="r", run_dir=tmp_path / "runs" / "r")

    monkeypatch.setattr(common, "load_config", lambda agent=None, model=None: config)
    monkeypatch.setattr("hillclimb.cli.run.run_fleet", fake_run_fleet)
    result = CliRunner().invoke(cli.app, ["run", "circle-packing", "--agent", "dummy"])
    assert result.exit_code == 0, result.output
    assert calls[0]["budget"] == "900s"


def test_reset_runs_keeps_the_run_defaults(tmp_path, monkeypatch):
    from hillclimb import cli

    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / "runs" / RUNS_CONFIG).write_text("budget: {total_s: 900}\n")
    (folder / "runs" / "some-run").mkdir()
    monkeypatch.chdir(folder)
    result = CliRunner().invoke(cli.app, ["reset", "--runs", "--yes"])
    assert result.exit_code == 0, result.output
    assert not (folder / "runs" / "some-run").exists()
    assert (folder / "runs" / RUNS_CONFIG).read_text() == "budget: {total_s: 900}\n"


def test_a_whole_config_splits_into_the_folders_and_the_run_defaults():
    from hillclimb.config import split_config

    general, runs = split_config({
        "agent": "dummy", "sandbox": {"enabled": False},
        "concurrency": {"parallel_agents": 2, "machine_max_agents": 4}, "budget": {"total_s": 60},
    })
    assert general == {"sandbox": {"enabled": False}, "concurrency": {"machine_max_agents": 4}}
    assert runs == {"agent": "dummy", "concurrency": {"parallel_agents": 2}, "budget": {"total_s": 60}}


def test_an_explicit_path_reads_the_run_defaults_beside_it(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "hillclimb.yaml").write_text("sandbox: {enabled: false}\n")
    (tmp_path / "runs" / RUNS_CONFIG).write_text("budget: {total_s: 60}\nagent: dummy\n")
    config = Config.load(path=tmp_path / "hillclimb.yaml")
    assert config.budget.total_s == 60 and config.agent == "dummy" and not config.sandbox.enabled


def test_the_machines_cap_is_refused_in_the_run_defaults(tmp_path):
    folder = scaffold_hillclimb_dir(tmp_path / "hc")
    (folder / "runs" / RUNS_CONFIG).write_text("concurrency: {machine_max_agents: 2}\n")
    with pytest.raises(ConfigError, match="caps every search on this machine"):
        Config.load(start=folder)


def test_a_folder_pin_is_unpinned_in_both_its_files():
    from hillclimb.connect import unpin_split

    runs, folder = unpin_split("agent: codex\n", "agent_auth: subscription\n", "codex")
    assert runs == "# agent: codex\n" and folder == "# agent_auth: subscription\n"
    runs, folder = unpin_split("agent: claude-code\n", "agent_auth: subscription\n", "codex")
    assert runs == "agent: claude-code\n" and folder == "agent_auth: subscription\n"  # another agent's pin stays
