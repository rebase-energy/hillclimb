"""hillclimb-dir discovery, machine dirs, config precedence, init, run specs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hillclimb.config import Config
from hillclimb.project import (
    HillclimbDirNotFound,
    find_hillclimb_dir,
    machine_cache_dir,
    user_config_path,
)


def make_hillclimb_dir(root: Path, config: dict | None = None) -> Path:
    """Create `<root>/hillclimb/config.yaml`; returns the holding folder."""
    marker = root / "hillclimb" / "config.yaml"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(yaml.safe_dump(config or {}))
    return root


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep discovery and machine dirs away from the real environment."""
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.delenv("HILLCLIMB_WORKSPACE", raising=False)
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "machine-cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


class TestDiscovery:
    def test_finds_marker_from_nested_dir(self, tmp_path):
        root = make_hillclimb_dir(tmp_path / "ws")
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        assert find_hillclimb_dir(nested) == root / "hillclimb"

    def test_finds_it_from_inside_itself(self, tmp_path):
        root = make_hillclimb_dir(tmp_path / "ws")
        assert find_hillclimb_dir(root / "hillclimb") == root / "hillclimb"

    def test_none_without_marker(self, tmp_path):
        assert find_hillclimb_dir(tmp_path) is None

    def test_env_pin_short_circuits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HILLCLIMB_DIR", str(tmp_path / "pinned"))
        assert find_hillclimb_dir(tmp_path) == (tmp_path / "pinned").resolve()

    def test_legacy_env_pin_names_the_parent(self, tmp_path, monkeypatch):
        """HILLCLIMB_WORKSPACE named the folder holding hillclimb/, not the dir."""
        monkeypatch.setenv("HILLCLIMB_WORKSPACE", str(tmp_path / "pinned"))
        assert find_hillclimb_dir(tmp_path) == (tmp_path / "pinned").resolve() / "hillclimb"

    def test_machine_dirs_honor_env(self, tmp_path, monkeypatch):
        assert machine_cache_dir() == tmp_path / "machine-cache"
        monkeypatch.delenv("HILLCLIMB_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
        assert machine_cache_dir() == tmp_path / "xdg-cache" / "hillclimb"
        assert user_config_path() == tmp_path / "xdg-config" / "hillclimb" / "config.yaml"


class TestConfigPrecedence:
    def test_paths_resolve_against_the_holding_folder(self, tmp_path, monkeypatch):
        root = make_hillclimb_dir(tmp_path / "ws")
        nested = root / "deep"
        nested.mkdir()
        monkeypatch.chdir(nested)
        config = Config.load()
        assert config.hillclimb_dir == root / "hillclimb"
        assert config.paths.runs_dir == root / "hillclimb" / "runs"
        assert config.paths.problems_dir == root / "hillclimb" / "problems"

    def test_hillclimb_dir_overrides_user_config(self, tmp_path, monkeypatch):
        user = user_config_path()
        user.parent.mkdir(parents=True)
        user.write_text(yaml.safe_dump({"model": "haiku", "agent": "dummy"}))
        root = make_hillclimb_dir(tmp_path / "ws", {"model": "opus"})
        monkeypatch.chdir(root)
        config = Config.load()
        assert config.model == "opus"  # the hillclimb dir wins
        assert config.agent == "dummy"  # user fills the gap

    def test_overrides_beat_the_config_file(self, tmp_path, monkeypatch):
        root = make_hillclimb_dir(tmp_path / "ws", {"model": "opus"})
        monkeypatch.chdir(root)
        assert Config.load(model="haiku").model == "haiku"

    def test_missing_hillclimb_dir_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(HillclimbDirNotFound):
            Config.load()

    def test_require_dir_false_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = Config.load(require_dir=False)
        assert config.hillclimb_dir is None
        assert config.agent == "claude-code"

    def test_explicit_path_bypasses_discovery(self, tmp_path):
        explicit = tmp_path / "custom.yaml"
        explicit.write_text(yaml.safe_dump({"model": "opus"}))
        config = Config.load(path=explicit)
        assert config.model == "opus"
        assert config.hillclimb_dir is None


class TestInit:
    def run_init(self, *args):
        from hillclimb.cli import app

        return CliRunner().invoke(app, ["init", *args])

    def test_creates_layout(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self.run_init()
        assert result.exit_code == 0
        assert (tmp_path / "hillclimb" / "config.yaml").exists()
        assert (tmp_path / "hillclimb" / "problems").is_dir()
        assert (tmp_path / "hillclimb" / "runs").is_dir()
        assert not (tmp_path / "hillclimb" / "specs").exists()  # a run carries its own spec.yaml
        ignored = (tmp_path / ".gitignore").read_text().splitlines()
        # the record of a run is committed, its bulk is not, keys never
        assert "hillclimb/runs/" not in ignored
        assert "hillclimb/runs/*/searches/*/candidates/" in ignored
        assert "hillclimb/runs/*/logs/" in ignored
        assert "!hillclimb/runs/*/searches/*/best/solution.py" in ignored
        assert "hillclimb/.env" in ignored  # provider keys live there
        assert find_hillclimb_dir(tmp_path) == tmp_path / "hillclimb"
        # idempotent: a second init adds nothing
        before = (tmp_path / ".gitignore").read_text()
        assert self.run_init("--force").exit_code == 0
        assert (tmp_path / ".gitignore").read_text() == before

    def test_gitignore_rules_keep_the_record_and_drop_the_bulk(self, tmp_path, monkeypatch):
        """git itself decides: with the init rules, a run's record files are
        tracked and its candidates, logs and submission are not."""
        import subprocess

        monkeypatch.chdir(tmp_path)
        assert self.run_init().exit_code == 0
        search = tmp_path / "hillclimb" / "runs" / "r1" / "searches" / "p"
        for rel in (
            "run.yaml", "spec.yaml", "searches/p/search.yaml", "searches/p/journal.jsonl",
            "searches/p/status.json", "searches/p/knowledge_card.yaml", "searches/p/climber/climber.yaml",
            "searches/p/best/solution.py", "searches/p/best/params.json", "searches/p/best/submission.csv",
            "searches/p/candidates/c001/solution.py", "searches/p/candidates/c001/agent_stream.jsonl",
            "searches/p/control/stop.json", "logs/01-p.log",
        ):
            path = tmp_path / "hillclimb" / "runs" / "r1" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        (tmp_path / "hillclimb" / ".env").write_text("OPENROUTER_API_KEY=sk\n")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.split()
        tracked = {p for p in out if p.startswith("hillclimb/runs/r1/")}
        assert "hillclimb/runs/.gitkeep" in out  # the folder itself is versioned
        assert tracked == {
            "hillclimb/runs/r1/run.yaml", "hillclimb/runs/r1/spec.yaml",
            "hillclimb/runs/r1/searches/p/search.yaml", "hillclimb/runs/r1/searches/p/journal.jsonl",
            "hillclimb/runs/r1/searches/p/status.json", "hillclimb/runs/r1/searches/p/knowledge_card.yaml",
            "hillclimb/runs/r1/searches/p/climber/climber.yaml",
            "hillclimb/runs/r1/searches/p/best/solution.py", "hillclimb/runs/r1/searches/p/best/params.json",
        }
        assert "hillclimb/.env" not in out
        assert search.exists()

    def test_refuses_nested_without_force(self, tmp_path, monkeypatch):
        make_hillclimb_dir(tmp_path)
        inner = tmp_path / "inner"
        inner.mkdir()
        monkeypatch.chdir(inner)
        assert self.run_init().exit_code == 1
        assert self.run_init("--force").exit_code == 0


class TestRunSpecs:
    def test_dict_entries_and_single_target_form(self, tmp_path):
        from hillclimb.problem import load_suite

        config = Config(paths={"problems_dir": tmp_path})
        multi = tmp_path / "multi.yaml"
        multi.write_text(yaml.safe_dump({
            "problems": [
                "emflow://gefcom2014:wind",
                {"target": "emflow://gefcom2014:solar", "model": "opus", "budget": "2h"},
            ]
        }))
        suite = load_suite(multi, config)
        assert suite.problems[0].target == "emflow://gefcom2014:wind"
        assert suite.problems[0].model is None
        assert suite.problems[1].model == "opus"
        assert suite.problems[1].budget == "2h"

        single = tmp_path / "single.yaml"
        single.write_text(yaml.safe_dump({"target": "emflow://gefcom2014:solar", "model": "opus"}))
        suite = load_suite(single, config)
        assert len(suite.problems) == 1
        assert suite.problems[0].model == "opus"

        # an entry may pin its climber and carry `--set` pairs, which is what
        # lets a run's own spec.yaml say everything the launch said
        rich = tmp_path / "rich.yaml"
        rich.write_text(yaml.safe_dump({"problems": [
            {"target": "emflow://gefcom2014:solar", "climber": "gepa", "set": ["budget.max_evaluations=3"]},
        ]}))
        entry = load_suite(rich, config).problems[0]
        assert entry.climber == "gepa" and entry.set == ["budget.max_evaluations=3"]

    def test_a_run_writes_its_own_spec(self, tmp_path):
        """`spec_entry` + `write_run_spec`: every parameter the launch
        resolved to, None left out, the budget in seconds when it was a
        number, the seed path absolute, rerunnable as a suite."""
        from hillclimb.api import spec_entry, write_run_spec
        from hillclimb.problem import load_suite

        (tmp_path / "seed.py").write_text("x = 1\n")
        entries = [
            spec_entry("heilbronn-11", budget=600, agent="dummy", model="sonnet", climber="greedy",
                       parallel_agents=2, n_replicates=None, seed_from=tmp_path / "seed.py",
                       set=["budget.max_evaluations=3"]),
            spec_entry("heilbronn-11", name="gepa", budget="10m", climber="gepa"),
        ]
        path = write_run_spec(tmp_path / "run", entries, source="hillclimb/experiments/x.yaml") if (tmp_path / "run").mkdir() is None else None
        text = path.read_text()
        assert text.startswith("# The spec this run launched from")
        assert "# launched from: hillclimb/experiments/x.yaml" in text
        suite = load_suite(path, Config(paths={"problems_dir": tmp_path}))
        first, second = suite.problems
        assert (first.budget, first.agent, first.climber, first.parallel_agents) == ("600s", "dummy", "greedy", 2)
        assert first.n_replicates is None and first.seed_from == str((tmp_path / "seed.py").resolve())
        assert first.set == ["budget.max_evaluations=3"]
        assert (second.name, second.budget, second.climber, second.set) == ("gepa", "10m", "gepa", [])


class TestVenvHashing:
    def test_hash_keyed_and_source_sensitive(self):
        from hillclimb.api import default_venv_python

        config = Config()
        base = default_venv_python(config, "csv")
        assert base == default_venv_python(config, "csv")  # deterministic
        assert "csv-" in base.parts[-3]
        emflow_a = default_venv_python(config, "emflow")
        config.emflow.source = "-e ../somewhere-else"
        emflow_b = default_venv_python(config, "emflow")
        assert emflow_a != emflow_b  # source participates in the key
        assert base.parts[-3] != emflow_a.parts[-3]
