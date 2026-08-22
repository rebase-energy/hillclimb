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
        user.write_text(yaml.safe_dump({"model": "haiku", "backend": "dummy"}))
        root = make_hillclimb_dir(tmp_path / "ws", {"model": "opus"})
        monkeypatch.chdir(root)
        config = Config.load()
        assert config.model == "opus"  # the hillclimb dir wins
        assert config.backend == "dummy"  # user fills the gap

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
        assert config.backend == "claude-code"

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
        assert (tmp_path / "hillclimb" / "specs" / "example.yaml").exists()
        assert "hillclimb/runs/" in (tmp_path / ".gitignore").read_text()
        assert find_hillclimb_dir(tmp_path) == tmp_path / "hillclimb"

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
