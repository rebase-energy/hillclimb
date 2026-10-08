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
from tests.catalog_fixture import GEPA, GREEDY, class_ref


def make_hillclimb_dir(root: Path, config: dict | None = None) -> Path:
    """Create `<root>/hillclimb.yaml` (and the run defaults, for run keys);
    returns `root`, now a hillclimb dir."""
    from tests.folder_config import write_config

    write_config(root, config)
    return root


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep discovery and machine dirs away from the real environment."""
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    monkeypatch.setenv("HILLCLIMB_CACHE_DIR", str(tmp_path / "machine-cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


class TestDiscovery:
    def test_no_upward_search_from_a_nested_dir(self, tmp_path):
        """Commands run from the hillclimb dir's root; a folder below it is
        not the dir (nor is anything above a folder)."""
        root = make_hillclimb_dir(tmp_path / "ws")
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        assert find_hillclimb_dir(nested) is None

    def test_a_sibling_named_hillclimb_is_not_this_folders(self, tmp_path):
        """A hillclimb checkout beside your project (`Github/hillclimb/`
        holds a hillclimb.yaml) must not be taken for your project's dir."""
        make_hillclimb_dir(tmp_path / "hillclimb")
        mine = tmp_path / "my-project"
        mine.mkdir()
        assert find_hillclimb_dir(mine) is None

    def test_a_projects_hillclimb_subfolder_is_found_from_the_project_root(self, tmp_path):
        sub = make_hillclimb_dir(tmp_path / "proj" / "hillclimb")
        assert find_hillclimb_dir(tmp_path / "proj") == sub

    def test_finds_it_from_inside_itself(self, tmp_path):
        root = make_hillclimb_dir(tmp_path / "ws")
        assert find_hillclimb_dir(root) == root

    def test_any_folder_name_works(self, tmp_path):
        """The marker is the file, not the folder's name."""
        root = make_hillclimb_dir(tmp_path / "anything")
        assert find_hillclimb_dir(root) == root

    def test_a_plain_config_yaml_is_not_a_marker(self, tmp_path):
        """Plenty of repos have a config.yaml; only hillclimb.yaml counts."""
        (tmp_path / "config.yaml").write_text("model: opus\n")
        (tmp_path / "hillclimb").mkdir()
        (tmp_path / "hillclimb" / "config.yaml").write_text("model: opus\n")
        assert find_hillclimb_dir(tmp_path) is None

    def test_none_without_marker(self, tmp_path):
        assert find_hillclimb_dir(tmp_path) is None

    def test_env_pin_short_circuits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HILLCLIMB_DIR", str(tmp_path / "pinned"))
        assert find_hillclimb_dir(tmp_path) == (tmp_path / "pinned").resolve()

    def test_machine_dirs_honor_env(self, tmp_path, monkeypatch):
        assert machine_cache_dir() == tmp_path / "machine-cache"
        monkeypatch.delenv("HILLCLIMB_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
        assert machine_cache_dir() == tmp_path / "xdg-cache" / "hillclimb"
        assert user_config_path() == tmp_path / "xdg-config" / "hillclimb" / "config.yaml"


class TestConfigPrecedence:
    def test_paths_resolve_against_the_hillclimb_dir(self, tmp_path, monkeypatch):
        root = make_hillclimb_dir(tmp_path / "ws")
        monkeypatch.chdir(root)
        config = Config.load()
        assert config.hillclimb_dir == root
        assert config.paths.runs_dir == root / "runs"
        assert config.paths.problems_dir == root / "problems"
        assert config.store.sqlite_path == root / "store.sqlite"

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

    def test_creates_layout_in_the_current_folder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self.run_init()
        assert result.exit_code == 0, result.output
        assert (tmp_path / "hillclimb.yaml").exists()
        assert (tmp_path / "problems").is_dir()
        assert not (tmp_path / "problems" / "example").exists()  # picking a problem is the user's step
        assert (tmp_path / "runs").is_dir()
        # climbers/ is there from the start, empty until `hillclimb climber get`
        assert sorted(p.name for p in (tmp_path / "climbers").iterdir()) == [".gitkeep", ".hillclimb"]
        assert not (tmp_path / "hillclimb").exists()  # flat: no subfolder
        assert not (tmp_path / "specs").exists()  # a run carries its own spec.yaml
        ignored = (tmp_path / ".gitignore").read_text().splitlines()
        # the record of a run is committed, its bulk is not, keys never
        assert "/runs/" not in ignored
        assert "/runs/*/searches/*/candidates/" in ignored
        assert "/runs/*/logs/" in ignored
        assert "!/runs/*/searches/*/best/solution.py" in ignored
        assert "/.env" in ignored  # provider keys live there
        assert find_hillclimb_dir(tmp_path) == tmp_path
        # idempotent: a second init adds nothing
        before = (tmp_path / ".gitignore").read_text()
        assert self.run_init("--force").exit_code == 0
        assert (tmp_path / ".gitignore").read_text() == before

    def test_datastore_sqlite_writes_the_store_block(self, tmp_path, monkeypatch):
        """`init --datastore sqlite` is the database from the first run on;
        the default leaves the files store, with the choice commented in."""
        sqlite_dir, files_dir = tmp_path / "db", tmp_path / "plain"
        for folder in (sqlite_dir, files_dir):
            folder.mkdir()
        monkeypatch.chdir(sqlite_dir)
        result = self.run_init("--datastore", "sqlite")
        assert result.exit_code == 0, result.output
        assert "store.sqlite" in result.output
        assert Config.load().store.backend == "sqlite"
        monkeypatch.chdir(files_dir)
        assert self.run_init().exit_code == 0
        assert Config.load().store.backend == "files"
        assert "# store:" in (files_dir / "hillclimb.yaml").read_text()  # the option, discoverable
        assert self.run_init("--datastore", "postgres", "--force").exit_code == 2

    def test_a_directory_argument_makes_that_folder_the_dir(self, tmp_path, monkeypatch):
        """`hillclimb init hillclimb` is the tucked-away layout: everything
        in a subfolder, created if missing, with its own .gitignore."""
        monkeypatch.chdir(tmp_path)
        result = self.run_init("hillclimb")
        assert result.exit_code == 0, result.output
        folder = tmp_path / "hillclimb"
        assert (folder / "hillclimb.yaml").exists()
        assert (folder / "problems").is_dir() and (folder / "runs").is_dir()
        assert (folder / ".gitignore").exists()
        assert not (tmp_path / "hillclimb.yaml").exists()
        assert not (tmp_path / ".gitignore").exists()
        # found from the project's root, and from inside the subfolder
        assert find_hillclimb_dir(tmp_path) == folder
        assert find_hillclimb_dir(folder) == folder
        assert "cd hillclimb" in result.output

    def test_a_folder_with_folders_of_its_own_gets_a_hillclimb_subfolder(self, tmp_path, monkeypatch):
        """A code repo's own runs/ or knowledge/ must never be mixed with
        hillclimb's: hillclimb keeps to ./hillclimb/ instead, and finds it from
        the project root."""
        from hillclimb.project import OWNED_MARKER, find_hillclimb_dir

        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "mine.txt").write_text("x")
        (tmp_path / "knowledge").mkdir()
        monkeypatch.chdir(tmp_path)
        result = self.run_init()
        assert result.exit_code == 0, result.output
        assert "hillclimb keeps to" in result.output
        assert not (tmp_path / "hillclimb.yaml").exists()
        assert (tmp_path / "hillclimb" / "hillclimb.yaml").is_file()
        assert (tmp_path / "hillclimb" / "runs" / OWNED_MARKER).is_file()
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["mine.txt"]  # untouched
        assert find_hillclimb_dir(tmp_path) == tmp_path / "hillclimb"
        # made by hillclimb, so `reset` takes the whole subfolder and nothing of the project's
        from hillclimb.cli import main as cli_main

        monkeypatch.setattr("hillclimb.harness.orphans.live_engines", lambda: [])
        with pytest.raises(SystemExit) as exc:
            cli_main(["reset", "--yes"])
        assert exc.value.code == 0
        assert not (tmp_path / "hillclimb").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["knowledge", "runs"]
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["mine.txt"]

    def test_gitignore_rules_keep_the_record_and_drop_the_bulk(self, tmp_path, monkeypatch):
        """git itself decides: with the init rules, a run's record files are
        tracked and its candidates, logs and submission are not — also when
        the hillclimb dir is a subfolder of the repo."""
        import subprocess

        for folder, prefix in ((tmp_path / "flat", ""), (tmp_path / "nested", "hillclimb/")):
            folder.mkdir()
            monkeypatch.chdir(folder)
            assert self.run_init(prefix.rstrip("/") or ".").exit_code == 0
            hc = folder / prefix
            for rel in (
                "run.yaml", "spec.yaml", "searches/p/search.yaml", "searches/p/journal.jsonl",
                "searches/p/status.json", "searches/p/knowledge_card.yaml", "searches/p/climber/climber.yaml",
                "searches/p/best/solution.py", "searches/p/best/params.json", "searches/p/best/submission.csv",
                "searches/p/candidates/c001/solution.py", "searches/p/candidates/c001/agent_stream.jsonl",
                "searches/p/control/stop.json", "logs/01-p.log",
            ):
                path = hc / "runs" / "r1" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x")
            (hc / ".env").write_text("OPENROUTER_API_KEY=sk\n")
            (hc / "store.sqlite").write_text("x")
            subprocess.run(["git", "init", "-q"], cwd=folder, check=True)
            out = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"], cwd=folder, capture_output=True, text=True, check=True
            ).stdout.split()
            run = f"{prefix}runs/r1/"
            tracked = {p for p in out if p.startswith(run)}
            assert f"{prefix}runs/.gitkeep" in out  # the folder itself is versioned
            assert tracked == {
                run + "run.yaml", run + "spec.yaml",
                run + "searches/p/search.yaml", run + "searches/p/journal.jsonl",
                run + "searches/p/status.json", run + "searches/p/knowledge_card.yaml",
                run + "searches/p/climber/climber.yaml",
                run + "searches/p/best/solution.py", run + "searches/p/best/params.json",
            }, prefix
            assert f"{prefix}.env" not in out
            assert f"{prefix}store.sqlite" not in out
            assert f"{prefix}hillclimb.yaml" in out

    def test_refuses_a_folder_that_already_is_one(self, tmp_path, monkeypatch):
        make_hillclimb_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert self.run_init().exit_code == 1

    def test_a_folder_inside_another_dir_is_a_folder_of_its_own(self, tmp_path, monkeypatch):
        """No upward search: a subfolder of a hillclimb dir can become one
        without --force (the outer one is not this folder's)."""
        make_hillclimb_dir(tmp_path)
        inner = tmp_path / "inner"
        inner.mkdir()
        monkeypatch.chdir(inner)
        assert self.run_init().exit_code == 0
        assert (inner / "hillclimb.yaml").is_file()


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
            {"target": "emflow://gefcom2014:solar", "climber": str(GEPA), "set": ["budget.max_evaluations=3"]},
        ]}))
        entry = load_suite(rich, config).problems[0]
        assert entry.climber["loop"].endswith("gepa/loop.py:GepaLoop") and entry.set == ["budget.max_evaluations=3"]

    def test_a_spec_defines_the_climber_of_each_search(self, tmp_path):
        """The run config is where the climber is defined: an entry's
        `climber:` is the block (a bare name is a preset), a top-level one is
        the default for entries that name none, and file refs in it resolve
        from the spec's own folder — like `seed_from`."""
        from hillclimb.climber import resolve_climber
        from hillclimb.problem import load_suite

        config = Config(paths={"problems_dir": tmp_path})
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "mine.py").write_text(
            "class Mine:\n    def propose(self, view):\n        return None\n"
            "    def observe(self, view, candidate):\n        pass\n"
        )
        spec = specs / "run.yaml"
        spec.write_text(yaml.safe_dump({
            "climber": {"operator_policy": class_ref("openevolve", "Greedy"), "params": {"num_islands": 2}},
            "problems": [
                "emflow://gefcom2014:wind",
                {"target": "emflow://gefcom2014:solar", "climber": str(GEPA)},
                {"target": "emflow://gefcom2014:solar",
                 "climber": {"operator_policy": "mine.py", "tuner": "optuna", "operators": ["draft"]}},
            ],
        }))
        suite = load_suite(spec, config)
        default, preset, inline = (entry.climber for entry in suite.problems)
        assert (default["operator_policy"], default["params"]) == (class_ref("openevolve", "Greedy"), {"num_islands": 2})
        assert preset["loop"].endswith("gepa/loop.py:GepaLoop") and "policy" not in preset
        assert inline["operator_policy"] == str(specs / "mine.py") and inline["tuner"] == "optuna"
        assert resolve_climber(inline).build_loop().policy.name == "mine"
        single = specs / "single.yaml"
        gepa_loop = f"{GEPA.parent / 'loop.py'}:GepaLoop"
        single.write_text(yaml.safe_dump({"target": "emflow://gefcom2014:solar", "climber": {"loop": gepa_loop}}))
        assert load_suite(single, config).problems[0].climber["loop"] == gepa_loop
        spec.write_text(yaml.safe_dump({"problems": [{"target": "x", "climber": {"polcy": "greedy"}}]}))
        with pytest.raises(ValueError, match=r"run.yaml: problems\[1\].climber: .*polcy"):
            load_suite(spec, config)

    def test_a_run_writes_its_own_spec(self, tmp_path):
        """`spec_entry` + `write_run_spec`: every parameter the launch
        resolved to, None left out, the budget in seconds when it was a
        number, the seed path absolute, rerunnable as a suite."""
        from hillclimb.api import spec_entry, write_run_spec
        from hillclimb.problem import load_suite

        (tmp_path / "seed.py").write_text("x = 1\n")
        entries = [
            spec_entry("heilbronn-11", budget=600, agent="dummy", model="sonnet", climber=str(GREEDY),
                       parallel_agents=2, seed_from=tmp_path / "seed.py",
                       set=["budget.max_evaluations=3"]),
            spec_entry("heilbronn-11", name="gepa", budget="10m", climber=str(GEPA)),
        ]
        path = write_run_spec(tmp_path / "run", entries, source="hillclimb/experiments/x.yaml") if (tmp_path / "run").mkdir() is None else None
        text = path.read_text()
        assert text.startswith("# The spec this run launched from")
        assert "# launched from: hillclimb/experiments/x.yaml" in text
        suite = load_suite(path, Config(paths={"problems_dir": tmp_path}))
        first, second = suite.problems
        assert (first.budget, first.agent, first.parallel_agents) == ("600s", "dummy", 2)
        assert first.n_replicates is None and first.seed_from == str((tmp_path / "seed.py").resolve())
        assert first.set == ["budget.max_evaluations=3"]
        assert (second.name, second.budget, second.set) == ("gepa", "10m", [])
        # a run's spec carries each climber as its FULL block, however it was named
        written = yaml.safe_load(text)["problems"]
        assert written[0]["climber"] == first.climber and first.climber["operator_policy"] == class_ref("greedy", "Greedy")
        assert written[1]["climber"]["loop"].endswith("gepa/loop.py:GepaLoop") and second.climber["tuner"] == "random"


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


def test_a_folders_old_spelling_beats_the_user_levels_new_one(tmp_path, monkeypatch):
    """K1: `connect` writes `agent:` at the user level; a folder that still
    says `backend: dummy` must run dummy, not the user's paid agent. Old
    spellings are renamed per level before the levels merge."""
    from hillclimb.config import Config

    user = tmp_path / "user" / "config.yaml"
    user.parent.mkdir()
    user.write_text("agent: claude-code\nagent_auth: subscription\n")
    monkeypatch.setattr("hillclimb.config.user_config_path", lambda: user)
    folder = tmp_path / "proj"
    (folder / "runs").mkdir(parents=True)
    (folder / "hillclimb.yaml").write_text("")
    (folder / "runs" / "config.yaml").write_text("backend: dummy\n")
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    assert Config.load(start=folder).agent == "dummy"
