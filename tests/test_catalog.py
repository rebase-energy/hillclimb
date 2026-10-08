"""The catalog: the problems (and, from 0.9, the climbers) hillclimb ships
as EXAMPLES. One copy — the repository's own `problems/` and `climbers/` —
read in place from a checkout and bundled into the wheel by `hatch_build.py`;
the engine imports nothing from it."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import hillclimb
from hillclimb import catalog

REPO = Path(__file__).resolve().parents[1]


def _hook():
    """`hatch_build.py` loaded by path, as hatchling loads it."""
    spec = importlib.util.spec_from_file_location("hatch_build", REPO / "hatch_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_checkout_reads_its_own_folders():
    assert catalog.root() == REPO
    assert catalog.problems_dir() == REPO / "problems" and catalog.climbers_dir() == REPO / "climbers"
    for problem_id in catalog.PROBLEM_IDS:
        assert (catalog.problem_path(problem_id) / "problem.yaml").is_file(), problem_id
    with pytest.raises(ValueError, match="no catalog problem 'nope'"):
        catalog.problem_path("nope")
    # what stays in the repository: the reference meta-problem, the generators, the dev-only problems
    assert "meta-heilbronn" not in catalog.PROBLEM_IDS and "bin-packing" not in catalog.PROBLEM_IDS


def test_the_wheel_gets_exactly_the_catalog():
    """The build hook's file list: every shipped problem's files (no caches),
    nothing the id list leaves out, and every catalog climber folder."""
    included = _hook().catalog_files(REPO)
    inside = set(included.values())
    for problem_id in catalog.PROBLEM_IDS:
        assert f"hillclimb/_catalog/problems/{problem_id}/problem.yaml" in inside, problem_id
    assert not [p for p in inside if "__pycache__" in p or p.endswith(".pyc")]
    assert not [p for p in inside if "/meta-heilbronn/" in p or "/bin-packing/" in p or p.endswith("make_heilbronn.py")]
    for source in included:
        assert Path(source).is_file()
    for name in catalog.climber_names():
        assert f"hillclimb/_catalog/climbers/{name}/{catalog.CLIMBER_ENTRY}" in inside


def test_the_catalog_module_stays_stdlib_at_the_top():
    """The build hook loads it with nothing but hatchling installed."""
    import ast

    tree = ast.parse((REPO / "src/hillclimb/catalog.py").read_text())
    top = {
        (node.names[0].name if isinstance(node, ast.Import) else node.module).split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert top <= set(sys.stdlib_module_names), top - set(sys.stdlib_module_names)


def test_scaffold_is_engine_data_not_catalog(tmp_path):
    from hillclimb.scaffold import scaffold_problem, windows_verifier_path

    assert windows_verifier_path().is_file()
    created = scaffold_problem(tmp_path, "mine")
    assert (created / "problem.yaml").is_file() and "mine" in (created / "problem.yaml").read_text()


@pytest.mark.slow
def test_a_built_wheel_carries_the_catalog(tmp_path):
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], cwd=REPO, check=True, capture_output=True)
    wheel = next(tmp_path.glob("hillclimb-*.whl"))
    assert wheel.name == f"hillclimb-{hillclimb.__version__}-py3-none-any.whl"  # the one version, read at build time
    names = set(zipfile.ZipFile(wheel).namelist())
    for problem_id in catalog.PROBLEM_IDS:
        assert f"hillclimb/_catalog/problems/{problem_id}/problem.yaml" in names, problem_id
    assert not [n for n in names if n.startswith("hillclimb/demo/") or "__pycache__" in n]
    assert not [n for n in names if "meta-heilbronn" in n]


# --- the catalog climbers ----------------------------------------------------
# Each is ONE self-contained file (`policy.py`: both policies, every default,
# the Climber(...) that wires them) that `climber get` copies as it is; what
# the two share is duplicated on purpose, and held together here.

import inspect  # noqa: E402

from tests.catalog_fixture import GEPA, GREEDY, META_BASELINE, OPENEVOLVE  # noqa: E402

SCHEDULE_METHODS = (
    "schedule", "debuggable_tip", "prospective_branches", "in_ensemble_window",
    "should_combine", "combine_succeeded", "combine_candidates",
)
TAIL_MARKER = "# --- the climber ---"


def _method_sources(cls, names) -> dict[str, str]:
    # `getattr` unwraps staticmethods, so the source is the function's
    return {name: inspect.getsource(getattr(cls, name)) for name in names}


def _own_methods(cls) -> list[str]:
    return sorted(name for name, obj in vars(cls).items() if callable(getattr(cls, name)) and not name.startswith("__"))


def _modules():
    return catalog.module("greedy"), catalog.module("openevolve")


def test_the_catalog_climbers_are_the_folders():
    assert set(catalog.climber_names()) >= {"greedy", "openevolve"}
    assert catalog.climber_path("greedy") == REPO / "climbers" / "greedy"
    with pytest.raises(ValueError, match="no catalog climber 'nope'"):
        catalog.climber_path("nope")
    greedy = catalog.climber("greedy")
    assert greedy.name == "greedy" and greedy.module.Greedy is catalog.module("greedy").Greedy
    assert greedy.spec.operator_policy == f"{GREEDY}:Greedy" and greedy.spec.selector_policy == f"{GREEDY}:Best"
    assert greedy.spec.params == {} and greedy.spec.selector_params == {}  # the values are the classes' DEFAULTS
    assert greedy.prompts_dir is None  # no prompts/ in the catalog folder: the built-in templates render


def test_the_two_greedy_operator_policies_differ_only_in_their_defaults():
    greedy, openevolve = _modules()
    assert _own_methods(greedy.Greedy) == _own_methods(openevolve.Greedy)
    assert _method_sources(greedy.Greedy, _own_methods(greedy.Greedy)) == _method_sources(
        openevolve.Greedy, _own_methods(openevolve.Greedy)
    )
    assert inspect.getsource(greedy._trial_cost_s) == inspect.getsource(openevolve._trial_cost_s)
    assert greedy.MIN_TRIAL_COST_S == openevolve.MIN_TRIAL_COST_S
    assert greedy.Greedy.__doc__ == openevolve.Greedy.__doc__
    assert greedy.Greedy.name == openevolve.Greedy.name == "greedy"
    # what is different is the point of the second file: tuning off, map-elites picks
    assert {k: v for k, v in greedy.Greedy.defaults().items() if k != "tune_budget"} == {
        k: v for k, v in openevolve.Greedy.defaults().items() if k != "tune_budget"
    }
    assert (greedy.Greedy.defaults()["tune_budget"], openevolve.Greedy.defaults()["tune_budget"]) == (8, 0)


def test_the_two_selectors_share_the_schedule():
    greedy, openevolve = _modules()
    assert _method_sources(greedy.Best, SCHEDULE_METHODS) == _method_sources(openevolve.MapElites, SCHEDULE_METHODS)
    best, elites = greedy.Best.defaults(), openevolve.MapElites.defaults()
    best.pop("max_stale_children")  # greedy's `select` sets a stale candidate aside; MAP-Elites samples its archive
    assert set(best) <= set(elites)
    assert {k: v for k, v in best.items() if k != "ensemble"} == {k: elites[k] for k in best if k != "ensemble"}
    assert (best["ensemble"], elites["ensemble"]) == (True, False)


def test_the_greedy_file_is_a_one_file_climber_and_the_meta_baseline_is_its_head():
    """Only `hillclimb.sdk`, the facades and the standard library — the rule
    a meta-problem's candidate is held to (`hillclimb meta check`). The
    meta baseline is the file up to its `Climber(...)` tail: a candidate
    must not choose the tuner or the memory, the contract forbids it."""
    from hillclimb import meta

    assert meta.check_climber_source(META_BASELINE) == []
    # the whole catalog file is a CLIMBER, not a candidate: its tail's `from hillclimb import Climber`
    # is the one thing the candidate rule refuses (a candidate picks neither tuner nor memory)
    findings = meta.check_climber_source(GREEDY)
    assert len(findings) == 1 and "imports hillclimb.Climber" in findings[0]
    assert all("imports openevolve" in line or "imports hillclimb.Climber" in line for line in meta.check_climber_source(OPENEVOLVE))
    text, baseline = GREEDY.read_text(), META_BASELINE.read_text()
    assert text.startswith(baseline)
    assert text[len(baseline):].lstrip("\n").startswith(TAIL_MARKER)


def test_install_climber_copies_the_folder_and_adds_prompts(tmp_path):
    folder, created = catalog.install_climber(tmp_path / "climbers", "greedy")
    assert created and folder == tmp_path / "climbers" / "greedy"
    assert (folder / "policy.py").read_bytes() == GREEDY.read_bytes()  # the catalog's file, as it is
    assert sorted(p.name for p in (folder / "prompts").iterdir()) == sorted(
        ["draft.md", "research_cue.md", "debug.md", "improve.md", "ablation_cue.md", "ensemble.md", "README.md"]
    )
    (folder / "policy.py").write_text("edited")
    again, created = catalog.install_climber(tmp_path / "climbers", "greedy")
    assert not created and (again / "policy.py").read_text() == "edited"  # never overwritten
    # under another name the copy is called that (its `name=` line), nothing else changes
    renamed, _ = catalog.install_climber(tmp_path / "climbers", "greedy", as_name="mine")
    assert (renamed / "policy.py").read_text() == GREEDY.read_text().replace("name='greedy',", "name='mine',")


# --- no default climber -------------------------------------------------------


def test_a_folder_without_a_climber_is_told_to_fetch_one(tmp_path, monkeypatch):
    """The engine ships no climber: `run` refuses in the terminal (not in a
    detached child's log) and `hc.run` leaves no run folder behind."""
    from typer.testing import CliRunner

    from hillclimb import api, cli
    from hillclimb.config import Config
    from hillclimb.modules.refs import NoClimber

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    assert CliRunner().invoke(cli.app, ["init"]).exit_code == 0
    catalog.install_problem(tmp_path / "problems", "fitness-landscape")
    config = Config.load()
    assert config.climber is None
    with pytest.raises(NoClimber, match="hillclimb climber get greedy"):
        config.climber_block()
    with pytest.raises(NoClimber):
        api.run("fitness-landscape", config=config, agent="toy", budget="10s")
    assert not any(p.is_dir() for p in (tmp_path / "runs").iterdir())  # no run folder written before the refusal
    result = CliRunner().invoke(cli.app, ["run", "fitness-landscape", "--agent", "toy", "--budget", "10s"])
    assert result.exit_code == 1 and "hillclimb climber get greedy" in result.output
    # a knob alone has nothing to land on; naming a climber starts the block
    with pytest.raises(NoClimber, match="no climber to edit"):
        config.apply_overrides({"climber.params.tune_budget": 0})
    config.apply_overrides({"climber.operator_policy": str(GREEDY) + ":Greedy"})
    assert config.climber is not None and config.climber_block()["operator_policy"].endswith("policy.py:Greedy")


def test_catalog_climber_files_reach_the_harness_through_the_sdk_only():
    """Every `.py` of every catalog climber imports from `hillclimb.sdk`, the
    facades that name prebuilt pieces (`hillclimb.operators`, `.tuners`,
    `.memory`), `hillclimb` itself (for `Climber`) and its own folder — never
    the harness: a catalog climber is written as a third party would."""
    import ast

    allowed = {"hillclimb", "hillclimb.sdk", "hillclimb.operators", "hillclimb.tuners", "hillclimb.memory"}
    offenders = []
    for name in catalog.climber_names():
        for path in sorted(catalog.climber_path(name).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name
                else:
                    continue
                if module.split(".")[0] == "hillclimb" and module not in allowed and not module.startswith("hillclimb.sdk."):
                    offenders.append(f"{path.relative_to(REPO)}: {module}")
    assert offenders == []


# --- what a climber imports beyond hillclimb -----------------------------------


def _hillclimb_installs() -> set[str]:
    """The top-level module names of hillclimb's own dependencies: a climber
    may lean on what the engine's environment is guaranteed to hold."""
    import re
    import tomllib

    names = set()
    for requirement in tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]:
        names.add(re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower().replace("-", "_"))
    return names | {"yaml"}  # pyyaml imports as yaml


def _third_party_imports(folder: Path) -> set[str]:
    import ast
    import sys

    found = set()
    installs = _hillclimb_installs()
    for path in sorted(folder.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
            elif isinstance(node, ast.Import):
                top = node.names[0].name.split(".")[0]
            else:
                continue
            if top not in sys.stdlib_module_names and top != "hillclimb" and top.lower() not in installs:
                found.add(top)
    return found


def test_a_catalog_climber_names_what_it_imports_beyond_hillclimb():
    """A library a catalog climber imports beyond what hillclimb installs is
    declared in its own `requirements.txt`, like a problem's — never as an
    extra of the engine, which ships no climber. gepa and openevolve are the
    two today."""
    import tomllib

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"].get("optional-dependencies", {})
    assert "gepa" not in extras and "openevolve" not in extras
    for name in catalog.climber_names():
        folder = catalog.climber_path(name)
        imports = _third_party_imports(folder)
        requirements = folder / catalog.CLIMBER_REQUIREMENTS
        if not imports:
            assert not requirements.exists(), f"{name} declares requirements it never imports"
            continue
        declared = [line.strip() for line in requirements.read_text().splitlines() if line.strip() and not line.startswith("#")]
        for module in imports:
            assert any(line.startswith(module) for line in declared), f"{name} imports {module} without declaring it"
    assert _third_party_imports(catalog.climber_path("gepa")) == {"gepa"}
    assert _third_party_imports(catalog.climber_path("openevolve")) == {"openevolve"}


def test_install_climber_copies_the_requirements(tmp_path):
    folder, _ = catalog.install_climber(tmp_path / "climbers", "gepa")
    assert (folder / "requirements.txt").read_bytes() == (GEPA.parent / "requirements.txt").read_bytes()


def test_a_missing_library_is_told_with_the_requirements_line(tmp_path):
    """An import a climber's file cannot satisfy ends with the install line of
    the `requirements.txt` beside it — and with nothing more when there is none."""
    from hillclimb.modules.refs import ClimberLoadError, FileScope

    mine = tmp_path / "mine.py"
    mine.write_text("import no_such_library_xyz\n")
    with pytest.raises(ClimberLoadError) as bare:
        FileScope([mine]).import_file(mine)
    assert "no_such_library_xyz" in str(bare.value) and "pip install" not in str(bare.value)
    (tmp_path / "requirements.txt").write_text("no_such_library_xyz>=1\n")
    with pytest.raises(ClimberLoadError) as told:
        FileScope([mine]).import_file(mine)
    assert str(told.value).endswith(f"this climber's requirements: pip install -r {tmp_path / 'requirements.txt'}")
