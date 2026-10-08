"""`hillclimb climber get`: a preset copied out as a folder that IS a climber —
policy.py, the shipped file with the `Climber(...)` that wires it appended
(the whole climber as Python, no config file), the templates its operators
render and a README of what fills them — and made the folder's default. The
file and the folder holding it are both first-class ways of naming it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from hillclimb import cli
from hillclimb.cli.climber import pin_climber
from hillclimb.climber import ClimberSpec, load_climber, resolve_climber
from hillclimb.config import Config
from hillclimb.modules.operators.builtin import BUILTIN_OPERATORS, TOKEN_GUIDE
from hillclimb.prompts.render import TEMPLATE_DIR, tokens_in
from tests.catalog_fixture import GREEDY, class_ref

from hillclimb import catalog
TEMPLATES = ["draft", "research_cue", "debug", "improve", "ablation_cue", "ensemble"]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HILLCLIMB_DIR", raising=False)
    assert CliRunner().invoke(cli.app, ["init"]).exit_code == 0
    return tmp_path


def test_get_writes_the_folder_and_makes_it_the_default(folder):
    result = CliRunner().invoke(cli.app, ["climber", "get", "greedy"])
    assert result.exit_code == 0, result.output
    target = folder / "climbers" / "greedy"
    files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert files == sorted(["policy.py", *(f"prompts/{t}.md" for t in TEMPLATES), "prompts/README.md"])  # no config file
    assert (folder / "climbers" / ".hillclimb").is_file()  # hillclimb's folder: `reset` may delete it
    for template in TEMPLATES:  # the templates are the package's, byte for byte
        assert (target / "prompts" / f"{template}.md").read_bytes() == (TEMPLATE_DIR / f"{template}.md").read_bytes()
    # the catalog's file, byte for byte: nothing hidden, nothing rewritten — the
    # Climber(...) at its end makes it the whole climber; prompts/ beside it is its prompts dir
    assert (target / "policy.py").read_bytes() == GREEDY.read_bytes()
    assert "climber = Climber(" in GREEDY.read_text() and "name='greedy'," in GREEDY.read_text()
    # the file is this hillclimb dir's climber now: a run default
    assert re.search(r"^climber: climbers/greedy/policy.py$", (folder / "runs" / "config.yaml").read_text(), re.M)
    config = Config.load(start=folder)
    policy_file = (target / "policy.py").resolve()
    assert config.climber.operator_policy == f"{policy_file}:Greedy"
    assert config.climber.selector_policy == f"{policy_file}:Best"  # both policies, in the one file
    assert Path(config.climber.prompts).resolve() == (target / "prompts").resolve()
    assert config.climber.params == {} and config.climber.selector_params == {}  # the values are the classes' DEFAULTS
    climber = resolve_climber(config.climber, folder)
    assert climber.operator_set().names() == ("draft", "debug", "improve", "ensemble")
    assert climber.prompts_dir == target / "prompts"
    policy = climber.build_loop().policy
    assert policy.resolved_params()["tune_budget"] == 8 and policy.selector.resolved_params()["num_drafts"] == 3
    assert type(policy.selector).__module__ == type(policy).__module__  # the folder's Best, not the package's
    assert "Fetched greedy" in result.output and "climber: climbers/greedy" in result.output


def test_the_folder_is_the_same_climber_however_it_is_named(folder):
    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    by_folder = load_climber("climbers/greedy", folder)  # the folder names the policy.py in it
    by_file = load_climber("climbers/greedy/policy.py", folder)
    by_string = resolve_climber(ClimberSpec.model_validate("climbers/greedy", context={"base_dir": folder}), folder)
    by_cli = resolve_climber(Config.load(start=folder).climber, folder)
    assert by_folder.sha256 == by_file.sha256 == by_string.sha256 == by_cli.sha256
    assert by_folder.name == "greedy"
    from hillclimb import catalog

    # a copy is its own climber (its prompts/ is part of what it is): edits count, the catalog's stays
    assert by_folder.sha256 != catalog.climber("greedy").sha256
    assert len({by_folder.scope.root, by_string.scope.root}) == 1


def test_get_is_idempotent_and_never_overwrites(folder):
    runner = CliRunner()
    assert runner.invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    improve = folder / "climbers" / "greedy" / "prompts" / "improve.md"
    improve.write_text("MY OWN IMPROVE PROMPT for {{metric_name}}.\n\n{{contract}}\n")
    again = runner.invoke(cli.app, ["climber", "get", "greedy"])
    assert again.exit_code == 0 and "Already have greedy" in again.output
    assert improve.read_text().startswith("MY OWN IMPROVE PROMPT")
    (folder / "climbers" / "mine").mkdir()
    taken = runner.invoke(cli.app, ["climber", "get", "greedy", "--name", "mine"])
    assert taken.exit_code == 1 and "not a climber folder" in taken.output


def test_get_names_the_copy_and_can_leave_the_default_alone(folder):
    result = CliRunner().invoke(cli.app, ["climber", "get", "openevolve", "--name", "qd", "--no-default"])
    assert result.exit_code == 0, result.output
    climber = load_climber("climbers/qd", folder)
    spec = climber.spec
    assert spec.name == "qd" and spec.selector_policy.endswith("/qd/policy.py:MapElites") and spec.operator_policy.endswith("/qd/policy.py:Greedy")
    assert spec.params == {} and spec.selector_params == {}  # ensemble off, tune off: the classes' DEFAULTS
    assert "climber: climbers/qd" not in (folder / "hillclimb.yaml").read_text()
    pytest.importorskip("openevolve")
    policy = climber.build_loop().policy
    assert policy.selector.resolved_params()["ensemble"] is False and policy.resolved_params()["tune_budget"] == 0


def test_get_refuses_what_is_not_in_the_catalog_and_copies_a_loop_folder_whole(folder):
    runner = CliRunner()
    result = runner.invoke(cli.app, ["climber", "get", "nope"])
    assert result.exit_code == 1 and "no catalog climber 'nope'" in result.output
    # a loop climber is a folder of several files with its own prompts: copied as it is
    result = runner.invoke(cli.app, ["climber", "get", "gepa", "--no-default"])
    assert result.exit_code == 0, result.output
    gepa = folder / "climbers" / "gepa"
    assert {p.name for p in gepa.iterdir() if p.suffix == ".py"} >= {"policy.py", "loop.py", "driver.py", "operator.py"}
    assert (gepa / "prompts" / "gepa_reflect.md").is_file() and not (gepa / "prompts" / "README.md").exists()
    # what it imports beyond hillclimb rides along, and the fetch says how to install it
    assert (gepa / "requirements.txt").is_file() and "pip install -r climbers/gepa/requirements.txt" in result.output
    copy = load_climber("climbers/gepa", folder)
    assert copy.is_loop and copy.operator_set().names() == ("gepa-reflect",) and copy.prompts_dir == (gepa / "prompts").resolve()


def test_list_marks_the_folder_as_the_default_and_not_the_preset(folder):
    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    result = CliRunner().invoke(cli.app, ["climber", "list", "--json"])
    assert result.exit_code == 0, result.output
    import json

    rows = {row["ref"]: row for row in json.loads(result.output)}
    assert rows["climbers/greedy"]["origin"] == "folder" and rows["climbers/greedy"]["default"]
    assert not rows["greedy"]["default"]


def test_run_climbs_with_the_folder(folder, monkeypatch):
    """`--climber climbers/greedy` and the folder's default both name the copy."""
    from hillclimb.cli.run import _spec_climber

    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    config = Config.load(start=folder)
    policy_file = (folder / "climbers" / "greedy" / "policy.py").resolve()
    for ref in ("climbers/greedy", "climbers/greedy/policy.py"):
        named = _spec_climber(config, ref)
        assert named["operator_policy"] == f"{policy_file}:Greedy"
        assert named["selector_policy"] == f"{policy_file}:Best"
        assert Path(named["prompts"]).resolve() == policy_file.parent / "prompts"
        assert config.climber_block() == named


@pytest.mark.parametrize(
    "text, expected",
    [
        ("model: sonnet\n# climber: climbers/greedy/policy.py   # HOW to climb\n# climber:\n#   tuner: random\n",
         "model: sonnet\nclimber: climbers/greedy\n# climber:\n#   tuner: random\n"),
        ("model: sonnet\nclimber: openevolve\n", "model: sonnet\nclimber: climbers/greedy\n"),
        ("model: sonnet\nclimber: greedy  # a note\n", "model: sonnet\nclimber: climbers/greedy\n"),
        ("# only comments\n", "# only comments\nclimber: climbers/greedy\n"),
        ("model: sonnet\nclimber:\n  operator_policy: greedy\n", None),  # a block: the person's edit
        ("model: sonnet\nclimber: {operator_policy: greedy}\n", None),
    ],
)
def test_pin_climber(text, expected):
    assert pin_climber(text, "climbers/greedy") == expected


def test_the_token_guide_and_the_templates_agree():
    """Every token the built-in templates carry has a row, every row is a
    token some template carries, and each operator declares exactly the
    templates its source renders."""
    used: set[str] = set()
    for operator in BUILTIN_OPERATORS:
        assert operator.templates, operator.name
        for template in operator.templates:
            used |= tokens_in((TEMPLATE_DIR / f"{template}.md").read_text())
    assert used == set(TOKEN_GUIDE)
    import inspect

    text = Path(inspect.getsourcefile(BUILTIN_OPERATORS[0])).read_text()  # builtin.py
    rendered = set(re.findall(r'ctx\.render\(\s*"([a-z_]+)"', text))
    assert rendered == {t for op in BUILTIN_OPERATORS for t in op.templates}


def test_the_readme_covers_every_token_of_its_templates(folder):
    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    prompts = folder / "climbers" / "greedy" / "prompts"
    readme = (prompts / "README.md").read_text()
    for template in TEMPLATES:
        assert f"`{template}.md`" in readme
        for token in tokens_in((prompts / f"{template}.md").read_text()):
            assert f"| `{{{{{token}}}}}` |" in readme, token
    assert "until 3 roots are scored" in readme and "8 extra trials" in readme
    assert "`debug: true`" in readme
    assert readme.rstrip().endswith("as it was sent.")


def test_a_folder_climber_snapshots_and_checks(folder):
    """What a search records of a folder climber is the folder's content;
    `climber check` lints its prompts clean."""
    from hillclimb.climber import load_snapshot, snapshot_climber

    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    climber = load_climber("climbers/greedy", folder)
    snapshot = snapshot_climber(climber, folder / "search")
    recorded = yaml.safe_load((snapshot / "climber.yaml").read_text())
    assert (recorded["operator_policy"], recorded["selector_policy"]) == ("files/policy.py:Greedy", "files/policy.py:Best")
    assert sorted(p.name for p in (snapshot / "prompts").iterdir()) == sorted([*(f"{t}.md" for t in TEMPLATES), "README.md"])
    assert load_snapshot(folder / "search").sha256 == climber.sha256
    check = CliRunner().invoke(cli.app, ["climber", "check", "--climber", "climbers/greedy/policy.py"])
    assert check.exit_code == 0, check.output


def test_a_climber_file_names_its_prompts_dir_or_takes_the_one_beside_it(folder):
    """`prompts/` beside a climber file is its prompts dir; `Climber(prompts_dir=…)`
    points elsewhere; `prompts=` was the keyword's spelling before 0.9."""
    from hillclimb import Climber

    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    target = folder / "climbers" / "greedy"
    (target / "words").mkdir()
    (target / "words" / "draft.md").write_text("OTHER WORDS for {{metric_name}}.\n\n{{contract}}\n")
    text = (target / "policy.py").read_text()
    (target / "policy.py").write_text(text.replace("    name='greedy',", "    prompts_dir=Path(__file__).parent / 'words',\n    name='greedy',")
                                      .replace("from hillclimb import Climber", "from pathlib import Path\n\nfrom hillclimb import Climber"))
    assert load_climber("climbers/greedy", folder).prompts_dir == (target / "words").resolve()
    with pytest.raises(TypeError, match="`prompts=` is the old spelling of `prompts_dir=`"):
        Climber(operator_policy=class_ref("greedy", "Greedy"), prompts=target / "words")


def test_new_from_a_folder_copies_the_whole_climber(folder):
    """`climber new mine --from climbers/greedy`: the file builds the whole
    climber, so the copy is a folder too — policy.py renamed inside, the
    prompts beside it — and it is its own climber from the first edit."""
    runner = CliRunner()
    assert runner.invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    result = runner.invoke(cli.app, ["climber", "new", "mine", "--from", "climbers/greedy"])
    assert result.exit_code == 0, result.output
    mine = folder / "climbers" / "mine"
    assert (mine / "policy.py").is_file() and sorted(p.name for p in (mine / "prompts").iterdir()) == sorted([*(f"{t}.md" for t in TEMPLATES), "README.md"])
    assert "name='mine'," in (mine / "policy.py").read_text() and "name='greedy'," not in (mine / "policy.py").read_text()
    assert "climbers/mine/policy.py" in result.output
    copy, original = load_climber("climbers/mine", folder), load_climber("climbers/greedy", folder)
    assert copy.name == "mine" and copy.sha256 != original.sha256
    assert copy.prompts_dir == (mine / "prompts").resolve()  # the prompts beside the copy, not the original's
    assert copy.build_loop().policy.resolved_params() == original.build_loop().policy.resolved_params()
