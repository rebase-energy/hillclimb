"""`hillclimb climber get`: a preset copied out as a folder that IS a climber —
its block with the defaults spelled out, the policy's source, the templates
its operators render and a README of what fills them — and made the
folder's default. A climber folder is a first-class way of naming a climber."""

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
    assert files == sorted(["climber.yaml", "policy.py", *(f"prompts/{t}.md" for t in TEMPLATES), "prompts/README.md"])
    assert (folder / "climbers" / ".hillclimb").is_file()  # hillclimb's folder: `reset` may delete it
    for template in TEMPLATES:  # the templates are the package's, byte for byte
        assert (target / "prompts" / f"{template}.md").read_bytes() == (TEMPLATE_DIR / f"{template}.md").read_bytes()
    block = yaml.safe_load((target / "climber.yaml").read_text())
    assert block["operator_policy"] == "policy.py:Greedy" and block["prompts"] == "prompts"
    assert block["selector_params"]["num_drafts"] == 3 and block["params"]["tune_budget"] == 8  # spelled out
    # the folder is this hillclimb dir's climber now
    assert re.search(r"^climber: climbers/greedy$", (folder / "hillclimb.yaml").read_text(), re.M)
    config = Config.load(start=folder)
    assert config.climber.operator_policy == "climbers/greedy/policy.py:Greedy"
    assert config.climber.prompts == "climbers/greedy/prompts"
    climber = resolve_climber(config.climber, folder)
    assert climber.operator_set().names() == ("draft", "debug", "improve", "ensemble")
    assert climber.prompts_dir == target / "prompts"
    assert climber.build_loop().policy.resolved_params()["tune_budget"] == 8
    assert "Fetched greedy" in result.output and "climber: climbers/greedy" in result.output


def test_the_folder_is_the_same_climber_however_it_is_named(folder):
    assert CliRunner().invoke(cli.app, ["climber", "get", "greedy"]).exit_code == 0
    by_folder = load_climber("climbers/greedy", folder)
    by_string = resolve_climber(ClimberSpec.model_validate("climbers/greedy", context={"base_dir": folder}), folder)
    by_cli = resolve_climber(Config.load(start=folder).climber, folder)
    assert by_folder.sha256 == by_string.sha256 == by_cli.sha256
    assert by_folder.name == "greedy"
    assert by_folder.sha256 != load_climber("greedy").sha256  # a copy is its own climber: edits count
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
    block = yaml.safe_load((folder / "climbers" / "qd" / "climber.yaml").read_text())
    assert block["name"] == "qd" and block["selector_policy"] == "map-elites"
    assert block["selector_params"]["ensemble"] is False and block["params"]["tune_budget"] == 0
    assert "climber: climbers/qd" not in (folder / "hillclimb.yaml").read_text()
    assert load_climber("climbers/qd", folder).selector().resolved_params()["ensemble"] is False


def test_get_refuses_what_is_not_a_preset_or_is_a_loop(folder):
    runner = CliRunner()
    assert runner.invoke(cli.app, ["climber", "get", "nope"]).exit_code == 1
    result = runner.invoke(cli.app, ["climber", "get", "gepa"])
    assert result.exit_code == 1 and "is a loop" in result.output
    assert not (folder / "climbers" / "gepa").exists()


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
    named = _spec_climber(config, "climbers/greedy")
    assert named["operator_policy"] == f"{folder / 'climbers' / 'greedy' / 'policy.py'}:Greedy"
    assert named["prompts"] == str(folder / "climbers" / "greedy" / "prompts")
    assert config.climber_block() == named


@pytest.mark.parametrize(
    "text, expected",
    [
        ("model: sonnet\n# climber: greedy   # HOW to climb\n# climber:\n#   tuner: random\n",
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
    assert yaml.safe_load((snapshot / "climber.yaml").read_text())["operator_policy"] == "files/policy.py:Greedy"
    assert sorted(p.name for p in (snapshot / "prompts").iterdir()) == sorted([*(f"{t}.md" for t in TEMPLATES), "README.md"])
    assert load_snapshot(folder / "search").sha256 == climber.sha256
    check = CliRunner().invoke(cli.app, ["climber", "check", "--climber", "climbers/greedy"])
    assert check.exit_code == 0, check.output
