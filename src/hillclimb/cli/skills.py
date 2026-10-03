"""`hillclimb skills …`: the skills operators are given, global and local."""

from __future__ import annotations

import shutil

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, say

skills_app = typer.Typer(
    cls=HillclimbGroup,
    help="The skills operators are given: the global layer and this folder's own",
    invoke_without_command=True,
)

app.add_typer(skills_app, name="skills")


@skills_app.callback()
def skills_list(
    ctx: typer.Context,
    generated: bool = typer.Option(False, "--generated", help="Only skills an operator or a plugin wrote"),
    manual: bool = typer.Option(False, "--manual", help="Only skills a person wrote or kept"),
):
    """List the skills operators get: layer, origin and what each is for.

    Global skills (`~/.config/hillclimb/agent/skills/`) reach every hillclimb
    dir on this machine; local ones (`agent/skills/` here) reach this folder's
    searches and override a global skill of the same name. Each skill is a
    folder with a SKILL.md, rendered for whichever coding agent runs (Claude
    Code, codex, pi). A skill you put there yourself is `manual`; one an
    operator created is added to the local layer as `generated`, with a
    skill.yaml saying which run, candidate and agent wrote it (`generated,
    edited` once changed since). `hillclimb skills keep` makes one yours.
    """
    if ctx.invoked_subcommand is not None:
        return
    from hillclimb.harness.agent_context import global_dir, local_dir, skills

    config = common.load_config(require_dir=False)
    found = list(skills(config).values())
    if generated:
        found = [s for s in found if s.origin != "manual"]
    if manual:
        found = [s for s in found if s.origin == "manual"]
    local = local_dir(config)
    say()
    if found:
        common.table(
            [("skill", "cmd"), ("layer", None), ("origin", None), ("from", None), ("description", None)],
            [
                (_m(s.name), _m(s.layer), _m(s.origin), _m(_written_by(s)), _m(s.description))
                for s in found
            ],
        )
        say()
    else:
        say("  [note]no skills yet[/]")
    say(f"  [head]global[/]: [path]{_m(global_dir() / 'skills')}[/]")
    if local is not None:
        say(f"  [head]local[/]: [path]{_m(local / 'skills')}[/]")
    say()


def _written_by(skill) -> str:
    """Which call wrote a generated skill: `<run>/<search> <candidate>`."""
    made = skill.record.get("created_by") or {}
    if not made.get("candidate"):
        return ""
    return f"{made.get('run', '?')}/{made.get('search', '?')} {made['candidate']}"


@skills_app.command("keep")
def skills_keep(
    name: str = typer.Argument(..., help="The skill's name (its folder)"),
):
    """Mark a generated skill as reviewed and kept: it is `manual` from now on.

    Its skill.yaml keeps the record of what generated it, and when it was kept.
    """
    from hillclimb.harness.agent_context import keep, skills

    config = common.load_config(require_dir=False)
    skill = skills(config).get(name)
    if skill is None:
        fail(f"no skill {name!r}")
        raise typer.Exit(1)
    if skill.origin == "manual":
        say(f"[head]{_m(name)}[/] is already manual")
        return
    keep(skill)
    say(f"[head]kept[/] {_m(name)} [note]({_m(skill.layer)}; manual from now on)[/]")


@skills_app.command("delete")
def skills_delete(
    name: str = typer.Argument(..., help="The skill's name (its folder)"),
    global_: bool = typer.Option(False, "--global", help="Delete it from the global layer instead of this folder's"),
):
    """Delete a skill from this folder's layer (or, with --global, from the global one)."""
    from hillclimb.harness.agent_context import global_dir, local_dir

    config = common.load_config(require_dir=not global_)
    root = global_dir() if global_ else local_dir(config)
    path = root / "skills" / name if root is not None else None
    if path is None or "/" in name or name.startswith(".") or not path.is_dir():
        fail(f"no skill {name!r} in the {'global' if global_ else 'local'} layer")
        raise typer.Exit(1)
    shutil.rmtree(path)
    say(f"[head]deleted[/] [path]{_m(path)}[/]")
