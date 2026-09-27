"""`hillclimb meta evaluate|check` — the meta-problem kit's commands. Hidden
until the meta-problem launch: nothing in `--help` or the docs names them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, say

meta_app = typer.Typer(
    cls=HillclimbGroup,
    hidden=True,
    help=(
        "Meta-problems — score a one-file climber by the inner searches it runs "
        "(the improver role). `evaluate` is what a meta-problem's verifier.sh calls."
    ),
)


app.add_typer(meta_app, name="meta", hidden=True)


@meta_app.command("evaluate")
def meta_evaluate(
    climber: Path = typer.Option(
        None, "--climber", help="The one-file climber to score (default: $HILLCLIMB_SOLUTION)"
    ),
    spec: Path = typer.Option(
        Path("problem") / "meta.yaml", "--spec", help="meta.yaml: inner problems, budget, repeats"
    ),
    result: Path = typer.Option(
        None, "--result", help="Where to write {score, instances} (default: $HILLCLIMB_RESULT, else stdout)"
    ),
    workdir: Path = typer.Option(None, "--workdir", help="Where the inner searches run (default: cwd)"),
):
    """Run the inner searches of a meta-problem with CLIMBER and report the
    gap they closed — the verifier of a problem whose solution is a climber.

    Reads the hillclimb dir it runs under (`$HILLCLIMB_DIR`, set by the
    engine for every verifier) for the user's agent, model and problems,
    runs one `hillclimb run` per inner problem and repeat in a nested
    hillclimb dir under the working directory, and writes the score.
    """
    from hillclimb.meta import MetaError, evaluate, load_meta_spec

    climber = climber or (Path(os.environ["HILLCLIMB_SOLUTION"]) if os.environ.get("HILLCLIMB_SOLUTION") else None)
    if climber is None:
        raise typer.BadParameter("--climber is required outside a verifier ($HILLCLIMB_SOLUTION unset)")
    result = result or (Path(os.environ["HILLCLIMB_RESULT"]) if os.environ.get("HILLCLIMB_RESULT") else None)
    config = common.load_config()
    params = None
    if os.environ.get("HILLCLIMB_PARAMS"):
        # the trial's values of an improver's params.json: the outer harness
        # tunes a climber's knobs through the same seam as any solution's
        from hillclimb import spaces

        try:
            params = spaces.params_values(spaces.load_params_file(os.environ["HILLCLIMB_PARAMS"]))
        except spaces.ParamsError as exc:
            fail(f"params.json: {_m(exc)}")
            raise typer.Exit(1) from exc
    try:
        meta_spec = load_meta_spec(spec)
        outcome = evaluate(
            meta_spec, climber, config, workdir or Path.cwd(),
            log=lambda line: say(f"[note]{_m(line)}[/]", err=True), params=params,
        )
    except MetaError as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc
    payload = json.dumps(outcome.to_result())
    if result is not None:
        result.write_text(payload)
    else:
        typer.echo(payload)
    say(f"[head]gap closed:[/] {outcome.score:.4f}  " + "  ".join(
        f"{_m(key)}={value:.4f}" for key, value in outcome.instances.items()
    ), err=True)


@meta_app.command("check")
def meta_check(
    ctx: typer.Context,
    climber: Path = typer.Option(Path("solution.py"), "--climber", help="The one-file climber to check"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """The cheap pre-verifier for an improver's candidate: the file may
    import only hillclimb.sdk, hillclimb.spaces and the standard library,
    and then everything `hillclimb climber check` verifies — it loads, has
    exactly one policy, replays recorded searches deterministically and
    never writes. Exit 1 on any breach; no agent, no verifier, no inner search.
    """
    from hillclimb.cli.climber import climber_check
    from hillclimb.meta import check_climber_source

    problems = check_climber_source(climber)
    if problems:
        if as_json:
            typer.echo(json.dumps({"ok": False, "imports": problems}, indent=2))
        else:
            for line in problems:
                fail(_m(line))
        raise typer.Exit(1)
    if not as_json:
        say("[ok]imports: OK[/] (hillclimb.sdk and the standard library only)")
    climber_check(
        ctx, climber=str(climber), policy=None, problem=None, set_=[], limit=20,
        smoke=False, smoke_budget="2m", as_json=as_json,
    )
