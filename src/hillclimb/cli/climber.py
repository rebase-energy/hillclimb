"""`hillclimb climber list|new|check`."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, say, warn
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.store import key_for, open_store
from hillclimb.problem import load_problem

climber_app = typer.Typer(
    cls=HillclimbGroup,
    help=(
        "Climbers — the shareable bundle that decides HOW to hillclimb (policy, operators, "
        "prompts, tuner): list them, start your own, check one before spending budget on it"
    ),
    short_help="Climbers, the swappable search methods: list, copy, check.",
)


app.add_typer(climber_app, name="climber")




LOCAL_CLIMBERS_DIRNAME = "climbers"  # <hillclimb dir>/climbers/<name>/ — where `climber new` writes


def _local_climbers_dir(config: Config) -> Path | None:
    return config.hillclimb_dir / LOCAL_CLIMBERS_DIRNAME if config.hillclimb_dir is not None else None


def _climber_ref(path: Path, base_dir: Path | None) -> str:
    """How to name a local climber on the command line: relative to the
    folder holding the hillclimb dir, the anchor every relative ref resolves from."""
    if base_dir is not None:
        try:
            return str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            pass
    return str(path)


def _is_legacy_dir(path: Path) -> bool:
    return (path / "climber.yaml").is_file()


@climber_app.command("list")
def climber_list(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """The climbers a bare name stands for: the presets, and every one-file
    climber under climbers/."""
    from hillclimb.climber import ClimberLoadError, climber_base_dir, climber_label, load_climber, presets

    config = common.load_config()
    base_dir = climber_base_dir(config)
    default = config.climber.label  # the folder's climber, by what views call it
    refs = [(name, "preset") for name in presets()]
    local = _local_climbers_dir(config)
    if local is not None and local.is_dir():
        for path in sorted(local.iterdir()):
            if _is_legacy_dir(path) or (path.is_file() and path.suffix == ".py"):
                refs.append((_climber_ref(path, base_dir), "local"))
    rows = []
    for ref, origin in refs:
        try:
            climber = load_climber(ref, base_dir)
            kind, description = ("loop" if climber.is_loop else "policy"), climber.description
        except (ClimberLoadError, ValueError) as exc:
            rows.append({"ref": ref, "origin": origin, "kind": "?", "description": f"BROKEN: {exc}",
                         "sha256": None, "default": climber_label(ref) == default})
            continue
        if base_dir is not None and _is_legacy_dir(base_dir / ref):
            description = f"pre-0.6 directory: `hillclimb climber show {ref}` prints it as a block"
        rows.append({
            "ref": ref, "origin": origin, "kind": kind,
            "description": description, "sha256": climber.sha256[:12],
            "default": climber_label(ref) == default,
        })
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    width = max(len(row["ref"]) for row in rows)
    for row in rows:
        mark = "*" if row["default"] else " "
        say(
            f"{mark} [path]{_m(row['ref']):<{width}}[/]  {_m(row['origin']):<7} {_m(row['kind']):<6} "
            f"[note]{_m(row['description'])}[/]"
        )
    say("\n[note]* = this folder's default.[/]  Run one:        [cmd]hillclimb run <problem> --climber <name>[/]")
    say("                              See its block:  [cmd]hillclimb climber show <name>[/]")
    say("                              Start your own: [cmd]hillclimb climber new <name> --from greedy[/]")


def _portable_block(climber, base_dir: Path | None) -> dict:
    """The climber's block with file refs written the way a config beside
    `base_dir` would write them: relative when the file is below it."""
    from hillclimb.modules import refs as module_refs

    def relative(text: str) -> str:
        if base_dir is not None:
            try:
                return Path(text).resolve().relative_to(base_dir.resolve()).as_posix()
            except ValueError:
                pass
        return text

    def portable(ref: str) -> str:
        path = module_refs.ref_path(ref)
        if path is None:
            return ref
        attr = module_refs.split_file_ref(ref)[1]
        return relative(str(path)) + (f":{attr}" if attr else "")

    spec = climber.spec.map_refs(portable)
    if spec.prompts:
        spec = spec.model_copy(update={"prompts": relative(spec.prompts)})
    return spec.block()


@climber_app.command("show")
def climber_show(
    ref: str = typer.Argument(None, help="A preset, a .py file, or a pre-0.6 climber directory (default: this folder's climber)"),
):
    """Print a climber as the block a run config takes.

    Paste it under `climber:` in a run spec or in hillclimb.yaml and edit
    it there: the block IS the climber. A pre-0.6 directory holding
    climber.yaml comes out as its block too — this is how one is migrated.
    """
    import yaml

    from hillclimb.climber import ClimberLoadError, climber_base_dir, load_climber, resolve_climber

    config = common.load_config()
    base_dir = climber_base_dir(config)
    try:
        climber = load_climber(ref, base_dir) if ref else resolve_climber(config.climber, base_dir)
        climber.brain  # noqa: B018 — resolve it, so a broken ref is reported here
    except (ClimberLoadError, ValueError) as exc:
        fail(_m(exc))
        raise typer.Exit(2) from exc
    typer.echo(yaml.safe_dump({"climber": _portable_block(climber, base_dir)}, sort_keys=False), nl=False)


@climber_app.command("new")
def climber_new(
    name: str = typer.Argument(..., help="Name of the new climber (becomes climbers/<name>.py)"),
    from_: str = typer.Option(
        "greedy", "--from", help="What to copy: a preset's name or a .py file"
    ),
):
    """Start your own climber from a copy of an existing one.

    Copies the source of the policy (or loop) into climbers/<name>.py — a
    one-file climber you can edit — and prints the block that runs it.
    """
    import inspect
    import re as _re
    import shutil

    import yaml

    from hillclimb.climber import ClimberLoadError, climber_base_dir, load_climber

    config = common.load_config()
    local = _local_climbers_dir(config)
    if local is None:
        raise typer.BadParameter("no hillclimb dir here — run `hillclimb init` (or `hillclimb problem get`) first")
    if not _re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise typer.BadParameter(f"{name!r}: a climber name is letters, digits, - and _")
    base_dir = climber_base_dir(config)
    try:
        source = load_climber(from_, base_dir)
        source_file = Path(inspect.getsourcefile(source.brain.target))
    except (ClimberLoadError, ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    target = (local / name).with_suffix(".py")
    if target.exists() or (local / name).exists():
        raise typer.BadParameter(f"{_climber_ref(target, base_dir)} already exists")
    local.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target)
    ref = _climber_ref(target, base_dir)
    brain = "loop" if source.is_loop else "policy"
    class_name = getattr(source.brain.target, "__name__", "")
    block = {**_portable_block(source, base_dir), "name": name, brain: f"{ref}:{class_name}" if class_name else ref}
    try:
        load_climber(block[brain], base_dir).brain  # noqa: B018
    except (ClimberLoadError, ValueError) as exc:  # never leave a broken copy behind
        target.unlink()
        raise typer.BadParameter(f"the copy does not load: {exc}") from exc
    say(f"[head]Created[/] [path]{_m(ref)}[/] from [path]{_m(from_)}[/]")
    say("[head]Its block[/] [note](paste under `climber:` in a run spec or hillclimb.yaml):[/]")
    typer.echo(yaml.safe_dump({"climber": block}, sort_keys=False), nl=False)
    say(f"[head]Next:[/] edit it, then   [cmd]hillclimb climber check --climber {_m(block[brain])}[/]")
    say(f"                       [cmd]hillclimb run <problem> --climber {_m(block[brain])}[/]")


@climber_app.command("check")
def climber_check(
    climber: str = typer.Option(None, "--climber", help="A preset or one .py file (default: this folder's `climber:` block)"),
    problem: str = typer.Option(
        None, "--problem", help="Replay only this problem's recorded searches (default: every search)"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Config override, dotted: --set climber.params.num_drafts=1",
    ),
    limit: int = typer.Option(20, "--limit", help="Newest recorded searches to replay"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Then run a dummy-agent search on --problem (no LLM, real verifier)"
    ),
    smoke_budget: str = typer.Option("2m", "--smoke-budget", help="Wall clock for the smoke search"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Conformance check for a climber — the cheap pre-verifier.

    Replays every recorded journal (plus an empty one) through the climber's
    policy with no agent or verifier: two fresh instances must propose the
    same action at every budget point (the resume contract), every referenced
    candidate must exist, the policy must never write, and the climber's
    prompts must lint clean. Exit 1 on any breach. `--smoke` follows up with
    a short `--agent dummy` search so the whole loop — prompts included —
    runs once before an agent hour is spent on it.
    """
    from hillclimb.api import run_search
    from hillclimb.climber import ClimberLoadError, climber_base_dir, load_climber
    from hillclimb.modules.policies.check import JournalCase, check_policy

    from hillclimb.climber import as_spec, resolve_climber

    config = common.load_config()
    base_dir = climber_base_dir(config)
    name = climber or config.climber.label
    source = None
    if climber and climber.endswith(".py"):  # a one-file climber: say so before anything is imported
        source = Path(climber).expanduser()
        if not source.is_absolute() and base_dir is not None:
            source = base_dir / source
    if source is not None and not source.is_file():
        raise typer.BadParameter(f"climber file not found: {source}")
    try:
        if climber:
            try:
                config.climber = as_spec(climber)  # a preset, one file
            except ClimberLoadError:
                # a pre-0.6 directory: read it as the block it is
                config.climber = load_climber(climber, base_dir).spec
        config.apply_overrides(common._parse_set(set_ or []))  # edits the block, like `hillclimb run --set`
        loaded = resolve_climber(config.climber, base_dir)
        loaded.brain  # noqa: B018 — every module must resolve, before an agent hour is spent
        loaded.operator_set()
        loaded.tuner()
        loaded.graph_module()
    except (ClimberLoadError, ValueError, KeyError) as exc:
        fail(_m(exc))
        raise typer.Exit(2) from exc
    params = dict(loaded.spec.params)
    if loaded.is_loop:
        fail(
            f"{_m(name)} brings its own Loop; "
            "the conformance check covers climbers built on a Policy"
        )
        raise typer.Exit(2)

    def make_policy():
        # a fresh policy per call, exactly as a search builds it
        return loaded.build_loop(log=lambda *_: None).policy

    problem_key = None
    if problem:
        problem_key = load_problem(problem, config).problem_key
    cases: list[JournalCase] = []
    with closing(open_store(config)) as store:
        records = store.searches(problem_key=problem_key)
        for record in sorted(records, key=lambda r: r.meta.started_at, reverse=True)[:limit]:
            cases.append(
                JournalCase(
                    label=record.ref,
                    journal=Journal(store.journal(record.key)),
                    higher_is_better=record.meta.higher_is_better,
                    total_s=record.meta.budget_s or 3600,
                    search_dir=record.search_dir,
                )
            )
    report = check_policy(
        make_policy, cases, config, prompts_dir=loaded.prompts_dir, operators=loaded.operator_set()
    )
    if report.ok:
        resolved = getattr(make_policy(), "resolved_params", None)
        resolved_params = resolved() if callable(resolved) else params
    else:
        resolved_params = params  # the policy may not even construct
    if source is not None:
        report.policy = f"{report.policy} ({source})"
    smoke_result: dict | None = None
    if smoke and report.ok:
        if not problem:
            raise typer.BadParameter("--smoke needs --problem")
        smoke_config = common.load_config(agent="dummy")
        smoke_config.climber = config.climber  # the block just checked, --set edits included
        smoke_config.learning.enabled = False
        outcome = run_search(
            problem,
            budget_s=common.parse_budget(smoke_budget),
            name="climber-check",
            config=smoke_config,
            log=(lambda *_: None) if as_json else common.engine_log,
        )
        with closing(open_store(smoke_config)) as store:
            journal = Journal(store.journal(key_for(outcome.search_dir)))
        smoke_result = {
            "search": outcome.ref,
            "state": outcome.state,
            "candidates": len(journal.candidates),
            "scored": len(journal.scored_candidates()),
            "best": outcome.selected.val_score if outcome.selected is not None else None,
        }
    if as_json:
        payload = report.to_dict()
        payload["resolved_params"] = resolved_params
        payload["journals"] = [c.label for c in cases]
        if smoke_result is not None:
            payload["smoke"] = smoke_result
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo(report.render())
        say(f"[head]resolved params:[/] {_m(json.dumps(resolved_params, default=str))}")
        say(f"[head]replayed[/] {len(cases)} recorded journal(s)")
        if smoke_result is not None:
            say(
                f"[head]smoke [path]{_m(smoke_result['search'])}[/]:[/] {_m(smoke_result['state'])}, "
                f"{smoke_result['candidates']} candidate(s), {smoke_result['scored']} scored, "
                f"best={_m(smoke_result['best'])}"
            )
        elif smoke:
            warn("smoke skipped: fix the breaches above first")
    if not report.ok or (smoke_result is not None and smoke_result["state"] != "done"):
        raise typer.Exit(1)
