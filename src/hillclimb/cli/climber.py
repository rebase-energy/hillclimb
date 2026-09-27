"""`hillclimb climber list|new|check` (and the hidden old `policy check`)."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
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
)


app.add_typer(climber_app, name="climber")


policy_app = typer.Typer(cls=HillclimbGroup, hidden=True, help="Old spelling of `hillclimb climber`")


app.add_typer(policy_app, name="policy", hidden=True)


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


@climber_app.command("list")
def climber_list(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """The climbers `hillclimb run --climber` accepts: the bundled ones and
    every directory or one-file climber under hillclimb/climbers/."""
    from hillclimb.climber import ClimberLoadError, bundled_climbers, load_climber
    from hillclimb.modules.policies import policy_base_dir

    config = common.load_config()
    base_dir = policy_base_dir(config)
    refs = [(name, "bundled") for name in bundled_climbers()]
    local = _local_climbers_dir(config)
    if local is not None and local.is_dir():
        for path in sorted(local.iterdir()):
            if (path / "climber.yaml").is_file() or (path.is_file() and path.suffix == ".py"):
                refs.append((_climber_ref(path, base_dir), "local"))
    rows = []
    for ref, origin in refs:
        try:
            climber = load_climber(ref, base_dir)
        except ClimberLoadError as exc:
            rows.append({"ref": ref, "origin": origin, "kind": "?", "description": f"BROKEN: {exc}",
                         "sha256": None, "default": ref == config.climber.ref})
            continue
        rows.append({
            "ref": ref, "origin": origin, "kind": "loop" if climber.is_loop else "policy",
            "description": climber.manifest.description, "sha256": climber.sha256[:12],
            "default": ref == config.climber.ref,
        })
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    width = max(len(row["ref"]) for row in rows)
    for row in rows:
        mark = "*" if row["default"] else " "
        typer.echo(f"{mark} {row['ref']:<{width}}  {row['origin']:<7} {row['kind']:<6} {row['description']}")
    typer.echo("\n* = config climber.ref.  Run one:        hillclimb run <problem> --climber <ref>")
    typer.echo("                          Start your own: hillclimb climber new <name> --from greedy")


_COPIED_MODULE_KEYS = ("policy", "loop")


@climber_app.command("new")
def climber_new(
    name: str = typer.Argument(..., help="Name of the new climber (becomes hillclimb/climbers/<name>/)"),
    from_: str = typer.Option(
        "greedy", "--from", help="Climber to copy: a bundled name, a directory holding climber.yaml, or a .py file"
    ),
):
    """Start your own climber from a copy of an existing one.

    Copies the manifest, the policy (or loop) source and the prompts into
    hillclimb/climbers/<name>/ so every part is a file you can edit, then
    prints how to check and run it. A bundled climber's `module:Class`
    policy is copied in as `<module>.py:Class` — edit that file.
    """
    import re as _re
    import shutil

    import yaml

    from hillclimb.climber import ClimberLoadError, load_climber
    from hillclimb.modules.policies import policy_base_dir

    config = common.load_config()
    local = _local_climbers_dir(config)
    if local is None:
        raise typer.BadParameter("no hillclimb dir here — run `hillclimb init` (or `hillclimb problem get`) first")
    if not _re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise typer.BadParameter(f"{name!r}: a climber name is letters, digits, - and _")
    base_dir = policy_base_dir(config)
    try:
        source = load_climber(from_, base_dir)
    except ClimberLoadError as exc:
        raise typer.BadParameter(str(exc)) from exc
    target = local / name
    if target.exists() or target.with_suffix(".py").exists():
        raise typer.BadParameter(f"{_climber_ref(target, base_dir)} already exists")
    local.mkdir(parents=True, exist_ok=True)
    if source.root is None:  # a one-file climber stays one file
        target = target.with_suffix(".py")
        shutil.copy2(source.source, target)
    else:
        shutil.copytree(
            source.root, target,
            ignore=lambda _dir, names: [n for n in names if n == "__pycache__" or n.startswith(".")],
        )
        manifest_path = target / "climber.yaml"
        data = yaml.safe_load(manifest_path.read_text()) or {}
        data["name"] = name
        # a bundled manifest names its brain by package module; bring the
        # source in so the copy is editable without touching the package
        for key in _COPIED_MODULE_KEYS:
            ref = data.get(key)
            if not isinstance(ref, str) or ":" not in ref or not ref.startswith("hillclimb."):
                continue
            from hillclimb._moved import modernize

            module_name, cls = modernize(ref).split(":", 1)
            import importlib

            module_file = Path(importlib.import_module(module_name).__file__)
            shutil.copy2(module_file, target / module_file.name)
            data[key] = f"{module_file.name}:{cls}"
        manifest_path.write_text(yaml.safe_dump(data, sort_keys=False))
    ref = _climber_ref(target, base_dir)
    try:
        load_climber(ref, base_dir)
    except ClimberLoadError as exc:  # never leave a broken copy behind
        shutil.rmtree(target) if target.is_dir() else target.unlink()
        raise typer.BadParameter(f"the copy does not load: {exc}") from exc
    typer.echo(f"Created {ref} from {from_}")
    files = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]
    for path in files:
        typer.echo(f"  {path.relative_to(target if target.is_dir() else target.parent)}")
    typer.echo(f"Next: edit it, then   hillclimb climber check --climber {ref}")
    typer.echo(f"      and climb with   hillclimb run <problem> --climber {ref}")


@climber_app.command("check")
@policy_app.command("check", hidden=True)
def climber_check(
    ctx: typer.Context,
    climber: str = typer.Option(None, "--climber", help="Climber ref (default: config climber.ref)"),
    policy: str = typer.Option(None, "--policy", hidden=True),
    problem: str = typer.Option(
        None, "--problem", help="Replay only this problem's recorded searches (default: every search)"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Config override, dotted: --set climber.params.num_drafts=1",
    ),
    limit: int = typer.Option(20, "--limit", help="Newest recorded searches to replay"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Then run a dummy-backend search on --problem (no LLM, real verifier)"
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
    a short `--backend dummy` search so the whole loop — prompts included —
    runs once before an agent hour is spent on it.
    """
    if ctx.parent is not None and ctx.parent.info_name == "policy":
        typer.echo("note: `hillclimb policy check` is now `hillclimb climber check`", err=True)
    if policy:
        typer.echo("note: `--policy` is now `--climber` (same values)", err=True)
    from hillclimb.api import run_search
    from hillclimb.climber import ClimberLoadError, load_climber
    from hillclimb.modules.policies import policy_base_dir, policy_path
    from hillclimb.modules.policies.check import JournalCase, check_policy

    config = common.load_config()
    config.apply_overrides(common._parse_set(set_ or []))
    name = climber or policy or config.climber.ref
    params = dict(config.climber.params)  # the user's overlay; the manifest's params are the base
    base_dir = policy_base_dir(config)
    source = policy_path(name, base_dir)
    if source is not None and not source.is_file():
        raise typer.BadParameter(f"climber file not found: {source}")
    try:
        loaded = load_climber(name, base_dir)
        loaded.graph_module()  # `graph:` must resolve too, before an agent hour is spent
    except (ClimberLoadError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if loaded.root is not None and "memory: knowledge-graph" in (loaded.root / "climber.yaml").read_text():
        typer.echo("note: `memory: knowledge-graph` is now `memory: files` (the old spelling still loads)", err=True)
    if loaded.is_loop:
        typer.echo(
            f"{name} brings its own SearchLoop; "
            "the conformance check covers climbers built on a SearchPolicy",
            err=True,
        )
        raise typer.Exit(2)

    def make_policy():
        # a fresh policy per call, exactly as a search builds it
        return loaded.build_loop(params=params, log=lambda *_: None).policy

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
    report = check_policy(make_policy, cases, config, prompts_dir=loaded.prompts_dir)
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
        smoke_config = common.load_config(backend="dummy")
        smoke_config.apply_overrides(common._parse_set(set_ or []))
        smoke_config.climber.ref = name
        smoke_config.learning.enabled = False
        outcome = run_search(
            problem,
            budget_s=common.parse_budget(smoke_budget),
            name="climber-check",
            config=smoke_config,
            log=(lambda *_: None) if as_json else typer.echo,
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
        typer.echo(f"resolved params: {json.dumps(resolved_params, default=str)}")
        typer.echo(f"replayed {len(cases)} recorded journal(s)")
        if smoke_result is not None:
            typer.echo(
                f"smoke {smoke_result['search']}: {smoke_result['state']}, "
                f"{smoke_result['candidates']} candidate(s), {smoke_result['scored']} scored, "
                f"best={smoke_result['best']}"
            )
        elif smoke:
            typer.echo("smoke skipped: fix the breaches above first")
    if not report.ok or (smoke_result is not None and smoke_result["state"] != "done"):
        raise typer.Exit(1)
