"""`hillclimb climber list|get|show|new|check`."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, legend, next_steps, say, warn
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.store import key_for, open_store
from hillclimb.problem import load_problem

climber_app = typer.Typer(
    cls=HillclimbGroup,
    help=(
        "Climbers — the shareable bundle that decides HOW to hillclimb (policies, operators, "
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


def _is_climber_dir(path: Path) -> bool:
    """A climber folder: `hillclimb climber get` wrote it (its `policy.py` is
    the climber), or a pre-0.9 folder / pre-0.6 manifest (`climber.yaml`)."""
    return (path / "policy.py").is_file() or (path / "climber.yaml").is_file()


def _is_pre_06_manifest(path: Path) -> bool:
    """Does the folder's manifest use the 0.4/0.5 keys (`policy`, `holdout_timing`, …)?"""
    import yaml

    try:
        data = yaml.safe_load((path / "climber.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and any(key in data for key in ("policy", "holdout_timing", "description", "similarity"))


def _say_check_report(report) -> None:
    """`climber check`'s verdict and findings in the CLI's voice: a green
    `ok` or a red `FAIL` per check, the journal it was replayed on dimmed."""
    verdict = (
        "[ok]conforms[/]" if report.ok
        else f"[bad]{len(report.failures)} contract breach{'' if len(report.failures) == 1 else 'es'}[/]"
    )
    say(f"[head]policy {_m(report.policy)}[/] [note]{_m(report.params or '{}')}[/]: {verdict}")
    for finding in report.findings:
        mark = "[ok]ok  [/]" if finding.ok else "[bad]FAIL[/]"
        where = f" [note]{_m('[' + finding.journal + ']')}[/]" if finding.journal else ""
        say(f"  {mark} [head]{_m(finding.check)}[/]{where}: {_m(finding.detail)}")


@climber_app.command("list")
def climber_list(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")):
    """The catalog's climbers (`climber get` copies one out) and every
    climber under climbers/."""
    from hillclimb.climber import ClimberLoadError, climber_base_dir, load_climber

    from hillclimb.climber import resolve_climber

    config = common.load_config()
    base_dir = climber_base_dir(config)
    try:  # the folder's climber, by identity: a catalog climber and a copy of it are two climbers
        default = resolve_climber(config.climber, base_dir).sha256
    except (ClimberLoadError, ValueError):
        default = None
    from hillclimb import catalog

    refs = [(name, "catalog") for name in catalog.climber_names()]
    local = _local_climbers_dir(config)
    if local is not None and local.is_dir() and local.resolve() != catalog.climbers_dir().resolve():
        for path in sorted(local.iterdir()):
            if _is_climber_dir(path):
                refs.append((_climber_ref(path, base_dir), "folder"))
            elif path.is_file() and path.suffix == ".py":
                refs.append((_climber_ref(path, base_dir), "local"))
    rows = []
    for ref, origin in refs:
        try:
            climber = catalog.climber(ref) if origin == "catalog" else load_climber(ref, base_dir)
            kind, description = ("loop" if climber.is_loop else "operator policy"), climber.description
        except (ClimberLoadError, ValueError) as exc:
            rows.append({"ref": ref, "origin": origin, "kind": "?", "description": f"BROKEN: {exc}",
                         "sha256": None, "default": False})
            continue
        if base_dir is not None and _is_climber_dir(base_dir / ref) and _is_pre_06_manifest(base_dir / ref):
            description = f"pre-0.6 directory: `hillclimb climber show {ref}` prints it as a block"
        rows.append({
            "ref": ref, "origin": origin, "kind": kind,
            "description": description, "sha256": climber.sha256[:12],
            "default": climber.sha256 == default,
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
    # what a block is built from: every registered module, by slot
    from hillclimb.modules import refs as module_refs

    say("\n[head]Building blocks[/] [note](a `climber:` block names one per slot; a .py file or package.module:Class works too)[/]")
    for slot, kind in (
        ("selector_policy", "selector_policy"), ("operator_policy", "operator_policy"), ("loop", "loop"),
        ("operators", "operator"), ("tuner", "tuner"), ("memory", "memory"),
    ):
        say(f"  {slot:<16} [path]{_m(', '.join(module_refs.registered_names(kind)))}[/]")
    say("\n[note]* = this folder's default.[/]  Fetch one from the catalog: [cmd]hillclimb climber get greedy[/]  (the whole climber as Python, plus its prompts)")
    say("                              Run it:          [cmd]hillclimb run <problem> --climber climbers/greedy/policy.py[/]")
    say("                              See its block:   [cmd]hillclimb climber show climbers/greedy[/]")
    say("                              Start your own:  [cmd]hillclimb climber new <name> --from climbers/greedy[/]")


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
    ref: str = typer.Argument(None, help="A .py file, a climber folder, or a pre-0.6 climber directory (default: this folder's climber)"),
):
    """Print a climber as the block a run config takes.

    Paste it under `climber:` in a run spec or in runs/config.yaml and edit
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


@climber_app.command("get")
def climber_get(
    preset: str = typer.Argument("greedy", help="A catalog climber to copy out: greedy | openevolve | gepa (see: hillclimb climber list)"),
    name: str = typer.Option(None, "--name", help="The folder's name under climbers/ (default: the catalog's)"),
    default: bool = typer.Option(
        True, "--default/--no-default",
        help="Make the copy this folder's climber (`climber:` in runs/config.yaml), so `hillclimb run` uses it",
    ),
):
    """Copy a catalog climber into climbers/<name>/: the whole climber as Python you can read and edit.

    hillclimb ships no climber of its own — a catalog of examples to copy
    from, like `problem get`. policy.py IS the climber: the selector policy
    (which candidate the next attempt starts from) and the operator policy
    (which operator makes it), every decision and every default (`DEFAULTS`)
    written out, and at the end the `Climber(...)` that wires them to the
    operators, tuner and memory — no config file; prompts/ holds the
    templates its operators render — the words the coding agents get — plus
    a README of what the harness fills into each template's tokens. It
    becomes this folder's climber, so edit policy.py or a template and the
    next `hillclimb run` climbs with it. An existing folder is never
    overwritten.
    """
    import re as _re

    from hillclimb import catalog
    from hillclimb.cli.problem import _hillclimb_dir_or_offer
    from hillclimb.climber import ClimberLoadError, climber_base_dir, load_climber
    from hillclimb.project import RUNS_CONFIG, ensure_owned_dir

    if preset not in catalog.climber_names():
        fail(f"error: no catalog climber {_m(repr(preset))} [note](available: {_m(', '.join(catalog.climber_names()))})[/]")
        raise typer.Exit(1)
    name = name or preset
    if not _re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise typer.BadParameter(f"{name!r}: a climber name is letters, digits, - and _", param_hint="--name")
    config = _hillclimb_dir_or_offer(preset, f"climber get {preset}")
    local = _local_climbers_dir(config)
    base_dir = climber_base_dir(config)
    target = local / name
    folder_ref = _climber_ref(target, base_dir)
    ref = _climber_ref(target / "policy.py", base_dir)  # the file is the climber; the folder names it too
    if target.exists():
        if not _is_climber_dir(target):
            fail(f"error: [path]{_m(target)}[/] exists and is not a climber folder [note](never overwritten)[/]")
            raise typer.Exit(1)
        say(f"[head]Already have {_m(name)}[/] at [path]{_m(target)}[/] [note](never overwritten)[/]")
        written = sorted(
            p.relative_to(target).as_posix() for p in target.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
        )
    else:
        try:
            ensure_owned_dir(local)  # marked: `reset` may delete it (an existing one stays the user's)
            catalog.install_climber(local, preset, as_name=name)
            load_climber(ref, base_dir).operator_set()  # the copy loads, or it is not left behind
        except (ClimberLoadError, ValueError) as exc:
            import shutil

            shutil.rmtree(target, ignore_errors=True)
            fail(f"error: {_m(exc)}")
            raise typer.Exit(1) from exc
        written = sorted(
            p.relative_to(target).as_posix() for p in target.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
        )
        say(f"[head]Fetched {_m(preset)}[/] as [path]{_m(folder_ref)}[/]")
    legend([(path, note) for path, note in _folder_legend(written)])
    if default and config.hillclimb_dir is not None:
        # the climber is a run default: it goes in runs/config.yaml
        run_defaults = config.paths.runs_dir / RUNS_CONFIG
        shown = f"runs/{RUNS_CONFIG}"
        pinned = pin_climber(run_defaults.read_text() if run_defaults.exists() else "", ref)
        if pinned is None:
            warn(f"{shown} defines a `climber:` block of its own; set `climber: {ref}` there yourself")
        else:
            run_defaults.parent.mkdir(parents=True, exist_ok=True)
            run_defaults.write_text(pinned)
            say(f"[head]Default:[/] `climber: {_m(ref)}` in [path]{_m(shown)}[/] [note](every `hillclimb run` here climbs with it)[/]")
    next_steps([
        (f"cat {folder_ref}/prompts/README.md", "how a prompt is made, and what fills each template"),
        (f"hillclimb climber check --climber {ref}", "after editing policy.py or a template"),
        ("hillclimb run <problem> --budget 10m", "climb with it"),
    ])


def _folder_legend(written: list[str]) -> list[tuple[str, str]]:
    """One note per file of a climber folder, prompts grouped."""
    notes = {
        "policy.py": (
            "the climber, as Python: the selector policy (which candidate next), the operator policy "
            "(which operator on it), their defaults, and the Climber(...) that wires them to the operators, tuner, memory and prompts"
        ),
        "prompts/README.md": "how a prompt is made; what fills every token",
    }
    rows = [(path, notes[path]) for path in notes if path in written]
    templates = [Path(path).name for path in written if path.startswith("prompts/") and path not in notes]
    if templates:
        rows.append(("prompts/*.md", f"the templates the operators render, the coding agents' words: {', '.join(templates)}"))
    return rows


def pin_climber(text: str, ref: str) -> str | None:
    """`climber: <ref>` set in runs/config.yaml, comments intact: the
    commented `# climber: greedy` line `init` leaves is uncommented in
    place, an active scalar replaced, none at all added. None when the file
    defines an active `climber:` BLOCK — replacing its first line would
    leave the block's keys dangling, so that edit is the person's."""
    import re as _re

    from hillclimb.connect import apply_config_defaults

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _re.match(r"^climber:", line):
            after = line.split(":", 1)[1].split("#", 1)[0].strip()
            if after and not after.startswith(("{", "[")):
                lines[index] = f"climber: {ref}"
                return "\n".join(lines) + "\n"
            return None
    return apply_config_defaults(text, {"climber": ref})


@climber_app.command("new")
def climber_new(
    name: str = typer.Argument(..., help="Name of the new climber (becomes climbers/<name>.py)"),
    from_: str = typer.Option(
        "greedy", "--from", help="What to copy: a catalog climber's name or a local climber (.py file or folder)"
    ),
):
    """Start your own climber from a copy of an existing one.

    A climber folder (`hillclimb climber get` wrote it: policy.py builds the
    whole Climber, prompts/ beside it) is copied as climbers/<name>/. Any other
    source's file — the policies, or the loop — is copied into climbers/<name>.py,
    a one-file climber you can edit, and the block that runs it is printed.
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
        from hillclimb import catalog

        source = catalog.climber(from_) if from_ in catalog.climber_names() else load_climber(from_, base_dir)
        source_file = Path(inspect.getsourcefile(source.brain.target))
    except (ClimberLoadError, ValueError, TypeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    target = (local / name).with_suffix(".py")
    if target.exists() or (local / name).exists():
        raise typer.BadParameter(f"{_climber_ref(target, base_dir)} already exists")
    from hillclimb.project import ensure_owned_dir

    ensure_owned_dir(local)  # marked: `reset` may delete it (an existing one stays the user's)
    if "Climber(" in source_file.read_text(encoding="utf-8", errors="replace"):
        # the file builds the whole climber (a folder `climber get` wrote): the
        # copy is a folder too — the file, renamed inside, with its prompts beside it
        folder = local / name
        folder.mkdir()
        text = source_file.read_text(encoding="utf-8")
        if source.spec.name:
            text = text.replace(f"name={source.spec.name!r},", f"name={name!r},", 1)
        (folder / "policy.py").write_text(text, encoding="utf-8")
        if source.prompts_dir is not None:
            shutil.copytree(source.prompts_dir, folder / "prompts")
        ref = _climber_ref(folder / "policy.py", base_dir)
        try:
            load_climber(ref, base_dir).brain  # noqa: B018
        except (ClimberLoadError, ValueError) as exc:  # never leave a broken copy behind
            shutil.rmtree(folder, ignore_errors=True)
            raise typer.BadParameter(f"the copy does not load: {exc}") from exc
        say(f"[head]Created[/] [path]{_m(_climber_ref(folder, base_dir))}[/] from [path]{_m(from_)}[/]")
        say(f"[head]Run it:[/] `climber: {_m(ref)}` in runs/config.yaml, or [cmd]--climber {_m(ref)}[/]")
        say(f"[head]Next:[/] edit [path]{_m(ref)}[/] or a template beside it, then   [cmd]hillclimb climber check --climber {_m(ref)}[/]")
        return
    shutil.copy2(source_file, target)
    ref = _climber_ref(target, base_dir)
    brain = "loop" if source.is_loop else "operator_policy"
    class_name = getattr(source.brain.target, "__name__", "")
    block = {**_portable_block(source, base_dir), "name": name, brain: f"{ref}:{class_name}" if class_name else ref}
    selector_cls = None if source.is_loop else source._selector_target()
    if inspect.isclass(selector_cls):
        try:
            same_file = Path(inspect.getsourcefile(selector_cls)) == source_file
        except TypeError:
            same_file = False
        if same_file:  # the copy carries the selector policy too: name it there, not in the source
            block["selector_policy"] = f"{ref}:{selector_cls.__name__}"
    try:
        load_climber(block[brain], base_dir).brain  # noqa: B018
    except (ClimberLoadError, ValueError) as exc:  # never leave a broken copy behind
        target.unlink()
        raise typer.BadParameter(f"the copy does not load: {exc}") from exc
    say(f"[head]Created[/] [path]{_m(ref)}[/] from [path]{_m(from_)}[/]")
    say("[head]Its block[/] [note](paste under `climber:` in a run spec or runs/config.yaml):[/]")
    typer.echo(yaml.safe_dump({"climber": block}, sort_keys=False), nl=False)
    say(f"[head]Next:[/] edit it, then   [cmd]hillclimb climber check --climber {_m(block[brain])}[/]")
    say(f"                       [cmd]hillclimb run <problem> --climber {_m(block[brain])}[/]")


@climber_app.command("check")
def climber_check(
    spec: str = typer.Argument(
        None, help="A run spec: check the climber of every entry (default: this folder's `climber:` block)"
    ),
    climber: str = typer.Option(None, "--climber", help="A .py file or climber folder, instead of the folder's block"),
    problem: str = typer.Option(
        None, "--problem", help="Replay only this problem's recorded searches (default: every search)"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Config override, dotted: --set climber.params.num_drafts=1",
    ),
    limit: int = typer.Option(20, "--limit", help="Newest recorded searches to replay"),
    smoke: bool = typer.Option(
        False, "--smoke", help="Then run a dummy-coding-agent search on --problem (no LLM, real verifier)"
    ),
    smoke_budget: str = typer.Option("2m", "--smoke-budget", help="Wall clock for the smoke search"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Conformance check for a climber — the cheap pre-verifier.

    Resolves every module the block names, then replays every recorded
    journal (plus an empty one) through the climber's policies with no coding agent or
    verifier: two fresh instances must propose the same action at every
    budget point and a resumed one must agree with a live one (the resume
    contract), every referenced candidate must exist, the policies must never
    write, and the climber's prompts must lint clean. Exit 1 on any breach.
    `--smoke` follows up with a short `--agent dummy` search so the whole
    loop — prompts included — runs once before a coding agent hour is spent on it.
    Given a run spec, every entry's climber is checked.
    """
    from hillclimb.climber import ClimberLoadError, as_spec, climber_base_dir, load_climber
    from hillclimb.problem import load_suite

    config = common.load_config()
    base_dir = climber_base_dir(config)
    overrides = common._parse_set(set_ or [])
    source = None
    if climber and climber.endswith(".py"):  # a one-file climber: say so before anything is imported
        source = Path(climber).expanduser()
        if not source.is_absolute() and base_dir is not None:
            source = base_dir / source
    if source is not None and not source.is_file():
        raise typer.BadParameter(f"climber file not found: {source}")
    # what to check: (label, the config whose `climber:` block it is)
    targets: list[tuple[str | None, Config]] = []
    try:
        if spec:
            if climber:
                raise typer.BadParameter("give a run spec or --climber, not both")
            seen: list[dict] = []
            for index, entry in enumerate(load_suite(spec, config).problems, 1):
                entry_config = config.model_copy(deep=True)
                entry_config.hillclimb_dir = config.hillclimb_dir
                if entry.climber is not None:
                    entry_config.climber = as_spec(entry.climber)
                entry_config.apply_overrides(common._parse_set(entry.set))
                entry_config.apply_overrides(overrides)
                block = entry_config.climber_block()  # NoClimber when neither the entry nor the folder names one
                if block not in seen:  # several entries on one climber: checked once
                    seen.append(block)
                    targets.append((f"{entry.name or entry.target} [{index}]", entry_config))
        else:
            if climber:
                try:
                    config.climber = as_spec(climber)  # one file, a climber folder
                except ClimberLoadError:
                    # a pre-0.6 directory: read it as the block it is
                    config.climber = load_climber(climber, base_dir).spec
            config.apply_overrides(overrides)  # edits the block, like `hillclimb run --set`
            targets.append((None, config))
    except (ClimberLoadError, ValueError, KeyError, FileNotFoundError) as exc:
        if isinstance(exc, typer.BadParameter):
            raise
        fail(_m(exc))
        raise typer.Exit(2) from exc

    payloads, worst = [], 0
    for label, target_config in targets:
        if label is not None and not as_json:
            say(f"[head]{_m(label)}[/]")
        code, payload = _check_climber(
            target_config, problem=problem, limit=limit, smoke=smoke, smoke_budget=smoke_budget,
            as_json=as_json, source=source,
        )
        worst = max(worst, code)
        if payload is not None:
            payloads.append({"entry": label, **payload} if label is not None else payload)
    if as_json and payloads:
        typer.echo(json.dumps(payloads if spec else payloads[0], indent=2, default=str))
    if worst:
        raise typer.Exit(worst)


def _check_climber(config: Config, *, problem, limit, smoke, smoke_budget, as_json, source) -> tuple[int, dict | None]:
    """Check the climber `config.climber` defines. -> (exit code, the JSON
    payload when one was asked for): 2 it does not load, 1 a breach, 0 conforms."""
    from hillclimb.api import run_search
    from hillclimb.climber import ClimberLoadError, climber_base_dir, resolve_climber
    from hillclimb.modules.policies.check import JournalCase, check_policy

    try:
        loaded = resolve_climber(config.climber, climber_base_dir(config))
        loaded.brain  # noqa: B018 — every module must resolve, before a coding agent hour is spent
        loaded.operator_set()
        loaded.tuner()
        loaded.graph_module()
    except (ClimberLoadError, ValueError, KeyError) as exc:
        fail(_m(exc))
        return 2, None
    params = dict(loaded.spec.params)
    if loaded.is_loop:
        fail(
            f"{_m(loaded.name)} brings its own Loop; "
            "the conformance check covers climbers built on an OperatorPolicy"
        )
        return 2, None

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
        policy = make_policy()
        resolved = getattr(policy, "resolved_params", None)
        resolved_params = resolved() if callable(resolved) else params
        # the exploration process is the policy's knobs AND the selector's schedule
        selector = getattr(policy, "selector", None)
        if selector is not None and callable(getattr(selector, "resolved_params", None)):
            resolved_params = {**selector.resolved_params(), **resolved_params}
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
            "passing": sum(
                c.status == "passing" for c in journal.candidates.values() if c.kind not in ("baseline", "seed")
            ),
        }
        if not as_json:
            common.warn_if_none_passed(journal, outcome.ref, "dummy")
    payload = None
    if as_json:
        payload = report.to_dict()
        payload["resolved_params"] = resolved_params
        payload["journals"] = [c.label for c in cases]
        if smoke_result is not None:
            payload["smoke"] = smoke_result
    else:
        _say_check_report(report)
        say(f"[head]resolved params:[/] {_m(json.dumps(resolved_params, default=str))}")
        say(f"[head]replayed[/] {len(cases)} recorded journal(s)")
        if smoke_result is not None:
            say(
                f"[head]smoke [path]{_m(smoke_result['search'])}[/]:[/] {_m(smoke_result['state'])}, "
                f"{smoke_result['candidates']} candidate(s), {smoke_result['scored']} scored, "
                f"{smoke_result['passing']} attempt(s) passing, "
                f"best={_m(smoke_result['best'])}"
            )
        elif smoke:
            warn("smoke skipped: fix the breaches above first")
    breach = not report.ok or (smoke_result is not None and smoke_result["state"] != "done")
    return (1 if breach else 0), payload
