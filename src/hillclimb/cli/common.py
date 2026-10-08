"""What every command shares: config loading, search lookup, budget parsing, the
hillclimb-dir scaffold and the CLI's voice. Command modules call these as
`common.<name>()` so a test that patches `hillclimb.cli.common` reaches them all."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from hillclimb.config import Config
from hillclimb.tui.palette import CYAN, GREEN, RED, YELLOW
from hillclimb.harness.store import DataStore, SearchRecord, open_store, resolve_search

_CONSOLE = None
_ERR_CONSOLE = None

# The CLI's voice. Every sentence hillclimb speaks goes through `say` (or
# `warn`/`fail` for the two stderr registers) with rich markup on the
# theme's styles, so a reader tells hillclimb's words from a verifier's
# stdout or a coding agent's log at a glance:
#
#   [head]   the lead of a line — the verdict, the thing that happened
#            ("Deleted", "Search r1/alpha", "No hillclimb dir here.")
#   [path]   paths, search refs, candidate ids, file names, env var names
#   [cmd]    a command the reader could type
#   [note]   the explanation beside it, in dim ink
#   [ok] [warn] [bad]   verdicts: green, yellow, bold red
#
# The colours are the palette's hex, not ANSI names: a named colour is drawn
# in the terminal's own palette, which is pastel in a stock Ghostty.
#
# Every dynamic value is wrapped in `_m()` so a `[` in a path or an id is
# not read as markup. Data the reader pipes elsewhere — `--json` output,
# diffs, file bodies, a candidate's stdout tail — is NOT the voice: it
# stays on `typer.echo`, byte-exact, with no styling.
THEME = {
    "cmd": f"bold {CYAN}", "path": CYAN, "note": "dim", "head": "bold",
    "ok": GREEN, "warn": YELLOW, "bad": f"bold {RED}",
}


def _make_console(stderr: bool, width: int | None = None):
    from rich.console import Console
    from rich.theme import Theme

    # emoji off: a `:tag:` inside a path or a coding agent's summary must print
    # as written. No color when the stream is not a terminal (a pipe, a
    # test), exactly like the banner.
    return Console(stderr=stderr, highlight=False, emoji=False, theme=Theme(THEME), width=width)


def _console():
    """One rich console for the CLI's own messages, on the theme's palette:
    commands bold cyan, paths cyan, explanations dim."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = _make_console(stderr=False)
    return _CONSOLE


def _err_console():
    """The same voice on stderr, for `warn` and `fail`."""
    global _ERR_CONSOLE
    if _ERR_CONSOLE is None:
        _ERR_CONSOLE = _make_console(stderr=True)
    return _ERR_CONSOLE


def say(text: str = "", *, err: bool = False) -> None:
    """Print with rich markup: [cmd]hillclimb init[/], [path]…[/], [note]…[/], [head]…[/].
    `err=True` speaks on stderr."""
    (_err_console() if err else _console()).print(text, soft_wrap=True)


def warn(text: str) -> None:
    """A warning on stderr: the whole line in the warn colour, markup inside honoured."""
    say(f"[warn]{text}[/]", err=True)


def fail(text: str) -> None:
    """An error on stderr: the whole line in the bad colour, markup inside honoured.
    The caller raises `typer.Exit(1)` itself, so a command can say several things first."""
    say(f"[bad]{text}[/]", err=True)


def warn_if_none_passed(journal, ref: str, agent: str | None = None) -> bool:
    """Warn when a search made attempts and not one of them passed: `done`
    then only means the budget ran out, not that anything was found. The
    baseline and the seed are the harness's floor and do not count. Returns
    whether it warned."""
    attempts = [
        c for c in journal.candidates.values()
        if c.kind not in ("baseline", "seed") and c.status in ("passing", "failing", "buggy")
    ]
    if not attempts or any(c.status == "passing" for c in attempts):
        return False
    buggy = sum(c.status == "buggy" for c in attempts)
    warn(
        f"none of the {len(attempts)} attempt(s) passed ({buggy} buggy, {len(attempts) - buggy} failing): "
        "the search found nothing beyond its floor"
    )
    say(f"[note]See why with[/] [cmd]hillclimb show {_m(ref)} {_m(attempts[0].candidate_id)}[/]", err=True)
    if agent == "dummy":
        say(
            "[note]The dummy agent writes a canned solution.py that knows nothing of this problem's "
            "interface: it tests the loop, not the score. For a free run that climbs, try[/] "
            "[cmd]--agent toy[/] [note]on[/] [path]fitness-landscape[/]",
            err=True,
        )
    return True


def _m(text) -> str:
    """Escape a value (a path, an id) for rich markup."""
    from rich.markup import escape

    return escape(str(text))


_CLOCK = re.compile(r"^(\s*)(\[[^\]]*\bleft\])(.*)$")
_PREFIX = re.compile(r"^(\s*)([a-z][a-z ]{0,23}[a-z]):(\s.*|$)")
_CANDIDATE = re.compile(r"\b(c\d{3,})\b")
_TROUBLE = re.compile(r"\b(failed|crashed|refused|abandoned|TIMEOUT|buggy|out of credits)\b")


def engine_line(line: str) -> str:
    """An engine log line in the CLI's voice, as markup. The engine speaks
    in a few stable shapes and this marks them without parsing anything
    else: a leading `[N minutes left]` clock goes dim, a `word: rest` lead
    (`learning:`, `new selection:`, `baseline written:`) goes bold,
    candidate ids go in the path colour, and a line reporting trouble
    (`failed`, `crashed`, `refused`, `TIMEOUT`) is warned as a whole. The
    text itself is escaped first, so a `[` in a coding agent's summary is never
    read as markup."""
    text = _m(line)
    if _TROUBLE.search(line):
        return f"[warn]{text}[/]"
    lead = ""
    if m := _CLOCK.match(text):
        lead, text = f"{m.group(1)}[note]{m.group(2)}[/]", m.group(3)
    elif m := _PREFIX.match(text):
        lead, text = f"{m.group(1)}[head]{m.group(2)}:[/]", m.group(3)
    text = _CANDIDATE.sub(r"[path]\1[/]", text)
    return lead + text


CLOCK_GUTTER = 13  # `1:05:00 left` fits; the message column starts after it


def split_engine_line(line: str) -> tuple[str, str]:
    """`(clock, message)` for the log's two columns: the `m:ss left` text of
    a leading `[… left]` clock (empty for a line without one) and the rest
    as [`engine_line`] markup. A line under a clock keeps its own indent,
    so `  new selection:` still nests under the operator that made it."""
    m = _CLOCK.match(line)
    if not m:
        return "", engine_line(line)
    clock = m.group(2)[1:-1]  # without the brackets: the gutter is the frame
    return clock, engine_line(m.group(1) + m.group(3).lstrip(" ") if m.group(3) else "")


def engine_log(line: str) -> None:
    """The `log=` callback for a foreground engine. On a terminal each line
    is two columns — the clock, right-aligned and dim in a gutter of
    `CLOCK_GUTTER` cells, then the message — so a message that wraps folds
    under itself and the clocks stand in one column down the log. Off a
    terminal (a pipe, a test) the line prints whole, marked up."""
    console = _console()
    if not console.is_terminal:
        say(engine_line(line))
        return
    from rich.table import Table

    clock, message = split_engine_line(line)
    grid = Table.grid(padding=(0, 2, 0, 0))
    grid.add_column(width=CLOCK_GUTTER, justify="right", no_wrap=True, style="note")
    grid.add_column(ratio=1, overflow="fold")  # a long path folds, never an ellipsis
    grid.add_row(clock, message)
    console.print(grid)


def table(columns, rows, *, err: bool = False) -> None:
    """A listing in the CLI's voice: bold headings over a single rule, no
    borders, cells in whatever markup the caller gives (already escaped
    with `_m`). `columns` are `(heading, style | None)` pairs — the style
    colours a whole column, e.g. `("problem", "path")`; a heading may be
    `""` for a marker column, and a third item aligns it (`("best", None,
    "right")`). One helper so `connect`, `problem list` and
    every later listing look like one tool."""
    from rich import box
    from rich.table import Table

    grid = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="head", show_edge=False)
    for heading, style, *justify in columns:
        # a cell wraps inside its own column, and a long word (a path, an
        # id) folds rather than being cut with an ellipsis; a third item
        # aligns the column ("right" for numbers)
        grid.add_column(heading, style=style, overflow="fold", justify=justify[0] if justify else "left")
    for row in rows:
        grid.add_row(*row)
    console = _err_console() if err else _console()
    # A terminal has a width to fit, so long cells wrap there. A pipe (a
    # test, `| grep`) does not: rich would still squeeze the table into 80
    # columns and crop cells with an ellipsis, so there the table is printed
    # through a console as wide as its natural width and every cell prints
    # whole (`Console.print(width=…)` cannot widen, only narrow).
    if not console.is_terminal:
        from rich.measure import Measurement

        natural = Measurement.get(console, console.options.update(max_width=10_000), grid).maximum
        if natural > console.width:
            console = _make_console(err, width=natural)
    console.print(grid)


def markdown(text: str) -> None:
    """A Markdown report: rendered in the theme on a terminal (headings,
    tables), the Markdown itself when piped — so it pastes into notes."""
    console = _console()
    if not console.is_terminal:
        typer.echo(text)
        return
    from rich.markdown import Markdown

    console.print(Markdown(text))


def short_path(path) -> str:
    """A path as a person reads it: the home folder as `~`."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + "/") else text


def legend(rows, indent: int = 2) -> None:
    """Aligned `key — note` rows: the key in the path color, the note dim."""
    width = max(len(key) for key, _ in rows)
    for key, note in rows:
        say(f"{' ' * indent}[path]{_m(key):<{width}}[/]  [note]— {_m(note)}[/]")


def next_steps(rows) -> None:
    """`Next:` then one command per line with a dim note beside it."""
    width = max(len(cmd) for cmd, _ in rows)
    for index, (cmd, note) in enumerate(rows):
        lead = "[head]Next:[/]" if index == 0 else "     "
        say(f"{lead} [cmd]{_m(cmd):<{width}}[/]  [note]{_m(note)}[/]")


def load_config(*, raise_not_found: bool = False, **overrides) -> Config:
    """Config.load with the no-hillclimb-dir hint rendered for the CLI
    (or re-raised, for commands that have a fallback)."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return Config.load(**overrides)
    except HillclimbDirNotFound as exc:
        if raise_not_found:
            raise
        say_no_hillclimb_dir(exc)
        raise typer.Exit(1) from exc


def require_sandbox(config: Config, overrides: dict | None = None) -> None:
    """Before a command starts coding agents or a verifier: exit with the fix when
    the sandbox is on and cannot start here, and say so when what follows
    runs without one (`sandbox: off`, or an OS that has none)."""
    from hillclimb.harness import sandbox

    if overrides:
        config = config.model_copy(deep=True)
        config.apply_overrides(overrides)
    try:
        if not sandbox.enabled(config):
            reason = "off"
        elif sandbox.backend() is None:
            reason = sandbox.no_sandbox_reason()
        else:
            return
    except sandbox.SandboxUnavailable as exc:
        fail(f"Not started: {_m(exc)}")
        raise typer.Exit(1) from exc
    warn(f"sandbox: {reason} — coding agents and solutions run with your full user rights")


def say_no_hillclimb_dir(exc) -> None:
    """The no-hillclimb-dir hint in the CLI's voice, on stderr: the same
    words as the exception's, with the path, the command and the env var
    marked up."""
    from hillclimb.project import MARKER_FILE

    say(f"[head]No {MARKER_FILE} in[/] [path]{_m(exc.start)}[/].", err=True)
    say(
        f"Run hillclimb from the root of a hillclimb dir [note](the folder holding {MARKER_FILE})[/], "
        f"run [cmd]hillclimb init[/] to make this folder one [note](writes ./{MARKER_FILE})[/], "
        "or set [path]HILLCLIMB_DIR[/] to an existing one.",
        err=True,
    )


# What `hillclimb init` writes lives in hillclimb.project (the Python API makes
# a fresh folder a hillclimb dir too); the CLI reaches it as common.<name>.
from hillclimb.project import (  # noqa: E402,F401
    INIT_CONFIG, INIT_GITIGNORE, SCAFFOLD_DIRS, OWNED_DIR_NAMES, DATASTORES, init_config, scaffold_blockers, scaffold_target, scaffold_hillclimb_dir,
)


def owned_paths(root: Path, config, *, runs_only: bool = False) -> tuple[list[Path], list[Path]]:
    """What `hillclimb reset` does with the hillclimb dir `root`, as
    (deleted, kept). Deleted: the config, the sqlite store, and every folder
    hillclimb created there (it carries a `.hillclimb` marker). Kept: a
    folder of one of hillclimb's names without the marker — the user's own,
    or one from before markers existed. A hillclimb dir may be a code repo's
    root; nothing else in it is touched, and a configured runs_dir or
    problems_dir counts only when it lies inside `root`. `runs_only` is
    `reset --runs`: just what searches produced (runs/, the store, knowledge/),
    so the config, problems/ and climbers/ are not candidates at all."""
    from hillclimb.experiment import EXPERIMENTS_DIRNAME
    from hillclimb.project import MARKER_FILE, is_owned_dir

    files = [root / MARKER_FILE, *sorted(config.store.sqlite_path.parent.glob(config.store.sqlite_path.name + "*"))]
    folders = [
        config.paths.problems_dir,
        config.paths.runs_dir,
        root / "knowledge",
        root / "climbers",  # cli/climber.py's LOCAL_CLIMBERS_DIRNAME (common imports no command module)
        root / EXPERIMENTS_DIRNAME,
    ]
    if runs_only:
        files = files[1:]
        folders = [config.paths.runs_dir, root / "knowledge"]
    deleted: list[Path] = []
    kept: list[Path] = []
    resolved_root = root.resolve()
    for path in [*files, *folders]:
        if path.resolve() == resolved_root or not path.resolve().is_relative_to(resolved_root):
            continue
        if not (path.exists() or path.is_symlink()) or path in deleted or path in kept:
            continue
        if path in folders and path.is_dir() and not path.is_symlink() and not is_owned_dir(path):
            kept.append(path)
        else:
            deleted.append(path)
    return deleted, kept


def parse_budget(value: str) -> int:
    from hillclimb.harness.budget import parse_budget as parse

    try:
        return parse(value)
    except ValueError:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)") from None


def open_search(config: Config, ref: str | None) -> tuple[DataStore, SearchRecord]:
    """The configured store and the search `ref` names in it, with the
    lookup message as a usage error."""
    store = open_store(config)
    try:
        return store, resolve_search(store, ref)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc


def resolve_search_dir(config: Config, ref: str | None) -> Path:
    return open_search(config, ref)[1].search_dir


def _spec_provenance(config: Config, suite_path: Path) -> str:
    """Spec path recorded in run.yaml, relative to the hillclimb dir
    (absolute if the spec lives outside it)."""
    if config.hillclimb_dir is not None:
        try:
            return str(suite_path.relative_to(config.hillclimb_dir))
        except ValueError:
            pass
    return str(suite_path)


def _parse_experiment_set(pairs: list[str]) -> dict[str, list[str]]:
    """`EXPERIMENT:KEY=VALUE` strings (the `--experiment-set` flag) →
    experiment name -> its `--set` pairs, validated the way `--set` is."""
    out: dict[str, list[str]] = {}
    for item in pairs:
        name, sep, pair = item.partition(":")
        if not sep or not name.strip() or "=" not in pair:
            raise typer.BadParameter(f"--experiment-set expects EXPERIMENT:KEY=VALUE, got {item!r}")
        _parse_set([pair])
        out.setdefault(name.strip(), []).append(pair)
    return out


def _parse_set(pairs: list[str]) -> dict:
    from hillclimb.config import parse_set_overrides

    try:
        return parse_set_overrides(pairs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def folder_problem(config: Config, problem: str | None):
    """The problem a folder-level command means: the one named, else the
    only problem this hillclimb dir has searched, else the only one in its
    problems/ — a usage error naming the choices when that is ambiguous."""
    from hillclimb.problem import load_problem

    if problem:
        return load_problem(problem, config)
    store = open_store(config)
    try:
        records = store.searches()
    finally:
        store.close()
    by_key: dict[str, SearchRecord] = {}
    for record in records:  # oldest first: the newest record wins its key
        by_key[record.meta.problem_key or record.meta.problem_id] = record
    if len(by_key) == 1:
        return load_problem(next(iter(by_key.values())).meta.problem, config)
    if not by_key:
        folders = sorted(
            p.name for p in config.paths.problems_dir.iterdir() if (p / "problem.yaml").is_file()
        ) if config.paths.problems_dir.is_dir() else []
        if len(folders) == 1:
            return load_problem(folders[0], config)
        choices = folders
    else:
        choices = sorted(by_key)
    raise typer.BadParameter(
        "which problem? " + (f"this folder has {', '.join(choices)}" if choices else "this folder has none")
        + " — name one with --problem"
    )


def _ensure_matplotlib(python: Path) -> None:
    """matplotlib in a problem's runtime venv, installed once on first plot
    (the solution package sets do not carry it — a search never plots)."""
    import subprocess

    from hillclimb.harness.oscompat import lock_file

    def has_it() -> bool:
        return subprocess.run([str(python), "-c", "import matplotlib"], capture_output=True).returncode == 0

    if has_it():
        return
    venv_dir = python.parents[1]
    with open(f"{venv_dir}.lock", "w") as lock:  # the lock the venv build takes
        lock_file(lock)
        if has_it():
            return
        say(f"[note]Adding matplotlib to the problem's runtime venv ({_m(venv_dir.name)}), once ...[/]")
        done = subprocess.run(
            ["uv", "pip", "install", "matplotlib", "--python", str(python)], capture_output=True, text=True
        )
        if done.returncode != 0:
            fail("Could not add matplotlib to the runtime venv (first use needs network):")
            say(f"    [note]{_m(done.stderr.strip()[-600:])}[/]", err=True)
            raise typer.Exit(1)


def open_file(path: Path) -> None:
    """Open a file in the system's own viewer, best effort."""
    import os
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False, capture_output=True)
    except OSError:
        pass


def show_solution_plot(
    config: Config, problem, solution_dir: Path, where: str, out: Path, *, open_it: bool = True
) -> None:
    """Draw `solution_dir` with the problem's plot.py — matplotlib, run in the
    problem's runtime venv like its verifier — save it to `out` and open it;
    or say why there is nothing to draw."""
    import subprocess
    from importlib import resources

    from hillclimb.api import ensure_runtime_venv

    if problem.plot_path is None:
        say(
            f"[warn]{_m(problem.problem_id)} ships no plot.py[/] [note]— add one to the problem dir "
            "(def plot(solution_dir, ax) drawing its output files with matplotlib) to see its solutions[/]"
        )
        raise typer.Exit(1)
    python = ensure_runtime_venv(
        config, kind=problem.runtime, requirements=problem.requirements_file,
        log=lambda line: say(f"[note]{_m(line)}[/]"),
    )
    _ensure_matplotlib(python)
    title = f"{problem.problem_id} · {where}"
    with resources.as_file(resources.files("hillclimb.runtime") / "plot_solution.py") as runner:
        done = subprocess.run(
            [str(python), str(runner), str(problem.plot_path), str(Path(solution_dir).resolve()), str(out), title],
            capture_output=True, text=True,
        )
    if done.returncode != 0 or not out.is_file():
        fail(f"Cannot plot {_m(where)}: {_m(problem.plot_path.name)} failed")
        for line in (done.stderr or done.stdout).strip().splitlines()[-12:]:
            say(f"    [note]{_m(line)}[/]", err=True)
        raise typer.Exit(1)
    caption = next(
        (line[len("caption: "):] for line in done.stdout.splitlines() if line.startswith("caption: ")), ""
    )
    say(f"[head]{_m(problem.problem_id)}[/] · {_m(where)}" + (f": {_m(caption)}" if caption else ""))
    say(f"  wrote [path]{_m(out)}[/]")
    if open_it:
        open_file(out)


def ensure_agents_ready(config: Config, agent: str | None = None, model: str | None = None) -> None:
    """Before a command uses coding agents: every one it would call — the
    default and any per-operator route — has a live login, or, at a
    terminal, is offered a fresh one ("Log in again now?"). Declined, or no
    terminal to ask at, the command stops here with the fix, before anything
    is spent (`connect.ensure_agent_ready`)."""
    import sys

    from hillclimb import connect

    default = (agent or config.agent, config.agent_auth, model or config.model)
    routes = [
        (route.agent or default[0], route.agent_auth or default[1], route.model or default[2])
        for route in config.routing.values()
    ]
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    def ask(question: str) -> bool:
        warn(question.split(". Log in again now?")[0] + ".")
        return typer.confirm("Log in again now?", default=True)

    for name, auth, chosen in dict.fromkeys([default, *routes]):
        try:
            connect.ensure_agent_ready(
                name, auth, chosen, ask=ask if interactive else None,
                say=lambda line: say(f"[note]{_m(line)}[/]"),
                models_file=config.pi.models_file,
            )
        except connect.AgentLoginError as exc:
            fail(str(exc))
            raise typer.Exit(1) from exc
