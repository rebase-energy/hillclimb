"""What every command shares: config loading, search lookup, budget parsing, the
hillclimb-dir scaffold and the CLI's voice. Command modules call these as
`common.<name>()` so a test that patches `hillclimb.cli.common` reaches them all."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from hillclimb.config import Config
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
# Every dynamic value is wrapped in `_m()` so a `[` in a path or an id is
# not read as markup. Data the reader pipes elsewhere — `--json` output,
# diffs, file bodies, a candidate's stdout tail — is NOT the voice: it
# stays on `typer.echo`, byte-exact, with no styling.
THEME = {
    "cmd": "bold cyan", "path": "cyan", "note": "dim", "head": "bold",
    "ok": "green", "warn": "yellow", "bad": "bold red",
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
    `""` for a marker column. One helper so `connect`, `problem list` and
    every later listing look like one tool."""
    from rich import box
    from rich.table import Table

    grid = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="head", show_edge=False)
    for heading, style in columns:
        # a cell wraps inside its own column, and a long word (a path, an
        # id) folds rather than being cut with an ellipsis
        grid.add_column(heading, style=style, overflow="fold")
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
            reason = "none exists for this operating system"
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

    say(f"[head]No {MARKER_FILE} found[/] from [path]{_m(exc.start)}[/] upward.", err=True)
    say(
        f"Run [cmd]hillclimb init[/] to make this folder a hillclimb dir [note](writes ./{MARKER_FILE})[/], "
        "or set [path]HILLCLIMB_DIR[/] to an existing one.",
        err=True,
    )


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir (problems/ and runs/
# sit beside it); commands work from any subdirectory below it. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# agent: claude-code

# climber: greedy          # HOW to climb, this folder's default. A preset: greedy | openevolve | gepa,
#                          # or one .py file. A run spec's own `climber:` replaces it
# climber:                 # ...or the whole block (`hillclimb climber show greedy` prints one to edit)
#   selector_policy: best  # which candidate to build on next: best | map-elites, or a file / package.module:Class
#   selector_params: {num_drafts: 3}
#   operator_policy: greedy  # which operator to use on it: a name, a file (mine.py or mine.py:Class) or package.module:Class
#   operators: [draft, debug, improve, ensemble]
#   tuner: random          # random | optuna (parameter tuning of candidates that declare params.json)
#   memory: files          # files | none

# budget:
#   total_s: 7200
#   deadline: graceful     # `hard` aborts in-flight operators when total_s runs out
#   max_evaluations: 0     # verifier trials the climber may spend (0 = unlimited)

# evaluation:
#   n_replicates: 1        # seeded runs per trial (median is the trial's score)
#   replicate_mode: parallel # `serial` when the metric measures the machine (time!)
#   noise_k: 0             # require gains > k x the measured noise floor
#   min_improvement: 0     # ...or an absolute floor, in metric units

# concurrency:
#   parallel_agents: 1   # >1 runs concurrent coding agents
#   machine_max_agents: 8  # cap across every search on this machine (default min(8, cores-2))

# holdout:
#   enabled: true
#   top_k: 5             # holdout scored only for top-k-by-val candidates

# similarity:            # `hillclimb similarity scores`: name (or my_score.py) -> params
#   scores:
#     solution-card: {card_model: anthropic/claude-haiku-4.5, embedding_model: voyageai/voyage-4}
#     api-calls: {}

# learning:
#   enabled: true        # knowledge cards in knowledge/ inform new searches
#   max_cards: 3
#   complexity_prior: false
#   live: true           # concurrent searches in one run share discoveries mid-flight

# report:
#   enabled: true        # inject eval breakdowns (per-zone/horizon/quantile) into improve prompts
"""


# What `hillclimb init` adds to the hillclimb dir's .gitignore. The RECORD
# of every run is committed — run.yaml, spec.yaml, each search's search.yaml,
# journal, status, knowledge card, climber snapshot, and the best solution —
# so `git log` explains every run and `hillclimb chart` works on a fresh
# clone. The BULK is not: candidates (coding agent streams, replicate outputs,
# runtime data), engine logs, the control queue, the rest of best/ (a
# submission can be large), the sqlite store and the derived knowledge graph.
# Keys never are. Leading slashes anchor each rule at the hillclimb dir.
INIT_GITIGNORE = (
    "# hillclimb: the record of every run is committed, its bulk is not",
    "/.env",
    "/runs/*/logs/",
    "/runs/*/searches/*/candidates/",
    "/runs/*/searches/*/control/",
    "/runs/*/searches/*/best/*",
    "!/runs/*/searches/*/best/solution.py",
    "!/runs/*/searches/*/best/params.json",
    "/store.sqlite*",
    "/knowledge/graph.json",
)

# The folders `init` creates beside hillclimb.yaml.
SCAFFOLD_DIRS = ("problems", "runs")


def scaffold_blockers(folder: Path) -> list[Path]:
    """What stops `folder` from becoming a hillclimb dir: a problems/ or
    runs/ that is already there and not hillclimb's (a code repo's own).
    Empty when the folder is free or already a hillclimb dir."""
    from hillclimb.project import MARKER_FILE

    if (folder / MARKER_FILE).exists():
        return []
    return [folder / sub for sub in SCAFFOLD_DIRS if (folder / sub).exists()]


def scaffold_hillclimb_dir(folder: Path) -> Path:
    """Make `folder` (created if missing) a hillclimb dir: hillclimb.yaml,
    empty problems/ and runs/ beside it, and the gitignore rules that keep
    run artifacts and keys out of git while the record of every run goes in
    (`INIT_GITIGNORE`). No problem is added: picking one (`hillclimb problem
    get`) is the user's first real choice. Idempotent on the folder layout;
    never overwrites an existing config, only adds ignore rules that are
    missing. Callers check `scaffold_blockers` first."""
    from hillclimb.project import MARKER_FILE

    for sub in SCAFFOLD_DIRS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(INIT_CONFIG)
    gitignore = folder / ".gitignore"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    present = existing_ignore.splitlines()
    missing = [line for line in INIT_GITIGNORE if line not in present]
    if missing == [INIT_GITIGNORE[0]]:  # every rule is there, only the heading is not
        missing = []
    if missing:
        gitignore.write_text(
            existing_ignore.rstrip("\n")
            + ("\n" if existing_ignore else "")
            + "\n".join(missing)
            + "\n"
        )
    return folder


def owned_paths(root: Path, config) -> list[Path]:
    """What `hillclimb reset` deletes from the hillclimb dir `root`: the
    config and everything hillclimb writes beside it, and nothing else — a
    hillclimb dir may be a code repo's root. A configured runs_dir or
    problems_dir counts only when it lies inside `root`."""
    from hillclimb.experiment import EXPERIMENTS_DIRNAME
    from hillclimb.project import MARKER_FILE

    candidates = [
        root / MARKER_FILE,
        config.paths.problems_dir,
        config.paths.runs_dir,
        root / "knowledge",
        root / "climbers",  # cli/climber.py's LOCAL_CLIMBERS_DIRNAME (common imports no command module)
        root / EXPERIMENTS_DIRNAME,
        *sorted(config.store.sqlite_path.parent.glob(config.store.sqlite_path.name + "*")),
    ]
    owned: list[Path] = []
    resolved_root = root.resolve()
    for path in candidates:
        if path.resolve() == resolved_root or not path.resolve().is_relative_to(resolved_root):
            continue
        if (path.exists() or path.is_symlink()) and path not in owned:
            owned.append(path)
    return owned


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
