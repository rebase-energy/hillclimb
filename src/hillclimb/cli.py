from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer
import typer.core
import typer.rich_utils

from hillclimb.tui.banner import (  # noqa: F401 — re-exported: tests/test_cli.py and
    BANNER_LINES,  # hillclimb-web/scripts/generate_wordmark.py read the art from here
    LOGO_LINES,
    WORDMARK_LINES,
    print_banner,
)

from hillclimb.api import (
    child_launch_context,
    create_problem_run,
    create_run,
    create_search,
    build_evaluator,
    build_executor,
    build_unit_test_runner,
    build_holdout_scorer,
    ensure_runtime_venv,
    execute_search,
    new_run_id,
    resume_spent_seconds,
    FleetEngine,
    mixed_fleet,
    run_fleet,
    spawn_search_proc,
)
from hillclimb.backends import get_backend
from hillclimb.harness.budget import BudgetManager
from hillclimb.config import Config, RouteConfig
from hillclimb.harness.control import request_prune, request_stop
from hillclimb.harness.journal import Journal
from hillclimb.problem import (
    ProblemSpec,
    load_problem,
    resolve_target,
    suite_problem_targets,
)
from hillclimb.harness.run import (
    load_run_meta,
    load_search_meta,
    RunMeta,
    search_ref,
)
from hillclimb.harness.core import Harness
from hillclimb.modules.policies.base import Action
from hillclimb.harness.status import read_status
from hillclimb.harness.store import (
    DataStore,
    SearchRecord,
    key_for,
    latest_search,
    open_store,
    resolve_search,
    running_searches,
)

# Typer's default rich theme paints "Usage:" and every `<...>` metavar yellow,
# which clashes with the cyan command/option column. Repaint both in the same
# cyan family so the help screen reads as one palette. These are module-level
# globals that typer.rich_utils reads at render time, so assigning them here
# (before any help is formatted) is enough.
typer.rich_utils.STYLE_USAGE = "bold cyan"
typer.rich_utils.STYLE_TYPES = "cyan"


class HillclimbGroup(typer.core.TyperGroup):
    """Command listing and the completion flags, the way this CLI wants them."""

    # `ctx` is a click Context and get_params returns click Parameters, but
    # typer >=0.27 vendors click as `typer._click` and hillclimb does not
    # depend on the standalone package — so these stay unannotated rather than
    # importing a module that is not guaranteed to be installed.
    def list_commands(self, ctx) -> list[str]:
        """Typer lists commands in declaration order; list them alphabetically."""
        return sorted(self.commands)

    def get_params(self, ctx) -> list:
        """Drop `--show-completion`, and keep `--install-completion` working but
        unlisted — shell completion is a one-time setup step documented in the
        README, not something worth a third of the top-level options panel."""
        params = []
        for param in super().get_params(ctx):
            if param.name == "show_completion":
                continue
            if param.name == "install_completion":
                param.hidden = True
            params.append(param)
        return params


app = typer.Typer(
    cls=HillclimbGroup,
    help="Hillclimbing on verifier-defined problems: a code-generation harness for model development with long-running agents.",
    no_args_is_help=True,
    # Subcommands inherit help_option_names from the parent click Context, so
    # `-h` works on every command in the tree, not just the top level.
    context_settings={"help_option_names": ["--help", "-h"]},
)

_CONSOLE = None


def _console():
    """One rich console for the CLI's own messages, on the theme's palette:
    commands bold cyan, paths cyan, explanations dim. No color when stdout
    is not a terminal (a pipe, a test), exactly like the banner."""
    global _CONSOLE
    if _CONSOLE is None:
        from rich.console import Console
        from rich.theme import Theme

        _CONSOLE = Console(highlight=False, theme=Theme({
            "cmd": "bold cyan", "path": "cyan", "note": "dim", "head": "bold",
            "ok": "green", "warn": "yellow", "bad": "bold red",
        }))
    return _CONSOLE


def say(text: str = "") -> None:
    """Print with rich markup: [cmd]hillclimb init[/], [path]…[/], [note]…[/], [head]…[/]."""
    _console().print(text, soft_wrap=True)


def _m(text) -> str:
    """Escape a value (a path, an id) for rich markup."""
    from rich.markup import escape

    return escape(str(text))


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
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir; commands work from
# any subdirectory below it. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# backend: claude-code

# climber: greedy          # HOW to climb: greedy | openevolve | gepa | hillclimb/climbers/<name>
# climber:                 # ...or with your overrides on the climber's own params
#   ref: greedy
#   params: {num_drafts: 3}
#   tuner: random          # random | optuna (parameter tuning of candidates that declare params.json)

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
#   parallel_operators: 1   # >1 runs concurrent operators
#   machine_max_operators: 8  # cap across every search on this machine (default min(8, cores-2))

# holdout:
#   enabled: true
#   top_k: 5             # holdout scored only for top-k-by-val candidates

# similarity:            # `hillclimb similarity scores`: name (or my_score.py) -> params
#   scores:
#     solution-card: {card_model: anthropic/claude-haiku-4.5, embedding_model: voyageai/voyage-4}
#     api-calls: {}

# learning:
#   enabled: true        # knowledge cards in hillclimb/knowledge/ inform new searches
#   max_cards: 3
#   complexity_prior: false
#   live: true           # concurrent searches in one run share discoveries mid-flight

# report:
#   enabled: true        # inject eval breakdowns (per-zone/horizon/quantile) into improve prompts
"""

INIT_PROBLEM_YAML = """\
problem_id: example
metric: score
higher_is_better: true
description: description.md
time_budget_s: 900
# verifier: verifier.sh   # the default; a problem IS its verifier
# holdout: true           # engine also runs `verifier.sh --holdout`
# unit_tests:             # optional frozen correctness gate, run once per trial
#   root: tests
#   command: ["{python}", "-m", "pytest", "-q", "{tests}"]
# baseline: baseline.py   # scored at t=0 as the floor to beat (or a number, e.g. 0.5)
# requirements: requirements.txt
# interface: interface.py  # optional machine-checked I/O declaration (hillclimb spaces)
"""

INIT_PROBLEM_DESCRIPTION = """\
# Example problem

Replace this with what the solution has to do, what data it gets, and how it
is judged. The agent reads this file verbatim.

The toy objective below: write `solution.py` that prints a number. Bigger wins.
"""

INIT_PROBLEM_VERIFIER = """\
#!/usr/bin/env bash
# A problem is defined by this file. hillclimb runs it in the candidate's
# working directory (./solution.py, ./problem/ and ./data/ are present) and
# reads one thing back: the score.
#
#   exit 0                -> the candidate is valid
#   $HILLCLIMB_RESULT     -> where the score goes: a bare number, or
#                            {"score": <float>, "report": {...}}
#
# Also available: $HILLCLIMB_PYTHON (the managed venv interpreter — use it
# instead of bare `python`), $HILLCLIMB_SOLUTION, $HILLCLIMB_SPLIT,
# $HILLCLIMB_REPLICATE_SEED. `--holdout` is passed when scoring the hidden split.
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION" > solution_out.txt

# Score whatever the solution produced. Do the real checking here: a verifier
# that cannot fail is a verifier the search will learn to cheat.
tail -n 1 solution_out.txt > "$HILLCLIMB_RESULT"
"""

INIT_SPEC_EXAMPLE = """\
# Example run spec — committed run parameters (`hillclimb run hillclimb/specs/example.yaml`).
# Single search:
#   target: emflow://gefcom2014:solar
#   model: opus
#   budget: 2h
# Or several, with per-search parameters:
# problems:
#   - target: emflow://gefcom2014:solar
#     model: opus
#     budget: 2h
#   - target: emflow://gefcom2014:wind
#     budget: 1h
"""


def scaffold_hillclimb_dir(root: Path, *, example: bool = True) -> Path:
    """Create `<root>/hillclimb/` with config, the example problem, an
    example spec, and gitignore entries for runs/ and the .env that carries
    provider keys. `example=False` (what `problem get` does when it has to
    create the dir) leaves out the example problem and spec: the user asked
    for one bundled problem, not a scaffold. Idempotent on the folder
    layout; never overwrites an existing config."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE

    folder = root / MARKER_DIR
    for sub in ("problems", "specs", "runs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(INIT_CONFIG)
    if example:
        (folder / "specs" / "example.yaml").write_text(INIT_SPEC_EXAMPLE)
        example_dir = folder / "problems" / "example"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "problem.yaml").write_text(INIT_PROBLEM_YAML)
        (example_dir / "description.md").write_text(INIT_PROBLEM_DESCRIPTION)
        (example_dir / "verifier.sh").write_text(INIT_PROBLEM_VERIFIER)
        (example_dir / "verifier.sh").chmod(0o755)
    gitignore = root / ".gitignore"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    present = existing_ignore.splitlines()
    missing = [
        line
        for line in (f"{MARKER_DIR}/runs/", f"{MARKER_DIR}/.env")
        if line not in present
    ]
    if missing:
        gitignore.write_text(
            existing_ignore.rstrip("\n")
            + ("\n" if existing_ignore else "")
            + "\n".join(missing)
            + "\n"
        )
    return folder


problem_app = typer.Typer(
    cls=HillclimbGroup,
    help="Ready-made verifier problems shipped with hillclimb.",
    no_args_is_help=True,
)
app.add_typer(problem_app, name="problem")


def _compact_duration(seconds: int) -> str:
    """A problem budget in its shortest exact CLI spelling."""
    if seconds <= 0:
        return "-"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@problem_app.command("list")
def problem_list():
    """List every problem bundled with hillclimb."""
    import yaml

    from hillclimb.demo import BUNDLED_PROBLEM_IDS, demo_problem_resource

    rows = []
    for problem_id in BUNDLED_PROBLEM_IDS:  # ladder order, as declared
        resource = demo_problem_resource(problem_id) / "problem.yaml"
        metadata = yaml.safe_load(resource.read_text()) or {}
        rows.append((
            problem_id,
            str(metadata.get("metric", "-")),
            "maximize" if metadata.get("higher_is_better", True) else "minimize",
            _compact_duration(int(metadata.get("time_budget_s", 0))),
            _best_known(metadata.get("chart_baselines") or {}, bool(metadata.get("higher_is_better", True))),
        ))

    headers = ("problem", "metric", "direction", "budget", "best known")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    typer.echo("  ".join(f"{header:<{widths[index]}}" for index, header in enumerate(headers)).rstrip())
    for row in rows:
        typer.echo("  ".join(f"{cell:<{widths[index]}}" for index, cell in enumerate(row)).rstrip())
    typer.echo("\nGet one with: hillclimb problem get <problem>")


def _best_known(chart_baselines: dict, higher_is_better: bool) -> str:
    """The frontier a chart draws, as `value who`: the best of the declared
    reference lines in the metric's direction (never the problem's own floor)."""
    lines = {label: value for label, value in chart_baselines.items() if label != "baseline"}
    if not lines:
        return "-"
    label, value = (max if higher_is_better else min)(lines.items(), key=lambda item: item[1])
    if label.startswith("best known"):  # "best known (who)" -> "who"
        label = label[len("best known"):].strip(" ()")
    return f"{value:.6g}  {label}".rstrip()


@problem_app.command("get")
def problem_get(
    problem: str = typer.Argument(
        "circle-packing",
        help="A bundled problem id (see: hillclimb problem list)",
    ),
):
    """Copy a ready-made problem into hillclimb/problems/.

    Creates the hillclimb/ dir here if there is none, then copies the
    problem's files in and lists them — read them before you run: the
    verifier IS the problem. An existing folder is never overwritten.
    """
    from hillclimb.demo import BUNDLED_PROBLEM_IDS, install_demo_problem
    from hillclimb.project import MARKER_DIR, find_hillclimb_dir

    if problem not in BUNDLED_PROBLEM_IDS:
        available = ", ".join(BUNDLED_PROBLEM_IDS)
        typer.echo(f"error: no bundled problem {problem!r} (available: {available})", err=True)
        raise typer.Exit(1)
    if find_hillclimb_dir() is None:
        say(f"[head]No hillclimb dir here.[/] {_m(problem)} needs one: a [path]hillclimb/[/] folder holding")
        say("[path]config.yaml[/], [path]problems/[/] [note](where the problem goes)[/] and [path]runs/[/] [note](where searches land)[/].")
        if _stdin_is_tty() and not typer.confirm(f"Create {Path.cwd() / MARKER_DIR}?", default=True):
            say("Not created. Run [cmd]hillclimb init[/] where you want it, then [cmd]hillclimb problem get[/] again.")
            raise typer.Exit(1)
        folder = scaffold_hillclimb_dir(Path.cwd(), example=False)
        say(f"Created [path]{_m(folder)}[/] [note](config.yaml, problems/, runs/)[/]")
    config = load_config()
    problem_dir, created = install_demo_problem(config.paths.problems_dir, problem)
    verb = "Fetched" if created else "Already have"
    say(f"[head]{verb} {_m(problem)}[/] at [path]{_m(problem_dir)}[/]")
    legend([(name, what) for name, what in PROBLEM_FILES if (problem_dir / name).exists()])
    next_steps([
        (f"hillclimb verify {problem}", "scores the floor; the spread it prints is the noise"),
        (f"hillclimb run {problem} --budget 10m", "then climb"),
    ])


@app.command(hidden=True)
def intro(
    reset: bool = typer.Option(
        False, "--reset", help="Forget that the intro played; it plays again on the next run"
    ),
):
    """Replay the first-run 3D intro animation."""
    from hillclimb.tui.intro import intro_marker_path, play_intro

    if reset:
        intro_marker_path().unlink(missing_ok=True)
        typer.echo("Intro re-armed: it plays on the next hillclimb command.")
        return
    play_intro()


@app.command(hidden=True)
def fetch(
    problem: str = typer.Argument(
        "circle-packing",
        help="A bundled problem id (see: hillclimb problem list)",
    ),
):
    """Deprecated spelling of `hillclimb problem get`."""
    typer.echo("note: `hillclimb fetch` is now `hillclimb problem get`", err=True)
    problem_get(problem)


def _stdin_is_tty() -> bool:
    """Whether there is a person to ask; a script or a pipe gets the default."""
    return sys.stdin.isatty()


PROBLEM_FILES = (
    ("problem.yaml", "metric, direction, budget — the problem's identity"),
    ("description.md", "what the agents read before drafting"),
    ("contract.md", "the interface solution.py must implement"),
    ("interface.py", "the output format, machine-checked (hillclimb spaces)"),
    ("verifier.sh", "the ONLY process hillclimb starts: drives solution.py and reports the score"),
    ("verify.py", "the scorer — writes the score to $HILLCLIMB_RESULT"),
    ("baseline.py", "the starting solution scored at t=0"),
    ("sample_submission.csv", "a valid, weak submission: the floor the search starts from"),
    ("requirements.txt", "the solution venv"),
)


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to create the hillclimb/ dir"),
    force: bool = typer.Option(False, "--force", help="Create one even inside an existing hillclimb dir"),
):
    """Create a hillclimb dir.

    A hillclimb/ folder holding config, problems, run specs, and runs.
    """
    from hillclimb.project import MARKER_DIR, MARKER_FILE, find_hillclimb_dir

    root = directory.resolve()
    existing = find_hillclimb_dir(root)
    if existing is not None and not force:
        where = "This already has" if existing.parent == root else f"{existing.parent} already has"
        typer.echo(
            f"{where} a hillclimb dir ({existing / MARKER_FILE} exists). "
            "Next: hillclimb verify example — or --force to nest another one here.",
            err=True,
        )
        raise typer.Exit(1)
    folder = scaffold_hillclimb_dir(root)
    say(f"[head]Initialized hillclimb dir[/] at [path]{_m(folder)}[/]")
    legend([
        (f"{MARKER_DIR}/{MARKER_FILE}", "config (edit defaults here)"),
        (f"{MARKER_DIR}/problems/", "problem definitions (example/ is a working one)"),
        (f"{MARKER_DIR}/specs/", "committed run specs"),
        (f"{MARKER_DIR}/runs/", "search artifacts (gitignored)"),
    ])
    next_steps([
        ("hillclimb connect", "which agent runs the operators, and who pays"),
        ("hillclimb problem get heilbronn-11", "a bundled problem; `problem list` shows them all"),
        ("hillclimb verify example", "or your own: edit problems/example, then run it"),
    ])


@app.command()
def verify(
    target: str = typer.Argument(..., help="Problem folder/name to check"),
    solution: Path = typer.Option(
        None, "--solution", help="solution.py to score (default: the problem's baseline)"
    ),
    repeat: int = typer.Option(1, "--repeat", "-n", help="Score it N times to see the noise"),
    holdout: bool = typer.Option(False, "--holdout", help="Also score the hidden split"),
):
    """Run a problem's verifier once, outside a search.

    The fastest way to check a new `verifier.sh`: it reports the score the
    engine would climb on, and with `--repeat` how much that score moves
    between identical runs — an improvement smaller than that spread is noise,
    not progress.
    """
    import statistics
    import tempfile

    from hillclimb.harness.dirs import create_candidate_dir

    config = load_config()
    problem = load_problem(target, config)
    source = solution.read_text() if solution else problem.baseline_text
    floor_files = {} if solution else problem.baseline_files
    if source is None and floor_files:
        # the floor is a set of files scored as they are (heilbronn's
        # sample_submission.csv), so the "solution" has nothing to do
        source = "# the problem's declared floor: its baseline_files, scored as they are\n"
    if source is None:
        typer.echo(
            f"{problem.problem_id} ships no baseline — pass --solution <file> to score one",
            err=True,
        )
        raise typer.Exit(1)
    scores: list[float] = []
    with tempfile.TemporaryDirectory(prefix="hillclimb-verify-") as tmp:
        root = Path(tmp)
        from hillclimb.harness.unit_tests import freeze_for_run

        problem.unit_tests = freeze_for_run(problem, root)
        executor = build_executor(config, problem)
        test_runner = build_unit_test_runner(config, problem)
        say(f"[head]{_m(problem.problem_id)}[/]: [path]{_m(' '.join(problem.verifier_cmd))}[/]")
        for index in range(max(1, repeat)):
            candidate_dir = create_candidate_dir(
                root, f"v{index}", problem.data_dir, problem.problem_dir,
                unit_tests_dir=(problem.unit_tests.root if problem.unit_tests else None),
            )
            script = candidate_dir / "solution.py"
            script.write_text(source)
            for dest, src in floor_files.items():
                shutil.copy2(src, candidate_dir / dest)
            # distinct seeds, exactly as the engine's repeated trials run, so
            # the floor reported here is the one the search will face
            result = executor.execute(
                script, candidate_dir, config.budget.exec_timeout_s,
                seed=index if repeat > 1 else None,
            )
            if not result.ok:
                reason = "timed out" if result.timed_out else f"exit {result.returncode}"
                if result.val_score is None and not result.timed_out:
                    reason += "; no score in eval_result.json"
                typer.echo(f"  run {index}: FAILED ({reason})", err=True)
                typer.echo(f"  logs: {result.stdout_path}", err=True)
                raise typer.Exit(1)
            scores.append(result.val_score)
            say(f"  run {index}: {_m(problem.metric_name)} = [head]{result.val_score:.6g}[/]")
            if index == 0 and test_runner is not None:
                remaining = config.budget.exec_timeout_s - result.duration_s
                if remaining <= 0:
                    typer.echo("  unit tests: BUGGY (no execution time remaining)", err=True)
                    raise typer.Exit(1)
                test_result = test_runner.run(script, candidate_dir, remaining)
                if test_result.timed_out:
                    typer.echo("  unit tests: BUGGY (timed out)", err=True)
                    raise typer.Exit(1)
                if test_result.returncode is None or test_result.returncode < 0:
                    typer.echo(
                        f"  unit tests: BUGGY (crashed with {test_result.returncode})",
                        err=True,
                    )
                    raise typer.Exit(1)
                if not test_result.passed:
                    typer.echo(
                        f"  unit tests: FAILING (exit {test_result.returncode}); "
                        f"logs: {candidate_dir / 'tests_stdout.log'}",
                        err=True,
                    )
                    raise typer.Exit(1)
                typer.echo("  unit tests: PASSING")
            if index == 0 and problem.interface_path:
                # authoring lint: the baseline the verifier just accepted must
                # also satisfy the declared interface — the two drifting apart
                # is exactly the bug this catches
                from hillclimb import spaces

                module = spaces.load_interface(problem.interface_path)
                violations = module.output.check(candidate_dir) if getattr(
                    module, "output", None
                ) else []
                if violations:
                    for violation in violations:
                        typer.echo(f"  interface: {violation}", err=True)
                    raise typer.Exit(1)
                say("  interface: [ok]OK[/]")
            if holdout:
                scorer = build_holdout_scorer(config, problem, root)
                if scorer is None:
                    typer.echo("  holdout: not configured for this problem")
                else:
                    value, error, _cpu = scorer.score(candidate_dir)
                    typer.echo(f"  holdout: {error if error else format(value, '.6g')}")
    if len(scores) > 1:
        centre = statistics.median(scores)
        mad = statistics.median([abs(value - centre) for value in scores])
        spread = max(scores) - min(scores)
        say(
            f"\n{len(scores)} runs: median [head]{centre:.6g}[/], spread [head]{spread:.6g}[/], "
            f"noise floor (MAD) [head]{mad:.6g}[/]"
        )
        if mad == 0:
            say("[ok]deterministic across runs — any improvement is real[/]")
            return
        typer.echo(
            f"an improvement smaller than ~{2 * mad:.3g} cannot be told from noise. "
            "To stop the search climbing it:"
        )
        typer.echo(f"  evaluation:\n    n_replicates: {max(3, repeat)}\n    noise_k: 2")
        typer.echo(
            "  add `replicate_mode: serial` if this metric measures the machine "
            "(time, throughput, memory) — parallel trials would measure each other"
        )



store_app = typer.Typer(
    cls=HillclimbGroup,
    help="The record store behind the cross-run views (`store.backend` in config.yaml: files | sqlite)",
)
app.add_typer(store_app, name="store")


@store_app.command("sync")
def store_sync():
    """Import the hillclimb folder's searches into the configured store.

    Searches the store already holds are left alone, so this is safe to
    repeat. Run it after switching `store.backend` to `sqlite` so history
    written as files shows up in the chart and best-ever views.
    """
    from hillclimb.harness.store import FileDataStore, open_store, sync_store

    config = load_config()
    if config.store.backend == "files":
        typer.echo("store.backend is `files`: the hillclimb folder is the store, nothing to import")
        return
    store = open_store(config)
    try:
        counts = sync_store(FileDataStore(config.paths.runs_dir), store)
    finally:
        store.close()
    typer.echo(
        f"imported {counts['runs']} run(s), {counts['searches']} search(es), "
        f"{counts['records']} journal record(s) into {config.store.sqlite_path}"
    )


@store_app.command("searches")
def store_searches(
    problem: str | None = typer.Option(None, "--problem", help="Only searches on this problem key"),
):
    """List the searches the store knows, best score per search."""
    from hillclimb.harness.direction import better
    from hillclimb.harness.store import open_store

    config = load_config()
    store = open_store(config)
    try:
        records = store.searches(problem_key=problem)
        if not records:
            typer.echo("no searches recorded" + (f" for {problem}" if problem else ""))
            return
        typer.echo(f"{'search':40} {'problem':24} {'state':8} {'best':>12}  run")
        for record in records:
            best = None
            for cand in Journal(store.journal(record.key)).candidates.values():
                if cand.pruned or cand.val_score is None:
                    continue
                if best is None or better(cand.val_score, best, record.meta.higher_is_better):
                    best = cand.val_score
            shown = f"{best:.6g}" if best is not None else "-"
            typer.echo(
                f"{record.ref:40} {record.meta.problem_key:24} {record.state:8} {shown:>12}  {record.run_name}"
            )
    finally:
        store.close()


knowledge_app = typer.Typer(cls=HillclimbGroup, help="Cross-search learning: cards distilled from finished searches")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("backfill")
def knowledge_backfill():
    """Distill cards from every finished search that lacks one.

    Walks runs/ and bootstraps learning from pre-existing history.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.knowledge import distill_card, write_card
    from hillclimb.harness.run import load_search_meta

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    written = 0
    store = open_store(config)
    for record in store.searches():
            meta, search_dir = record.meta, record.search_dir
            # any finished search teaches something — parked and stopped
            # searches included; "unknown" covers pre-upgrade status files
            if record.state == "running":
                continue
            journal = Journal(store.journal(record.key))
            if not journal.scored_candidates():
                continue
            problem = SimpleNamespace(
                problem_id=meta.problem_id,
                metric_name=meta.metric,
                higher_is_better=meta.higher_is_better,
            )
            target = meta.problem if meta.problem.startswith("emflow://") else ""
            card = distill_card(
                journal, problem=problem, run_ref=search_ref(search_dir),
                target=target, budget_s=meta.budget_s,
                selection=config.holdout.selection,
            )
            path = write_card(knowledge_dir, card)
            written += 1
            typer.echo(f"  {search_ref(search_dir)} -> {path.relative_to(knowledge_dir)}")
    typer.echo(f"{written} knowledge card(s) written to {knowledge_dir}")


@knowledge_app.command("live")
def knowledge_live(run: str = typer.Argument("latest", help="Run id, or `latest`")):
    """Show the live cards concurrent searches in a run are sharing.

    The discoveries a sibling's next operator would receive.
    """
    from hillclimb.modules.memory.knowledge import load_live_cards, render_live_experience

    config = load_config()
    runs_dir = config.paths.runs_dir
    store = open_store(config)
    if run == "latest":
        latest = latest_search(store)
        if latest is None:
            typer.echo(f"No searches found in {runs_dir}", err=True)
            raise typer.Exit(1)
        run_dir = latest.search_dir.parents[1]
    else:
        run_dir = runs_dir / run
        if not any(r.run_id == run for r in store.runs()):
            raise typer.BadParameter(f"No run named {run!r} in {runs_dir}")
    cards = load_live_cards(run_dir)
    if not cards:
        typer.echo(f"no live cards under {run_dir / 'knowledge'}")
        raise typer.Exit(0)
    typer.echo(f"Run {run_dir.name} — {len(cards)} live card(s)")
    for card in cards:
        val = f"{card.selected_val:.5g}" if card.selected_val is not None else "-"
        typer.echo(
            f"  {card.run_ref}: {card.n_ok} passing / {card.n_failing} failing / "
            f"{card.n_buggy} buggy of "
            f"{card.n_candidates}, best {card.metric or 'score'} {val}"
        )
    typer.echo("")
    typer.echo(render_live_experience(cards, max_cards=len(cards)))


@knowledge_app.command("show")
def knowledge_show(target: str = typer.Argument(..., help="Problem target, e.g. emflow://gefcom2014:solar")):
    """Render the prior experience a new search would receive.

    Scoped to this target.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.knowledge import load_cards, problem_family, render_prior_experience

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    problem = load_problem(target, config)
    cards = load_cards(
        knowledge_dir, problem_id=problem.problem_id,
        family=problem_family(problem.problem_id, str(target)),
    )
    if not cards:
        typer.echo(f"no knowledge cards for {problem.problem_id} in {knowledge_dir}")
        raise typer.Exit(0)
    typer.echo(render_prior_experience(cards, max_cards=config.learning.max_cards))


@knowledge_app.command("distill")
def knowledge_distill(
    search: str = typer.Argument(
        "latest", help="Search ref (run-id/search-id), or `latest`"
    ),
    backfill: bool = typer.Option(
        False, "--backfill", help="Extract claims for every knowledge card that has none"
    ),
):
    """Run the LLM claims pass on a finished search.

    Distills typed claims (entities, concepts) — or, with `--backfill`,
    extracts them across existing cards.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.claims import distill_claims, distill_claims_from_card
    from hillclimb.modules.memory.knowledge import SCHEMA_VERSION, KnowledgeCard, distill_card, write_card

    import yaml as _yaml

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)

    if backfill:
        distilled = 0
        for path in sorted(knowledge_dir.glob("*/*.yaml")):
            try:
                data = _yaml.safe_load(path.read_text()) or {}
                if data.get("schema_version") != SCHEMA_VERSION or data.get("claims"):
                    continue
                card = KnowledgeCard.model_validate(data)
            except Exception:  # noqa: BLE001
                continue
            work_dir = knowledge_dir / ".distill" / path.stem
            card.claims = distill_claims_from_card(
                card, work_dir=work_dir, knowledge_dir=knowledge_dir,
                config=config, log=typer.echo,
            )
            if card.claims:
                write_card(knowledge_dir, card)
                distilled += 1
                typer.echo(f"  {path.relative_to(knowledge_dir)}: {len(card.claims)} claim(s)")
        typer.echo(f"{distilled} card(s) backfilled with claims")
        return

    store = open_store(config)
    if search == "latest":
        record = latest_search(store)
        if record is None:
            typer.echo(f"No searches found in {config.paths.runs_dir}", err=True)
            raise typer.Exit(1)
    else:
        run_id, _, search_id = search.partition("/")
        record = store.search((run_id, search_id))
        if record is None:
            raise typer.BadParameter(f"No search at {store.search_dir((run_id, search_id))}")
    meta, search_dir = record.meta, record.search_dir
    journal = Journal(store.journal(record.key))
    if not journal.scored_candidates():
        typer.echo("search has no scored candidates — nothing to distill", err=True)
        raise typer.Exit(1)
    problem = SimpleNamespace(
        problem_id=meta.problem_id,
        metric_name=meta.metric,
        higher_is_better=meta.higher_is_better,
    )
    target = meta.problem if meta.problem.startswith("emflow://") else ""
    card = distill_card(
        journal, problem=problem, run_ref=search_ref(search_dir),
        target=target, budget_s=meta.budget_s,
        selection=config.holdout.selection,
    )
    card.claims = distill_claims(
        journal, problem=problem, card=card, search_dir=search_dir,
        knowledge_dir=knowledge_dir, config=config, log=typer.echo,
    )
    path = write_card(knowledge_dir, card)
    typer.echo(f"{len(card.claims)} claim(s) -> {path}")


@knowledge_app.command("query")
def knowledge_query(
    terms: str = typer.Argument(..., help="Keywords, e.g. 'gradient boosting' or a technique slug"),
    family: str = typer.Option("", "--family", help="Restrict claims to one problem family"),
    limit: int = typer.Option(5, "--limit"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Read-only memory lookup (no model calls).

    Also advertised to operator agents so they can consult accumulated
    knowledge mid-search.
    """
    import json as _json

    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import load_or_build_graph, query_graph, render_query_hits

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    hits = query_graph(load_or_build_graph(knowledge_dir), terms, family=family, limit=limit)
    if as_json:
        typer.echo(_json.dumps(hits, indent=1))
    else:
        typer.echo(render_query_hits(hits))


@knowledge_app.command("consolidate")
def knowledge_consolidate(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would generalize / which playbooks would rewrite; no writes, no agent calls"
    ),
):
    """The sleep phase: generalize claims, rewrite playbooks.

    Lifts multi-family claims up the concept hierarchy (mechanical) and
    rewrites per-concept playbooks (one agent call per qualifying concept,
    routing key `consolidate`). Playbook rewrites land as reviewable git
    diffs.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.consolidate import consolidate

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    summary = consolidate(knowledge_dir, config, typer.echo, dry_run=dry_run)
    verb = "would generalize" if dry_run else "generalized"
    typer.echo(f"{verb} {len(summary['generalized'])} claim(s)")
    if dry_run:
        typer.echo(
            "playbook candidates: " + (", ".join(summary["playbook_concepts"]) or "(none)")
        )
    else:
        typer.echo(f"{len(summary['playbooks_written'])} playbook(s) written")


@knowledge_app.command("rebuild")
def knowledge_rebuild():
    """Force-rebuild knowledge/graph.json from cards and registries.

    The graph is a derived index — always safe to rebuild, never hand-edit.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import graph_path, graph_stats, rebuild_graph

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    graph = rebuild_graph(knowledge_dir)
    typer.echo(f"rebuilt {graph_path(knowledge_dir)}")
    typer.echo(graph_stats(graph))


paper_app = typer.Typer(
    cls=HillclimbGroup,
    help="Distill PDF papers into knowledge claims that seed future searches",
)
app.add_typer(paper_app, name="paper")


def _paper_knowledge_dir() -> tuple[Config, Path]:
    from hillclimb.api import resolve_knowledge_dir

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    return config, knowledge_dir


@paper_app.command("add")
def paper_add(
    pdfs: list[Path] = typer.Argument(..., help="PDF paper(s) to distill into claims"),
    problem: str = typer.Option(
        None, "--problem", help="Scope the claims: emflow://pkg:name or a local problem id. "
        "Omitted, claims are global and reach searches through concept overlap only"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-distill even when this exact PDF was already ingested"
    ),
):
    """Distill papers into typed claims and rebuild the knowledge graph.

    One agent pass per paper (routing key `paper`, default model sonnet)
    writes knowledge/papers/<slug>.yaml; the claims then ride the normal
    retrieval and credit paths — inspect the wiring with `hillclimb graph`
    before starting a run.
    """
    from hillclimb.modules.memory.graph import rebuild_graph
    from hillclimb.modules.memory.papers import distill_paper

    config, knowledge_dir = _paper_knowledge_dir()
    ingested = 0
    for pdf in pdfs:
        if not pdf.exists():
            typer.echo(f"no such file: {pdf}", err=True)
            raise typer.Exit(1)
        record = distill_paper(
            config, knowledge_dir, pdf, problem=problem, force=force, log=typer.echo
        )
        if record is not None:
            ingested += 1
    if ingested:
        rebuild_graph(knowledge_dir)
        typer.echo("knowledge graph rebuilt")
    if ingested < len(pdfs):
        raise typer.Exit(1)


@paper_app.command("list")
def paper_list():
    """Ingested papers: slug, scope, claim count, and ingestion date."""
    from hillclimb.modules.memory.papers import load_papers

    _config, knowledge_dir = _paper_knowledge_dir()
    papers = load_papers(knowledge_dir)
    if not papers:
        typer.echo("no papers ingested yet — add one with `hillclimb paper add <pdf>`")
        return
    for paper in papers:
        scope = paper.problem_id or paper.family or "global"
        title = f"  {paper.title!r}" if paper.title else ""
        typer.echo(
            f"{paper.slug}  [{scope}]  {len(paper.claims)} claim(s)  "
            f"added {paper.added_at[:10]}{title}"
        )


@knowledge_app.command("graph")
def knowledge_graph(
    stats: bool = typer.Option(False, "--stats", help="Print index stats instead of the TUI"),
):
    """Explore the knowledge graph.

    Default: the interactive TUI screen (zoom/pan/click, time scrubber);
    `--stats` prints a text summary.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.modules.memory.graph import graph_stats, load_or_build_graph

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    if stats:
        typer.echo(graph_stats(load_or_build_graph(knowledge_dir)))
        return
    try:
        from hillclimb.tui.graphview import GraphApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb knowledge graph` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc
    GraphApp(config).run()


experiment_app = typer.Typer(
    cls=HillclimbGroup, help="Compare setups: named arms of config overrides × repeats on a problem"
)
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def experiment_run(
    spec: str = typer.Argument(..., help="Spec YAML path, or a name under hillclimb/experiments/"),
    budget: str = typer.Option(None, "--budget", help="Per-search budget, e.g. 10m (overrides the spec)"),
    repeats: int = typer.Option(None, "--repeats", help="Override the spec's repeat count"),
    parallel: bool = typer.Option(
        None, "--parallel/--sequential",
        help="Launch every search detached at once, or one after another (default: the spec's schedule)",
    ),
    max_concurrent: int = typer.Option(
        None, "--max-concurrent", min=1,
        help="Parallel, but at most N searches alive at once: launch detached in job order, "
        "wait on the children before starting the next (overrides the spec's max_concurrent)",
    ),
    run_id: str = typer.Option(
        None, "--run-id",
        help="Append the searches to this existing experiment run instead of creating a new one",
    ),
    first_repeat: int = typer.Option(
        1, "--first-repeat", min=1,
        help="Number the repeats from K (with --run-id: add repeats K.. to a finished run)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the jobs, start nothing"),
):
    """Run an experiment: every arm × every problem × N repeats.

    Sequential (the default) runs jobs in a fair order — repeat by repeat,
    arms round-robin inside — so shared state such as cross-search memory
    is seen by every arm at the same point; use it whenever an arm touches
    shared state. Parallel launches all jobs detached at once (machine
    slots still cap concurrency) — fine for stateless comparisons such as
    policy or model. Bounded parallel (--max-concurrent N, or the spec's
    max_concurrent) keeps at most N alive so no search spends its wall
    clock waiting for a machine slot; the launcher stays up until the last
    child exits, then prints the report — run it under nohup or in tmux.
    Unlike sequential it runs the whole matrix rather than stopping at the
    first failure; its exit code is 1 if any child failed, else 2 if any
    parked, else 0. Real agent runs — the repeat count is your cost dial.
    """
    from hillclimb.experiment import expand, load_experiment, resolve_experiment_path, resolved_seed

    config = load_config()
    try:
        spec_path = resolve_experiment_path(spec, config.hillclimb_dir)
        experiment = load_experiment(spec_path)
        seed_path = resolved_seed(experiment)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if repeats is not None:
        experiment = experiment.model_copy(update={"repeats": repeats})
    for problem_target in experiment.problems:
        if resolve_target(problem_target, config).kind == "suite":
            raise typer.BadParameter(f"experiments take problems, not suites ({problem_target!r})")
    jobs = expand(experiment, first_repeat)
    if parallel is False and max_concurrent is not None:
        raise typer.BadParameter("--sequential and --max-concurrent contradict each other")
    schedule = "parallel" if parallel else "sequential" if parallel is False else experiment.schedule
    limit = max_concurrent if max_concurrent is not None else experiment.max_concurrent
    if max_concurrent is not None:
        schedule = "parallel"  # the flag implies detached launches
    if schedule != "parallel":
        limit = None
    child_budget = budget or experiment.budget
    repeats_text = (
        f"{experiment.repeats} repeat(s)" if first_repeat == 1
        else f"repeats {first_repeat}..{first_repeat + experiment.repeats - 1}"
    )
    typer.echo(
        f"Experiment {experiment.name}: {len(experiment.arms)} arms × {len(experiment.problems)} "
        f"problem(s) × {repeats_text} = {len(jobs)} searches, {schedule}"
        + (f" (at most {limit} at once)" if limit else "")
    )
    if seed_path is not None:
        import hashlib

        digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()[:12]
        typer.echo(f"  shared seed: {seed_path} (sha256 {digest})")
    for job in jobs:
        settings = ", ".join(f"{k}={v}" for k, v in job.overrides.items()) or "(defaults)"
        typer.echo(f"  {job.index:2d}. {job.problem} · {job.arm} · r{job.repeat}  {settings}")
    if dry_run:
        return
    if run_id:
        run_dir = config.paths.runs_dir / run_id
        existing = load_run_meta(run_dir)
        if existing is None or existing.kind != "experiment":
            raise typer.BadParameter(f"--run-id {run_id!r} is not an existing experiment run")
        run_name = existing.name
        typer.echo(f"Appending to run {run_id}")
    else:
        run_name = experiment.name
        run_id = new_run_id(run_name)
        run_dir = create_run(
            config,
            RunMeta(
                run_id=run_id, name=run_name, kind="experiment", target=spec,
                spec=_spec_provenance(config, spec_path),
                problem_ids=list(dict.fromkeys(load_problem(p, config).problem_id for p in experiment.problems)),
            ),
        )
    launched = []
    alive: dict[int, tuple[subprocess.Popen, str]] = {}  # bounded parallel: pid -> (child, slug)
    exits: dict[str, int] = {}  # bounded parallel: slug -> non-zero exit code
    for job in jobs:
        argv = [
            job.problem, "--run-id", run_id, "--run-name", run_name,
            "--experiment", experiment.name, "--arm", job.arm, "--repeat", str(job.repeat),
        ]
        if seed_path is not None:
            argv += ["--seed-from", str(seed_path)]
        if child_budget:
            argv += ["--budget", child_budget]
        for key, value in job.overrides.items():
            argv += ["--set", f"{key}={_set_value(value)}"]
        slug = f"{Path(job.problem).name}-{job.arm}-r{job.repeat}"
        if schedule == "parallel":
            if limit:
                _reap_until_below(alive, limit, exits, typer.echo)
            proc, log_path = _spawn_search_proc(config, run_dir, job.index, slug, argv)
            launched.append((slug, proc.pid, log_path))
            if limit:
                alive[proc.pid] = (proc, slug)
                typer.echo(f"=== {job.index}/{len(jobs)}: {slug} started (pid {proc.pid}, log {log_path})")
            continue
        typer.echo(f"=== {job.index}/{len(jobs)}: {slug} ===")
        cwd, env = _child_launch_context(config)
        result = subprocess.run([sys.executable, "-m", "hillclimb.cli", "run", *argv], cwd=cwd, env=env)
        if result.returncode != 0:
            hint = " (parked — resume it, then `experiment report`)" if result.returncode == 2 else ""
            typer.echo(f"{slug} exited {result.returncode}{hint}; stopping the experiment", err=True)
            raise typer.Exit(result.returncode)
    if schedule == "parallel" and not limit:
        typer.echo(f"Run {run_id}: launched {len(launched)} searches (`hillclimb experiment report` when done)")
        for slug, pid, log_path in launched:
            typer.echo(f"  pid={pid} {slug}  log={log_path}")
        return
    if limit:
        _reap_until_below(alive, 1, exits, typer.echo)  # drain: every child has exited
        typer.echo(f"Run {run_id}: {len(launched)} searches finished, {len(exits)} with a non-zero exit")
        for slug, code in exits.items():
            hint = "parked — resume it" if code == 2 else f"exit {code}"
            typer.echo(f"  {slug}: {hint}; log under {run_dir / 'logs'}")
    typer.echo("")
    _experiment_report_impl(config, experiment.name, "", spec_path=spec_path)
    if exits:
        raise typer.Exit(1 if any(code != 2 for code in exits.values()) else 2)


# How often the bounded experiment launcher polls its children for exits.
_REAP_POLL_S = 5.0


def _reap_until_below(
    alive: dict[int, tuple[subprocess.Popen, str]], limit: int, exits: dict[str, int], log
) -> None:
    """Block until fewer than `limit` of the detached children in `alive`
    (pid -> (child, slug)) are still running, reaping each one that exits and
    recording non-zero exit codes in `exits`. `limit=1` drains them all.
    Polls the Popen objects themselves: a dropped Popen gets reaped behind
    our back by the next subprocess call, and its exit code with it."""
    while len(alive) >= limit:
        for pid, (proc, slug) in list(alive.items()):
            code = proc.poll()
            if code is None:
                continue
            alive.pop(pid)
            log(f"    finished: {slug} (exit {code})")
            if code != 0:
                exits[slug] = code
        if len(alive) >= limit:
            time.sleep(_REAP_POLL_S)


def _set_value(value) -> str:
    """An override value as `--set` text: scalars verbatim, mappings/lists as
    JSON (which `--set` parses back as YAML)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@experiment_app.command("report")
def experiment_report(
    experiment: str = typer.Argument(None, help="Experiment name or spec (default: every experiment)"),
    problem: str = typer.Option("", "--problem", help="Filter to one problem id"),
    control: str = typer.Option(None, "--control", help="Arm to compare against (default: the first)"),
    noise_floor: float = typer.Option(
        None, "--noise-floor", help="Gap below which arms are not different (default: the spec's)"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable: the summaries as JSON (a meta-verifier reads the gaps)"
    ),
):
    """Compare the arms of an experiment.

    On the selected candidate's holdout score (val when holdout was off):
    per arm n/mean/median/spread, best-of-repeat wins, time to best and
    tokens; then every arm against the control, with the gap judged against
    the noise floor (`hillclimb verify <problem> --repeat 5` measures it).
    """
    from hillclimb.experiment import resolve_experiment_path

    config = load_config()
    spec_path = None
    if experiment:
        try:
            spec_path = resolve_experiment_path(experiment, config.hillclimb_dir)
        except FileNotFoundError:
            spec_path = None  # a name with no spec on disk: report by tag alone
    _experiment_report_impl(
        config, experiment, problem, spec_path=spec_path, control=control,
        noise_floor=noise_floor, as_json=as_json,
    )


def _experiment_report_impl(
    config: Config,
    experiment: str | None,
    problem_id: str,
    *,
    spec_path: Path | None = None,
    control: str | None = None,
    noise_floor: float | None = None,
    as_json: bool = False,
) -> None:
    from hillclimb.experiment import (
        collect_results,
        load_experiment,
        render_report,
        summaries_to_dict,
        summarize,
    )

    floors: dict[str, float | None] = {}
    name = experiment
    if spec_path is not None:
        spec = load_experiment(spec_path)
        name = spec.name
        control = control or spec.control
        for target in spec.problems:
            try:
                pid = load_problem(target, config).problem_id
            except Exception:  # noqa: BLE001 — a problem that no longer loads still has rows
                pid = Path(target).name
            floors[pid] = spec.noise_for(pid)
    with closing(open_store(config)) as store:
        rows = collect_results(store, experiment=name, problem_id=problem_id)
    if noise_floor is not None:
        floors = {row.problem_id: noise_floor for row in rows}
    summaries = summarize(rows, control=control, noise_floor=floors)
    if as_json:
        typer.echo(json.dumps(summaries_to_dict(summaries), indent=2))
    else:
        typer.echo(render_report(summaries))


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

    config = load_config()
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

    config = load_config()
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

    config = load_config()
    config.apply_overrides(_parse_set(set_ or []))
    name = climber or policy or config.climber.ref
    params = dict(config.climber.params)  # the user's overlay; the manifest's params are the base
    base_dir = policy_base_dir(config)
    source = policy_path(name, base_dir)
    if source is not None and not source.is_file():
        raise typer.BadParameter(f"climber file not found: {source}")
    try:
        loaded = load_climber(name, base_dir)
    except ClimberLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
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
        smoke_config = load_config(backend="dummy")
        smoke_config.apply_overrides(_parse_set(set_ or []))
        smoke_config.climber.ref = name
        smoke_config.learning.enabled = False
        outcome = run_search(
            problem,
            budget_s=parse_budget(smoke_budget),
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


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


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


# The child-engine launcher lives in api.py (hosted runs use it too); these
# names stay for the call sites in this module.
_child_launch_context = child_launch_context
_spawn_search_proc = spawn_search_proc
_create_problem_run = create_problem_run


def _spawn_search(config: Config, run_dir: Path, index: int, slug: str, run_argv: list[str]) -> tuple[int, Path]:
    """`api.spawn_search_proc` for launchers that only keep the pid."""
    proc, log_path = _spawn_search_proc(config, run_dir, index, slug, run_argv)
    return proc.pid, log_path


def _read_knowledge_context(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text() or None
    except OSError as exc:
        raise typer.BadParameter(f"--knowledge-context-file: {exc}") from exc


def _execute(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    budget: BudgetManager,
    seed_from: Path | None = None,
    knowledge_context: str | None = None,
) -> None:
    """CLI shell over api.execute_search: messages + exit codes."""
    outcome = execute_search(
        config, problem, search_dir, budget, log=typer.echo, seed_from=seed_from,
        knowledge_context=knowledge_context,
    )
    ref = outcome.ref
    if outcome.state == "parked":
        typer.echo(f"\nParked: {outcome.error}")
        typer.echo(f"Resume later with: hillclimb resume {ref}")
        raise typer.Exit(2)
    if outcome.state == "stopped":
        typer.echo("\nStopped.")
        typer.echo(f"Resume with: hillclimb resume {ref}")
        raise typer.Exit(2)
    selected = outcome.selected
    if selected is not None:
        scores = f"val_score={selected.val_score}"
        if selected.holdout_score is not None:
            scores += f", holdout={selected.holdout_score:.5g}"
        say(
            f"\n[head]Done.[/] Selected candidate [head]{_m(selected.candidate_id)}[/]: {_m(scores)} "
            f"[note]({_m(problem.metric_name)}, {'higher' if problem.higher_is_better else 'lower'} is better)[/]"
        )
    else:
        say("\n[head]Done.[/] No scored solution; best/ holds the t=0 baseline.")
    # the solution is always the artifact; a submission file only exists
    # where the problem's verifier asks for one
    artifact = "solution.py"
    say(f"Best artifact: [path]{_m(search_dir / 'best' / artifact)}[/]")
    say(f"Inspect with:  [cmd]hillclimb status {_m(ref)}[/]")


def _run_problem(
    target: str,
    config: Config,
    budget: str | None,
    run_id: str | None = None,
    run_name: str | None = None,
    seed_from: Path | None = None,
    experiment: str | None = None,
    arm: str | None = None,
    repeat: int = 0,
    arm_overrides: dict | None = None,
    knowledge_context: str | None = None,
) -> None:
    if arm_overrides:
        try:
            config.apply_overrides(arm_overrides)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    problem = load_problem(target, config)
    if run_id is None:
        run_name = run_name or problem.problem_id
        run_dir = _create_problem_run(config, run_name, target, problem.problem_id)
        run_id = run_dir.name
    else:
        # suite child: the parent already wrote run.yaml
        run_name = run_name or run_id
        run_dir = config.paths.runs_dir / run_id
    total_s = parse_budget(budget) if budget else problem.time_budget_s
    search_dir = create_search(
        config, problem, run_dir, run_id, total_s, seed_from=seed_from,
        experiment=experiment, arm=arm, repeat=repeat, arm_overrides=arm_overrides,
    )
    tag = f", experiment={experiment}/{arm}" + (f" r{repeat}" if repeat else "") if experiment else ""
    typer.echo(
        f"Search {search_ref(search_dir)} (run={run_name}, problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s{tag})"
    )
    _execute(
        config, problem, search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        seed_from=seed_from,
        knowledge_context=knowledge_context,
    )


def _spec_provenance(config: Config, suite_path: Path) -> str:
    """Spec path recorded in run.yaml, relative to the folder holding the
    hillclimb dir (absolute if the spec lives outside it)."""
    if config.hillclimb_dir is not None:
        try:
            return str(suite_path.relative_to(config.hillclimb_dir.parent))
        except ValueError:
            pass
    return str(suite_path)


def _run_suite(
    target: str,
    config: Config,
    budget: str | None,
    backend: str | None,
    model: str | None,
    holdout: bool,
    name: str | None,
    climber: str | None = None,
    parallel_operators: int | None = None,
    n_replicates: int | None = None,
    seed_from: Path | None = None,
    learning: bool = True,
    set_: list[str] | None = None,
) -> None:
    resolved = resolve_target(target, config)
    if resolved.kind != "suite" or resolved.suite is None:
        raise typer.BadParameter(f"{target!r} is not a suite")
    suite = resolved.suite
    run_name = name or suite.suite_id
    run_id = new_run_id(run_name)
    problem_targets = suite_problem_targets(suite, config)
    # a problem may appear more than once (two models on one problem, say):
    # each entry is its own search, and search ids get a -2/-3 suffix
    problem_ids = list(dict.fromkeys(
        load_problem(problem_target, config).problem_id for problem_target in problem_targets
    ))
    run_dir = create_run(
        config,
        RunMeta(
            run_id=run_id,
            name=run_name,
            kind="suite",
            target=target,
            spec=_spec_provenance(config, suite.suite_path),
            problem_ids=problem_ids,
        ),
    )
    # Snapshot every suite before launching the first child search. A child
    # only reuses these run-owned bundles; it never observes later live edits.
    from hillclimb.harness.unit_tests import freeze_for_run

    for problem_target in problem_targets:
        freeze_for_run(load_problem(problem_target, config), run_dir)
    launched = []
    for index, (entry, problem_target) in enumerate(zip(suite.problems, problem_targets), 1):
        slug = Path(problem_target).name or f"problem-{index}"
        cmd = [problem_target, "--run-id", run_id, "--run-name", run_name]
        # CLI flags override the spec entry's committed values
        child_budget = budget or entry.budget
        child_backend = backend or entry.backend
        child_model = model or entry.model
        child_parallel = parallel_operators if parallel_operators is not None else entry.parallel_operators
        child_replicates = n_replicates if n_replicates is not None else entry.n_replicates
        child_seed = seed_from or entry.seed_from
        if child_budget:
            cmd += ["--budget", child_budget]
        if child_backend:
            cmd += ["--backend", child_backend]
        if child_model:
            cmd += ["--model", child_model]
        if climber:
            cmd += ["--climber", climber]
        if child_parallel is not None:
            cmd += ["--parallel-operators", str(child_parallel)]
        if child_replicates is not None:
            cmd += ["--n-replicates", str(child_replicates)]
        if child_seed:
            # spec-relative paths resolve against the spec's own directory
            seed_path = Path(child_seed)
            if not seed_path.is_absolute():
                seed_path = (suite.suite_path.parent / seed_path).resolve()
            cmd += ["--seed-from", str(seed_path)]
        if not holdout:
            cmd.append("--no-holdout")
        if not learning:
            cmd.append("--no-learning")
        for pair in set_ or []:
            cmd += ["--set", pair]
        pid, log_path = _spawn_search(config, run_dir, index, slug, cmd)
        launched.append((problem_target, pid, log_path))
    typer.echo(f"Run {run_id}: launched {len(launched)} searches")
    for problem_target, pid, log_path in launched:
        typer.echo(f"  pid={pid} {problem_target}  log={log_path}")


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    climber: list[str] = typer.Option(
        None, "--climber",
        help=(
            "The climber (default: greedy): a bundled name, a directory holding climber.yaml, "
            "or one .py file; params via config climber.params. Repeat it "
            "(--climber greedy --climber gepa) for a mixed fleet: one search per climber "
            "on the problem, under one run, each tagged as an arm"
        ),
    ),
    policy: list[str] = typer.Option(None, "--policy", hidden=True),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    learning: bool = typer.Option(
        True, "--learning/--no-learning",
        help="Cross-search memory (cards/claims injection + distillation); off = memory-blind arm",
    ),
    name: str = typer.Option(None, "--name", help="Run name shown in the TUI"),
    parallel_operators: int = typer.Option(
        None, "--parallel-operators", help="Concurrent operators per search (worker pool)"
    ),
    parallel_searches: int = typer.Option(
        1, "--parallel-searches", min=1,
        help="Independent searches on the problem at once (>1 runs them detached, in the background)",
    ),
    n_replicates: int = typer.Option(
        None, "--n-replicates", "--n-trials",
        help="Seeded runs per trial (the median is the trial's score; --n-trials is the old spelling)",
    ),
    seed_from: Path = typer.Option(
        None, "--seed-from", help="Incumbent solution.py scored as the floor candidate"
    ),
    set_: list[str] = typer.Option(
        None, "--set", help="Any config setting, dotted: --set climber.ref=openevolve --set learning.enabled=false",
    ),
    arm_set: list[str] = typer.Option(
        None, "--arm-set",
        help=(
            "A setting for one arm of a mixed fleet, ARM:KEY=VALUE: "
            "--arm-set gepa:concurrency.parallel_operators=1 (applied after --set)"
        ),
    ),
    experiment: str = typer.Option(
        None, "--experiment",
        help="Tag the search as one arm of an experiment (with --arm); names a mixed fleet's experiment",
    ),
    arm: str = typer.Option(None, "--arm", help="The arm name (with --experiment)"),
    repeat: int = typer.Option(0, "--repeat", hidden=True),
    run_id: str = typer.Option(None, "--run-id", hidden=True),
    run_name: str = typer.Option(None, "--run-name", hidden=True),
    knowledge_context_file: Path = typer.Option(
        None, "--knowledge-context-file", hidden=True,
        help="Pre-built knowledge context (markdown) injected into every operator prompt",
    ),
):
    """Start a run on a problem or a run-spec YAML."""
    config = load_config(backend=backend, model=model)
    if not holdout:
        config.holdout.enabled = False
    if not learning:
        config.learning.enabled = False
    if policy:
        typer.echo("note: `--policy` is now `--climber` (same values)", err=True)
    climbers = [*(climber or []), *(policy or [])]
    mixed = len(climbers) > 1
    arm_overrides = _parse_arm_set(arm_set or [])
    if arm_overrides and not mixed:
        raise typer.BadParameter("--arm-set needs a mixed fleet (two or more --climber)")
    single_climber = None if mixed else (climbers[0] if climbers else None)
    if single_climber is not None:
        config.climber.ref = single_climber
    if parallel_operators is not None:
        config.concurrency.parallel_operators = parallel_operators
    if n_replicates is not None:
        config.evaluation.n_replicates = n_replicates
    overrides = _parse_set(set_ or [])
    if mixed:
        if arm or run_id:
            raise typer.BadParameter("a mixed fleet names its arms itself; --arm/--run-id do not apply")
    elif (experiment is None) != (arm is None):
        raise typer.BadParameter("--experiment and --arm go together")
    resolved = resolve_target(target, config)
    if resolved.kind == "suite":
        if mixed:
            raise typer.BadParameter("a spec takes one --climber; mixed fleets run on a single problem")
        _run_suite(
            target, config, budget, backend, model, holdout, name,
            climber=single_climber, parallel_operators=parallel_operators, n_replicates=n_replicates,
            seed_from=seed_from, learning=learning, set_=set_,
        )
        return
    if mixed:
        try:
            engines = mixed_fleet(climbers, repeats=parallel_searches, arm_overrides=arm_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            backend=backend, model=model, climber=None, parallel_operators=parallel_operators,
            n_replicates=n_replicates, holdout=holdout, learning=learning, set_=set_ or [],
            seed_from=seed_from, knowledge_context_file=knowledge_context_file,
            engines=engines, experiment=experiment,
        )
        return
    if parallel_searches > 1:
        if experiment or run_id:
            raise typer.BadParameter("--parallel-searches does not combine with --experiment/--run-id")
        _run_problem_fleet(
            target, config, budget, parallel_searches, name,
            backend=backend, model=model, climber=single_climber, parallel_operators=parallel_operators,
            n_replicates=n_replicates, holdout=holdout, learning=learning, set_=set_ or [],
            seed_from=seed_from, knowledge_context_file=knowledge_context_file,
        )
        return
    _run_problem(
        target,
        config,
        budget,
        run_id=run_id,
        run_name=run_name or name,
        seed_from=seed_from,
        experiment=experiment,
        arm=arm,
        repeat=repeat,
        arm_overrides=overrides,
        knowledge_context=_read_knowledge_context(knowledge_context_file),
    )


def _run_problem_fleet(
    target: str,
    config: Config,
    budget: str | None,
    parallel_searches: int,
    name: str | None,
    *,
    backend: str | None,
    model: str | None,
    climber: str | None,
    parallel_operators: int | None,
    n_replicates: int | None,
    holdout: bool,
    learning: bool,
    set_: list[str],
    seed_from: Path | None = None,
    knowledge_context_file: Path | None = None,
    engines: list[FleetEngine] | None = None,
    experiment: str | None = None,
) -> Path:
    """CLI shell over api.run_fleet: N independent searches on one problem
    (or one per `engines` entry — a mixed fleet), each its own detached
    engine under one run. Returns the run dir."""
    fleet = run_fleet(
        target,
        config=config,
        parallel_searches=parallel_searches,
        run_name=name,
        budget=budget,
        backend=backend, model=model, climber=climber,
        parallel_operators=parallel_operators, n_replicates=n_replicates,
        holdout=holdout, learning=learning, seed_from=seed_from,
        knowledge_context_file=knowledge_context_file, overrides=set_,
        engines=engines, experiment=experiment,
        log=typer.echo,
    )
    operators = parallel_operators if parallel_operators is not None else config.concurrency.parallel_operators
    if engines:
        arms = ", ".join(dict.fromkeys(engine.arm for engine in engines))
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {len(engines)} searches ({_m(arms)}) x {operators} operators "
            f"running in the background"
        )
    else:
        say(
            f"[head]Run {_m(fleet.run_id)}[/]: {parallel_searches} searches x {operators} operators "
            f"running in the background"
        )
    say(f"Engine logs in [path]{_m(fleet.run_dir / 'logs')}[/]")
    steps = [
        ("hillclimb watch", "every agent, what it is doing, its candidate's score"),
        ("hillclimb chart", "best score so far against time"),
        ("hillclimb stop --all", "end the run; the best solution of every search stays in runs/"),
    ]
    if engines:
        steps.insert(2, (f"hillclimb experiment report {experiment or fleet.run_id}", "compare the arms"))
    next_steps(steps)
    return fleet.run_dir


def _parse_arm_set(pairs: list[str]) -> dict[str, list[str]]:
    """`ARM:KEY=VALUE` strings (the `--arm-set` flag) → arm name -> its
    `--set` pairs, validated the way `--set` is."""
    out: dict[str, list[str]] = {}
    for item in pairs:
        arm, sep, pair = item.partition(":")
        if not sep or not arm.strip() or "=" not in pair:
            raise typer.BadParameter(f"--arm-set expects ARM:KEY=VALUE, got {item!r}")
        _parse_set([pair])
        out.setdefault(arm.strip(), []).append(pair)
    return out


def _parse_set(pairs: list[str]) -> dict:
    from hillclimb.config import parse_set_overrides

    try:
        return parse_set_overrides(pairs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


RESUMABLE_STATES = ("parked", "stopped", "crashed")


def _spawn_resume(config: Config, record: SearchRecord) -> tuple[int, Path]:
    """Start a detached `hillclimb resume <ref>` engine for this search,
    logging to <run>/logs/resume-<search-id>.log. Returns (pid, log path)."""
    run_dir = record.search_dir.parents[1]
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"resume-{record.meta.search_id}.log"
    cwd, env = _child_launch_context(config)
    cmd = [sys.executable, "-m", "hillclimb.cli", "resume", record.ref]
    with log_path.open("a") as out:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid, log_path


@app.command()
def resume(
    search: str = typer.Argument("latest"),
    all_: Annotated[
        bool, typer.Option("--all", help="Resume every parked/stopped/crashed search, each detached")
    ] = False,
    detach: Annotated[
        bool,
        typer.Option("--detach", help="Resume in a detached background engine instead of the foreground"),
    ] = False,
):
    """Resume a parked or interrupted search.

    SEARCH is `<run-id>/<search-id>`, `<run-id>`, or `latest`. `--all` resumes
    everything resumable (the counterpart of `hillclimb stop --all`), each as
    its own detached engine — the pause/resume flow for changing code or env
    under a live project.
    """
    config = load_config()
    if all_:
        store = open_store(config)
        targets = [r for r in store.searches() if r.state in RESUMABLE_STATES]
        if not targets:
            typer.echo("No parked, stopped, or crashed searches to resume.")
            raise typer.Exit(1)
        for record in targets:
            pid, log_path = _spawn_resume(config, record)
            typer.echo(f"Resuming {record.ref} ({record.state}) detached: pid {pid}, log {log_path}")
        return
    store, record = open_search(config, search)
    if detach:
        pid, log_path = _spawn_resume(config, record)
        typer.echo(f"Resuming {record.ref} ({record.state}) detached: pid {pid}, log {log_path}")
        return
    meta, search_dir = record.meta, record.search_dir
    config = load_config(backend=meta.backend, model=meta.model)
    config.holdout.enabled = meta.holdout_enabled
    # the search resumes under the policy/routing it started with, not
    # whatever the live config currently says
    config.climber.ref = meta.climber
    config.climber.params = meta.climber_params
    if meta.climber_sha256 is not None:
        from hillclimb.climber import ClimberLoadError, load_climber, load_snapshot
        from hillclimb.modules.policies import policy_base_dir

        # the search resumes from the snapshot in its own folder, so an
        # edited (or deleted) live climber changes nothing — but say so
        snapshot = load_snapshot(search_dir)
        try:
            now = load_climber(meta.climber, policy_base_dir(config)).sha256
        except ClimberLoadError as exc:
            if snapshot is None:  # nothing left to resume WITH
                raise typer.BadParameter(
                    f"climber {meta.climber} is gone and this search has no snapshot of it: {exc}"
                ) from exc
            now = None
        if snapshot is None:
            typer.echo("note: this search predates climber snapshots; resuming from the live climber", err=True)
        if now is not None and now != meta.climber_sha256:
            typer.echo(
                f"note: climber {meta.climber} changed since the search started "
                f"({meta.climber_sha256[:12]} -> {now[:12]}); resuming the version it started with",
                err=True,
            )
    if meta.tuner is not None:
        config.climber.tuner = meta.tuner
    config.climber.tuner_params = meta.tuner_params
    config.routing = {op: RouteConfig(**route) for op, route in meta.routing.items()}
    problem = load_problem(meta.problem, config)
    from hillclimb.harness.unit_tests import restore_frozen

    problem = restore_frozen(
        problem,
        search_dir.parents[1],
        bundle_path=getattr(meta, "unit_tests_bundle", None),
        command=getattr(meta, "unit_tests_command", []),
        sha256=getattr(meta, "unit_tests_sha256", None),
    )
    journal = Journal(store.journal(record.key))
    spent = resume_spent_seconds(store.read_status(record.key), journal)
    typer.echo(
        f"Resuming {search_ref(search_dir)}: {len(journal.candidates)} candidates, ~{int(spent)}s spent"
    )
    _execute(
        config,
        problem,
        search_dir,
        BudgetManager(meta.budget_s, config.budget.stop_margin_s, spent_s=spent),
    )


def _load_config_or_reap_orphans(all_: bool) -> Config:
    """`load_config()`, except that `--all` with no hillclimb dir in sight
    falls back to the live engines: the dir was deleted under them, so the
    control queue is gone and the only way to stop them is by signal."""
    from hillclimb.harness.orphans import kill_engines, orphan_engines
    from hillclimb.project import HillclimbDirNotFound

    try:
        return load_config(raise_not_found=True)
    except HillclimbDirNotFound as exc:
        if not all_:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        orphans = orphan_engines()
        if not orphans:
            typer.echo(str(exc), err=True)
            typer.echo("No orphaned engines running either.")
            raise typer.Exit(1)
        typer.echo("No hillclimb/ dir found, but engines whose hillclimb dir was deleted are still running:")
        for engine in orphans:
            typer.echo(f"  pid {engine.pid}  (was {engine.hillclimb_dir})")
        forced = kill_engines(orphans)
        typer.echo(
            f"Terminated {len(orphans)} engine process group(s) with their agents and verifiers"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
        raise typer.Exit(0)


def _search_targets(config: Config, search: str, all_: bool) -> tuple[DataStore, list[SearchRecord]]:
    """The searches a stop/kill applies to: every running one, or the ref."""
    if not all_:
        store, record = open_search(config, search)
        return store, [record]
    store = open_store(config)
    running = running_searches(store)
    if not running:
        typer.echo("No running searches.")
        raise typer.Exit(1)
    return store, running


@app.command()
def ps():
    """Every process hillclimb is responsible for on this machine.

    One block per live engine (`hillclimb run`), with its agents, verifiers
    and their children nested underneath. Engines whose hillclimb dir has
    been deleted are tagged `orphan` — `hillclimb stop --all` reaps those.
    """
    from hillclimb.harness.orphans import engine_trees, process_table

    table = process_table()
    trees = engine_trees(table)
    if not trees:
        typer.echo("No hillclimb engines running.")
        return
    total = 0
    for engine, kids in trees:
        proc = table[engine.pid]
        where = str(engine.hillclimb_dir) if engine.hillclimb_dir else "?"
        tag = "  [orphan: dir deleted]" if engine.hillclimb_dir and not engine.hillclimb_dir.exists() else ""
        argv = proc.command.split("hillclimb.cli run", 1)[-1].strip()
        typer.echo(f"engine pid {proc.pid}  up {proc.elapsed}  run {argv}")
        typer.echo(f"  dir {where}{tag}")
        for kid in kids:
            role = (
                "agent" if "claude -p" in kid.command or "claude --" in kid.command
                else "verifier" if "verifier.sh" in kid.command
                else "child"
            )
            typer.echo(
                f"  {role:8} pid {kid.pid:<6} cpu {kid.cpu:5.1f}%  mem {kid.rss_mb:6.0f}M  "
                f"up {kid.elapsed:>8}  {kid.command[:70]}"
            )
        total += 1 + len(kids)
    cpu = sum(table[e.pid].cpu for e, _ in trees) + sum(k.cpu for _, kids in trees for k in kids)
    mem = sum(table[e.pid].rss_mb for e, _ in trees) + sum(k.rss_mb for _, kids in trees for k in kids)
    typer.echo(f"{len(trees)} engine(s), {total} processes, {cpu:.0f}% cpu, {mem:.0f}M rss")


@app.command()
def stop(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Stop every running search"),
):
    """Gracefully stop a running engine.

    It finishes the current operator call, then parks. Resume later with
    `hillclimb resume`. `--all` stops every running search (e.g. the demo).
    """
    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    for record in targets:
        ref = record.ref
        outcome = request_stop(store, record.key, source="cli")
        if outcome is None:
            typer.echo(f"Search {ref} is {record.state}; nothing to stop.")
            raise typer.Exit(1)
        typer.echo(f"{outcome} (use `hillclimb kill {ref}` to interrupt now)")


@app.command()
def prune(
    search: str,
    candidate_id: str,
    reason: str = typer.Option("", help="Why this branch is being cut (recorded in the journal)"),
):
    """Prune a candidate and its whole subtree.

    The engine stops building on this lineage and it is excluded from
    selection. Statuses and scores stay visible in status/tree output.
    """
    config = load_config()
    store, record = open_search(config, search)
    higher = bool(record.meta.higher_is_better)
    try:
        outcome = request_prune(
            store,
            record.key,
            candidate_id,
            higher_is_better=higher,
            selection_mode=config.holdout.selection,
            reason=reason,
            source="cli",
        )
    except ValueError as exc:
        typer.echo(f"Cannot prune: {exc}")
        raise typer.Exit(1)
    typer.echo(outcome)


@app.command()
def kill(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Kill every running search"),
):
    """SIGTERM a running engine; it can be resumed.

    State is finalized on the way out. For a graceful stop that lets the
    current operator finish, use `hillclimb stop`. `--all` kills every
    running search.
    """
    config = _load_config_or_reap_orphans(all_)
    store, targets = _search_targets(config, search, all_)
    for record in targets:
        ref = record.ref
        state = record.state
        if state != "running":
            typer.echo(f"Search {ref} is {state}; nothing to kill.")
            raise typer.Exit(1)
        engine_pid = store.read_status(record.key).pid
        os.kill(engine_pid, signal.SIGTERM)
        typer.echo(f"Sent SIGTERM to engine pid {engine_pid} ({ref}).")
        typer.echo(f"Resume with: hillclimb resume {ref}")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation"),
):
    """Kill every engine of THIS hillclimb dir and delete the dir.

    The hillclimb dir is the one found from the current directory (or
    `HILLCLIMB_DIR`). Only engines pinned to that exact dir are signalled —
    their agents and verifiers go with them — then the folder is removed.
    Searches of other folders on the machine are untouched. `runs_dir` or
    `problems_dir` configured outside the hillclimb dir are left in place and
    reported.
    """
    import shutil

    from hillclimb.harness.orphans import engines_for, kill_engines, live_engines

    config = load_config()
    root = config.hillclimb_dir
    if root is None:  # pragma: no cover - Config.load always sets it via discovery
        typer.echo("No hillclimb dir to reset.", err=True)
        raise typer.Exit(1)
    engines = live_engines()
    mine = engines_for(root, engines)
    unknown = [e for e in engines if e.hillclimb_dir is None]

    say(f"[head]Will delete[/] [path]{_m(root)}[/]")
    if mine:
        typer.echo(f"and terminate {len(mine)} engine(s) running against it (with their agents and verifiers):")
        for engine in mine:
            typer.echo(f"  pid {engine.pid}")
    else:
        typer.echo("No engines are running against it.")
    if unknown:
        typer.echo(
            f"Note: {len(unknown)} engine(s) whose hillclimb dir could not be read will be left alone: "
            + ", ".join(f"pid {e.pid}" for e in unknown)
        )
    outside = []
    for label, path in (("runs_dir", config.paths.runs_dir), ("problems_dir", config.paths.problems_dir)):
        if path.exists() and not path.resolve().is_relative_to(root.resolve()):
            outside.append((label, path))
    for label, path in outside:
        typer.echo(f"Note: {label} {path} lives outside the hillclimb dir and will be left in place.")
    if not yes and not typer.confirm("Proceed?", default=False):
        typer.echo("Aborted.")
        raise typer.Exit(1)

    if mine:
        forced = kill_engines(mine)
        typer.echo(
            f"Terminated {len(mine)} engine process tree(s)"
            + (f"; {len(forced)} needed SIGKILL." if forced else ".")
        )
    shutil.rmtree(root)
    say(f"[head]Deleted[/] [path]{_m(root)}[/]")


@app.command()
def status(search: str = typer.Argument("latest")):
    """Show the candidate tree of a search."""
    config = load_config()
    store, record = open_search(config, search)
    search_dir = record.search_dir
    journal = Journal(store.journal(record.key))
    search_status = store.read_status(record.key)
    state = record.state
    if search_status is not None:
        remaining = int(search_status.budget.remaining_s)
        line = f"state={state}  budget: {int(search_status.budget.spent_s)}s spent / {remaining}s left"
        budget = search_status.budget
        if budget.max_evaluations:
            line += f"  evaluations {budget.evaluations}/{budget.max_evaluations}"
        if budget.max_tokens:
            line += f"  tokens {budget.tokens:,}/{budget.max_tokens:,}"
        if search_status.current:
            active = " · ".join(
                f"{c.candidate_id}({c.operator}/{c.phase})" for c in search_status.current[:3]
            )
            if len(search_status.current) > 3:
                active += f" +{len(search_status.current) - 3}"
            line += f"  active: {active}"
        typer.echo(line)
    typer.echo(f"Search {search_ref(search_dir)} — {len(journal.candidates)} candidates")
    for candidate in journal.candidates.values():
        score = f"{candidate.val_score:.5f}" if candidate.val_score is not None else "-"
        hold = f" hold={candidate.holdout_score:.5f}" if candidate.holdout_score is not None else ""
        marks = (" *SELECTED*" if candidate.is_selected else "") + (
            " *best-val*" if candidate.is_best else ""
        )
        if candidate.pruned:
            marks += " *PRUNED*"
        parent = f" <- {candidate.parent_id}" if candidate.parent_id else ""
        typer.echo(
            f"  {candidate.candidate_id} {candidate.operator:<9} {candidate.status:<9} "
            f"val={score}{hold}{marks}{parent}  {candidate.summary[:70]}"
        )
    scored = [
        c
        for c in journal.candidates.values()
        if c.val_score is not None and c.holdout_score is not None
    ]
    if scored:
        gaps = [abs(c.val_score - c.holdout_score) for c in scored]
        typer.echo(
            f"val→holdout gap: mean {sum(gaps)/len(gaps):.5g}, max {max(gaps):.5g} over {len(scored)} candidates"
        )
    floor = journal.noise_floor()
    if floor is not None:
        typer.echo(
            f"noise floor: {floor:.5g} (median trial spread) — gains below "
            f"~{2 * floor:.3g} are not measurable"
        )


def _summit(config: Config, problem: str | None, dest: Path):
    """The best solution for `problem` across every run, copied into `dest`.

    Ranks each search's selected candidate (the one whose files its best/
    holds) and copies the winning search's best/ files. Reads journals as
    they are, so it is safe mid-climb. Returns (search record, candidate,
    copied file names)."""
    with closing(open_store(config)) as store:
        records = store.searches(problem_key=problem)
        if not records:
            known = sorted({r.meta.problem_key or r.meta.problem_id for r in store.searches()})
            what = f"no searches for {problem!r}" if problem else "no searches in this hillclimb dir"
            hint = f". Problems here: {', '.join(known)}" if known else ""
            raise typer.BadParameter(f"{what}{hint}.")
        keys = sorted({r.meta.problem_key or r.meta.problem_id for r in records})
        if len(keys) > 1:
            raise typer.BadParameter(
                f"several problems in this hillclimb dir ({', '.join(keys)}) — "
                "name one: hillclimb summit <problem>"
            )
        best: tuple[SearchRecord, object] | None = None
        for record in records:
            journal = Journal(store.journal(record.key))
            candidate = journal.selected_candidate(
                record.meta.higher_is_better, config.holdout.selection
            )
            if candidate is None or candidate.val_score is None:
                continue
            if best is None or (
                candidate.val_score > best[1].val_score
                if record.meta.higher_is_better
                else candidate.val_score < best[1].val_score
            ):
                best = (record, candidate)
    if best is None:
        raise typer.BadParameter(
            "no scored candidate yet — nothing to copy. Try again once the climb lands one."
        )
    record, candidate = best
    source = record.search_dir / "best"
    summit_files = list(dict.fromkeys(["solution.py", *record.meta.output_artifacts]))
    copied = [name for name in summit_files if (source / name).is_file()]
    if not copied:
        raise typer.BadParameter(f"{source} holds no solution files yet; try again in a moment.")
    for name in copied:
        shutil.copy(source / name, dest / name)
    return record, candidate, copied


@app.command()
def summit(
    problem: str = typer.Argument(
        None, help="Problem key; defaults to the only problem in the hillclimb dir"
    ),
    to: Path = typer.Option(
        None, "--to", help="Destination folder (default: the folder holding hillclimb/)"
    ),
):
    """Copy the best solution found so far next to your hillclimb/ folder.

    Ranks every search of the problem, across all runs, by its selected
    candidate and copies that search's solution.py plus its declared output
    artifacts into the destination. Run it at any point, even
    mid-climb — you always get the best discovered so far.
    """
    config = load_config()
    dest = (to or config.hillclimb_dir.parent).resolve()
    already_there = (
        {path.name for path in dest.iterdir() if path.is_file()} if dest.is_dir() else set()
    )
    dest.mkdir(parents=True, exist_ok=True)
    record, candidate, copied = _summit(config, problem, dest)
    key = record.meta.problem_key or record.meta.problem_id
    typer.echo(
        f"summit of {key}: {record.meta.metric} {candidate.val_score:.6g} — "
        f"{candidate.candidate_id} ({candidate.operator}) from {record.ref}"
    )
    for name in copied:
        verb = "refreshed" if name in already_there else "wrote"
        typer.echo(f"  {verb} {dest / name}")


@app.command()
def show(
    search: str = typer.Argument("latest", help="latest, <run-id>, or <run-id>/<search-id>"),
    candidate_id: str = typer.Argument(..., metavar="CANDIDATE", help="Candidate id, e.g. c007"),
):
    """Everything known about one candidate.

    Metadata, scores, the evaluation breakdown (the same report the improve
    operator receives), the code diff vs its parent, notes, and execution
    output.
    """
    import difflib

    from hillclimb.harness.report import candidate_report, render_delta, render_report
    from hillclimb.harness.evaluation import tail

    config = load_config()
    store, record = open_search(config, search)
    search_dir = record.search_dir
    journal = Journal(store.journal(record.key))
    cand = journal.candidates.get(candidate_id)
    if cand is None:
        known = ", ".join(journal.candidates) or "(none)"
        raise typer.BadParameter(
            f"No candidate {candidate_id!r} in {search_ref(search_dir)}; known: {known}"
        )
    meta = record.meta
    metric = meta.metric
    higher = bool(meta.higher_is_better)
    parent = journal.candidates.get(cand.parent_id) if cand.parent_id else None

    marks = [
        name
        for name, on in (
            ("SELECTED", cand.is_selected), ("best-val", cand.is_best), ("PRUNED", cand.pruned),
        )
        if on
    ]
    header = f"{cand.candidate_id}  {cand.operator}"
    if cand.complexity:
        header += f"[{cand.complexity}]"
    header += f"  status={cand.status}"
    if cand.parent_id:
        header += f"  <- {cand.parent_id}"
    if marks:
        header += f"  [{', '.join(marks)}]"
    typer.echo(header)
    if cand.summary:
        typer.echo(f"summary: {cand.summary}")
    backend = cand.backend
    if backend.name:
        agent_line = f"agent: {backend.name}"
        if backend.cost_usd is not None:
            agent_line += f", ${backend.cost_usd:.2f}"
        if backend.num_turns is not None:
            agent_line += f", {backend.num_turns} turns"
        typer.echo(agent_line)
    for trial in cand.trials:
        parts = [f"val={trial.val_score if trial.val_score is not None else '-'}"]
        if trial.params:
            parts.append("params=" + json.dumps(trial.params, sort_keys=True))
        if trial.holdout_score is not None:
            parts.append(f"holdout={trial.holdout_score:.5g}")
        if trial.holdout_error:
            parts.append(f"holdout_error={trial.holdout_error[:60]}")
        mark = "*" if trial.is_best and len(cand.trials) > 1 else ""
        typer.echo(f"trial {trial.index}{mark}: {'  '.join(parts)}")
        for replicate in trial.replicates:
            rparts = [f"val={replicate.val_score if replicate.val_score is not None else '-'}"]
            if replicate.seed is not None:
                rparts.append(f"seed={replicate.seed}")
            if replicate.duration_s is not None:
                rparts.append(f"{replicate.duration_s:.0f}s")
            if replicate.returncode not in (0, None):
                rparts.append(f"rc={replicate.returncode}")
            if replicate.timed_out:
                rparts.append("TIMEOUT")
            typer.echo(f"  replicate {replicate.seed if replicate.seed is not None else 0}: {'  '.join(rparts)}")
    if cand.tunable:
        typer.echo("tunable: yes")
    elif cand.params_error:
        typer.echo(f"tunable: no — {cand.params_error}")
    scores = f"val_score={cand.val_score}"
    if cand.holdout_score is not None:
        scores += f"  holdout={cand.holdout_score:.5g}"
    typer.echo(f"{scores}  ({metric}, {'higher' if higher else 'lower'} is better)")

    report = candidate_report(cand)
    typer.echo("\n# Evaluation breakdown (validation split)\n")
    typer.echo(
        render_report(report, metric)
        or "(no evaluation report — pre-feature candidate or non-emflow problem)"
    )
    delta = render_delta(candidate_report(parent), report, higher)
    if delta:
        typer.echo(f"\n# Where it moved vs parent {parent.candidate_id}\n")
        typer.echo(delta)

    if cand.metrics or cand.policy_meta:
        typer.echo("\n# search metadata\n")
        if cand.metrics:
            typer.echo("metrics: " + json.dumps(cand.metrics, sort_keys=True))
        if cand.policy_meta:
            typer.echo("policy_meta: " + json.dumps(cand.policy_meta, sort_keys=True))

    candidate_dir = Path(cand.candidate_dir) if cand.candidate_dir else None
    solution = candidate_dir / "solution.py" if candidate_dir else None
    if parent is not None:
        typer.echo(f"\n# solution.py diff vs {parent.candidate_id}\n")
        parent_solution = Path(parent.candidate_dir) / "solution.py" if parent.candidate_dir else None
        if solution is None or not solution.exists() or parent_solution is None or not parent_solution.exists():
            typer.echo("(candidate_dir not available on this machine)")
        else:
            diff = "".join(
                difflib.unified_diff(
                    parent_solution.read_text().splitlines(keepends=True),
                    solution.read_text().splitlines(keepends=True),
                    fromfile=f"{parent.candidate_id}/solution.py",
                    tofile=f"{cand.candidate_id}/solution.py",
                )
            )
            typer.echo(diff.rstrip() or "(identical)")
    if candidate_dir is not None and (candidate_dir / "notes.md").exists():
        typer.echo("\n# notes.md\n")
        typer.echo((candidate_dir / "notes.md").read_text().rstrip())
    if candidate_dir is not None and (candidate_dir / "exec_stdout.log").exists():
        typer.echo("\n# stdout (tail)\n")
        typer.echo(tail(candidate_dir / "exec_stdout.log").rstrip())


def _watch_app():
    try:
        from hillclimb.tui.watch import WatchApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb watch` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc
    return WatchApp


watch_app = typer.Typer(
    cls=HillclimbGroup,
    invoke_without_command=True,
    help="Live TUI: runs, searches, candidates, and candidate details.",
)
app.add_typer(watch_app, name="watch")


@watch_app.callback()
def watch(ctx: typer.Context):
    """Live TUI: runs, searches, candidates, and candidate details.

    Keys: enter=open/details, esc=close/back, +/-=resize details, s=stop
    search, x=prune candidate, g=knowledge graph, q=quit.
    `hillclimb watch candidates` opens straight on a search's candidates.
    """
    if ctx.invoked_subcommand is not None:
        return
    _watch_app()(load_config()).run()


@watch_app.command("candidates")
def watch_candidates(
    search: str = typer.Argument("latest", help="latest, <run-id>, or <run-id>/<search-id>"),
):
    """Open the TUI straight on one search's candidates.

    Esc backs out to the run's searches and the run list as usual.
    """
    config = load_config()
    search_dir = resolve_search_dir(config, search)
    _watch_app()(config, search_dir=search_dir).run()


@app.command()
def chart(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
    detail: bool = typer.Option(
        False, "--detail", "-d", help="One search only, with its exploration tree drawn on the curve"
    ),
    holdout: bool = typer.Option(
        False, "--holdout", help="Force the holdout view. Default: a holdout-scored problem "
        "opens on holdout (the split its reference baselines live on), others on validation; "
        "h toggles either way"
    ),
    cost: bool = typer.Option(
        False, "--cost", help="Overlay cumulative agent tokens and verifier CPU-minutes "
        "on right-hand axes — what the climb cost as it climbed"
    ),
):
    """Live hillclimb chart: best score so far vs tested candidates.

    With no argument and several charts to show (more than one problem, or
    the same problem in more than one run), first a table of them — one row
    per problem worked in a run, newest activity first; enter opens that
    chart, esc comes back to the table. `hillclimb watch` reaches the same
    chart with `c` from its runs, searches and candidates tables.

    One staircase across every search of the problem, every scored candidate
    a dot (bright where it set a new best, dim where it missed), plus optional
    problem-config baselines; an experiment gets one line per arm instead.
    Refreshes as candidates land.
    Keys: r=refresh, t=toggle improvement text, d=detail
    (every scored candidate as a mark, parent edges, accepted lineage bold),
    h=toggle holdout/validation, c=cost overlay, p=switch problem,
    esc=back to the table, q=quit.
    """
    try:
        from hillclimb.tui.chart import ChartApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb chart` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    # --holdout forces the view; omitted, the chart decides per problem
    ChartApp(load_config(), search, detail=detail, holdout=holdout or None, cost=cost).run()


@app.command()
def tree(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """Live exploration tree of one search: what was expanded, what was left.

    Roots across the top, one row per operator step; colour is the operator,
    silhouette is the fate (filled = expanded, ring = discontinued, diamond =
    best, dot = failed). Scroll zooms, drag pans, click a node for details,
    enter opens it in the candidate screen, j/k scrub through time, n/p switch
    search, `?` keys.
    """
    try:
        from hillclimb.tui.treeview import TreeApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    TreeApp(load_config(), search).run()


@app.command()
def tree2(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """Live archive tree of one search, drawn like the Darwin Gödel Machine's.

    Same layout as `tree`; each circle carries its candidate number (`c017` → 17) inside, is filled with its score on a viridis ramp (bright =
    best; hollow = no working solution), and ringed by what the search did
    with it: white = expanded (the spine the policy walked), no ring =
    scored and never built on, red = failed. The final best is a white
    star, and its parent chain is drawn bold in the same white. Circles never overlap: they
    are sized to the zoom, and the numbers appear as they grow. Scroll
    zooms, drag pans, click a node for details, enter opens it, j/k scrub
    through time, n/p switch search, `?` keys.
    """
    try:
        from hillclimb.tui.tree2view import Tree2App
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree2` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    Tree2App(load_config(), search).run()


@app.command()
def archive(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """Archive tree beside the progress chart, live — the Darwin Gödel
    Machine's two-panel figure for one search.

    Left, the `tree2` archive tree; right, every scored candidate as a dot
    at (candidate number, score) with the best-so-far staircase, a brighter dot
    where a candidate set a new best, and the lineage of the final best as
    a thick line — the same parent chain drawn bold in the tree, one
    circle and one dot per candidate number on both. j/k scrub both panels through
    time together (a cursor marks the candidate on the chart); click a node
    to ring its dot on the chart and see its detail, enter opens it, n/p
    switch search, `?` keys.
    """
    try:
        from hillclimb.tui.archiveview import ArchiveApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb archive` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    ArchiveApp(load_config(), search).run()


@app.command()
def surface(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """Live 3D fitness surface of one search: candidates on the problem's terrain.

    Needs a problem that ships `landscape.py` (elevation(x, y) + grid(n)) and
    journals each candidate's position as extra numeric keys next to the
    score (`surface_metrics` in problem.yaml, default x/y). Candidates are
    coloured by their tree fate, the accepted lineage is draped along the
    terrain, a white marker sits on the summit. Drag rotates, scroll zooms,
    n/p switch search, q quits.
    """
    try:
        from hillclimb.tui.surfaceview import SurfaceApp, surface_unavailable
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb surface` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    config = load_config()
    _, record = open_search(config, search)
    try:
        problem = load_problem(record.meta.problem, config)
    except Exception as exc:  # noqa: BLE001 — a moved/deleted problem dir
        typer.echo(f"Cannot load problem {record.meta.problem!r}: {exc}")
        return
    if problem.landscape_path is None:
        typer.echo(surface_unavailable(problem))
        return
    SurfaceApp(config, search).run()


class DefaultCommandGroup(HillclimbGroup):
    """A group whose bare form and any first token that is not one of its
    commands run `default_command` — `similarity <search>` is
    `similarity map <search>`, `similarity --help` still lists both."""

    default_command = "map"

    def parse_args(self, ctx, args):
        if not args or (args[0] not in self.commands and args[0] not in ("-h", "--help")):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


similarity_app = typer.Typer(
    cls=DefaultCommandGroup,
    invoke_without_command=True,
    help="Live 3D similarity views of a search's candidates: `map` (pairwise, default) and `reference`.",
)
app.add_typer(similarity_app, name="similarity")

_SIMILARITY_SEARCH = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>")
_SIMILARITY_SINGLE = typer.Option(False, "--single", help="One search only, even when it is an experiment arm")


@similarity_app.callback()
def similarity(ctx: typer.Context):
    """Live 3D similarity views of one search's candidates.

    `map` (the default) embeds every candidate by pairwise distance, so
    nearby dots are alike; `reference` places each candidate at its
    distance from one reference candidate. Both derive everything from
    what candidates already produced (the problem's fingerprint.py if it
    ships one, else submission or evaluator report; solution.py; the
    journal) and store nothing. `v` swaps between them in the TUI.
    """


@similarity_app.command("map")
def similarity_map(
    search: str = _SIMILARITY_SEARCH,
    single: bool = _SIMILARITY_SINGLE,
    metric: str = typer.Option(
        "behavioral", "--metric", "-m", help="Pairwise distance to lay out by: behavioral, structural, blend"
    ),
):
    """Candidates embedded by pairwise distance: nearby means alike.

    One 3D graph: lineage edges join parent to child, the best-so-far
    sequence is a gold trail, colour is score rank (cold to hot; the
    champion a gold diamond, the origin a white open diamond), shape the
    tree's fate. An experiment arm opens its whole run instead, coloured by
    arm like `chart` (--single for the one-search view). Keys: m cycles
    behavioral/structural/blend, v opens the reference view, space
    replays the search growing, s toggles the idle spin; hover reads a
    candidate's distances to the selected one, click dims everything
    outside its lineage. Drag rotates, scroll zooms, q quits.
    """
    _open_similarity(search, single, view="map", metric=metric)


@similarity_app.command("reference")
def similarity_reference(search: str = _SIMILARITY_SEARCH, single: bool = _SIMILARITY_SINGLE):
    """Candidates at their distance from one reference candidate.

    Every candidate sits at (behavioral, structural, lineage) distance from
    a reference candidate — the origin the search grew from (its seed, else
    its baseline) by default, `c` toggles to the current champion — coloured
    by score rank (cold to hot; the champion is gold, the reference white).
    A search that is an experiment arm opens the whole run instead: every
    search of that problem in one cube, measured from the shared seed,
    coloured by arm like `chart`, n/p stepping through the run's problems
    (--single for the one-search view). v opens the map. Drag rotates,
    scroll zooms, q quits.
    """
    _open_similarity(search, single, view="reference")


@similarity_app.command("scores")
def similarity_scores(
    search: str = _SIMILARITY_SEARCH,
    score: list[str] = typer.Option(
        None, "--score", "-s",
        help="Score to compute (repeatable): a registry name, a .py file, or module:Class. "
        "Default: similarity.scores from config",
    ),
    candidates: str = typer.Option(None, "--candidates", "-c", help="Comma-separated candidate ids (default: all)"),
    files: list[Path] = typer.Option(None, "--file", "-f", help="Compare these solution files instead of a search (repeatable)"),
    explain: bool = typer.Option(False, "--explain", help="Print each solution's representation text (e.g. its solution card)"),
    as_json: bool = typer.Option(False, "--json", help="Emit the matrices as JSON"),
    list_scores: bool = typer.Option(False, "--list", help="List the registered scores and exit"),
):
    """Pairwise similarity matrices between solutions, one per score.

    Scores are pluggable: `solution-card` (an LLM writes a method card per
    solution, cards are embedded, cosine between them — needs
    OPENROUTER_API_KEY), `api-calls` (imports + library calls, no LLM),
    `code-tokens` (token overlap), or your own `SimilarityScore` subclass in
    a .py file. 1.0 = the same; representations are cached per file content
    in ~/.cache/hillclimb/similarity/, nothing is written into the run.
    """
    from hillclimb.modules.policies import policy_base_dir
    from hillclimb.tui.similarity import dir_for
    from hillclimb.modules.similarity import (
        SimilarityUnavailable,
        Solution,
        get_score,
        registered_scores,
        similarity_matrix,
    )

    if list_scores:
        for name, cls in sorted(registered_scores().items()):
            typer.echo(f"{name:15} {cls.description}")
        return
    config = _config_or_default() if files else load_config()
    if files:
        if search:
            raise typer.BadParameter("pass a search or --file, not both")
        solutions = [Solution.from_file(path, id=sid) for path, sid in zip(files, _file_ids(files))]
    else:
        store, record = _similarity_anchor(config, search)
        journal = Journal(store.journal(record.key)).candidates
        wanted = [c.strip() for c in candidates.split(",")] if candidates else None
        if wanted:
            missing = [cid for cid in wanted if cid not in journal]
            if missing:
                raise typer.BadParameter(f"not in {record.ref}: {', '.join(missing)}")
        solutions = []
        for cid in wanted or list(journal):
            cand = journal[cid]
            solution = Solution(id=cid, dir=dir_for(record.search_dir, cand), candidate=cand)
            if solution.path.is_file():
                solutions.append(solution)
            elif wanted:
                typer.echo(f"{cid}: no solution.py — skipped", err=True)
    if len(solutions) < 2:
        raise typer.BadParameter("need at least two solutions with a solution.py to compare")

    requested = {name: config.similarity.scores.get(name, {}) for name in score} if score else config.similarity.scores
    results = []
    for name, params in requested.items():
        try:
            instance = get_score(name, params, base_dir=policy_base_dir(config))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        try:
            matrix = similarity_matrix(instance, solutions)
        except SimilarityUnavailable as exc:
            typer.echo(f"{instance.name}: unavailable — {exc}", err=True)
            continue
        explanations = {s.id: instance.explain(s) for s in solutions} if explain else {}
        results.append((matrix, explanations))

    if as_json:
        typer.echo(json.dumps([
            {**m.to_dict(), **({"explain": e} if explain else {})} for m, e in results
        ], indent=2))
        return
    for matrix, explanations in results:
        typer.echo(_format_similarity_matrix(matrix))
        for sid, text in explanations.items():
            if text:
                typer.echo(f"\n--- {sid}\n{text}")
        typer.echo("")


def _config_or_default() -> Config:
    """--file works outside a hillclimb dir: built-in defaults then."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return load_config(raise_not_found=True)
    except HillclimbDirNotFound:
        return Config()


def _file_ids(paths: list[Path]) -> list[str]:
    """Each file's trailing path, at the shortest depth that tells all of
    them apart (`c001/solution.py`, `c002/solution.py`)."""
    parts = [Path(p).resolve().parts for p in paths]
    for depth in range(1, max(len(p) for p in parts) + 1):
        tails = [str(Path(*p[-depth:])) for p in parts]
        if len(set(tails)) == len(tails):
            return tails
    return [str(p) for p in paths]


def _format_similarity_matrix(matrix) -> str:
    width = max(8, *(len(i) for i in matrix.ids)) + 1
    lines = [f"== {matrix.score}  (similarity, 1.0 = same)"]
    lines.append(" " * width + "".join(i.rjust(width) for i in matrix.ids))
    for row_id, row in zip(matrix.ids, matrix.values):
        cells = "".join(("—" if v != v else f"{v:.3f}").rjust(width) for v in row)
        lines.append(row_id.ljust(width) + cells)
    if matrix.unrepresented:
        lines.append(f"unrepresented: {', '.join(matrix.unrepresented)}")
    return "\n".join(lines)


def _open_similarity(search: str | None, single: bool, view: str, metric: str = "behavioral") -> None:
    try:
        from hillclimb.tui.similarity import build_run_similarity, build_similarity
        from hillclimb.tui.similarity_map import METRICS, build_map, build_run_map
        from hillclimb.tui.similarityview import SimilarityApp, run_inputs, search_inputs
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb similarity` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    if metric not in METRICS:
        raise typer.BadParameter(f"--metric must be one of {', '.join(METRICS)}")
    config = load_config()
    store, record = _similarity_anchor(config, search)
    meta = record.meta
    try:
        fingerprint_path = load_problem(meta.problem, config).fingerprint_path
    except Exception:  # noqa: BLE001 — a provider or moved problem: no fingerprint module
        fingerprint_path = None
    higher = bool(meta.higher_is_better)
    common = dict(output_artifacts=meta.output_artifacts, fingerprint_path=fingerprint_path)

    if meta.experiment and meta.arm and not single:
        inputs = search_inputs(store, run_inputs(store, record.run_id, meta.problem_key))
        reference: str | None = None
        if view == "map":
            unavailable = build_run_map(inputs, higher, metric=metric, problem_key=meta.problem_key, **common).unavailable
            reference = "seed" if unavailable is None else None
        else:
            for reference in ("seed", "champion"):
                unavailable = build_run_similarity(
                    inputs, higher, reference=reference, problem_key=meta.problem_key, **common,
                ).unavailable
                if unavailable is None:
                    break
            else:
                reference = None
        if reference is not None:
            SimilarityApp(
                config, reference=reference, run=(record.run_id, meta.problem_key), view=view, metric=metric,
            ).run()
            return
        typer.echo(f"run view unavailable ({unavailable}); opening {record.ref} alone")

    candidates = list(Journal(store.journal(record.key)).candidates.values())
    if view == "map":
        unavailable = build_map(candidates, record.search_dir, higher, metric=metric, **common).unavailable
        reference = "baseline"
    else:
        # open on a reference that has something to measure against (a declared
        # baseline ships no artifacts); nothing from either -> print why, return
        for reference in ("baseline", "champion"):
            unavailable = build_similarity(candidates, record.search_dir, higher, reference=reference, **common).unavailable
            if unavailable is None:
                break
    if unavailable is not None:
        typer.echo(unavailable)
        return
    SimilarityApp(config, record.ref, reference=reference, view=view, metric=metric).run()


def _similarity_anchor(config: Config, search: str | None) -> tuple[DataStore, SearchRecord]:
    """`open_search`, except that a bare run id holding several searches
    anchors on its most recent one instead of asking the user to pick — the
    run view shows them all anyway."""
    store = open_store(config)
    try:
        return store, resolve_search(store, search)
    except LookupError as exc:
        records = store.searches(run_id=search) if search and "/" not in search else []
        if not records:
            raise typer.BadParameter(str(exc)) from exc
        return store, max(records, key=lambda r: (r.activity_at, r.ref))


@app.command()
def graph():
    """Live knowledge graph (same screen as `hillclimb knowledge graph`).

    Problems, searches, techniques, and claims, growing as searches finish.
    Drag rotates, scroll zooms, click a node for details, `?` lists keys.
    """
    knowledge_graph(stats=False)


def _demo_preflight(backend: str) -> None:
    """Fail fast, with the fix, on the two tools the engine shells out to."""
    import shutil

    missing = []
    if shutil.which("uv") is None:
        missing.append("uv is not on PATH (it builds the solution venv): pip install uv")
    if backend == "claude-code" and shutil.which("claude") is None:
        missing.append(
            "claude (Claude Code CLI) is not on PATH — the agents run through it:\n"
            "    npm install -g @anthropic-ai/claude-code && claude login"
        )
    if backend == "codex" and shutil.which("codex") is None:
        missing.append(
            "codex (Codex CLI) is not on PATH — install it and run `codex login`"
        )
    if backend == "pi" and shutil.which("pi") is None:
        missing.append("pi (pi coding-agent CLI) is not on PATH — install pi before running this backend")
    if missing:
        for line in missing:
            typer.echo(f"error: {line}", err=True)
        typer.echo("`hillclimb connect` checks every backend's credential.", err=True)
        raise typer.Exit(1)


def _print_demo_intro(folder: Path, parallel_searches: int, parallel_operators: int, budget: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(highlight=False)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    for command, what in DEMO_COMMANDS:
        table.add_row(command, what)
    body = Table.grid(padding=(0, 0))
    body.add_row(
        f"Circle packing: 26 circles in the unit square, maximize the sum of radii.\n"
        f"{parallel_searches} searches of {budget} are climbing in parallel in [cyan]{folder}[/],\n"
        f"each running {parallel_operators} operators at a time, "
        f"starting from a one-circle baseline (sum of radii 0.5).\n"
    )
    body.add_row("They run in the background — watch them from this terminal:\n")
    body.add_row(table)
    body.add_row("\n[bold cyan]hillclimb stop --all[/] ends the demo; the best solutions stay in runs/.")
    console.print(Panel(body, title="hillclimb demo", border_style="cyan", expand=False))
    console.print()


DEMO_COMMANDS = (
    ("hillclimb watch candidates", "one search's candidates: agents drafting, debugging, improving"),
    ("hillclimb watch", "all the searches side by side"),
    ("hillclimb chart", "the hillclimb curve: best score vs candidates, live"),
    ("hillclimb tree", "one search's exploration tree: expanded vs discontinued lineages"),
    ("hillclimb graph", "the knowledge graph growing as searches finish"),
)


@app.command()
def demo(
    budget: str = typer.Option("10m", help="Wall-clock budget per search, e.g. 10m"),
    parallel_searches: int = typer.Option(3, "--parallel-searches", min=1, help="Searches to run at once"),
    parallel_operators: int = typer.Option(
        3, "--parallel-operators", min=1, help="Concurrent operators (one candidate each) per search"
    ),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
):
    """Try hillclimb in one command: agents climb the circle-packing problem.

    Creates a hillclimb/ dir here if there is none, installs the bundled
    problem, starts several searches in parallel in the background, and
    prints the commands that show them live — run those right here.
    """
    from hillclimb.demo import DEMO_PROBLEM_ID, install_demo_problem
    from hillclimb.project import find_hillclimb_dir

    print_banner()
    if find_hillclimb_dir() is None:
        folder = scaffold_hillclimb_dir(Path.cwd())
        typer.echo(f"Created hillclimb dir at {folder}")
    config = load_config(backend=backend, model=model)
    problem_dir, created = install_demo_problem(config.paths.problems_dir)
    if created:
        typer.echo(f"Installed the {DEMO_PROBLEM_ID} problem at {problem_dir}")
    _demo_preflight(config.backend)
    run_dir = _run_problem_fleet(
        DEMO_PROBLEM_ID, config, budget, parallel_searches, "demo",
        backend=backend, model=model, climber=None, parallel_operators=parallel_operators,
        n_replicates=None, holdout=True, learning=True, set_=[],
    )
    _print_demo_intro(config.hillclimb_dir, parallel_searches, parallel_operators, budget)
    typer.echo(f"Engine logs in {run_dir / 'logs'}")


# ---------------------------------------------------------------- connect --

connect_app = typer.Typer(
    cls=HillclimbGroup,
    invoke_without_command=True,
    help="Connect the agents that run operators (claude, codex, pi) and the OpenRouter route that pays for them.",
)
app.add_typer(connect_app, name="connect")

_CONNECT_AUTH = typer.Option(
    None, "--auth", help="Who pays: subscription (default) | api-key | openrouter"
)
# claude-code has no OpenRouter route — offering it in the help would be a lie
_CONNECT_AUTH_CLAUDE = typer.Option(
    None, "--auth", help="Who pays: subscription (default) | api-key"
)
_CONNECT_MODEL = typer.Option(None, "--model", help="Model to ping with (default: the configured one)")
_CONNECT_PROBE = typer.Option(
    True, "--probe/--no-probe", help="Make one tool-free agent call to prove the route works"
)
_CONNECT_DEFAULT = typer.Option(
    None,
    "--default/--no-default",
    help="Write this backend into config.yaml. Default: only when no backend is pinned there yet",
)
_CONNECT_USER = typer.Option(
    False, "--user", help="Write the defaults to ~/.config/hillclimb/config.yaml instead of the hillclimb dir"
)


def _connect_config() -> Config:
    """`connect` runs before `hillclimb init` too: checking a credential
    needs no hillclimb dir, only writing defaults and keys does."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return load_config(raise_not_found=True)
    except HillclimbDirNotFound:
        return Config()


def _connect_target_config(config: Config, *, user: bool) -> Path | None:
    """The config.yaml `connect` would write defaults into."""
    from hillclimb.project import MARKER_FILE, user_config_path

    if user:
        return user_config_path()
    if config.hillclimb_dir is None:
        return None
    return config.hillclimb_dir / MARKER_FILE


def _write_defaults(path: Path | None, updates: dict[str, str], *, wanted: bool | None) -> None:
    """Persist `backend`/`backend_auth`, unless the config already pins a
    backend on purpose — connecting a second agent to try it out must not
    silently repoint an existing setup."""
    from hillclimb import connect as connect_mod

    if wanted is False:
        return
    if path is None:
        if wanted:
            typer.echo(
                "error: no hillclimb dir to write to — run `hillclimb init`, or pass --user",
                err=True,
            )
            raise typer.Exit(1)
        return
    text = path.read_text() if path.exists() else ""
    if wanted is None and connect_mod.pins_backend(text):
        current = f"{updates['backend']}/{updates['backend_auth']}"
        typer.echo(
            f"{path} already pins a backend — left as is "
            f"(`hillclimb connect … --default` switches it to {current})"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(connect_mod.apply_config_defaults(text, updates))
    settings = ", ".join(f"{key}: {value}" for key, value in updates.items())
    typer.echo(f"wrote {settings} to {path}")


def _print_status(status) -> None:
    """One connection, coloured by verdict. Provider text is escaped: an
    error message full of brackets is not rich markup."""
    from rich.console import Console
    from rich.markup import escape

    colour = "green" if status.ok else ("yellow" if status.state != "error" else "red")
    detail = f" — {escape(status.detail)}" if status.detail else ""
    Console(highlight=False).print(
        f"[bold]{status.target}[/] ({status.auth}): [{colour}]{status.state}[/]{detail}"
    )
    if not status.ok and status.fix:
        typer.echo(f"  fix: {status.fix}")


def _run_probe(backend: str, auth: str, model: str, config: Config) -> bool:
    """One real call through the connected route. Returns whether it worked;
    a failure here is the whole reason the command exists, so it is loud."""
    from hillclimb import connect as connect_mod

    typer.echo(f"pinging {backend} with model {model} …")
    result = connect_mod.ping(backend, auth, model, models_file=config.pi.models_file)
    if not result.ok:
        detail = result.error_message or result.error_kind or "unknown error"
        typer.echo(f"error: the ping failed ({result.error_kind or 'error'}): {detail.strip()[:400]}", err=True)
        # the model is the usual culprit: aliases like `sonnet` mean nothing
        # to codex, and pi resolves a bare one against whichever provider
        # matches first
        typer.echo(f"  pinged model {model!r} — `--model <id>` tries another", err=True)
        return False
    tokens = result.total_tokens or 0
    model_id = result.model_id or model
    typer.echo(f"ping ok: {model_id}, {tokens} tokens, {result.duration_s:.1f}s")
    return True


def _connect_backend(
    target: str,
    *,
    auth: str | None,
    model: str | None,
    probe: bool,
    login: bool,
    default: bool | None,
    user: bool,
) -> None:
    """Shared flow for claude / codex / pi: check, log in, import, ping, pin."""
    from hillclimb import connect as connect_mod

    config = _connect_config()
    auth = auth or connect_mod.configured_auth(config, target) or "subscription"
    if auth not in connect_mod.AUTHS_FOR[target]:
        raise typer.BadParameter(
            f"{target} has no {auth} route (one of {', '.join(connect_mod.AUTHS_FOR[target])})",
            param_hint="--auth",
        )
    backend = connect_mod.BACKEND_FOR[target]

    status = connect_mod.check(target, auth)
    if status.state == "missing-cli":
        _print_status(status)
        raise typer.Exit(1)
    if not status.ok and login and connect_mod.login_command(target):
        typer.echo(f"{status.detail} — starting `{' '.join(connect_mod.login_command(target))}`")
        connect_mod.run_login(target)
        status = connect_mod.check(target, auth)
    if not status.ok:
        _print_status(status)
        raise typer.Exit(1)

    home = connect_mod.import_credentials(target, auth, config.pi.models_file)
    if home is not None:
        typer.echo(f"credentials staged for searches in {home}")
    _print_status(status)

    model = model or config.model
    if probe:
        if auth == "openrouter" and "/" not in model:
            typer.echo(
                f"no OpenRouter model id to ping with ({model!r}) — "
                "re-run with --model <provider>/<model> to check the route"
            )
        elif not _run_probe(backend, auth, model, config):
            raise typer.Exit(1)

    _write_defaults(
        _connect_target_config(config, user=user),
        {"backend": backend, "backend_auth": auth},
        wanted=default,
    )


@connect_app.callback()
def connect(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="The same rows as data"),
):
    """Which agents this machine can run operators with, and who pays.

    A bare `hillclimb connect` checks every target — the credential is read
    through the same environment an operator gets, so an inherited
    `ANTHROPIC_API_KEY` shadowing your subscription shows up here instead of
    on a bill. `●` marks the backend this config runs by default.

    `hillclimb connect <claude|codex|pi|openrouter>` sets one up: it runs the
    agent's own login, stages the credentials searches will read, pings the
    route with one tool-free call, and pins the defaults in config.yaml.
    `hillclimb smoke` is the next step up — a whole DRAFT on a real problem.
    """
    if ctx.invoked_subcommand is not None:
        return
    from hillclimb import connect as connect_mod
    from rich.console import Console
    from rich.table import Table

    config = _connect_config()
    rows = connect_mod.status_rows(config)
    if as_json:
        typer.echo(
            json.dumps(
                [{**status.as_dict(), "default": is_default} for status, is_default in rows],
                indent=2,
            )
        )
        return
    from rich.markup import escape

    console = Console(highlight=False)
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("")
    table.add_column("target", style="bold cyan")
    table.add_column("billing")
    table.add_column("state")
    table.add_column("")
    for status, is_default in rows:
        colour = "green" if status.ok else ("red" if status.state == "error" else "yellow")
        table.add_row(
            "[cyan]●[/]" if is_default else " ",
            status.target,
            status.auth,
            f"[{colour}]{status.state}[/]",
            escape(status.detail),
        )
    console.print()
    console.print(table)
    console.print()
    for status, _ in rows:
        if not status.ok and status.fix:
            console.print(f"  [bold]{status.target}[/]: {status.fix}")
    if config.hillclimb_dir is None:
        console.print("  no hillclimb dir here — `hillclimb init` before connecting anything")
    console.print()


@connect_app.command("claude")
def connect_claude(
    auth: str = _CONNECT_AUTH_CLAUDE,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(True, "--login/--no-login", help="Run `claude auth login` when logged out"),
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """Claude Code as the operator backend, billed to your Claude subscription.

    The login is Claude Code's own (`claude auth login`); hillclimb only
    checks it the way an operator will — with `ANTHROPIC_API_KEY` stripped,
    so a key left in the environment cannot masquerade as the subscription.
    `--auth api-key` keeps the key instead, for headless machines.
    """
    _connect_backend(
        "claude", auth=auth, model=model, probe=probe, login=login, default=default, user=user
    )


@connect_app.command("codex")
def connect_codex(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    login: bool = typer.Option(True, "--login/--no-login", help="Run `codex login` when logged out"),
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """The Codex CLI as the operator backend.

    Runs `codex login`, then copies the credential into the isolated
    `CODEX_HOME` searches use, so your personal `~/.codex` settings change
    neither a search's results nor its token bill. `--auth openrouter` bills
    OpenRouter credits instead (`hillclimb connect openrouter` first).
    """
    _connect_backend(
        "codex", auth=auth, model=model, probe=probe, login=login, default=default, user=user
    )


@connect_app.command("pi")
def connect_pi(
    auth: str = _CONNECT_AUTH,
    model: str = _CONNECT_MODEL,
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """The pi coding agent as the operator backend — the one that can sample.

    pi logs in inside its own TUI, so this imports what that login wrote
    (`~/.pi/agent/auth.json`) into pi's isolated hillclimb home, together
    with `pi.models_file` if the config names one.
    """
    _connect_backend(
        "pi", auth=auth, model=model, probe=probe, login=False, default=default, user=user
    )


@connect_app.command("openrouter")
def connect_openrouter(
    key: str = typer.Option(None, "--key", help="The API key; omitted, connect asks for it (input hidden)"),
    backend: str = typer.Option(None, "--backend", help="Also route this backend through OpenRouter: codex | pi"),
    model: str = typer.Option(None, "--model", help="OpenRouter model id, e.g. qwen/qwen3-coder"),
    probe: bool = _CONNECT_PROBE,
    default: bool = _CONNECT_DEFAULT,
    user: bool = _CONNECT_USER,
):
    """OpenRouter credits as the bill for codex or pi operators.

    The only credential hillclimb stores itself: the key is validated against
    OpenRouter (one unbilled call), then written to the `.env` beside
    config.yaml that `hillclimb init` gitignores — never into config.yaml,
    where it could be journaled. `--backend codex` also pins the route.
    """
    from hillclimb import connect as connect_mod
    from hillclimb.backends.openrouter import OpenRouterError, key_info

    config = _connect_config()
    # an ambient key is already usable — only a key typed here gets stored
    ambient = os.environ.get("OPENROUTER_API_KEY")
    provided = key
    if provided is None and not ambient:
        provided = typer.prompt("OpenRouter API key", hide_input=True).strip()
    candidate = provided or ambient
    try:
        info = key_info(candidate)
    except OpenRouterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"key accepted: {connect_mod.describe_key(info)}")

    if provided:
        env_path = connect_mod.env_file(config)
        if env_path is None:
            typer.echo(
                "error: no hillclimb dir to store the key in — run `hillclimb init`, "
                "or export OPENROUTER_API_KEY yourself",
                err=True,
            )
            raise typer.Exit(1)
        connect_mod.write_env_key(env_path, "OPENROUTER_API_KEY", candidate)
        os.environ["OPENROUTER_API_KEY"] = candidate  # usable by the ping below
        typer.echo(f"stored OPENROUTER_API_KEY in {env_path}")
    else:
        typer.echo("key came from the environment — nothing stored")

    if backend is None:
        typer.echo("pin it to a backend with: hillclimb connect openrouter --backend codex --model <id>")
        return
    if backend not in ("codex", "pi"):
        raise typer.BadParameter("OpenRouter runs through codex or pi", param_hint="--backend")
    _connect_backend(
        backend, auth="openrouter", model=model, probe=probe, login=False, default=default, user=user
    )


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None, help="Operator backend: claude-code | codex | pi | dummy"),
):
    """One real DRAFT call through the selected backend, end to end.

    Executes the result and reports — verifies auth, JSON field names, and
    the filesystem contract.
    """
    config = load_config(backend=backend, model=model)
    problem = load_problem(target, config)
    _demo_preflight(config.backend)
    version_cmd = {
        "claude-code": ["claude", "-v"],
        "codex": ["codex", "--version"],
        "pi": ["pi", "--version"],
    }.get(config.backend)
    if version_cmd:
        version = subprocess.run(
            version_cmd, capture_output=True, text=True
        ).stdout.strip()
        typer.echo(f"{config.backend} version: {version}")
    run_id = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = create_run(
        config,
        RunMeta(
            run_id=run_id,
            name=run_id,
            kind="problem",
            target=target,
            problem_ids=[problem.problem_id],
        ),
    )
    search_dir = create_search(config, problem, run_dir, run_id, total_s=1800)
    from hillclimb.harness.routing import BackendPool, Router

    backend_instance = get_backend(
        config.backend, auth=config.backend_auth, pi_models_file=config.pi.models_file
    )
    backends = BackendPool(pi_models_file=config.pi.models_file)
    backends.seed(config.backend, config.backend_auth, backend_instance)
    journal = Journal(open_store(config).journal(key_for(search_dir)))
    evaluator = build_evaluator(config, problem, search_dir, journal, log=typer.echo)
    harness = Harness(
        problem=problem,
        config=config,
        journal=journal,
        backend=backend_instance,
        router=Router(config),
        backends=backends,
        executor=evaluator.executor,
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=typer.echo,
        evaluator=evaluator,
    )
    typer.echo(
        f"Running one {config.backend} DRAFT in the foreground; "
        "this can take several minutes."
    )
    typer.echo("To follow it live, open another terminal and run: hillclimb watch")
    outcome = harness.run(Action(operator="draft"))
    if outcome.candidate is None:
        typer.echo(f"the harness refused the draft: {outcome.ticket.rejected}")
        raise typer.Exit(1)
    # the journal's own record: the smoke report shows holdout, which a
    # loop-facing Outcome never carries
    candidate = journal.get(outcome.candidate.candidate_id)
    trial = candidate.last_trial
    typer.echo(f"\ncandidate:   {candidate.candidate_id} status={candidate.status}")
    typer.echo(f"val_score:   {candidate.val_score}")
    typer.echo(f"holdout:     {candidate.holdout_score} (error: {trial.holdout_error if trial else '-'})")
    typer.echo(f"session_id:  {candidate.backend.session_id}")
    typer.echo(f"cost_usd:    {candidate.backend.cost_usd}")
    typer.echo(f"num_turns:   {candidate.backend.num_turns}")
    typer.echo(f"error_kind:  {candidate.backend.error_kind}")
    typer.echo(f"raw output:  {Path(candidate.candidate_dir) / 'agent_raw.json'}")
    if candidate.backend.session_id is None and candidate.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")


def main(argv: list[str] | None = None) -> None:
    """Console-script entry: front the help screen with the wordmark.

    A bare `hillclimb` is a request to see what the tool can do, not a usage
    error — so it prints the banner and the full command list instead of
    typer's "Missing command".
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--skip-intro" in args:
        # Opt out of the first-run intro for good; `hillclimb intro` still plays it.
        from hillclimb.tui.intro import mark_intro_shown

        args = [arg for arg in args if arg != "--skip-intro"]
        mark_intro_shown()
    elif not args or args[0] != "intro":  # `hillclimb intro` plays it itself
        from hillclimb.tui.intro import maybe_play_intro

        maybe_play_intro()
    if not args or args in (["--help"], ["-h"]):
        print_banner()
        args = ["--help"]
    app(args=args, prog_name="hillclimb")


if __name__ == "__main__":
    main()
