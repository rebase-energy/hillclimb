from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import typer
import typer.core
import typer.rich_utils

from hillclimb.api import (
    create_search,
    build_executor,
    build_holdout_scorer,
    ensure_runtime_venv,
    execute_search,
    new_run_id,
    search_ref,
    resume_spent_seconds,
)
from hillclimb.backends import get_backend
from hillclimb.budget import BudgetManager
from hillclimb.config import Config, RouteConfig
from hillclimb.control import request_prune, request_stop
from hillclimb.journal import Journal
from hillclimb.problem import (
    ProblemSpec,
    load_problem,
    resolve_target,
    suite_problem_targets,
)
from hillclimb.run import (
    SEARCHES_DIRNAME,
    iter_run_dirs,
    RunMeta,
    iter_search_dirs,
    latest_search_dir,
    load_run_meta,
    load_search_meta,
    write_run_meta,
)
from hillclimb.search import GreedySearcher
from hillclimb.status import effective_state, read_status
from hillclimb.dirs import create_run_dir

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

# ANSI-shadow "HILLCLIMB", printed above the command list on a bare `hillclimb`
# and on `--help`, the way `rebase` fronts the toolkit CLI. Same bold cyan as
# the command and option columns below it, so the whole help screen reads as
# one palette.
BANNER_STYLE = "bold cyan"
BANNER_LINES = [
    "██╗  ██╗ ██╗ ██╗      ██╗       ██████╗ ██╗      ██╗ ███╗   ███╗ ██████╗ ",
    "██║  ██║ ██║ ██║      ██║      ██╔════╝ ██║      ██║ ████╗ ████║ ██╔══██╗",
    "███████║ ██║ ██║      ██║      ██║      ██║      ██║ ██╔████╔██║ ██████╔╝",
    "██╔══██║ ██║ ██║      ██║      ██║      ██║      ██║ ██║╚██╔╝██║ ██╔══██╗",
    "██║  ██║ ██║ ███████╗ ███████╗ ╚██████╗ ███████╗ ██║ ██║ ╚═╝ ██║ ██████╔╝",
    "╚═╝  ╚═╝ ╚═╝ ╚══════╝ ╚══════╝  ╚═════╝ ╚══════╝ ╚═╝ ╚═╝     ╚═╝ ╚═════╝ ",
]
BANNER_WIDTH = max(len(line) for line in BANNER_LINES)


def print_banner() -> None:
    """Print the wordmark, or a plain-text fallback in a terminal too narrow for it."""
    from rich.console import Console

    console = Console(highlight=False)
    console.print()
    if console.width < BANNER_WIDTH:
        console.print("hillclimb", style=BANNER_STYLE)
        console.print()
        return
    for line in BANNER_LINES:
        console.print(line, style=BANNER_STYLE)
    console.print()


def load_config(**overrides) -> Config:
    """Config.load with the no-hillclimb-dir hint rendered for the CLI."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return Config.load(**overrides)
    except HillclimbDirNotFound as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


INIT_CONFIG = """\
# hillclimb config — this file marks the hillclimb dir; commands work from
# any subdirectory below it. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# backend: claude-code

# budget:
#   total_s: 7200

# search:
#   parallel_agents: 1   # >1 runs concurrent operators
#   n_trials: 1          # evals per candidate (median is the climbing score)
#   trial_mode: parallel # `serial` when the metric measures the machine (time!)
#   noise_k: 0           # require gains > k x the measured noise floor
#   min_improvement: 0   # ...or an absolute floor, in metric units

# holdout:
#   enabled: true
#   top_k: 5             # holdout scored only for top-k-by-val candidates

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
# baseline: baseline.py   # scored at t=0 as the floor to beat (or a number, e.g. 0.5)
# requirements: requirements.txt
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
# $HILLCLIMB_TRIAL_SEED. `--holdout` is passed when scoring the hidden split.
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


def scaffold_hillclimb_dir(root: Path) -> Path:
    """Create `<root>/hillclimb/` with config, the example problem, an
    example spec, and a gitignore entry for runs/. Idempotent on the folder
    layout; never overwrites an existing config."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE

    folder = root / MARKER_DIR
    for sub in ("problems", "specs", "runs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    if not (folder / MARKER_FILE).exists():
        (folder / MARKER_FILE).write_text(INIT_CONFIG)
    (folder / "specs" / "example.yaml").write_text(INIT_SPEC_EXAMPLE)
    example = folder / "problems" / "example"
    example.mkdir(parents=True, exist_ok=True)
    (example / "problem.yaml").write_text(INIT_PROBLEM_YAML)
    (example / "description.md").write_text(INIT_PROBLEM_DESCRIPTION)
    (example / "verifier.sh").write_text(INIT_PROBLEM_VERIFIER)
    (example / "verifier.sh").chmod(0o755)
    gitignore = root / ".gitignore"
    ignore_line = f"{MARKER_DIR}/runs/"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    if ignore_line not in existing_ignore.splitlines():
        gitignore.write_text(existing_ignore.rstrip("\n") + ("\n" if existing_ignore else "") + ignore_line + "\n")
    return folder


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
    typer.echo(f"Initialized hillclimb dir at {folder}")
    typer.echo(f"  {MARKER_DIR}/{MARKER_FILE}    — config (edit defaults here)")
    typer.echo(f"  {MARKER_DIR}/problems/      — problem definitions (example/ is a working one)")
    typer.echo(f"  {MARKER_DIR}/specs/         — committed run specs")
    typer.echo(f"  {MARKER_DIR}/runs/          — search artifacts (gitignored)")
    typer.echo("Next: hillclimb verify example   (then: hillclimb run example --budget 30m)")


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

    from hillclimb.dirs import create_candidate_dir

    config = load_config()
    problem = load_problem(target, config)
    source = solution.read_text() if solution else problem.baseline_text
    if source is None:
        typer.echo(
            f"{problem.problem_id} ships no baseline — pass --solution <file> to score one",
            err=True,
        )
        raise typer.Exit(1)
    executor = build_executor(config, problem)
    scores: list[float] = []
    with tempfile.TemporaryDirectory(prefix="hillclimb-verify-") as tmp:
        root = Path(tmp)
        typer.echo(f"{problem.problem_id}: {' '.join(problem.verifier_cmd)}")
        for index in range(max(1, repeat)):
            candidate_dir = create_candidate_dir(
                root, f"v{index}", problem.data_dir, problem.problem_dir
            )
            script = candidate_dir / "solution.py"
            script.write_text(source)
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
            typer.echo(f"  run {index}: {problem.metric_name} = {result.val_score:.6g}")
            if holdout:
                scorer = build_holdout_scorer(config, problem, root)
                if scorer is None:
                    typer.echo("  holdout: not configured for this problem")
                else:
                    value, error = scorer.score(candidate_dir)
                    typer.echo(f"  holdout: {error if error else format(value, '.6g')}")
    if len(scores) > 1:
        centre = statistics.median(scores)
        mad = statistics.median([abs(value - centre) for value in scores])
        spread = max(scores) - min(scores)
        typer.echo(
            f"\n{len(scores)} runs: median {centre:.6g}, spread {spread:.6g}, "
            f"noise floor (MAD) {mad:.6g}"
        )
        if mad == 0:
            typer.echo("deterministic across runs — any improvement is real")
            return
        typer.echo(
            f"an improvement smaller than ~{2 * mad:.3g} cannot be told from noise. "
            "To stop the search climbing it:"
        )
        typer.echo(f"  search:\n    n_trials: {max(3, repeat)}\n    noise_k: 2")
        typer.echo(
            "  add `trial_mode: serial` if this metric measures the machine "
            "(time, throughput, memory) — parallel trials would measure each other"
        )



knowledge_app = typer.Typer(cls=HillclimbGroup, help="Cross-search learning: cards distilled from finished searches")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("backfill")
def knowledge_backfill():
    """Distill cards from every finished search that lacks one.

    Walks runs/ and bootstraps learning from pre-existing history.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.knowledge import distill_card, write_card
    from hillclimb.run import iter_run_dirs, iter_search_dirs, load_search_meta

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    written = 0
    for run_dir in iter_run_dirs(config.paths.runs_dir):
        for search_dir in iter_search_dirs(run_dir):
            meta = load_search_meta(search_dir)
            state = effective_state(search_dir)
            # any finished search teaches something — parked and stopped
            # searches included; "unknown" covers pre-upgrade status files
            if meta is None or state == "running":
                continue
            journal = Journal(search_dir / "journal.jsonl")
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
    from hillclimb.knowledge import load_live_cards, render_live_experience

    config = load_config()
    runs_dir = config.paths.runs_dir
    if run == "latest":
        latest = latest_search_dir(runs_dir)
        if latest is None:
            typer.echo(f"No searches found in {runs_dir}", err=True)
            raise typer.Exit(1)
        run_dir = latest.parents[1]
    else:
        run_dir = runs_dir / run
        if load_run_meta(run_dir) is None:
            raise typer.BadParameter(f"No run named {run!r} in {runs_dir}")
    cards = load_live_cards(run_dir)
    if not cards:
        typer.echo(f"no live cards under {run_dir / 'knowledge'}")
        raise typer.Exit(0)
    typer.echo(f"Run {run_dir.name} — {len(cards)} live card(s)")
    for card in cards:
        val = f"{card.selected_val:.5g}" if card.selected_val is not None else "-"
        typer.echo(
            f"  {card.run_ref}: {card.n_ok} ok / {card.n_buggy} buggy of "
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
    from hillclimb.knowledge import load_cards, problem_family, render_prior_experience

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
    from hillclimb.claims import distill_claims, distill_claims_from_card
    from hillclimb.knowledge import SCHEMA_VERSION, KnowledgeCard, distill_card, write_card

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

    if search == "latest":
        search_dir = latest_search_dir(config.paths.runs_dir)
        if search_dir is None:
            typer.echo(f"No searches found in {config.paths.runs_dir}", err=True)
            raise typer.Exit(1)
    else:
        run_id, _, search_id = search.partition("/")
        search_dir = config.paths.runs_dir / run_id / SEARCHES_DIRNAME / search_id
    meta = load_search_meta(search_dir)
    if meta is None:
        raise typer.BadParameter(f"No search at {search_dir}")
    journal = Journal(search_dir / "journal.jsonl")
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
    from hillclimb.graph import load_or_build_graph, query_graph, render_query_hits

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
    from hillclimb.consolidate import consolidate

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
    from hillclimb.graph import graph_path, graph_stats, rebuild_graph

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    graph = rebuild_graph(knowledge_dir)
    typer.echo(f"rebuilt {graph_path(knowledge_dir)}")
    typer.echo(graph_stats(graph))


@knowledge_app.command("graph")
def knowledge_graph(
    stats: bool = typer.Option(False, "--stats", help="Print index stats instead of the TUI"),
):
    """Explore the knowledge graph.

    Default: the interactive TUI screen (zoom/pan/click, time scrubber);
    `--stats` prints a text summary.
    """
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.graph import graph_stats, load_or_build_graph

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no hillclimb/knowledge dir resolvable", err=True)
        raise typer.Exit(1)
    if stats:
        typer.echo(graph_stats(load_or_build_graph(knowledge_dir)))
        return
    try:
        from hillclimb.graphview import GraphApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb knowledge graph` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc
    GraphApp(config).run()


bench_app = typer.Typer(cls=HillclimbGroup, help="Learning A/B benchmark: does cross-search memory help?")
app.add_typer(bench_app, name="bench")


@bench_app.command("run")
def bench_run(
    target: str = typer.Argument(..., help="One problem (not a suite)"),
    pairs: int = typer.Option(1, "--pairs", help="Number of off/on pairs to run"),
    budget: str = typer.Option(None, "--budget", help="Per-search budget, e.g. 10m"),
    backend: str = typer.Option(None, "--backend"),
    model: str = typer.Option(None, "--model"),
):
    """Run paired searches: a memory-blind arm, then a memory-full one.

    Sequentially per pair, off (`--no-learning`) first, so a pair's blind arm
    never sees what its sibling learned; the on-arm accumulates knowledge
    between pairs exactly as production searches do. Real agent runs —
    subscription-billed; `--pairs` is your cost dial.
    """
    from hillclimb.bench import bench_run_name, slugify_target

    config = load_config()
    resolved = resolve_target(target, config)
    if resolved.kind == "suite":
        raise typer.BadParameter("bench runs one problem at a time, not a suite")
    problem = load_problem(target, config)
    slug = slugify_target(problem.problem_id)
    child_cwd = config.hillclimb_dir.parent if config.hillclimb_dir else Path.cwd()
    child_env = {**os.environ, "HILLCLIMB_DIR": str(config.hillclimb_dir or child_cwd / "hillclimb")}
    for pair in range(1, pairs + 1):
        for learning in (False, True):
            arm = "on" if learning else "off"
            run_name = bench_run_name(slug, pair, learning)
            cmd = [sys.executable, "-m", "hillclimb.cli", "run", target, "--name", run_name]
            if budget:
                cmd += ["--budget", budget]
            if backend:
                cmd += ["--backend", backend]
            if model:
                cmd += ["--model", model]
            if not learning:
                cmd.append("--no-learning")
            typer.echo(f"=== pair {pair}/{pairs}, {arm} arm: {run_name} ===")
            result = subprocess.run(cmd, cwd=child_cwd, env=child_env)
            if result.returncode != 0:
                hint = " (parked — resume it, then rerun bench report)" if result.returncode == 2 else ""
                typer.echo(f"{arm} arm exited {result.returncode}{hint}; stopping bench", err=True)
                raise typer.Exit(result.returncode)
    typer.echo("")
    _bench_report_impl(config, problem.problem_id, include_all=False)


@bench_app.command("report")
def bench_report(
    problem: str = typer.Option("", "--problem", help="Filter to one problem id"),
    include_all: bool = typer.Option(
        False, "--all", help="Group EVERY finished search by its learning flag, not just bench-* runs"
    ),
):
    """Compare learning-on vs learning-off arms.

    On the selected candidate's holdout score (falls back to val when
    holdout was off).
    """
    _bench_report_impl(load_config(), problem, include_all=include_all)


def _bench_report_impl(config: Config, problem_id: str, *, include_all: bool) -> None:
    from hillclimb.bench import collect_bench_results, pair_and_summarize, render_bench_report

    rows = collect_bench_results(
        config.paths.runs_dir, problem_id=problem_id, include_all=include_all
    )
    typer.echo(render_bench_report(pair_and_summarize(rows)))


def parse_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([hms]?)", value.strip())
    if not match:
        raise typer.BadParameter(f"Cannot parse budget {value!r} (use e.g. 2h, 30m, 3600s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


def resolve_search_dir(config: Config, ref: str | None) -> Path:
    """Resolve a search reference:

    - `latest` (or empty) — the most recently active search across v2 runs
    - `<run-id>/<search-id>` — exact address
    - `<run-id>` — the run's only search; error listing choices if several
    """
    runs_dir = config.paths.runs_dir
    if not ref or ref == "latest":
        latest = latest_search_dir(runs_dir)
        if latest is None:
            raise typer.BadParameter(f"No searches found in {runs_dir}")
        return latest
    if "/" in ref:
        run_id, _, search_id = ref.partition("/")
        search_dir = runs_dir / run_id / SEARCHES_DIRNAME / search_id
        if load_search_meta(search_dir) is None:
            raise typer.BadParameter(f"No search at {search_dir}")
        return search_dir
    run_dir = runs_dir / ref
    if load_run_meta(run_dir) is None:
        raise typer.BadParameter(f"No run named {ref!r} in {runs_dir}")
    searches = iter_search_dirs(run_dir)
    if not searches:
        raise typer.BadParameter(f"Run {ref} has no searches")
    if len(searches) > 1:
        choices = "\n".join(f"  {ref}/{s.name}" for s in searches)
        raise typer.BadParameter(f"Run {ref} has {len(searches)} searches; pick one:\n{choices}")
    return searches[0]


def _execute(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    budget: BudgetManager,
    seed_from: Path | None = None,
) -> None:
    """CLI shell over api.execute_search: messages + exit codes."""
    outcome = execute_search(config, problem, search_dir, budget, log=typer.echo, seed_from=seed_from)
    ref = outcome.ref
    if outcome.state == "parked":
        typer.echo(f"\nRate limited: {outcome.error}")
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
        typer.echo(
            f"\nDone. Selected candidate {selected.candidate_id}: {scores} "
            f"({problem.metric_name}, {'higher' if problem.higher_is_better else 'lower'} is better)"
        )
    else:
        typer.echo("\nDone. No scored solution; best/ holds the t=0 baseline.")
    # the solution is always the artifact; a submission file only exists
    # where the problem's verifier asks for one
    artifact = "solution.py"
    typer.echo(f"Best artifact: {search_dir / 'best' / artifact}")
    typer.echo(f"Inspect with: hillclimb status {ref}")


def _run_problem(
    target: str,
    config: Config,
    budget: str | None,
    run_id: str | None = None,
    run_name: str | None = None,
    seed_from: Path | None = None,
) -> None:
    problem = load_problem(target, config)
    if run_id is None:
        run_name = run_name or problem.problem_id
        run_id = new_run_id(run_name)
        run_dir = create_run_dir(config.paths.runs_dir, run_id)
        write_run_meta(
            run_dir,
            RunMeta(
                run_id=run_id,
                name=run_name,
                kind="problem",
                target=target,
                problem_ids=[problem.problem_id],
            ),
        )
    else:
        # suite child: the parent already wrote run.yaml
        run_name = run_name or run_id
        run_dir = config.paths.runs_dir / run_id
    total_s = parse_budget(budget) if budget else problem.time_budget_s
    search_dir = create_search(config, problem, run_dir, run_id, total_s)
    typer.echo(
        f"Search {search_ref(search_dir)} (run={run_name}, problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s)"
    )
    _execute(
        config, problem, search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        seed_from=seed_from,
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
    policy: str | None = None,
    parallel_agents: int | None = None,
    n_trials: int | None = None,
    seed_from: Path | None = None,
    learning: bool = True,
) -> None:
    resolved = resolve_target(target, config)
    if resolved.kind != "suite" or resolved.suite is None:
        raise typer.BadParameter(f"{target!r} is not a suite")
    suite = resolved.suite
    run_name = name or suite.suite_id
    run_id = new_run_id(run_name)
    problem_targets = suite_problem_targets(suite, config)
    problem_ids = [load_problem(problem_target, config).problem_id for problem_target in problem_targets]
    duplicates = {p for p in problem_ids if problem_ids.count(p) > 1}
    if duplicates:
        # search ids are problem ids, unique within a run
        raise typer.BadParameter(
            f"Suite {target!r} lists duplicate problem ids: {', '.join(sorted(duplicates))}"
        )
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    write_run_meta(
        run_dir,
        RunMeta(
            run_id=run_id,
            name=run_name,
            kind="suite",
            target=target,
            spec=_spec_provenance(config, suite.suite_path),
            problem_ids=problem_ids,
        ),
    )
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    child_cwd = config.hillclimb_dir.parent if config.hillclimb_dir else Path.cwd()
    child_env = {**os.environ, "HILLCLIMB_DIR": str(config.hillclimb_dir or child_cwd / "hillclimb")}
    launched = []
    for index, (entry, problem_target) in enumerate(zip(suite.problems, problem_targets), 1):
        slug = Path(problem_target).name or f"problem-{index}"
        log_path = log_dir / f"{index:02d}-{slug}.log"
        cmd = [
            sys.executable,
            "-m",
            "hillclimb.cli",
            "run",
            problem_target,
            "--run-id",
            run_id,
            "--run-name",
            run_name,
        ]
        # CLI flags override the spec entry's committed values
        child_budget = budget or entry.budget
        child_backend = backend or entry.backend
        child_model = model or entry.model
        child_parallel = parallel_agents if parallel_agents is not None else entry.parallel_agents
        child_trials = n_trials if n_trials is not None else entry.n_trials
        child_seed = seed_from or entry.seed_from
        if child_budget:
            cmd += ["--budget", child_budget]
        if child_backend:
            cmd += ["--backend", child_backend]
        if child_model:
            cmd += ["--model", child_model]
        if policy:
            cmd += ["--policy", policy]
        if child_parallel is not None:
            cmd += ["--parallel-agents", str(child_parallel)]
        if child_trials is not None:
            cmd += ["--n-trials", str(child_trials)]
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
        out = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=child_cwd,
            env=child_env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        out.close()
        launched.append((problem_target, proc.pid, log_path))
    typer.echo(f"Run {run_id}: launched {len(launched)} searches")
    for problem_target, pid, log_path in launched:
        typer.echo(f"  pid={pid} {problem_target}  log={log_path}")


@app.command()
def run(
    target: str,
    budget: str = typer.Option(None, help="Wall-clock budget, e.g. 2h / 30m"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | dummy"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    policy: str = typer.Option(
        None, "--policy", help="Search policy (default: greedy); params via config search.policy_params"
    ),
    holdout: bool = typer.Option(True, "--holdout/--no-holdout", help="Hidden selection holdout"),
    learning: bool = typer.Option(
        True, "--learning/--no-learning",
        help="Cross-search memory (cards/claims injection + distillation); off = memory-blind arm",
    ),
    name: str = typer.Option(None, "--name", help="Run name shown in the TUI"),
    parallel_agents: int = typer.Option(
        None, "--parallel-agents", help="Concurrent operators (worker pool)"
    ),
    n_trials: int = typer.Option(
        None, "--n-trials", help="Validation evals per candidate (mean climbs)"
    ),
    seed_from: Path = typer.Option(
        None, "--seed-from", help="Incumbent solution.py scored as the floor candidate"
    ),
    run_id: str = typer.Option(None, "--run-id", hidden=True),
    run_name: str = typer.Option(None, "--run-name", hidden=True),
    stop_margin_s: int = typer.Option(None, "--stop-margin-s", hidden=True),
):
    """Start a run on a problem or a run-spec YAML."""
    config = load_config(backend=backend, model=model)
    if stop_margin_s is not None:
        config.budget.stop_margin_s = stop_margin_s
    if not holdout:
        config.holdout.enabled = False
    if not learning:
        config.learning.enabled = False
    if policy is not None:
        config.search.policy = policy
    if parallel_agents is not None:
        config.search.parallel_agents = parallel_agents
    if n_trials is not None:
        config.search.n_trials = n_trials
    resolved = resolve_target(target, config)
    if resolved.kind == "suite":
        _run_suite(
            target, config, budget, backend, model, holdout, name,
            policy=policy, parallel_agents=parallel_agents, n_trials=n_trials,
            seed_from=seed_from, learning=learning,
        )
        return
    _run_problem(
        target,
        config,
        budget,
        run_id=run_id,
        run_name=run_name or name,
        seed_from=seed_from,
    )


@app.command()
def resume(search: str = typer.Argument("latest")):
    """Resume a parked or interrupted search.

    SEARCH is `<run-id>/<search-id>`, `<run-id>`, or `latest`.
    """
    config = load_config()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    if meta is None:
        raise typer.BadParameter(f"No valid search.yaml in {search_dir}")
    config = load_config(backend=meta.backend, model=meta.model)
    config.holdout.enabled = meta.holdout_enabled
    # the search resumes under the policy/routing it started with, not
    # whatever the live config currently says
    config.search.policy = meta.policy
    config.search.policy_params = meta.policy_params
    config.routing = {op: RouteConfig(**route) for op, route in meta.routing.items()}
    problem = load_problem(meta.problem, config)
    journal = Journal(search_dir / "journal.jsonl")
    spent = resume_spent_seconds(search_dir, journal)
    typer.echo(
        f"Resuming {search_ref(search_dir)}: {len(journal.candidates)} candidates, ~{int(spent)}s spent"
    )
    _execute(
        config,
        problem,
        search_dir,
        BudgetManager(meta.budget_s, config.budget.stop_margin_s, spent_s=spent),
    )


def _running_search_dirs(config: Config) -> list[Path]:
    return [
        s
        for run_dir in iter_run_dirs(config.paths.runs_dir)
        for s in iter_search_dirs(run_dir)
        if effective_state(s) == "running"
    ]


@app.command()
def stop(
    search: str = typer.Argument("latest"),
    all_: bool = typer.Option(False, "--all", help="Stop every running search"),
):
    """Gracefully stop a running engine.

    It finishes the current operator call, then parks. Resume later with
    `hillclimb resume`. `--all` stops every running search (e.g. the demo).
    """
    config = load_config()
    if all_:
        running = _running_search_dirs(config)
        if not running:
            typer.echo("No running searches.")
            raise typer.Exit(1)
        for s in running:
            typer.echo(request_stop(s, source="cli") or f"{search_ref(s)}: nothing to stop")
        return
    search_dir = resolve_search_dir(config, search)
    ref = search_ref(search_dir)
    outcome = request_stop(search_dir, source="cli")
    if outcome is None:
        typer.echo(f"Search {ref} is {effective_state(search_dir)}; nothing to stop.")
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
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    higher = bool(meta.higher_is_better) if meta else True
    try:
        outcome = request_prune(
            search_dir,
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
    config = load_config()
    if all_:
        running = _running_search_dirs(config)
        if not running:
            typer.echo("No running searches.")
            raise typer.Exit(1)
        for s in running:
            os.kill(read_status(s).pid, signal.SIGTERM)
            typer.echo(f"Sent SIGTERM to {search_ref(s)}")
        return
    search_dir = resolve_search_dir(config, search)
    ref = search_ref(search_dir)
    state = effective_state(search_dir)
    if state != "running":
        typer.echo(f"Search {ref} is {state}; nothing to kill.")
        raise typer.Exit(1)
    engine_pid = read_status(search_dir).pid
    os.kill(engine_pid, signal.SIGTERM)
    typer.echo(f"Sent SIGTERM to engine pid {engine_pid} ({ref}).")
    typer.echo(f"Resume with: hillclimb resume {ref}")


@app.command()
def status(search: str = typer.Argument("latest")):
    """Show the candidate tree of a search."""
    config = load_config()
    search_dir = resolve_search_dir(config, search)
    journal = Journal(search_dir / "journal.jsonl")
    search_status = read_status(search_dir)
    state = effective_state(search_dir)
    if search_status is not None:
        remaining = int(search_status.budget.remaining_s)
        line = f"state={state}  budget: {int(search_status.budget.spent_s)}s spent / {remaining}s left"
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

    from hillclimb.report import candidate_report, render_delta, render_report
    from hillclimb.search import tail

    config = load_config()
    search_dir = resolve_search_dir(config, search)
    journal = Journal(search_dir / "journal.jsonl")
    cand = journal.candidates.get(candidate_id)
    if cand is None:
        known = ", ".join(journal.candidates) or "(none)"
        raise typer.BadParameter(
            f"No candidate {candidate_id!r} in {search_ref(search_dir)}; known: {known}"
        )
    meta = load_search_meta(search_dir)
    metric = meta.metric if meta else "score"
    higher = bool(meta.higher_is_better) if meta else True
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
    for index, trial in enumerate(cand.trials):
        parts = [f"val={trial.val_score if trial.val_score is not None else '-'}"]
        if trial.seed is not None:
            parts.append(f"seed={trial.seed}")
        if trial.duration_s is not None:
            parts.append(f"{trial.duration_s:.0f}s")
        if trial.returncode not in (0, None):
            parts.append(f"rc={trial.returncode}")
        if trial.timed_out:
            parts.append("TIMEOUT")
        if trial.holdout_error:
            parts.append(f"holdout_error={trial.holdout_error[:60]}")
        typer.echo(f"trial {index}: {'  '.join(parts)}")
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
        import json

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


@app.command()
def tree(
    search: str = typer.Argument("latest"),
    out: Path = typer.Option(None, help="Output image path (.png/.svg/.pdf); default <search>/tree.png"),
):
    """Render the search's exploration tree to an image.

    Shows which candidates were created, built upon, or pruned.
    """
    try:
        from hillclimb.viz import render_tree
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    config = load_config()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    higher = bool(meta.higher_is_better) if meta else True
    journal = Journal(search_dir / "journal.jsonl")
    out_path = out or (search_dir / "tree.png")
    if meta is not None:
        title = f"{meta.problem_id}  ({meta.model}, budget {meta.budget_s}s)"
    else:
        title = search_dir.name
    render_tree(journal, higher, out_path, title)
    typer.echo(f"Wrote {out_path} ({len(journal.candidates)} candidates)")


def _watch_app():
    try:
        from hillclimb.watch import WatchApp
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
):
    """Live hillclimb chart: best score vs minutes into the search.

    One line per search of the same problem, so repeated searches sit on one
    pair of axes. Refreshes as candidates land. Keys: r=refresh, q=quit.
    """
    try:
        from hillclimb.chart import ChartApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb chart` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    ChartApp(load_config(), search).run()


@app.command()
def graph():
    """Live knowledge graph (same screen as `hillclimb knowledge graph`).

    Problems, searches, techniques, and claims, growing as searches finish.
    Drag rotates, scroll zooms, click a node for details, `?` lists keys.
    """
    knowledge_graph(stats=False)


DEMO_COMMANDS = (
    ("hillclimb watch", "watch the agents draft, debug, and improve solutions"),
    ("hillclimb chart", "the hillclimb curve: best score vs time, live"),
    ("hillclimb graph", "the knowledge graph growing as searches finish"),
)


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
    if missing:
        for line in missing:
            typer.echo(f"error: {line}", err=True)
        raise typer.Exit(1)


def _print_demo_intro(folder: Path, parallel: int, budget: str) -> None:
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
        f"{parallel} searches of {budget} are climbing in parallel in [cyan]{folder}[/],\n"
        f"each starting from a one-circle baseline (sum of radii 0.5).\n"
    )
    body.add_row("They run in the background — watch them from this terminal:\n")
    body.add_row(table)
    body.add_row("\n[bold cyan]hillclimb stop --all[/] ends the demo; the best solutions stay in runs/.")
    console.print(Panel(body, title="hillclimb demo", border_style="cyan", expand=False))
    console.print()


DEMO_COMMANDS = (
    ("hillclimb watch candidates", "one search's candidates: agents drafting, debugging, improving"),
    ("hillclimb watch", "all the searches side by side"),
    ("hillclimb chart", "the hillclimb curves: best score vs time, live"),
    ("hillclimb graph", "the knowledge graph growing as searches finish"),
)


def _launch_demo_search(config: Config, index: int, argv: list[str]) -> tuple[int, Path]:
    """One detached `hillclimb run` on the demo problem, logging to runs/.
    Returns (pid, log path)."""
    from hillclimb.demo import DEMO_PROBLEM_ID

    log_dir = config.paths.runs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"demo-{datetime.now():%Y%m%d-%H%M%S}-{index}.log"
    cmd = [
        sys.executable, "-m", "hillclimb.cli", "run", DEMO_PROBLEM_ID,
        "--name", f"demo-{index}", *argv,
    ]
    env = {**os.environ, "HILLCLIMB_DIR": str(config.hillclimb_dir)}
    with log_path.open("w") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=config.hillclimb_dir.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid, log_path


@app.command()
def demo(
    budget: str = typer.Option("10m", help="Wall-clock budget per search, e.g. 10m"),
    parallel: int = typer.Option(6, "--parallel", min=1, help="Searches to run at once"),
    model: str = typer.Option(None, help="Model for operator calls, e.g. sonnet / opus"),
    backend: str = typer.Option(None, help="Operator backend: claude-code | dummy"),
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
    # build the solution venv once, here, instead of N searches racing for it
    problem = load_problem(DEMO_PROBLEM_ID, config)
    ensure_runtime_venv(config, problem.runtime, log=typer.echo, requirements=problem.requirements_file)
    # a 10-minute search cannot afford the default 5-minute stop margin
    margin = min(config.budget.stop_margin_s, max(30, parse_budget(budget) // 10))
    argv = ["--budget", budget, "--stop-margin-s", str(margin)]
    if model:
        argv += ["--model", model]
    if backend:
        argv += ["--backend", backend]
    launched = [_launch_demo_search(config, index, argv) for index in range(1, parallel + 1)]
    _print_demo_intro(config.hillclimb_dir or problem_dir.parents[1], parallel, budget)
    typer.echo(f"{parallel} searches running in the background; engine logs in {config.paths.runs_dir}/demo-*.log")
    typer.echo("Try: hillclimb watch candidates")


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
):
    """One real DRAFT call through the claude-code backend, end to end.

    Executes the result and reports — verifies auth, JSON field names, and
    the filesystem contract.
    """
    config = load_config(backend="claude-code", model=model)
    problem = load_problem(target, config)
    version = subprocess.run(["claude", "-v"], capture_output=True, text=True).stdout.strip()
    typer.echo(f"claude version: {version}")
    run_id = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = create_run_dir(config.paths.runs_dir, run_id)
    write_run_meta(
        run_dir,
        RunMeta(
            run_id=run_id,
            name=run_id,
            kind="problem",
            target=target,
            problem_ids=[problem.problem_id],
        ),
    )
    search_dir = create_search(config, problem, run_dir, run_id, total_s=1800)
    journal = Journal(search_dir / "journal.jsonl")
    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=get_backend("claude-code"),
        executor=build_executor(config, problem),
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=typer.echo,
        holdout_scorer=build_holdout_scorer(config, problem, search_dir),
    )
    candidate = searcher.run_operator("draft", None)
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
    if not args or args in (["--help"], ["-h"]):
        print_banner()
        args = ["--help"]
    app(args=args, prog_name="hillclimb")


if __name__ == "__main__":
    main()
