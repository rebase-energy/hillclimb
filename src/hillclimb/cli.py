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

from hillclimb.api import (
    build_holdout,
    create_search,
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
from hillclimb.executor import LocalExecutor
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
from hillclimb.workspace import create_run_dir

app = typer.Typer(help="hillclimb: auto-hillclimbing for verifier-defined problems")


def load_config(**overrides) -> Config:
    """Config.load with the workspace-not-found hint rendered for the CLI."""
    from hillclimb.project import WorkspaceNotFound

    try:
        return Config.load(**overrides)
    except WorkspaceNotFound as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


INIT_CONFIG = """\
# hillclimb workspace config — this file marks the workspace root; commands
# work from any subdirectory. Precedence: CLI flags > this file >
# ~/.config/hillclimb/config.yaml > built-in defaults.

model: sonnet
# backend: claude-code

# budget:
#   total_s: 7200

# search:
#   parallel_agents: 1   # >1 runs concurrent operators
#   n_trials: 1          # validation evals per candidate

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


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Workspace root to initialize"),
    force: bool = typer.Option(False, "--force", help="Initialize even inside an existing workspace"),
):
    """Create a hillclimb workspace: a hillclimb/ folder holding config,
    problems, run specs, and runs."""
    from hillclimb.project import MARKER_DIR, MARKER_FILE, find_workspace_root

    root = directory.resolve()
    existing = find_workspace_root(root)
    if existing is not None and not force:
        typer.echo(
            f"Already inside the workspace at {existing} "
            f"({existing / MARKER_DIR / MARKER_FILE} exists). Use --force to nest anyway.",
            err=True,
        )
        raise typer.Exit(1)
    folder = root / MARKER_DIR
    for sub in ("problems", "specs", "runs"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / ".gitkeep").touch()
    (folder / MARKER_FILE).write_text(INIT_CONFIG)
    (folder / "specs" / "example.yaml").write_text(INIT_SPEC_EXAMPLE)
    gitignore = root / ".gitignore"
    ignore_line = f"{MARKER_DIR}/runs/"
    existing_ignore = gitignore.read_text() if gitignore.exists() else ""
    if ignore_line not in existing_ignore.splitlines():
        gitignore.write_text(existing_ignore.rstrip("\n") + ("\n" if existing_ignore else "") + ignore_line + "\n")
    typer.echo(f"Initialized hillclimb workspace at {root}")
    typer.echo(f"  {MARKER_DIR}/{MARKER_FILE}   — workspace config (edit defaults here)")
    typer.echo(f"  {MARKER_DIR}/problems/      — problem definitions")
    typer.echo(f"  {MARKER_DIR}/specs/         — committed run specs")
    typer.echo(f"  {MARKER_DIR}/runs/          — search artifacts (gitignored)")
    typer.echo("Next: hillclimb run <problem-or-spec> --budget 30m")


knowledge_app = typer.Typer(help="Cross-search learning: knowledge cards distilled from finished searches")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("backfill")
def knowledge_backfill():
    """Distill knowledge cards from every finished search under runs/ that
    doesn't have one yet — bootstraps learning from pre-existing history."""
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.knowledge import distill_card, write_card
    from hillclimb.run import iter_run_dirs, iter_search_dirs, load_search_meta

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no workspace/knowledge dir resolvable", err=True)
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
                lower_is_better=meta.lower_is_better,
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
    """Show the live cards concurrent searches in a run are sharing — the
    discoveries a sibling's next operator would receive."""
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
    """Render the prior-experience section a new search on this target
    would receive."""
    from hillclimb.api import resolve_knowledge_dir
    from hillclimb.knowledge import load_cards, problem_family, render_prior_experience

    config = load_config()
    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        typer.echo("learning is disabled or no workspace/knowledge dir resolvable", err=True)
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
            f"({problem.metric_name}, {'lower' if problem.lower_is_better else 'higher'} is better)"
        )
    else:
        typer.echo("\nDone. No scored solution; best/ holds the baseline submission.")
    artifact = "solution.py" if problem.kind in ("emflow", "evaluator") else "submission.csv"
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
    """Workspace-relative spec path recorded in run.yaml (absolute if the
    spec lives outside the workspace)."""
    if config.workspace_root is not None:
        try:
            return str(suite_path.relative_to(config.workspace_root))
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
    child_cwd = config.workspace_root or Path.cwd()
    child_env = {**os.environ, "HILLCLIMB_WORKSPACE": str(child_cwd)}
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
):
    """Start a hillclimb run on a problem folder/name or a run-spec YAML."""
    config = load_config(backend=backend, model=model)
    if not holdout:
        config.holdout.enabled = False
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
            seed_from=seed_from,
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
    """Resume a parked or interrupted search (`<run-id>/<search-id>`,
    `<run-id>`, or `latest`)."""
    config = load_config()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    if meta is None:
        raise typer.BadParameter(f"No valid search.yaml in {search_dir}")
    config = load_config(backend=meta.backend, model=meta.model)
    config.holdout.enabled = meta.holdout_enabled
    if meta.holdout_seed is not None:
        config.holdout.seed = meta.holdout_seed
    if meta.holdout_fraction is not None:
        config.holdout.fraction = meta.holdout_fraction
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


@app.command()
def stop(search: str = typer.Argument("latest")):
    """Gracefully stop a running engine: it finishes the current operator
    call, then parks. Resume later with `hillclimb resume`."""
    config = load_config()
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
    """Prune a candidate and its whole subtree: the engine stops building on
    this lineage and it is excluded from selection. Statuses and scores stay
    visible in status/tree output."""
    config = load_config()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    lower = bool(meta.lower_is_better) if meta else False
    try:
        outcome = request_prune(
            search_dir,
            candidate_id,
            lower_is_better=lower,
            selection_mode=config.holdout.selection,
            reason=reason,
            source="cli",
        )
    except ValueError as exc:
        typer.echo(f"Cannot prune: {exc}")
        raise typer.Exit(1)
    typer.echo(outcome)


@app.command()
def kill(search: str = typer.Argument("latest")):
    """SIGTERM a running engine; it finalizes state and can be resumed.
    For a graceful stop that lets the current operator finish, use
    `hillclimb stop`."""
    config = load_config()
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


@app.command()
def show(
    search: str = typer.Argument("latest", help="latest, <run-id>, or <run-id>/<search-id>"),
    candidate_id: str = typer.Argument(..., metavar="CANDIDATE", help="Candidate id, e.g. c007"),
):
    """Everything known about one candidate: metadata, scores, the evaluation
    breakdown (the same report the improve operator receives), the code diff
    vs its parent, notes, and execution output."""
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
    lower = bool(meta.lower_is_better) if meta else False
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
    typer.echo(f"{scores}  ({metric}, {'lower' if lower else 'higher'} is better)")

    report = candidate_report(cand)
    typer.echo("\n# Evaluation breakdown (validation split)\n")
    typer.echo(
        render_report(report, metric)
        or "(no evaluation report — pre-feature candidate or non-emflow problem)"
    )
    delta = render_delta(candidate_report(parent), report, lower)
    if delta:
        typer.echo(f"\n# Where it moved vs parent {parent.candidate_id}\n")
        typer.echo(delta)

    workspace = Path(cand.workspace) if cand.workspace else None
    solution = workspace / "solution.py" if workspace else None
    if parent is not None:
        typer.echo(f"\n# solution.py diff vs {parent.candidate_id}\n")
        parent_solution = Path(parent.workspace) / "solution.py" if parent.workspace else None
        if solution is None or not solution.exists() or parent_solution is None or not parent_solution.exists():
            typer.echo("(workspace not available on this machine)")
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
    if workspace is not None and (workspace / "notes.md").exists():
        typer.echo("\n# notes.md\n")
        typer.echo((workspace / "notes.md").read_text().rstrip())
    if workspace is not None and (workspace / "exec_stdout.log").exists():
        typer.echo("\n# stdout (tail)\n")
        typer.echo(tail(workspace / "exec_stdout.log").rstrip())


@app.command()
def tree(
    search: str = typer.Argument("latest"),
    out: Path = typer.Option(None, help="Output image path (.png/.svg/.pdf); default <search>/tree.png"),
):
    """Render the search's exploration tree (which candidates were created,
    built upon, or pruned) to an image."""
    try:
        from hillclimb.viz import render_tree
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    config = load_config()
    search_dir = resolve_search_dir(config, search)
    meta = load_search_meta(search_dir)
    lower = bool(meta.lower_is_better) if meta else False
    journal = Journal(search_dir / "journal.jsonl")
    out_path = out or (search_dir / "tree.png")
    if meta is not None:
        title = f"{meta.problem_id}  ({meta.model}, budget {meta.budget_s}s)"
    else:
        title = search_dir.name
    render_tree(journal, lower, out_path, title)
    typer.echo(f"Wrote {out_path} ({len(journal.candidates)} candidates)")


@app.command()
def watch():
    """Live TUI: runs, searches, candidates, and selected-candidate details.
    Keys: enter=open/details, esc=close/back, +/-=resize details, s=stop search,
    x=prune candidate, q=quit."""
    try:
        from hillclimb.watch import WatchApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb watch` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    WatchApp(load_config()).run()


@app.command()
def smoke(
    target: str = typer.Argument("circle-packing"),
    model: str = typer.Option(None),
):
    """One real DRAFT call through the claude-code backend, then execute and
    report — verifies auth, JSON field names, and the filesystem contract."""
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
        executor=LocalExecutor(ensure_runtime_venv(config)),
        budget=BudgetManager(1800, stop_margin_s=0),
        search_dir=search_dir,
        log=typer.echo,
        holdout=build_holdout(config, problem, search_dir),
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
    typer.echo(f"raw output:  {Path(candidate.workspace) / 'agent_raw.json'}")
    if candidate.backend.session_id is None and candidate.backend.error_kind is None:
        typer.echo("WARNING: session_id not parsed — check agent_raw.json for actual field names")


if __name__ == "__main__":
    app()
