"""The live views: `status`, `show`, `watch`, `chart`, `tree`, `tree`, `treeclimb`,
`surface`, `similarity …` and `graph`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli.common import _m, say
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.knowledge import knowledge_graph
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import search_ref
from hillclimb.harness.store import DataStore, SearchRecord, open_store, resolve_search
from hillclimb.problem import load_problem


SUMMARY_CHARS = 110  # a candidate's summary in the status table, before `…`


def _status_style(status: str) -> str:
    """The theme style for a candidate's verdict: passing is ok, a failure
    or a bug is bad, everything else (pending, abandoned) warns."""
    if status == "passing":
        return "ok"
    if status in ("failing", "buggy", "crashed"):
        return "bad"
    return "warn"


@app.command()
def status(search: str = typer.Argument("latest")):
    """Show the candidate tree of a search."""
    config = common.load_config()
    store, record = common.open_search(config, search)
    search_dir = record.search_dir
    journal = Journal(store.journal(record.key))
    search_status = store.read_status(record.key)
    state = record.state
    if search_status is not None:
        remaining = int(search_status.budget.remaining_s)
        line = f"[head]state={_m(state)}[/]  budget: {int(search_status.budget.spent_s)}s spent / {remaining}s left"
        budget = search_status.budget
        if budget.max_evaluations:
            line += f"  evaluations {budget.evaluations}/{budget.max_evaluations}"
        if budget.max_tokens:
            line += f"  tokens {budget.tokens:,}/{budget.max_tokens:,}"
        if search_status.current:
            active = " · ".join(
                f"[path]{_m(c.candidate_id)}[/]({_m(c.operator)}/{_m(c.phase)})"
                for c in search_status.current[:3]
            )
            if len(search_status.current) > 3:
                active += f" +{len(search_status.current) - 3}"
            line += f"  active: {active}"
        say(line)
    say(f"[head]Search[/] [path]{_m(search_ref(search_dir))}[/] — {len(journal.candidates)} candidates")
    candidates = list(journal.candidates.values())
    with_holdout = any(c.holdout_score is not None for c in candidates)
    # the CURRENT selection wears the star and the `selected` mark; a record's
    # own `is_selected` is historical (set when it landed, never cleared)
    current = journal.selected_candidate(bool(record.meta.higher_is_better))
    current_id = current.candidate_id if current is not None else None
    rows = []
    for candidate in candidates:
        is_current = candidate.candidate_id == current_id
        score = f"{candidate.val_score:.5f}" if candidate.val_score is not None else "-"
        hold = f"{candidate.holdout_score:.5f}" if candidate.holdout_score is not None else ""
        marks = " ".join(
            mark for mark, on in (
                ("[ok]selected[/]", is_current),
                ("[head]best[/]", candidate.is_best),
                ("[warn]pruned[/]", candidate.pruned),
            ) if on
        )
        summary = candidate.summary or ""
        if len(summary) > SUMMARY_CHARS:
            summary = summary[: SUMMARY_CHARS - 1].rstrip() + "…"
        row = [
            _m(candidate.candidate_id) + (" [warn]★[/]" if is_current else ""),
            _m(candidate.operator),
            f"[{_status_style(candidate.status)}]{_m(candidate.status)}[/]",
            score,
            *([hold] if with_holdout else []),
            marks,
            f"[path]{_m(candidate.parent_id)}[/]" if candidate.parent_id else "",
            f"[note]{_m(summary)}[/]",
        ]
        rows.append(row)
    # every cell wraps inside its own column, so a long summary folds under
    # itself and never spills under the id
    common.table(
        [("", "path"), ("operator", None), ("status", None), ("val", None),
         *([("holdout", None)] if with_holdout else []),
         ("", None), ("parent", "path"), ("summary", None)],
        rows,
    )
    scored = [
        c
        for c in journal.candidates.values()
        if c.val_score is not None and c.holdout_score is not None
    ]
    if scored:
        gaps = [abs(c.val_score - c.holdout_score) for c in scored]
        say(
            f"[head]val→holdout gap:[/] mean {sum(gaps)/len(gaps):.5g}, max {max(gaps):.5g} over {len(scored)} candidates"
        )
    floor = journal.noise_floor()
    if floor is not None:
        say(
            f"[head]noise floor:[/] {floor:.5g} [note](median trial spread)[/] — gains below "
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

    from hillclimb.harness.evaluation import tail
    from hillclimb.harness.report import candidate_report, render_delta, render_report

    config = common.load_config()
    store, record = common.open_search(config, search)
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
    header = f"[head][path]{_m(cand.candidate_id)}[/]  {_m(cand.operator)}"
    if cand.complexity:
        header += _m(f"[{cand.complexity}]")
    header += f"  status={_m(cand.status)}[/]"
    if cand.parent_id:
        header += f"  <- [path]{_m(cand.parent_id)}[/]"
    if marks:
        header += "  " + _m(f"[{', '.join(marks)}]")
    say(header)
    if cand.summary:
        say(f"[head]summary:[/] {_m(cand.summary)}")
    agent = cand.agent
    if agent.name:
        agent_line = f"agent: {agent.name}"
        if agent.cost_usd is not None:
            agent_line += f", ${agent.cost_usd:.2f}"
        if agent.num_turns is not None:
            agent_line += f", {agent.num_turns} turns"
        say(f"[head]agent:[/] {_m(agent_line[len('agent: '):])}")
    for trial in cand.trials:
        parts = [f"val={trial.val_score if trial.val_score is not None else '-'}"]
        if trial.params:
            parts.append("params=" + json.dumps(trial.params, sort_keys=True))
        if trial.holdout_score is not None:
            parts.append(f"holdout={trial.holdout_score:.5g}")
        if trial.holdout_error:
            parts.append(f"holdout_error={trial.holdout_error[:60]}")
        mark = "*" if trial.is_best and len(cand.trials) > 1 else ""
        say(f"[head]trial {trial.index}{mark}:[/] {_m('  '.join(parts))}")
        for replicate in trial.replicates:
            rparts = [f"val={replicate.val_score if replicate.val_score is not None else '-'}"]
            if replicate.seed is not None:
                rparts.append(f"seed={replicate.seed}")
            if replicate.duration_s is not None:
                rparts.append(f"{replicate.duration_s:.0f}s")
            if replicate.returncode not in (0, None):
                rparts.append(f"rc={replicate.returncode}")
            if replicate.timed_out:
                rparts.append("[bad]TIMEOUT[/]")  # the one part that is markup, not a value
            seed = replicate.seed if replicate.seed is not None else 0
            shown = "  ".join(r if r == "[bad]TIMEOUT[/]" else _m(r) for r in rparts)
            say(f"  [head]replicate {seed}:[/] {shown}")
    if cand.tunable:
        say("[head]tunable:[/] [ok]yes[/]")
    elif cand.params_error:
        say(f"[head]tunable:[/] no — [note]{_m(cand.params_error)}[/]")
    scores = f"val_score={cand.val_score}"
    if cand.holdout_score is not None:
        scores += f"  holdout={cand.holdout_score:.5g}"
    say(f"[head]{_m(scores)}[/]  [note]({_m(metric)}, {'higher' if higher else 'lower'} is better)[/]")

    report = candidate_report(cand)
    say("\n[head]# Evaluation breakdown (validation split)[/]\n")
    typer.echo(
        render_report(report, metric)
        or "(no evaluation report — pre-feature candidate or non-emflow problem)"
    )
    delta = render_delta(candidate_report(parent), report, higher)
    if delta:
        say(f"\n[head]# Where it moved vs parent [path]{_m(parent.candidate_id)}[/][/]\n")
        typer.echo(delta)

    if cand.metrics or cand.climber_meta:
        say("\n[head]# search metadata[/]\n")
        if cand.metrics:
            typer.echo("metrics: " + json.dumps(cand.metrics, sort_keys=True))
        if cand.climber_meta:
            typer.echo("climber_meta: " + json.dumps(cand.climber_meta, sort_keys=True))

    candidate_dir = Path(cand.candidate_dir) if cand.candidate_dir else None
    solution = candidate_dir / "solution.py" if candidate_dir else None
    if parent is not None:
        say(f"\n[head]# solution.py diff vs [path]{_m(parent.candidate_id)}[/][/]\n")
        parent_solution = Path(parent.candidate_dir) / "solution.py" if parent.candidate_dir else None
        if solution is None or not solution.exists() or parent_solution is None or not parent_solution.exists():
            say("[note](candidate_dir not available on this machine)[/]")
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
        say("\n[head]# notes.md[/]\n")
        typer.echo((candidate_dir / "notes.md").read_text().rstrip())
    if candidate_dir is not None and (candidate_dir / "exec_stdout.log").exists():
        say("\n[head]# stdout (tail)[/]\n")
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
    _watch_app()(common.load_config()).run()


@watch_app.command("candidates")
def watch_candidates(
    search: str = typer.Argument("latest", help="latest, <run-id>, or <run-id>/<search-id>"),
):
    """Open the TUI straight on one search's candidates.

    Esc backs out to the run's searches and the run list as usual.
    """
    config = common.load_config()
    search_dir = common.resolve_search_dir(config, search)
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
        False, "--cost", help="Overlay cumulative coding agent tokens and verifier CPU-minutes "
        "on right-hand axes — what the climb cost as it climbed"
    ),
):
    """Live hillclimb chart: best score so far vs tested candidates.

    One chart per problem. With no argument and more than one problem (or
    study) in the folder, first a table of them, newest activity first;
    enter opens that chart, esc comes back. `hillclimb watch` reaches the
    same chart with `c`.

    The climb: one staircase across every run of the problem, every scored
    candidate a dot (bright where it set a new best, dim where it missed).
    With several runs, `v` cycles three views: the plain climb (the
    default), the same climb with each run's dots in its own colour, named in
    the legend (`#2 15:08`: number and start time, or the name you gave the
    run), and a comparison — one line per run from its own first candidate,
    ending in a dot at its best. `]` / `[` focus one run (the others fade to
    grey) and step back to all; in the plain climb they open the run colours. A study gets one line per experiment. Problem-config
    baselines are dashed lines. Refreshes as candidates land.
    Keys: v=climb/runs/compare, ]/[=focus next/previous run, r=refresh,
    t=toggle improvement text, d=detail (the focused run's search: every
    scored candidate as a mark, parent edges, accepted lineage bold),
    h=toggle holdout/validation, c=cost overlay, p=switch problem, esc=back
    to the table, q=quit.
    """
    try:
        from hillclimb.tui.chart import ChartApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb chart` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    # --holdout forces the view; omitted, the chart decides per problem
    ChartApp(common.load_config(), search, detail=detail, holdout=holdout or None, cost=cost).run()


@app.command()
def tree(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """Live tree of one search, drawn like the Darwin Gödel Machine's archive.

    Roots across the top, one row per operator step; each circle carries its
    candidate number (`c017` → 17) inside, is filled with its score on a cyan
    ramp (bright = best; hollow = no working solution), and ringed by what the search did
    with it: white = expanded (the spine the policy walked), no ring =
    scored and never built on, red = failed. The final best is a white-ringed
    star, and its parent chain is drawn in cyan. Circles never overlap: they
    are sized to the zoom, and the numbers appear as they grow. Scroll
    zooms, drag pans, click a node for details, enter opens it, j/k scrub
    through time, n/p switch search, `?` keys.
    """
    try:
        from hillclimb.tui.treedrawview import TreeDrawApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb tree` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    TreeDrawApp(common.load_config(), search).run()


@app.command("treeclimb", short_help="The tree and the climb chart of one search, side by side, live.")
def treeclimb(
    search: str = typer.Argument(None, help="latest (default), <run-id>, or <run-id>/<search-id>"),
):
    """The tree and the climb chart of one search, side by side, live — the
    Darwin Gödel Machine's two-panel figure.

    Left, the `tree` tree; right, every scored candidate as a dot at
    (candidate number, score) with the best-so-far staircase, a brighter dot
    where a candidate set a new best, and the lineage of the final best as
    a thick line — the same parent chain drawn bold in the tree, one
    circle and one dot per candidate number on both. j/k scrub both panels through
    time together (a cursor marks the candidate on the chart); click a node
    to ring its dot on the chart and see its detail, enter opens it, n/p
    switch search, `?` keys.
    """
    try:
        from hillclimb.tui.treeclimbview import TreeclimbApp
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb treeclimb` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    TreeclimbApp(common.load_config(), search).run()


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

    config = common.load_config()
    _, record = common.open_search(config, search)
    try:
        problem = load_problem(record.meta.problem, config)
    except Exception as exc:  # noqa: BLE001 — a moved/deleted problem dir
        say(f"[warn]Cannot load problem [path]{_m(repr(record.meta.problem))}[/]: {_m(exc)}[/]")
        return
    if problem.landscape_path is None:
        say(f"[warn]{_m(surface_unavailable(problem))}[/]")
        return
    SurfaceApp(config, search).run()


@app.command()
def plot(
    target: str = typer.Argument(
        None,
        help="what to draw: nothing = the solution in this folder (what `summit` wrote), "
        "a folder holding a solution's files, or a search (latest, <run-id> or <run-id>/<search-id>)",
    ),
    candidate: str = typer.Argument(None, help="a candidate of that search (e.g. c003); default: its best"),
    problem: str = typer.Option(None, "--problem", "-p", help="the problem, when the folder has several"),
    out: Path = typer.Option(None, "--out", "-o", help="where to save the PNG"),
    no_open: bool = typer.Option(False, "--no-open", help="save the PNG without opening it"),
):
    """Draw a solution the way its problem says: the problem's plot.py (matplotlib).

    With no argument, the solution in this folder — what `hillclimb summit`
    copied here — saved beside it as solution.png; a directory, the solution
    files in it; a search, its best (or the CANDIDATE named), saved under the
    machine cache. The PNG opens in your image viewer. plot.py runs in the
    problem's runtime venv, like its verifier (matplotlib is added there
    once), and reads the solution's output files (submission.csv, …) — it
    never reruns solution.py. A problem without plot.py says so; `hillclimb
    summit --plot` copies the best and draws it in one go.
    """
    from hillclimb.project import machine_cache_dir

    config = common.load_config()
    path = Path(target) if target else None
    if target is None or (path is not None and path.is_dir()):
        spec = common.folder_problem(config, problem)
        solution_dir = (path or config.hillclimb_dir).resolve()
        where = "this folder" if path is None else str(path)
        default_out = solution_dir / "solution.png"
    else:
        store, record = common.open_search(config, target)
        spec = load_problem(problem or record.meta.problem, config)
        if candidate:
            from hillclimb.tui.watch import resolve_candidate_dir

            found = Journal(store.journal(record.key)).candidates.get(candidate)
            if found is None:
                raise typer.BadParameter(f"{record.ref} has no candidate {candidate!r}")
            solution_dir = resolve_candidate_dir(record.search_dir, candidate, found.candidate_dir)
            where = f"{record.ref} {candidate}"
        else:
            solution_dir = record.search_dir / "best"
            where = f"{record.ref} best"
        # never into the run: the engine is the single writer of its search dir
        name = f"{record.run_id}__{record.search_id}__{candidate or 'best'}.png"
        default_out = machine_cache_dir() / "plots" / name
    common.show_solution_plot(config, spec, solution_dir, where, (out or default_out).resolve(), open_it=not no_open)


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


_SIMILARITY_SINGLE = typer.Option(False, "--single", help="One search only, even when it is one experiment of a study")


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
    tree's fate. A study's experiment opens its whole run instead, coloured by
    experiment like `chart` (--single for the one-search view). Keys: m cycles
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
    A search that is one experiment of a study opens the whole run instead: every
    search of that problem in one cube, measured from the shared seed,
    coloured by experiment like `chart`, n/p stepping through the run's problems
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
    from hillclimb.climber import climber_base_dir
    from hillclimb.modules.similarity import (
        SimilarityUnavailable,
        Solution,
        get_score,
        registered_scores,
        similarity_matrix,
    )
    from hillclimb.tui.similarity import dir_for

    if list_scores:
        for name, cls in sorted(registered_scores().items()):
            say(f"[path]{_m(f'{name:15}')}[/] [note]{_m(cls.description)}[/]")
        return
    config = _config_or_default() if files else common.load_config()
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
                common.warn(f"[path]{_m(cid)}[/]: no solution.py — skipped")
    if len(solutions) < 2:
        raise typer.BadParameter("need at least two solutions with a solution.py to compare")

    requested = {name: config.similarity.scores.get(name, {}) for name in score} if score else config.similarity.scores
    results = []
    for name, params in requested.items():
        try:
            instance = get_score(name, params, base_dir=climber_base_dir(config))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        try:
            matrix = similarity_matrix(instance, solutions)
        except SimilarityUnavailable as exc:
            common.warn(f"[path]{_m(instance.name)}[/]: unavailable — {_m(exc)}")
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
        return common.load_config(raise_not_found=True)
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
        from hillclimb.tui.similarityview import (
            SimilarityApp,
            run_inputs,
            search_inputs,
        )
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "`hillclimb similarity` needs the TUI extra: pip install 'hillclimb[tui]'"
        ) from exc

    if metric not in METRICS:
        raise typer.BadParameter(f"--metric must be one of {', '.join(METRICS)}")
    config = common.load_config()
    store, record = _similarity_anchor(config, search)
    meta = record.meta
    try:
        fingerprint_path = load_problem(meta.problem, config).fingerprint_path
    except Exception:  # noqa: BLE001 — a provider or moved problem: no fingerprint module
        fingerprint_path = None
    higher = bool(meta.higher_is_better)
    shared = dict(output_artifacts=meta.output_artifacts, fingerprint_path=fingerprint_path)

    if meta.study and meta.experiment and not single:
        inputs = search_inputs(store, run_inputs(store, record.run_id, meta.problem_key))
        reference: str | None = None
        if view == "map":
            unavailable = build_run_map(inputs, higher, metric=metric, problem_key=meta.problem_key, **shared).unavailable
            reference = "seed" if unavailable is None else None
        else:
            for reference in ("seed", "champion"):
                unavailable = build_run_similarity(
                    inputs, higher, reference=reference, problem_key=meta.problem_key, **shared,
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
        say(f"[warn]run view unavailable ({_m(unavailable)})[/]; opening [path]{_m(record.ref)}[/] alone")

    candidates = list(Journal(store.journal(record.key)).candidates.values())
    if view == "map":
        unavailable = build_map(candidates, record.search_dir, higher, metric=metric, **shared).unavailable
        reference = "baseline"
    else:
        # open on a reference that has something to measure against (a declared
        # baseline ships no artifacts); nothing from either -> print why, return
        for reference in ("baseline", "champion"):
            unavailable = build_similarity(candidates, record.search_dir, higher, reference=reference, **shared).unavailable
            if unavailable is None:
                break
    if unavailable is not None:
        say(f"[warn]{_m(unavailable)}[/]")
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
