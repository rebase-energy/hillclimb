"""The live views: `status`, `show`, `watch`, `chart`, `tree`, `tree2`, `archive`,
`surface`, `similarity …` and `graph`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.knowledge import knowledge_graph
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.run import search_ref
from hillclimb.harness.store import DataStore, SearchRecord, open_store, resolve_search
from hillclimb.problem import load_problem


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
    ChartApp(common.load_config(), search, detail=detail, holdout=holdout or None, cost=cost).run()


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

    TreeApp(common.load_config(), search).run()


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

    Tree2App(common.load_config(), search).run()


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

    ArchiveApp(common.load_config(), search).run()


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
            typer.echo(f"{name:15} {cls.description}")
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

    if meta.experiment and meta.arm and not single:
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
        typer.echo(f"run view unavailable ({unavailable}); opening {record.ref} alone")

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
