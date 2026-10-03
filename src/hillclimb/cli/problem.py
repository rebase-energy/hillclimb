"""`hillclimb problem …`, `init`, `verify`, `summit`: the problem side of the CLI."""

from __future__ import annotations

import shutil
import sys
from contextlib import closing
from pathlib import Path

import typer

from hillclimb.api import build_executor, build_holdout_scorer, build_unit_test_runner
from hillclimb.cli import common
from hillclimb.cli._app import HillclimbGroup, app
from hillclimb.cli.common import _m, fail, legend, next_steps, say, warn
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.harness.store import SearchRecord, open_store
from hillclimb.problem import load_problem

problem_app = typer.Typer(
    cls=HillclimbGroup,
    help="Ready-made verifier problems shipped with hillclimb.",
    no_args_is_help=True,
)


app.add_typer(problem_app, name="problem")


@problem_app.command("list")
def problem_list():
    """List every problem bundled with hillclimb."""
    import yaml

    from hillclimb.demo import BUNDLED_PROBLEM_IDS, demo_problem_resource

    rows = []
    for problem_id in BUNDLED_PROBLEM_IDS:  # ladder order, as declared
        resource = demo_problem_resource(problem_id) / "problem.yaml"
        metadata = yaml.safe_load(resource.read_text()) or {}
        higher = bool(metadata.get("higher_is_better", True))
        value, who = _best_known(metadata.get("chart_baselines") or {}, higher)
        # no budget column: the problem's `time_budget_s` is only the default
        # `--budget`, the user's to set, and the clock is one of several budget
        # dimensions — a listing that showed it would read as the problem's size
        rows.append((
            _m(problem_id),
            _m(metadata.get("metric", "-")),
            "maximize" if higher else "minimize",
            _m(value),
            f"[note]{_m(who)}[/]" if who else "",
        ))
    say()
    common.table(
        [("problem", "path"), ("metric", None), ("direction", None), ("best known", None), ("", None)],
        rows,
    )
    say("\nGet one with: [cmd]hillclimb problem get <problem>[/]")


def _best_known(chart_baselines: dict, higher_is_better: bool) -> tuple[str, str]:
    """The frontier a chart draws, as `(value, who)`: the best of the declared
    reference lines in the metric's direction (never the problem's own floor)."""
    lines = {label: value for label, value in chart_baselines.items() if label != "baseline"}
    if not lines:
        return "-", ""
    label, value = (max if higher_is_better else min)(lines.items(), key=lambda item: item[1])
    if label.startswith("best known"):  # "best known (who)" -> "who"
        label = label[len("best known"):].strip(" ()")
    return f"{value:.6g}", label


@problem_app.command("get")
def problem_get(
    problem: str = typer.Argument(
        "circle-packing",
        help="A bundled problem id (see: hillclimb problem list)",
    ),
):
    """Copy a ready-made problem into problems/.

    Makes this folder a hillclimb dir if there is none, then copies the
    problem's files in and lists them — read them before you run: the
    verifier IS the problem. An existing folder is never overwritten.
    """
    from hillclimb.demo import BUNDLED_PROBLEM_IDS, install_demo_problem
    from hillclimb.project import MARKER_FILE, find_hillclimb_dir

    if problem not in BUNDLED_PROBLEM_IDS:
        available = ", ".join(BUNDLED_PROBLEM_IDS)
        fail(f"error: no bundled problem {_m(repr(problem))} [note](available: {_m(available)})[/]")
        raise typer.Exit(1)
    if find_hillclimb_dir() is None:
        here = Path.cwd().resolve()
        say(f"[head]No hillclimb dir here.[/] {_m(problem)} needs one: a folder holding")
        say(f"[path]{MARKER_FILE}[/], [path]problems/[/] [note](where the problem goes)[/] and [path]runs/[/] [note](where searches land)[/].")
        _refuse_blocked(here)
        if _stdin_is_tty() and not typer.confirm(f"Make {here} a hillclimb dir?", default=True):
            say("Not created. Run [cmd]hillclimb init[/] where you want it, then [cmd]hillclimb problem get[/] again.")
            raise typer.Exit(1)
        folder = common.scaffold_hillclimb_dir(here)
        say(f"Initialized [path]{_m(folder)}[/] [note]({MARKER_FILE}, problems/, runs/)[/]")
    config = common.load_config()
    problem_dir, created = install_demo_problem(config.paths.problems_dir, problem)
    verb = "Fetched" if created else "Already have"
    say(f"[head]{verb} {_m(problem)}[/] at [path]{_m(problem_dir)}[/]")
    legend([(name, what) for name, what in PROBLEM_FILES if (problem_dir / name).exists()])
    next_steps([
        (f"hillclimb verify {problem}", "scores the floor; the spread it prints is the noise"),
        (f"hillclimb run {problem} --budget 10m", "then climb"),
    ])


@app.command(hidden=True)
def fetch(
    problem: str = typer.Argument(
        "circle-packing",
        help="A bundled problem id (see: hillclimb problem list)",
    ),
):
    """Deprecated spelling of `hillclimb problem get`."""
    warn("note: `hillclimb fetch` is now `hillclimb problem get`")
    problem_get(problem)


def _stdin_is_tty() -> bool:
    """Whether there is a person to ask; a script or a pipe gets the default."""
    return sys.stdin.isatty()


PROBLEM_FILES = (
    ("problem.yaml", "metric, direction, budget — the problem's identity"),
    ("description.md", "what the coding agents read before drafting"),
    ("contract.md", "the interface solution.py must implement"),
    ("interface.py", "the output format, machine-checked (hillclimb spaces)"),
    ("verifier.sh", "the ONLY process hillclimb starts: drives solution.py and reports the score"),
    ("verifier.py", "the ONLY process hillclimb starts (Windows edition of verifier.sh, no bash needed)"),
    ("verify.py", "the scorer — writes the score to $HILLCLIMB_RESULT"),
    ("baseline.py", "the starting solution scored at t=0"),
    ("sample_submission.csv", "a valid, weak submission: the floor the search starts from"),
    ("requirements.txt", "the solution venv"),
)


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Folder to make a hillclimb dir (created if missing)"),
    force: bool = typer.Option(False, "--force", help="Create one even inside an existing hillclimb dir"),
):
    """Make a folder a hillclimb dir.

    Writes hillclimb.yaml (the config) with problems/ and runs/ beside it,
    plus the gitignore rules that commit the record of every run and not
    its bulk. The current folder by default; `hillclimb init hillclimb`
    keeps it all in a subfolder instead.
    """
    from hillclimb.project import MARKER_FILE, find_hillclimb_dir

    folder = directory.resolve()
    existing = find_hillclimb_dir(folder)
    if existing is not None and not force:
        where = "This is already" if existing == folder else f"{existing} is already"
        fail(f"{_m(where)} a hillclimb dir [note]({_m(existing / MARKER_FILE)} exists)[/].")
        next_steps([
            ("hillclimb problem get <problem>", "add a problem to it"),
            ("hillclimb init --force", "nest another one here"),
        ])
        raise typer.Exit(1)
    _refuse_blocked(folder)
    common.scaffold_hillclimb_dir(folder)
    say(f"[head]Initialized hillclimb dir[/] at [path]{_m(folder)}[/]")
    legend([
        (MARKER_FILE, "config (edit defaults here)"),
        ("problems/", "problem definitions (empty until you pick one)"),
        ("runs/", "one folder per run (records committed, artifacts gitignored)"),
    ])
    next_steps([
        *([(f"cd {_m(directory)}", "commands find the dir from inside it")] if folder != Path.cwd().resolve() else []),
        ("hillclimb connect", "which coding agent runs the operators, and who pays"),
        ("hillclimb problem list", "the bundled problems to choose from"),
        ("hillclimb problem get <problem>", "copy one into problems/"),
    ])


def _refuse_blocked(folder: Path) -> None:
    """Exit when `folder` already has a problems/ or runs/ that is not
    hillclimb's — scaffolding there would mix hillclimb's into it."""
    blockers = common.scaffold_blockers(folder)
    if not blockers:
        return
    names = " and ".join(f"{path.name}/" for path in blockers)
    fail(f"{_m(str(folder))} already has {_m(names)}, and it isn't a hillclimb dir.")
    next_steps([("hillclimb init hillclimb", "keep hillclimb in its own subfolder instead")])
    raise typer.Exit(1)


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

    config = common.load_config()
    common.require_sandbox(config)
    problem = load_problem(target, config)
    source = solution.read_text() if solution else problem.baseline_text
    floor_files = {} if solution else problem.baseline_files
    if source is None and floor_files:
        # the floor is a set of files scored as they are (heilbronn's
        # sample_submission.csv), so the "solution" has nothing to do
        source = "# the problem's declared floor: its baseline_files, scored as they are\n"
    if source is None:
        fail(f"{_m(problem.problem_id)} ships no baseline — pass [cmd]--solution <file>[/] to score one")
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
                fail(f"  run {index}: FAILED ({_m(reason)})")
                say(f"  logs: [path]{_m(result.stdout_path)}[/]", err=True)
                raise typer.Exit(1)
            scores.append(result.val_score)
            say(f"  run {index}: {_m(problem.metric_name)} = [head]{result.val_score:.6g}[/]")
            if index == 0 and test_runner is not None:
                remaining = config.budget.exec_timeout_s - result.duration_s
                if remaining <= 0:
                    fail("  unit tests: BUGGY (no execution time remaining)")
                    raise typer.Exit(1)
                test_result = test_runner.run(script, candidate_dir, remaining)
                if test_result.timed_out:
                    fail("  unit tests: BUGGY (timed out)")
                    raise typer.Exit(1)
                if test_result.returncode is None or test_result.returncode < 0:
                    fail(f"  unit tests: BUGGY (crashed with {_m(test_result.returncode)})")
                    raise typer.Exit(1)
                if not test_result.passed:
                    fail(
                        f"  unit tests: FAILING (exit {_m(test_result.returncode)}); "
                        f"logs: {_m(candidate_dir / 'tests_stdout.log')}"
                    )
                    raise typer.Exit(1)
                say("  unit tests: [ok]PASSING[/]")
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
                        fail(f"  interface: {_m(violation)}")
                    raise typer.Exit(1)
                say("  interface: [ok]OK[/]")
            if holdout:
                scorer = build_holdout_scorer(config, problem, root)
                if scorer is None:
                    say("  holdout: [note]not configured for this problem[/]")
                else:
                    value, error, _cpu = scorer.score(candidate_dir)
                    if error:
                        say(f"  holdout: [bad]{_m(error)}[/]")
                    else:
                        say(f"  holdout: [head]{value:.6g}[/]")
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
        say(
            f"[warn]an improvement smaller than ~{2 * mad:.3g} cannot be told from noise.[/] "
            "To stop the search climbing it:"
        )
        say(f"[path]  evaluation:\n    n_replicates: {max(3, repeat)}\n    noise_k: 2[/]")
        say(
            "[note]  add `replicate_mode: serial` if this metric measures the machine "
            "(time, throughput, memory) — parallel trials would measure each other[/]"
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
        None, "--to", help="Destination folder (default: the hillclimb dir)"
    ),
):
    """Copy the best solution found so far into your hillclimb dir.

    Ranks every search of the problem, across all runs, by its selected
    candidate and copies that search's solution.py plus its declared output
    artifacts into the destination. Run it at any point, even
    mid-climb — you always get the best discovered so far.
    """
    config = common.load_config()
    dest = (to or config.hillclimb_dir).resolve()
    already_there = (
        {path.name for path in dest.iterdir() if path.is_file()} if dest.is_dir() else set()
    )
    dest.mkdir(parents=True, exist_ok=True)
    record, candidate, copied = _summit(config, problem, dest)
    key = record.meta.problem_key or record.meta.problem_id
    say(
        f"[head]summit of {_m(key)}[/]: {_m(record.meta.metric)} [head]{candidate.val_score:.6g}[/] — "
        f"[path]{_m(candidate.candidate_id)}[/] [note]({_m(candidate.operator)})[/] from [path]{_m(record.ref)}[/]"
    )
    for name in copied:
        verb = "refreshed" if name in already_there else "wrote"
        say(f"  {verb} [path]{_m(dest / name)}[/]")
