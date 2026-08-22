"""Public programmatic API: run a hillclimb search in-process.

Typer-free — the CLI is a thin shell over these functions, and downstream
products embed them directly:

    import hillclimb
    outcome = hillclimb.run_search("emflow://gefcom2014:solar", budget_s=7200)
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from hillclimb.backends import get_backend
from hillclimb.budget import BudgetManager
from hillclimb.candidate import Candidate
from hillclimb.config import Config
from hillclimb.control import clear_stale_stops
from hillclimb.journal import Journal
from hillclimb.problem import ProblemSpec, load_problem
from hillclimb.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.search import GreedySearcher, ParkedSearch, StopRequested
from hillclimb.status import SearchStatus, StatusWriter
from hillclimb.dirs import create_run_dir, create_search_dir

Log = Callable[[str], None]


@dataclass
class SearchOutcome:
    run_dir: Path
    search_dir: Path
    selected: Candidate | None
    state: str  # done | parked | stopped
    error: str | None = None

    @property
    def ref(self) -> str:
        return search_ref(self.search_dir)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return cleaned or "run"


def new_run_id(name: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{slug(name)}"


def search_ref(search_dir: Path) -> str:
    """Human-facing `<run-id>/<search-id>` address of a search dir."""
    return f"{search_dir.parents[1].name}/{search_dir.name}"


def default_venv_python(config: Config, kind: str, requirements: Path | None = None) -> Path:
    """Shared machine venv path, keyed by a hash of the requirement set (and
    the emflow source, whose changes must also rebuild): built once per
    machine, shared by every hillclimb dir, new key = automatic rebuild.

    A problem-supplied `requirements` file keys on its CONTENT instead —
    editing the file rebuilds automatically, and problems with identical
    requirement sets share one venv."""
    import hashlib

    from hillclimb.project import machine_cache_dir
    from hillclimb.runtime import runtime_packages

    if requirements is not None:
        digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]
        return machine_cache_dir() / "venvs" / f"problem-{digest}" / "bin" / "python"
    text = "\n".join(runtime_packages(kind))
    if kind == "emflow":
        text += f"\n{config.emflow.source}"
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return machine_cache_dir() / "venvs" / f"{kind}-{digest}" / "bin" / "python"


def ensure_runtime_venv(
    config: Config, kind: str = "csv", log: Log = print, requirements: Path | None = None
) -> Path:
    """Create the solution-script venv for the problem kind on first use.
    Explicitly configured paths are used as-is (hosted image, tests); the
    default is a shared hash-keyed venv under the machine cache dir. A
    problem-supplied `requirements` file gets its own content-keyed venv
    (the config path overrides stay kind-scoped and do not apply)."""
    import fcntl
    from importlib import resources

    from hillclimb.runtime import requirements_resource

    if requirements is not None:
        python = default_venv_python(config, kind, requirements)
    else:
        python_path = (
            config.paths.emflow_runtime_python if kind == "emflow" else config.paths.runtime_python
        )
        python = python_path.absolute() if python_path else default_venv_python(config, kind)
    if python.exists():
        return python
    venv_dir = python.parents[1]
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    # concurrent searches on a cold machine must not race the build
    with open(f"{venv_dir}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if python.exists():
            return python
        log(f"Creating {kind} runtime venv at {venv_dir} ...")
        try:
            subprocess.run(["uv", "venv", "--python", "3.12", str(venv_dir)], check=True)
            if requirements is not None:
                subprocess.run(
                    ["uv", "pip", "install", "-r", str(requirements), "--python", str(python)],
                    check=True,
                )
            else:
                with resources.as_file(requirements_resource(kind)) as req:
                    subprocess.run(
                        ["uv", "pip", "install", "-r", str(req), "--python", str(python)],
                        check=True,
                    )
            if requirements is None and kind == "emflow":
                # flag forms ("-e ../emflow") split into args; requirement
                # specs ("emflow @ git+…", "/src/emflow") are ONE argument —
                # shlex would shred the PEP 508 " @ " form
                source = config.emflow.source.strip()
                source_args = shlex.split(source) if source.startswith("-") else [source]
                try:
                    subprocess.run(
                        ["uv", "pip", "install", *source_args, "--python", str(python)],
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Installing emflow from {config.emflow.source!r} failed — first use "
                        "needs network (or set `emflow.source` to a local checkout, e.g. '-e ../emflow')"
                    ) from exc
        except BaseException:
            # a partial venv would pass the exists() check forever
            import shutil

            shutil.rmtree(venv_dir, ignore_errors=True)
            raise
    return python


def build_executor(config: Config, problem: ProblemSpec, log: Log = print):
    """The problem's verifier command, wired to the runtime venv it needs."""
    from hillclimb.executor import CommandExecutor

    return CommandExecutor(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.verifier_cmd,
        env_extra=problem.verifier_env,
    )


def build_holdout_scorer(config: Config, problem: ProblemSpec, search_dir: Path, log: Log = print):
    """Hidden-split scorer, or None when holdout is off for this search —
    selection then climbs on validation alone."""
    if not config.holdout.enabled or problem.holdout_cmd is None:
        return None
    if problem.holdout_needs_credentials and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ):
        # fail fast: without credentials every holdout eval nans out and the
        # search burns debug cycles diagnosing the environment
        raise RuntimeError(
            "holdout scoring for this problem needs private data credentials: "
            "export HF_TOKEN (or HUGGINGFACE_TOKEN), or run with --no-holdout"
        )
    from hillclimb.executor import CommandHoldoutScorer

    return CommandHoldoutScorer(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.holdout_cmd,
        problem_dir=problem.problem_dir,
        data_dir=problem.data_dir,
        work_root=search_dir / "holdout-eval",
        timeout_s=config.budget.exec_timeout_s,
    )


def spent_seconds(journal: Journal) -> float:
    """Legacy resume accounting: sum of agent + trial work durations. Only a
    fallback — under parallel workers this overcounts wall-clock; prefer
    resume_spent_seconds."""
    return sum(
        (c.backend.agent_duration_s or 0) + sum(t.duration_s or 0 for t in c.trials)
        for c in journal.candidates.values()
    )


def resume_spent_seconds(search_dir: Path, journal: Journal) -> float:
    """Wall-clock already consumed by a search, for budget seeding on resume.
    status.json persists budget.spent_s on every heartbeat (15 s) and on
    finalize, so this is exact for parked/stopped searches and loses at most
    one heartbeat on crashes. Falls back to the work-duration sum for
    pre-upgrade searches without a persisted budget."""
    from hillclimb.status import read_status

    status = read_status(search_dir)
    if status is not None and status.budget.total_s > 0:
        return status.budget.spent_s
    return spent_seconds(journal)


def create_search(
    config: Config,
    problem: ProblemSpec,
    run_dir: Path,
    run_id: str,
    total_s: int,
    seed_from: Path | None = None,
) -> Path:
    search_dir = create_search_dir(run_dir, problem.problem_id)
    write_search_meta(
        search_dir,
        SearchMeta(
            search_id=problem.problem_id,
            run_id=run_id,
            problem=(
                f"emflow://{problem.emflow_problem}"
                if problem.emflow_problem
                else f"mlebench://{problem.mlebench_comp_id}"
                if problem.mlebench_comp_id
                else str(problem.problem_dir)
            ),
            problem_id=problem.problem_id,
            backend=config.backend,
            model=config.model,
            policy=config.search.policy,
            policy_params=config.search.policy_params,
            routing={
                op: route.model_dump(exclude_none=True)
                for op, route in config.routing.items()
            },
            metric=problem.metric_name,
            higher_is_better=problem.higher_is_better,
            budget_s=total_s,
            holdout_enabled=config.holdout.enabled and problem.holdout_cmd is not None,
            seed_from=str(seed_from) if seed_from else None,
            learning_enabled=config.learning.enabled,
        ),
    )
    return search_dir


def resolve_knowledge_dir(config: Config) -> Path | None:
    if not config.learning.enabled:
        return None
    if config.learning.dir is not None:
        return Path(config.learning.dir).absolute()
    if config.hillclimb_dir is not None:
        return config.hillclimb_dir / "knowledge"
    return None


def build_knowledge_context(
    config: Config, problem: ProblemSpec, target: str, log: Log
) -> tuple[str | None, int, list[str]]:
    """(prior-experience prompt section, draft-complexity offset, injected
    claim ids) from the hillclimb dir's knowledge cards. The claim ids feed credit
    assignment: whoever gets quoted in the prompt answers for the outcome."""
    from hillclimb.knowledge import (
        complexity_offset,
        load_cards,
        problem_family,
        render_prior_experience,
    )

    knowledge_dir = resolve_knowledge_dir(config)
    if knowledge_dir is None:
        return None, 0, []
    family = problem_family(problem.problem_id, target)
    cards = load_cards(knowledge_dir, problem_id=problem.problem_id, family=family)
    if not cards:
        return None, 0, []
    log(f"learning: {len(cards)} prior search card(s) inform this search")
    offset = complexity_offset(cards) if config.learning.complexity_prior else 0
    text = render_prior_experience(cards, max_cards=config.learning.max_cards)
    claim_ids: list[str] = []
    if config.learning.graph_retrieval:
        # graph-walk retrieval: distilled claims for this family plus
        # cross-family claims that share a concept with the problem. When a
        # consolidated playbook covers the problem's concepts it REPLACES the
        # raw claim list (evolved prose beats retrieved facts), and credit
        # flows to the claims the playbook was built from.
        # Best effort — the cards block above never depends on the graph.
        try:
            from hillclimb.claims import problem_concepts, render_claims
            from hillclimb.graph import load_or_build_graph, node_to_claim, retrieve_claims

            kind = "emflow" if target.startswith("emflow://") else getattr(problem, "kind", "csv")
            concepts = problem_concepts(kind, problem.metric_name)
            playbooks = []
            if config.learning.playbooks:
                from hillclimb.consolidate import load_playbooks, render_playbooks

                playbooks = load_playbooks(knowledge_dir, concepts)
            if playbooks:
                log(
                    "learning: playbook(s) inform this search: "
                    + ", ".join(p.concept for p in playbooks)
                )
                text = f"{text}\n\n{render_playbooks(playbooks)}"
                claim_ids = sorted({cid for p in playbooks for cid in p.source_claims})
            else:
                nodes = retrieve_claims(
                    load_or_build_graph(knowledge_dir),
                    family=family,
                    problem_id=problem.problem_id,
                    concepts=concepts,
                )
                claims_text = render_claims([node_to_claim(n) for n in nodes])
                if claims_text:
                    log(f"learning: {len(nodes)} distilled claim(s) inform this search")
                    text = f"{text}\n\n{claims_text}"
                    claim_ids = [n.id.removeprefix("claim:") for n in nodes]
        except Exception as exc:  # noqa: BLE001
            log(f"learning: graph retrieval failed (prior cards unaffected): {exc}")
    return text, offset, claim_ids


def _distill_knowledge(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    journal: Journal,
    *,
    target: str,
    budget_s: int,
    cost_usd: float,
    log: Log,
) -> None:
    """Best effort — learning must never fail a finished search."""
    from hillclimb.knowledge import CARD_FILENAME, distill_card, write_card, write_live_card
    from hillclimb.run import SEARCHES_DIRNAME

    try:
        card = distill_card(
            journal,
            problem=problem,
            run_ref=search_ref(search_dir),
            target=target,
            budget_s=budget_s,
            cost_usd=cost_usd,
            selection=config.holdout.selection,
        )
        knowledge_dir = resolve_knowledge_dir(config)
        if config.learning.claims and knowledge_dir is not None:
            # inner guard: a failed distill pass costs the claims, not the card
            try:
                from hillclimb.claims import distill_claims

                card.claims = distill_claims(
                    journal,
                    problem=problem,
                    card=card,
                    search_dir=search_dir,
                    knowledge_dir=knowledge_dir,
                    config=config,
                    log=log,
                )
                if card.claims:
                    log(f"learning: {len(card.claims)} claim(s) distilled")
            except Exception as exc:  # noqa: BLE001
                log(f"learning: claims distillation failed (card unaffected): {exc}")
        # always keep a copy with the search artifacts (synced for hosted runs)
        import yaml as _yaml

        (search_dir / CARD_FILENAME).write_text(
            _yaml.safe_dump(card.model_dump(exclude_none=True), sort_keys=False)
        )
        if knowledge_dir is not None:
            path = write_card(knowledge_dir, card)
            log(f"learning: knowledge card written to {path}")
        if (
            config.learning.enabled
            and config.learning.live
            and search_dir.parent.name == SEARCHES_DIRNAME
        ):
            # final refresh of the run-scoped live card: siblings still
            # running loaded their static prior cards before this search
            # finished, so the live channel is how its result reaches them
            write_live_card(search_dir.parents[1], card, search_dir.name)
        if (
            config.learning.credit
            and config.learning.graph_retrieval
            and knowledge_dir is not None
        ):
            # credit assignment: the claims this search's drafts were shown
            # share its outcome (see credit.py for the reward definition)
            try:
                from hillclimb.credit import (
                    CreditEvent,
                    read_injected_claims,
                    search_reward,
                    write_credit_event,
                )
                from hillclimb.knowledge import load_cards

                claim_ids = read_injected_claims(search_dir)
                if claim_ids:
                    prior_cards = [
                        c for c in load_cards(
                            knowledge_dir,
                            problem_id=card.problem_id,
                            family=card.family,
                        )
                        if c.run_ref != card.run_ref  # own card is already on disk
                    ]
                    reward, basis = search_reward(
                        journal, problem, prior_cards, selection=config.holdout.selection
                    )
                    write_credit_event(knowledge_dir, CreditEvent(
                        run_ref=card.run_ref,
                        problem_id=card.problem_id,
                        family=card.family,
                        claim_ids=claim_ids,
                        reward=reward,
                        basis=basis,
                    ))
                    log(
                        f"learning: credit {reward:g} ({basis}) recorded for "
                        f"{len(claim_ids)} injected claim(s)"
                    )
            except Exception as exc:  # noqa: BLE001
                log(f"learning: credit assignment failed (card unaffected): {exc}")
        if config.learning.skills and knowledge_dir is not None:
            # procedural memory: a scored winner joins the skill library
            try:
                from hillclimb.skills import harvest_skill

                skill_dir = harvest_skill(
                    journal, problem=problem, card=card,
                    knowledge_dir=knowledge_dir,
                    selection=config.holdout.selection, log=log,
                )
                if skill_dir is not None:
                    log(f"learning: skill harvested -> {skill_dir}")
            except Exception as exc:  # noqa: BLE001
                log(f"learning: skill harvest failed (card unaffected): {exc}")
        if knowledge_dir is not None:
            # keep the derived graph index fresh; cheap at this scale and
            # best-effort like everything else here
            try:
                from hillclimb.graph import rebuild_graph

                rebuild_graph(knowledge_dir)
            except Exception as exc:  # noqa: BLE001
                log(f"learning: graph rebuild failed (card unaffected): {exc}")
    except Exception as exc:  # noqa: BLE001
        log(f"learning: card distillation failed (search result unaffected): {exc}")


def _raise_stop_requested(signum, frame):
    raise StopRequested(f"signal {signal.Signals(signum).name}")


def execute_search(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    budget: BudgetManager,
    log: Log = print,
    seed_from: Path | None = None,
    knowledge_context: str | None = None,
    target: str = "",
) -> SearchOutcome:
    """Run the engine on an existing search dir. Returns the outcome for
    parked/stopped/done; unexpected engine crashes finalize `failed` and
    re-raise."""
    run_dir = search_dir.parents[1]
    clear_stale_stops(search_dir)
    journal = Journal(search_dir / "journal.jsonl")
    status = StatusWriter(
        search_dir,
        SearchStatus(
            search_id=search_dir.name,
            run_id=run_dir.name,
            state="running",
            pid=os.getpid(),
        ),
        budget=budget,
    )
    status.start_heartbeat()
    try:  # signal handlers are main-thread-only; embedded callers skip them
        signal.signal(signal.SIGTERM, _raise_stop_requested)
    except ValueError:
        pass
    import threading

    from hillclimb.slots import MachineSlots

    abort = threading.Event()
    _kc, _offset = (None, 0)
    if knowledge_context is None:
        _kc, _offset, _claim_ids = build_knowledge_context(config, problem, target, log)
        if _claim_ids:
            from hillclimb.credit import record_injected_claims

            record_injected_claims(search_dir, _claim_ids)
    reference_solution: Path | None = None
    reference_note = ""
    if config.learning.skills:
        _kdir = resolve_knowledge_dir(config)
        if _kdir is not None:
            try:
                from hillclimb.claims import problem_concepts
                from hillclimb.knowledge import problem_family
                from hillclimb.skills import SKILL_CODE_FILENAME, select_skill

                kind = "emflow" if target.startswith("emflow://") else getattr(problem, "kind", "csv")
                match = select_skill(
                    _kdir,
                    family=problem_family(problem.problem_id, target),
                    concepts=problem_concepts(kind, problem.metric_name),
                    higher_is_better=problem.higher_is_better,
                )
                if match is not None:
                    skill, skill_dir = match
                    reference_solution = skill_dir / SKILL_CODE_FILENAME
                    score = f"{skill.score:g} {skill.metric}" if skill.score is not None else "unscored"
                    reference_note = f"scored {score} on {skill.problem_id}"
                    log(f"learning: reference solution from {skill.run_ref} ({reference_note})")
            except Exception as exc:  # noqa: BLE001
                log(f"learning: skill selection failed (draft unaffected): {exc}")
    backend_obj = get_backend(config.backend, auth=config.backend_auth)
    if hasattr(backend_obj, "abort"):
        backend_obj.abort = abort
    from hillclimb.routing import BackendPool, Router

    backends = BackendPool(abort=abort)
    backends.seed(config.backend, config.backend_auth, backend_obj)
    from hillclimb.project import machine_cache_dir

    slots = (
        MachineSlots(machine_cache_dir() / "agent-slots", config.search.machine_max_agents)
        if config.search.machine_max_agents > 0
        else None
    )
    from hillclimb.policies import get_policy

    searcher = GreedySearcher(
        problem=problem,
        config=config,
        journal=journal,
        backend=backend_obj,
        executor=build_executor(config, problem, log),
        budget=budget,
        search_dir=search_dir,
        log=log,
        holdout_scorer=build_holdout_scorer(config, problem, search_dir, log),
        status=status,
        slots=slots,
        abort=abort,
        seed_solution=seed_from,
        knowledge_context=knowledge_context if knowledge_context is not None else _kc,
        reference_solution=reference_solution,
        reference_note=reference_note,
        complexity_start=_offset,
        policy=get_policy(
            config.search.policy, config.search.policy_params, complexity_start=_offset
        ),
        router=Router(config),
        backends=backends,
    )
    try:
        selected = searcher.run()
    except ParkedSearch as exc:
        status.finalize("parked", last_error=str(exc)[:500])
        return SearchOutcome(run_dir, search_dir, None, "parked", error=str(exc))
    except (StopRequested, KeyboardInterrupt) as exc:
        status.finalize("stopped", last_error=str(exc)[:500] or None)
        return SearchOutcome(run_dir, search_dir, None, "stopped", error=str(exc) or None)
    except Exception as exc:
        status.finalize("failed", last_error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    status.finalize("done")
    if problem.emflow_problem and selected is not None:
        _official_verify(config, problem, search_dir, journal, selected, log)
    if problem.mlebench_comp_id and selected is not None:
        _mlebench_grade(config, problem, search_dir, selected, log)
    _distill_knowledge(
        config, problem, search_dir, journal,
        target=target, budget_s=budget.total_s,
        cost_usd=searcher.total_cost_usd(), log=log,
    )
    return SearchOutcome(run_dir, search_dir, selected, "done")


def _official_verify(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    journal: Journal,
    selected: Candidate,
    log: Log,
) -> None:
    """One official emflow Verifier run on the selected model (scorecard +
    leaderboard row, with n_trials metadata for selection honesty). Best
    effort: a verify failure never fails a finished search."""
    from hillclimb.integrations.emflow.executor import official_verify

    try:
        python = ensure_runtime_venv(config, kind="emflow", log=log)
        official_verify(
            python,
            problem.emflow_problem,
            Path(selected.candidate_dir),
            search_dir / "holdout-eval" / "official",
            name=search_ref(search_dir),
            n_trials=len(journal.candidates),
            timeout_s=config.budget.exec_timeout_s,
            log=log,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"official verification failed (search result unaffected): {exc}")


def _mlebench_grade(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    selected: Candidate,
    log: Log,
) -> None:
    """One official `mlebench grade-sample` run on the selected candidate's
    submission, AFTER the search finishes — the private test set never
    influences selection (MLE-bench protocol). The report (score + medal
    flags) lands in mlebench-grade.json next to the search artifacts. Best
    effort: a grading failure never fails a finished search."""
    import json

    from hillclimb.grading import grade_submission

    try:
        submission = Path(selected.candidate_dir) / "submission.csv"
        if not submission.exists():
            raise FileNotFoundError(f"selected candidate has no submission.csv: {submission}")
        report = grade_submission(submission, problem.mlebench_comp_id, config)
        (search_dir / "mlebench-grade.json").write_text(json.dumps(report, indent=2))
        medal = next(
            (m for m in ("gold_medal", "silver_medal", "bronze_medal") if report.get(m)), None
        )
        log(
            f"mlebench grade: score={report.get('score')} "
            + (f"medal={medal.removesuffix('_medal')}" if medal else "no medal")
        )
    except Exception as exc:  # noqa: BLE001
        log(f"mlebench grading failed (search result unaffected): {exc}")


def run_search(
    target: str,
    *,
    budget_s: int | None = None,
    name: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
    config: Config | None = None,
    backend: str | None = None,
    model: str | None = None,
    holdout: bool = True,
    seed_from: Path | str | None = None,
    knowledge_context: str | None = None,
    log: Log = print,
) -> SearchOutcome:
    """Resolve a single-problem target, create the Run/Search dirs, and run
    the engine to completion. Suites are a CLI concern (parallel processes);
    this API runs exactly one search. `seed_from` scores an incumbent
    solution as the floor candidate a re-search must beat."""
    config = config or Config.load(backend=backend, model=model)
    if backend:
        config.backend = backend
    if model:
        config.model = model
    if not holdout:
        config.holdout.enabled = False
    problem = load_problem(target, config)
    if run_id is None:
        run_name = run_name or name or problem.problem_id
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
        run_dir = config.paths.runs_dir / run_id  # suite child: parent wrote run.yaml
    total_s = budget_s or problem.time_budget_s
    seed_path = Path(seed_from) if seed_from else None
    search_dir = create_search(config, problem, run_dir, run_id, total_s, seed_from=seed_path)
    log(
        f"Search {search_ref(search_dir)} (problem={problem.problem_id}, "
        f"backend={config.backend}, model={config.model}, budget={total_s}s)"
    )
    return execute_search(
        config,
        problem,
        search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        log,
        seed_from=seed_path,
        knowledge_context=knowledge_context,
        target=target,
    )
