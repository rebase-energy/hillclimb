"""Public programmatic API: run a hillclimb search in-process.

Typer-free — the CLI is a thin shell over these functions, and downstream
products embed them directly:

    import hillclimb
    outcome = hillclimb.run_search("emflow://gefcom2014:solar", budget_s=7200)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hillclimb.agents import get_agent
from hillclimb.agents.base import AgentRequest
from hillclimb.harness.budget import BudgetManager
from hillclimb.harness.candidate import Candidate
from hillclimb.config import Config
from hillclimb.harness.journal import Journal
from hillclimb.problem import ProblemSpec, load_problem
from hillclimb.harness.run import RunMeta, SearchMeta, new_search_uid
from hillclimb.harness.glue import (
    ParkedSearch,
    StopRequested,
    build_loop,
    build_operators,
    build_tuner,
    effective_memory,
    holdout_timing,
    search_climber,
)
from hillclimb.harness.oscompat import KILL_SIGNAL, new_group_kwargs, signal_group
from hillclimb.harness.status import SearchStatus, StatusWriter
from hillclimb.harness.store import key_for, open_store
from hillclimb.harness.dirs import allocate_search_dir, create_run_dir

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


from hillclimb.harness.run import search_ref  # noqa: E402,F401  (re-exported; the helper lives with harness/run.py's walkers)


def default_venv_python(config: Config, kind: str, requirements: Path | None = None) -> Path:
    """Shared machine venv path, keyed by a hash of the requirement set (and
    the emflow source, whose changes must also rebuild): built once per
    machine, shared by every hillclimb dir, new key = automatic rebuild.

    A problem-supplied `requirements` file keys on its CONTENT instead —
    editing the file rebuilds automatically, and problems with identical
    requirement sets share one venv."""
    import hashlib

    from hillclimb.harness.oscompat import venv_python
    from hillclimb.project import machine_cache_dir
    from hillclimb.runtime import runtime_packages

    if requirements is not None:
        digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]
        return venv_python(machine_cache_dir() / "venvs" / f"problem-{digest}")
    text = "\n".join(runtime_packages(kind))
    if kind == "emflow":
        text += f"\n{config.emflow.source}"
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return venv_python(machine_cache_dir() / "venvs" / f"{kind}-{digest}")


def ensure_runtime_venv(
    config: Config, kind: str = "csv", log: Log = print, requirements: Path | None = None
) -> Path:
    """Create the solution-script venv for the problem kind on first use.
    Explicitly configured paths are used as-is (hosted image, tests); the
    default is a shared hash-keyed venv under the machine cache dir. A
    problem-supplied `requirements` file gets its own content-keyed venv
    (the config path overrides stay kind-scoped and do not apply)."""
    from importlib import resources

    from hillclimb.harness.oscompat import lock_file
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
        lock_file(lock)
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


def interface_shim(log: Log = print) -> str | None:
    """PYTHONPATH entry making `from hillclimb import spaces` importable in
    the runtime venvs (always-on: one seam beats a conditional). Best-effort:
    a shim failure degrades to None — verifiers then simply lack the spaces
    library — rather than failing the search."""
    from hillclimb.runtime import ensure_interface_shim

    try:
        return str(ensure_interface_shim())
    except OSError as exc:
        log(f"interface shim unavailable ({exc}); verifiers run without hillclimb.spaces")
        return None


def build_executor(
    config: Config, problem: ProblemSpec, log: Log = print, search_dir: Path | None = None
):
    """The problem's verifier command, wired to the runtime venv it needs and
    confined to the sandbox (`harness/sandbox.py`)."""
    from hillclimb.harness.executor import CommandExecutor
    from hillclimb.harness.sandbox import verifier_policy

    env_extra = dict(problem.verifier_env)
    if config.hillclimb_dir is not None:
        # the hillclimb dir this verifier runs under: what a meta-problem's
        # verifier reads to give its inner searches the user's agent,
        # model and problems (`hillclimb meta evaluate`)
        env_extra.setdefault("HILLCLIMB_DIR", str(config.hillclimb_dir))
    return CommandExecutor(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.verifier_cmd,
        env_extra=env_extra,
        pythonpath=interface_shim(log),
        sandbox=verifier_policy(config, problem, search_dir),
    )


def build_unit_test_runner(
    config: Config, problem: ProblemSpec, log: Log = print, search_dir: Path | None = None
):
    """The run-frozen correctness gate, or None for verifier-only problems."""
    if problem.unit_tests is None:
        return None
    from hillclimb.harness.sandbox import verifier_policy
    from hillclimb.harness.unit_tests import UnitTestRunner

    return UnitTestRunner(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.unit_tests,
        pythonpath=interface_shim(log),
        sandbox=verifier_policy(config, problem, search_dir),
    )


def build_holdout_scorer(config: Config, problem: ProblemSpec, search_dir: Path, log: Log = print):
    """Hidden-split scorer, or None when holdout is off for this search —
    selection then climbs on validation alone."""
    if not config.holdout.enabled or problem.holdout_cmd is None:
        return None
    if problem.holdout_needs_credentials:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
            # fail fast: without credentials every holdout eval nans out and
            # the search burns debug cycles diagnosing the environment
            raise RuntimeError(
                "holdout scoring for this problem needs private data credentials: "
                "export HF_TOKEN (or HUGGINGFACE_TOKEN), or run with --no-holdout"
            )
        if not os.environ.get("HF_TOKEN"):
            # huggingface_hub only reads HF_TOKEN; without the mirror the
            # loader hits private datasets unauthenticated and the holdout
            # silently scores nan
            os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    from hillclimb.harness.executor import CommandHoldoutScorer
    from hillclimb.harness.sandbox import verifier_policy

    return CommandHoldoutScorer(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.holdout_cmd,
        problem_dir=problem.problem_dir,
        data_dir=problem.data_dir,
        work_root=search_dir / "holdout-eval",
        timeout_s=config.budget.exec_timeout_s,
        pythonpath=interface_shim(log),
        sandbox=verifier_policy(config, problem, search_dir, holdout=True),
    )


def build_evaluator(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    journal: "Journal",
    *,
    status=None,
    log: Log = print,
):
    """The host's evaluate service for one search: verifier trials plus the
    hidden split, scored at the timing the strategy's registry entry
    declares. Strategies receive this and never a holdout scorer."""
    from hillclimb.harness.evaluation import CandidateEvaluator

    return CandidateEvaluator(
        executor=build_executor(config, problem, log, search_dir),
        unit_test_runner=build_unit_test_runner(config, problem, log, search_dir),
        problem=problem,
        config=config,
        holdout_scorer=build_holdout_scorer(config, problem, search_dir, log),
        holdout_timing=holdout_timing(config, search_dir),
        journal=journal,
        status=status,
        log=log,
    )


def spent_seconds(journal: Journal) -> float:
    """Legacy resume accounting: sum of agent + trial work durations. Only a
    fallback — under parallel workers this overcounts wall-clock; prefer
    resume_spent_seconds."""
    return sum(
        (c.agent.agent_duration_s or 0)
        + sum(r.duration_s or 0 for t in c.trials for r in t.replicates)
        + sum(t.unit_tests.duration_s for t in c.trials if t.unit_tests is not None)
        for c in journal.candidates.values()
    )


def resume_spent_seconds(status, journal: Journal) -> float:
    """Wall-clock already consumed by a search, for budget seeding on resume.
    The status record persists budget.spent_s on every heartbeat (15 s) and
    on finalize, so this is exact for parked/stopped searches and loses at
    most one heartbeat on crashes. Falls back to the work-duration sum for
    pre-upgrade searches without a persisted budget."""
    if status is not None and status.budget.total_s > 0:
        return status.budget.spent_s
    return spent_seconds(journal)


def _sha256(path: Path) -> str | None:
    """Hex digest of a file's bytes; None when it cannot be read (the seed
    is validated later, where a missing file is a proper error)."""
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def create_run(config: Config, meta: RunMeta) -> Path:
    """The run dir, with the run recorded in the configured store."""
    run_dir = create_run_dir(config.paths.runs_dir, meta.run_id)
    with closing(open_store(config)) as store:
        store.record_run(meta)
    return run_dir


RUN_SPEC_FILE = "spec.yaml"


def spec_entry(
    target: str,
    *,
    name: str | None = None,
    budget: int | str | None = None,
    agent: str | None = None,
    model: str | None = None,
    climber: Any = None,
    parallel_agents: int | None = None,
    n_replicates: int | None = None,
    seed_from: Path | str | None = None,
    set: Sequence[str] = (),  # noqa: A002 — the spec key is `set`
) -> dict:
    """One `problems:` entry of a run spec, from the parameters a search
    actually launched with (see `write_run_spec`). Keys left None are left
    out; a seed path is made absolute, since the file will not sit next to
    the spec it came from; an integer budget is spelled in seconds. The
    climber (a name, a block, a `ClimberSpec`) is written as its full block —
    pass it with its file refs absolute (`Config.climber_block()`), for the
    same reason."""
    if isinstance(budget, int):
        budget = f"{budget}s"
    if seed_from is not None:
        seed_from = str(Path(seed_from).expanduser().resolve())
    if climber is not None:
        from hillclimb.climber import as_spec

        climber = as_spec(climber).block()
    entry = {
        "target": target, "name": name, "budget": budget, "agent": agent, "model": model,
        "climber": climber, "parallel_agents": parallel_agents, "n_replicates": n_replicates,
        "seed_from": seed_from, "set": list(set),
    }
    return {key: value for key, value in entry.items() if value not in (None, [])}


def write_run_spec(run_dir: Path, entries: list[dict], *, source: Path | str | None = None) -> Path:
    """`runs/<run-id>/spec.yaml`: the run's own spec, one `problems:` entry
    per search it launched, with every parameter the launch resolved to —
    so the run carries its recipe next to its record and artifacts, and
    `hillclimb run runs/<run-id>/spec.yaml` runs it again. It is generated
    from the resolved parameters, never copied from the spec a run was
    launched with (`source`, noted in the header): a copy would keep paths
    relative to a file that is not there."""
    import yaml

    lines = ["# The spec this run launched from — rerun it with: hillclimb run <this file>"]
    if source is not None:
        lines.append(f"# launched from: {source}")
    body = yaml.safe_dump({"problems": entries}, sort_keys=False, allow_unicode=True)
    path = run_dir / RUN_SPEC_FILE
    path.write_text("\n".join(lines) + "\n" + body)
    return path


def create_search(
    config: Config,
    problem: ProblemSpec,
    run_dir: Path,
    run_id: str,
    total_s: int,
    seed_from: Path | None = None,
    study: str | None = None,
    experiment: str | None = None,
    repeat: int = 0,
    experiment_overrides: dict | None = None,
) -> Path:
    import shutil

    from hillclimb import __version__
    from hillclimb.climber import ClimberLoadError, load_snapshot, snapshot_climber
    from hillclimb.harness.unit_tests import bundle_relative, freeze_for_run

    # a climber that cannot be loaded — a module that does not resolve, a
    # prompt that names a token nothing fills — fails here, before a search
    # dir exists
    climber = search_climber(config)
    climber.operator_set()
    climber.tuner()
    climber.graph_module()
    # ...and its loop must build: a param the policy does not have, a
    # selector setting that does not exist, a serial loop asked to run wide
    climber.build_loop(parallelism=max(1, config.concurrency.parallel_agents), log=lambda *_: None)
    problems = climber.lint_prompts()
    if problems:
        raise ValueError(f"climber {climber.name}: prompts do not lint clean:\n  " + "\n  ".join(problems))
    problem.unit_tests = freeze_for_run(problem, run_dir)
    search_dir = allocate_search_dir(run_dir, problem.problem_id)
    snapshot_climber(climber, search_dir)  # what the engine — and a resume — loads
    taken = load_snapshot(search_dir)
    if taken is None or taken.sha256 != climber.sha256:
        # the copy is not the climber: a file it needs was not reached (it is
        # imported some way the snapshot cannot follow). Better no search
        # than one that resumes as something else
        shutil.rmtree(search_dir, ignore_errors=True)
        raise ClimberLoadError(
            f"climber {climber.name}: its snapshot does not reproduce it — every local file it "
            "uses must be named in the block or imported relatively (`from .helpers import x`)"
        )
    meta = SearchMeta(
        search_id=search_dir.name,
        run_id=run_id,
        search_uid=new_search_uid(),
        problem=problem.target or str(problem.problem_dir),
        problem_id=problem.problem_id,
        problem_key=problem.problem_key,
        agent=config.agent,
        model=config.model,
        climber=climber.name,
        role="improver" if problem.solution_kind == "climber" else "solver",
        climber_sha256=climber.sha256,
        climber_spec=climber.spec.block(),
        hillclimb_version=__version__,
        routing={
            op: route.model_dump(exclude_none=True)
            for op, route in config.routing.items()
        },
        metric=problem.metric_name,
        higher_is_better=problem.higher_is_better,
        chart_baselines=problem.chart_baselines,
        provider_revision=problem.provider_revision,
        output_artifacts=problem.output_artifacts,
        unit_tests_bundle=(
            bundle_relative(problem.unit_tests, run_dir) if problem.unit_tests else None
        ),
        unit_tests_command=(list(problem.unit_tests.command) if problem.unit_tests else []),
        unit_tests_sha256=(problem.unit_tests.sha256 if problem.unit_tests else None),
        budget_s=total_s,
        holdout_enabled=config.holdout.enabled and problem.holdout_cmd is not None,
        seed_from=str(seed_from) if seed_from else None,
        seed_sha256=_sha256(seed_from) if seed_from else None,
        learning_enabled=effective_memory(config) != "none",
        study=study,
        experiment=experiment,
        repeat=repeat,
        experiment_overrides=dict(experiment_overrides or {}),
    )
    with closing(open_store(config)) as store:
        store.record_search(meta)
    return search_dir


def resolve_knowledge_dir(config: Config) -> Path | None:
    """Where the knowledge lives (None when learning is off or no dir is
    resolvable): `learning.dir`, else `<hillclimb dir>/knowledge`."""
    from hillclimb.modules.memory.files import knowledge_dir_for

    return knowledge_dir_for(config)


def build_knowledge_context(
    config: Config, problem: ProblemSpec, target: str, log: Log, *, search_dir: Path | None = None
) -> tuple[str | None, int, list[str]]:
    """(prior-experience prompt section, draft-complexity offset, injected
    claim ids) as the search's memory would hand them over — what `hillclimb
    knowledge …` shows and tests inspect. A memory other than `files` has no
    cards to show: (None, 0, [])."""
    from hillclimb.harness.glue import build_memory
    from hillclimb.modules.memory.base import MemoryEnv
    from hillclimb.modules.memory.files import FilesMemory

    memory = build_memory(config, search_dir)
    if not isinstance(memory, FilesMemory):
        return None, 0, []
    memory.bind(MemoryEnv(config=config, problem=problem, search_dir=search_dir, target=target, log=log))
    return memory.prior_experience()


def _raise_stop_requested(signum, frame):
    raise StopRequested(f"signal {signal.Signals(signum).name}")


def _preflight_sandbox(config: Config, log: Log) -> None:
    """Start the sandbox once before anything spends: a search whose agents
    and verifier cannot be confined does not start (SandboxUnavailable names
    the fix). Where the OS has none, say so and run without."""
    from hillclimb.harness import sandbox

    sandbox.log = log
    names = {config.agent} | {route.agent for route in config.routing.values() if route.agent}
    if not sandbox.enabled(config):
        log("sandbox: off — agents and solutions run with your full user rights")
    elif sandbox.backend() is None:
        log(
            "sandbox: none exists for this operating system — agents and "
            "solutions run with your full user rights"
        )
    else:
        log(f"sandbox: on ({sandbox.backend()})")
    if config.allow_internet_for_agents:
        return
    if names & {"claude-code", "pi"} and not sandbox.active(config):
        raise sandbox.SandboxUnavailable(sandbox.NEEDS_SANDBOX)
    log("agents: no internet")


def _preflight_pi_routes(
    config: Config, search_dir: Path, router, agents, log: Log, memory_passes: Sequence[str] = ()
) -> None:
    """Validate every statically reachable pi model/sampling combination.

    This deliberately happens before the baseline or first draft. Provider
    incompatibilities (notably models that reject temperature) therefore cost
    one tiny no-tools call rather than an entire failed candidate.
    """
    import hashlib
    import json

    from hillclimb.harness.sandbox import agent_policy

    # every operator this search's climber may call is routed by its own
    # name (`routing.draft`, `routing.gepa-reflect`, a climber's `crossover`)
    operators = set(build_operators(config, search_dir).names())
    operators.update(memory_passes)  # the agent calls the memory makes (claim distillation)
    operators.update(
        name
        for name in config.routing
        if name not in {"default", "distill", "consolidate"}
    )

    pending: dict[str, tuple] = {}
    for operator in sorted(operators):
        route = router.resolve(operator)
        if route.agent != "pi":
            continue
        models = [route.model]
        for layer in (
            config.routing.get(operator),
            config.routing.get("default"),
        ):
            if layer is None:
                continue
            if layer.models:
                models = list(layer.models)
                break
            if layer.model:
                break
        for model in models:
            key = json.dumps(
                [route.agent_auth, model, route.sampling],
                sort_keys=True,
                separators=(",", ":"),
            )
            pending.setdefault(key, (operator, route, model))

    for key, (operator, route, model) in pending.items():
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        work_dir = search_dir / "pi-preflight" / digest
        work_dir.mkdir(parents=True, exist_ok=True)
        agent = agents.get("pi", route.agent_auth)
        preflight = getattr(agent, "preflight", None)
        if preflight is None:
            raise RuntimeError("pi agent does not implement preflight")
        log(
            f"pi preflight: model={model}"
            + (f" sampling={route.sampling}" if route.sampling else "")
        )
        result = preflight(
            AgentRequest(
                operator=operator,
                prompt="Reply with exactly pong.",
                candidate_dir=work_dir,
                timeout_s=min(120, max(10, config.budget.agent_timeout_s)),
                model=model,
                sampling=route.sampling,
                # the same confinement the search's calls run under, so a
                # provider the proxy does not let through fails here
                allow_internet=config.allow_internet_for_agents,
                sandbox=agent_policy(config, search_dir, "pi"),
            )
        )
        (work_dir / "result.json").write_text(result.model_dump_json(indent=2))
        if not result.ok:
            detail = result.error_message or result.error_kind or "unknown provider error"
            raise RuntimeError(
                f"pi preflight failed for model={model}, sampling={route.sampling}: {detail}"
            )


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
    from hillclimb.harness.glue import build_memory
    from hillclimb.modules.memory.base import MemoryEnv

    # the memory this search runs under: its climber's, unless the user
    # switched learning off
    memory = build_memory(config, search_dir)
    if config.learning.enabled and not memory.enabled:
        config = config.model_copy(deep=True)
        config.learning.enabled = False  # this search neither reads nor writes memory
        log("memory: none (this climber runs without cross-search memory)")
    memory.bind(MemoryEnv(config=config, problem=problem, search_dir=search_dir, target=target, log=log))
    store = open_store(config)
    key = key_for(search_dir)
    store.clear_stale_stops(key)
    journal = Journal(store.journal(key))
    status = StatusWriter(
        lambda s: store.write_status(key, s),
        SearchStatus(
            search_id=search_dir.name,
            run_id=run_dir.name,
            state="running",
            pid=os.getpid(),
        ),
        budget=budget,
    )
    status.start_heartbeat()
    # SIGTERM before the harness exists just raises; once it does, the
    # handler latches it closed first (installed below)
    try:  # signal handlers are main-thread-only; embedded callers skip them
        signal.signal(signal.SIGTERM, _raise_stop_requested)
    except ValueError:
        pass
    import threading

    from hillclimb.harness.slots import MachineSlots

    abort = threading.Event()
    # what the search is handed before it starts: prior experience for the
    # prompts (an externally supplied section stands in for the memory's
    # own), a reference solution, and what memory learned about how to start
    retrieved = memory.retrieve(context=knowledge_context)
    agent_obj = get_agent(
        config.agent,
        auth=config.agent_auth,
        pi_models_file=config.pi.models_file,
    )
    if hasattr(agent_obj, "abort"):
        agent_obj.abort = abort
    from hillclimb.harness.routing import AgentPool, Router

    agents = AgentPool(
        abort=abort, pi_models_file=config.pi.models_file
    )
    agents.seed(config.agent, config.agent_auth, agent_obj)
    router = Router(config)

    try:
        _preflight_sandbox(config, log)
        _preflight_pi_routes(config, search_dir, router, agents, log, memory_passes=memory.agent_passes())
    except (StopRequested, KeyboardInterrupt) as exc:
        status.finalize("stopped", last_error=str(exc)[:500] or None)
        store.close()
        return SearchOutcome(run_dir, search_dir, None, "stopped", error=str(exc) or None)
    except Exception as exc:
        status.finalize("failed", last_error=f"{type(exc).__name__}: {exc}"[:500])
        store.close()
        raise
    from hillclimb.project import machine_cache_dir

    machine_max = config.concurrency.effective_machine_max_agents()
    slots = MachineSlots(machine_cache_dir() / "agent-slots", machine_max) if machine_max > 0 else None
    evaluator = build_evaluator(config, problem, search_dir, journal, status=status, log=log)
    from hillclimb.harness.core import Harness

    # validated before anything is scored: a bad climber costs nothing
    climber = search_climber(config, search_dir)
    problems = climber.lint_prompts()
    if problems:
        raise ValueError(f"climber {climber.name}: prompts do not lint clean: " + "; ".join(problems))
    # what memory learned about how to start, fixed when the search first
    # runs: a resume reads the record, never what the knowledge says by now
    record = store.search(key)
    priors = record.meta.memory_priors if record is not None else None
    if priors is None:
        priors = dict(retrieved.priors)
        if record is not None:
            record.meta.memory_priors = priors
            store.record_search(record.meta)
    loop = build_loop(config, priors=priors, log=log, search_dir=search_dir)
    harness = Harness(
        problem=problem,
        config=config,
        journal=journal,
        agent=agent_obj,
        executor=evaluator.executor,
        budget=budget,
        search_dir=search_dir,
        log=log,
        evaluator=evaluator,
        status=status,
        slots=slots,
        abort=abort,
        seed_solution=seed_from,
        memory=memory,
        retrieved=retrieved,
        router=router,
        agents=agents,
        drain_commands=lambda: store.drain_commands(key),
        operators=build_operators(config, search_dir),
        prompts_dir=climber.prompts_dir,
        tuner=build_tuner(config, search_dir),
    )

    def _stop_on_signal(signum, frame):
        reason = f"signal {signal.Signals(signum).name}"
        harness.request_stop(reason)
        raise StopRequested(reason)

    try:
        signal.signal(signal.SIGTERM, _stop_on_signal)
    except ValueError:
        pass

    def finalize(state: str, last_error: str | None = None) -> None:
        status.finalize(state, last_error=last_error)
        store.close()

    try:
        selected = harness.execute(loop)
        selected = _finish_holdout(config, problem, search_dir, journal, evaluator, selected, log)
    except ParkedSearch as exc:
        finalize("parked", last_error=str(exc)[:500])
        return SearchOutcome(run_dir, search_dir, None, "parked", error=str(exc))
    except (StopRequested, KeyboardInterrupt) as exc:
        finalize("stopped", last_error=str(exc)[:500] or None)
        return SearchOutcome(run_dir, search_dir, None, "stopped", error=str(exc) or None)
    except Exception as exc:
        finalize("failed", last_error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finalize("done")
    if problem.emflow_problem and selected is not None:
        _official_verify(config, problem, search_dir, journal, selected, log)
    if problem.mlebench_comp_id and selected is not None:
        _mlebench_grade(config, problem, search_dir, selected, log)
    # what the finished search leaves for the next ones (and its own card,
    # beside its artifacts, under any memory)
    memory.record(journal, budget_s=budget.total_s, cost_usd=harness.total_cost_usd())
    return SearchOutcome(run_dir, search_dir, selected, "done")


def _finish_holdout(
    config: Config,
    problem: ProblemSpec,
    search_dir: Path,
    journal: Journal,
    evaluator,
    selected: Candidate | None,
    log: Log,
) -> Candidate | None:
    """Host-side holdout for `after` timing: score the top-k hidden splits
    now that the strategy has returned, repoint best/ and re-select. A
    no-op for `inline` timing (every candidate was scored as it landed)."""
    if evaluator.holdout_timing != "after" or evaluator.holdout_scorer is None:
        return selected
    from hillclimb.harness.control import resync_best

    scored = evaluator.finalize_holdout(journal)
    if scored:
        log(f"holdout scored for {', '.join(c.candidate_id for c in scored)}")
    resync_best(
        search_dir, journal, problem.higher_is_better,
        config.holdout.selection, problem.output_artifacts,
    )
    return journal.selected_candidate(problem.higher_is_better, config.holdout.selection)


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
    from hillclimb.providers.emflow.executor import official_verify

    try:
        python = ensure_runtime_venv(config, kind="emflow", log=log)
        official_verify(
            python,
            problem.emflow_problem,
            Path(selected.candidate_dir),
            search_dir / "holdout-eval" / "official",
            name=search_ref(search_dir),
            n_trials=len(journal.candidates),  # emflow metadata: candidates tried, not replicates
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

    from hillclimb.harness.grading import grade_submission

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
    agent: str | None = None,
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
    config = config or Config.load(agent=agent, model=model)
    if agent:
        config.agent = agent
    if model:
        config.model = model
    if not holdout:
        config.holdout.enabled = False
    problem = load_problem(target, config)
    run_dir_is_new = run_id is None
    if run_id is None:
        run_name = run_name or name or problem.problem_id
        run_id = new_run_id(run_name)
        run_dir = create_run(
            config,
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
    if run_dir_is_new:
        write_run_spec(run_dir, [spec_entry(
            target, budget=total_s, agent=config.agent, model=config.model,
            climber=config.climber_block(), parallel_agents=config.concurrency.parallel_agents,
            n_replicates=config.evaluation.n_replicates, seed_from=seed_path,
        )])
    search_dir = create_search(config, problem, run_dir, run_id, total_s, seed_from=seed_path)
    log(
        f"Search {search_ref(search_dir)} (problem={problem.problem_id}, "
        f"agent={config.agent}, model={config.model}, budget={total_s}s)"
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


# -- fleets: N detached engines under one run -----------------------------------
#
# `run --parallel-searches` and the hosted platform: independent searches on one
# problem, each its own engine process, sharing discoveries through the run
# dir's live knowledge cards. Processes, not threads: status.json carries the
# engine's pid, so two searches in one process would be indistinguishable to
# every viewer.


def child_launch_context(config: Config) -> tuple[Path, dict[str, str]]:
    """(cwd, env) for a child `hillclimb run`: rooted at the hillclimb dir
    with HILLCLIMB_DIR pinned, so the child never has to search."""
    root = config.hillclimb_dir or Path.cwd()
    return root, {**os.environ, "HILLCLIMB_DIR": str(root)}


def spawn_search_proc(
    config: Config, run_dir: Path, index: int, slug: str, run_argv: list[str]
) -> tuple[subprocess.Popen, Path]:
    """Start a detached `hillclimb run <run_argv...>` as one search of
    `run_dir`, logging to <run>/logs/NN-<slug>.log. The one launcher behind
    suites and fleets. Returns (child, log path)."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{index:02d}-{slug}.log"
    cwd, env = child_launch_context(config)
    cmd = [sys.executable, "-m", "hillclimb.cli", "run", *run_argv]
    with log_path.open("w") as out:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            **new_group_kwargs(detached=True),
        )
    return proc, log_path


def create_problem_run(config: Config, run_name: str, target: str, problem_id: str) -> Path:
    """A new run dir + run.yaml for searches on one problem."""
    run_id = new_run_id(run_name)
    return create_run(
        config,
        RunMeta(run_id=run_id, name=run_name, kind="problem", target=target, problem_ids=[problem_id]),
    )


def climber_argv(climber: Any) -> list[str]:
    """How a climber reaches a child engine, which gets nothing but argv: a
    name (a preset, one .py file) as `--climber`, a block as a `--set
    climber=<json>` — it must come before the other `--set` pairs, which may
    edit its fields."""
    if climber is None:
        return []
    if isinstance(climber, str):
        return ["--climber", climber]
    from hillclimb.climber import as_spec

    return ["--set", "climber=" + json.dumps(as_spec(climber).block())]


def fleet_argv(
    target: str,
    run_dir: Path,
    run_name: str,
    *,
    budget: int | str | None = None,
    agent: str | None = None,
    model: str | None = None,
    climber: Any = None,
    parallel_agents: int | None = None,
    n_replicates: int | None = None,
    holdout: bool = True,
    learning: bool = True,
    seed_from: Path | str | None = None,
    knowledge_context_file: Path | str | None = None,
    overrides: list[str] | tuple[str, ...] = (),
    study: str | None = None,
    experiment: str | None = None,
    repeat: int = 0,
) -> list[str]:
    """The `hillclimb run` arguments an engine of a fleet is started with.
    `budget` is the CLI's wall-clock spelling (`2h`, `30m`) or plain seconds.
    `study`/`experiment` tag the search as one experiment of a mixed fleet
    (see `FleetEngine`), so `hillclimb experiment report` compares them."""
    argv = [target, "--run-id", run_dir.name, "--run-name", run_name]
    if study:
        argv += ["--study", study, "--experiment", experiment or study]
        if repeat:
            argv += ["--repeat", str(repeat)]
    if budget:
        argv += ["--budget", f"{budget}s" if isinstance(budget, int) else str(budget)]
    if agent:
        argv += ["--agent", agent]
    if model:
        argv += ["--model", model]
    argv += climber_argv(climber)
    if parallel_agents is not None:
        argv += ["--parallel-agents", str(parallel_agents)]
    if n_replicates is not None:
        argv += ["--n-replicates", str(n_replicates)]
    if not holdout:
        argv.append("--no-holdout")
    if not learning:
        argv.append("--no-learning")
    if seed_from:
        argv += ["--seed-from", str(seed_from)]
    if knowledge_context_file:
        argv += ["--knowledge-context-file", str(knowledge_context_file)]
    for pair in overrides:
        argv += ["--set", pair]
    return argv


@dataclass(frozen=True)
class FleetEngine:
    """One engine of a mixed fleet: the experiment it is tagged as, the climber it
    runs (a preset's name, one .py file, a block or a `ClimberSpec`), and the
    `--set` overrides that apply to this engine only (after the fleet-wide
    ones, so they win). `climber=None` keeps the fleet-wide climber.
    Overrides are the one per-experiment knob — `concurrency.parallel_agents=1`
    for a serial climber like GEPA, `climber.params.seed=7`, anything
    `Config.apply_overrides` accepts."""

    experiment: str
    climber: Any = None
    overrides: tuple[str, ...] = ()
    repeat: int = 0


def mixed_fleet(
    climbers: Sequence[Any],
    *,
    repeats: int = 1,
    experiment_overrides: Mapping[str, Sequence[str]] | None = None,
) -> list[FleetEngine]:
    """The engines of a fleet that runs one search per climber on the same
    problem. Experiments are named after their climber (a repeated climber
    gets a `-2`, `-3` suffix); `repeats` > 1 clones every experiment that
    many times, repeat-major so every experiment has seen the same shared
    state when it starts. `experiment_overrides` maps an experiment name to
    that experiment's `--set` pairs; a name that matches no experiment is an
    error."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not climbers:
        raise ValueError("a mixed fleet needs at least one climber")
    from hillclimb.climber import as_spec

    experiments: list[tuple[str, Any]] = []
    seen: dict[str, int] = {}
    for climber in climbers:
        label = as_spec(climber).label  # its name, else its policy's: a one-file climber's is its stem
        count = seen.get(label, 0) + 1
        seen[label] = count
        experiments.append((label if count == 1 else f"{label}-{count}", climber))
    overrides = {name: tuple(pairs) for name, pairs in (experiment_overrides or {}).items()}
    unknown = sorted(set(overrides) - {name for name, _ in experiments})
    if unknown:
        raise ValueError(
            f"experiment override for unknown experiment(s) {', '.join(unknown)}; "
            f"experiments are {', '.join(name for name, _ in experiments)}"
        )
    return [
        FleetEngine(experiment=name, climber=climber, overrides=overrides.get(name, ()), repeat=repeat if repeats > 1 else 0)
        for repeat in range(1, repeats + 1)
        for name, climber in experiments
    ]


@dataclass
class FleetHandle:
    """The engines of one fleet run. `wait` reaps them; exit codes follow the
    CLI's contract (0 done, 2 parked/stopped, anything else failed)."""

    run_dir: Path
    procs: list[subprocess.Popen]
    log_paths: list[Path]

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def alive(self) -> list[subprocess.Popen]:
        return [proc for proc in self.procs if proc.poll() is None]

    def wait(self, *, poll_s: float = 5.0, deadline_s: float | None = None) -> dict[int, int | None]:
        """Block until every engine has exited, or `deadline_s` seconds have
        passed. Returns pid -> exit code (None for engines still running)."""
        started = time.monotonic()
        while self.alive():
            if deadline_s is not None and time.monotonic() - started >= deadline_s:
                break
            time.sleep(poll_s)
        return {proc.pid: proc.poll() for proc in self.procs}

    def terminate(self, *, grace_s: float = 10.0) -> None:
        """SIGTERM every engine's process group (the engine parks its search
        on SIGTERM), then SIGKILL whatever is still alive after `grace_s`."""
        for proc in self.alive():
            _signal_group(proc, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while self.alive() and time.monotonic() < deadline:
            time.sleep(0.2)
        for proc in self.alive():
            _signal_group(proc, KILL_SIGNAL)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        signal_group(proc.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.send_signal(sig)


def run_fleet(
    target: str,
    *,
    config: Config,
    parallel_searches: int = 1,
    run_name: str | None = None,
    budget: int | str | None = None,
    agent: str | None = None,
    model: str | None = None,
    climber: Any = None,
    parallel_agents: int | None = None,
    n_replicates: int | None = None,
    holdout: bool = True,
    learning: bool = True,
    seed_from: Path | str | None = None,
    knowledge_context_file: Path | str | None = None,
    overrides: list[str] | tuple[str, ...] = (),
    engines: Sequence[FleetEngine] | None = None,
    study: str | None = None,
    log: Log = print,
) -> FleetHandle:
    """N independent searches on one problem, each its own detached engine
    under one run. Builds the solution venv once first so the engines do not
    race for it. Returns immediately; `FleetHandle.wait` reaps the engines.

    Two shapes. `parallel_searches=N`: N identical engines. `engines=[...]`
    (see `FleetEngine`, `mixed_fleet`): one engine per entry, each with its
    own climber and overrides on top of the fleet-wide arguments, tagged as
    an experiment of `study` (default: the run id) so `hillclimb experiment
    report <run-id>` compares them — three optimizers on one problem under
    one run. `parallel_searches` is ignored when `engines` is given."""
    if engines is not None and not engines:
        raise ValueError("engines must hold at least one FleetEngine")
    if engines is None and parallel_searches < 1:
        raise ValueError("parallel_searches must be >= 1")
    problem = load_problem(target, config)
    ensure_runtime_venv(config, problem.runtime, log=log, requirements=problem.requirements_file)
    name = run_name or problem.problem_id
    run_dir = create_problem_run(config, name, target, problem.problem_id)
    shared_entry = dict(
        budget=budget, agent=agent or config.agent, model=model or config.model,
        parallel_agents=parallel_agents, n_replicates=n_replicates, seed_from=seed_from,
    )
    from hillclimb.climber import as_spec

    def block(named: Any) -> dict:
        """The climber an engine runs, as the run's spec records it: the full
        block, relative file refs resolved from the hillclimb dir."""
        if named is None:
            return config.climber_block()
        return as_spec(named).anchored(config.hillclimb_dir).block()

    if engines is None:
        entries = [
            spec_entry(target, climber=block(climber), set=overrides, **shared_entry)
            for _ in range(parallel_searches)
        ]
    else:
        entries = [
            spec_entry(
                target, name=engine.experiment, climber=block(engine.climber or climber),
                set=(*overrides, *engine.overrides), **shared_entry,
            )
            for engine in engines
        ]
    write_run_spec(run_dir, entries)
    # Freeze before starting any child engine. Every search in the fleet then
    # binds to this same run-owned bundle, even if the live problem tree changes
    # while the fleet is running.
    from hillclimb.harness.unit_tests import freeze_for_run

    problem.unit_tests = freeze_for_run(problem, run_dir)
    shared = dict(
        budget=budget, agent=agent, model=model,
        parallel_agents=parallel_agents, n_replicates=n_replicates, holdout=holdout,
        learning=learning, seed_from=seed_from, knowledge_context_file=knowledge_context_file,
    )
    plan: list[tuple[str, list[str]]] = []  # (log slug, argv) per engine
    if engines is None:
        argv = fleet_argv(target, run_dir, name, climber=climber, overrides=overrides, **shared)
        plan = [(problem.problem_id, argv)] * parallel_searches
    else:
        study = study or run_dir.name
        for engine in engines:
            argv = fleet_argv(
                target, run_dir, name, climber=engine.climber or climber,
                overrides=(*overrides, *engine.overrides),
                study=study, experiment=engine.experiment, repeat=engine.repeat, **shared,
            )
            slug = f"{problem.problem_id}-{engine.experiment}" + (f"-r{engine.repeat}" if engine.repeat else "")
            plan.append((slug, argv))
    procs: list[subprocess.Popen] = []
    log_paths: list[Path] = []
    for index, (slug, argv) in enumerate(plan, 1):
        proc, log_path = spawn_search_proc(config, run_dir, index, slug, argv)
        procs.append(proc)
        log_paths.append(log_path)
    return FleetHandle(run_dir=run_dir, procs=procs, log_paths=log_paths)
