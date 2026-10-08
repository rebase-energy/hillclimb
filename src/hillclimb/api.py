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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from hillclimb.agents import get_agent
from hillclimb.agents.base import AgentRequest
from hillclimb.harness.budget import Budget, BudgetManager, resolve_budget
from hillclimb.harness.candidate import Candidate
from hillclimb.config import Config, parse_set_overrides
from hillclimb.harness.journal import Journal
from hillclimb.problem import Problem, ProblemSpec, load_problem
from hillclimb.results import SearchOutcome, open_search  # noqa: F401  (re-exported: the result of a search)
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

if TYPE_CHECKING:
    from hillclimb.harness.loop import Outcome
    from hillclimb.modules.policies.base import Action, SearchState

Log = Callable[[str], None]


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
        log(f"Preparing the {kind} runtime (first use on this machine; cached in {venv_dir}) ...")

        def uv(*args: str) -> None:
            # uv's download log stays out of the user's output unless it fails
            done = subprocess.run(["uv", *args], capture_output=True, text=True)
            if done.returncode != 0:
                tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-15:])
                raise subprocess.CalledProcessError(done.returncode, ["uv", *args], output=done.stdout, stderr=tail)

        try:
            uv("venv", "--python", "3.12", str(venv_dir))
            if requirements is not None:
                uv("pip", "install", "-r", str(requirements), "--python", str(python))
            else:
                with resources.as_file(requirements_resource(kind)) as req:
                    uv("pip", "install", "-r", str(req), "--python", str(python))
            if requirements is None and kind == "emflow":
                # flag forms ("-e ../emflow") split into args; requirement
                # specs ("emflow @ git+…", "/src/emflow") are ONE argument —
                # shlex would shred the PEP 508 " @ " form
                source = config.emflow.source.strip()
                source_args = shlex.split(source) if source.startswith("-") else [source]
                try:
                    uv("pip", "install", *source_args, "--python", str(python))
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Installing emflow from {config.emflow.source!r} failed — first use "
                        "needs network (or set `emflow.source` to a local checkout, e.g. '-e ../emflow')"
                    ) from exc
        except BaseException as exc:
            # a partial venv would pass the exists() check forever
            import shutil

            shutil.rmtree(venv_dir, ignore_errors=True)
            if isinstance(exc, subprocess.CalledProcessError):
                raise RuntimeError(
                    f"Building the {kind} runtime failed ({' '.join(map(str, exc.cmd[:3]))} …):\n{exc.stderr}"
                ) from exc
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
    from hillclimb.harness.sandbox import holdout_input_paths, private_paths, verifier_policy

    env_extra = dict(problem.verifier_env)
    if config.hillclimb_dir is not None:
        # the hillclimb dir this verifier runs under: what a meta-problem's
        # verifier reads to give its inner searches the user's coding agent,
        # model and problems (`hillclimb grade`)
        env_extra.setdefault("HILLCLIMB_DIR", str(config.hillclimb_dir))
    return CommandExecutor(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.verifier_cmd,
        env_extra=env_extra,
        pythonpath=interface_shim(log),
        sandbox=verifier_policy(config, problem, search_dir),
        score_argv=problem.score_cmd,
        private=private_paths(problem),
        holdout_inputs=holdout_input_paths(problem),
        time_limit_s=problem.solution_time_limit_s,
        cpus=config.concurrency.solution_cpus,
    )


def build_unit_test_runner(
    config: Config, problem: ProblemSpec, log: Log = print, search_dir: Path | None = None
):
    """The run-frozen correctness gate, or None for verifier-only problems."""
    if problem.unit_tests is None:
        return None
    from hillclimb.harness.sandbox import hidden_from_agents, verifier_policy
    from hillclimb.harness.unit_tests import UnitTestRunner

    policy = verifier_policy(config, problem, search_dir)
    return UnitTestRunner(
        ensure_runtime_venv(
            config, kind=problem.runtime, log=log, requirements=problem.requirements_file
        ),
        problem.unit_tests,
        pythonpath=interface_shim(log),
        # the tests run the solution: they read what it may, no more
        sandbox=policy.unreadable(*hidden_from_agents(problem)) if policy is not None else None,
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
    from hillclimb.harness.sandbox import holdout_input_paths, private_paths, verifier_policy

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
        score_argv=problem.score_cmd,
        private=private_paths(problem),
        holdout_inputs=holdout_input_paths(problem),
        time_limit_s=problem.solution_time_limit_s,
        cpus=config.concurrency.solution_cpus,
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
    """Legacy resume accounting: sum of coding agent + trial work durations. Only a
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
    taken = load_snapshot(search_dir) if climber.portable else climber
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
        climber_portable=climber.portable,
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
    """Start the sandbox once before anything spends: a search whose coding agents
    and verifier cannot be confined does not start (SandboxUnavailable names
    the fix). Where the OS has none, say so and run without."""
    from hillclimb.harness import sandbox

    sandbox.log = log
    names = {config.agent} | {route.agent for route in config.routing.values() if route.agent}
    if not sandbox.enabled(config):
        log("sandbox: off — coding agents and solutions run with your full user rights")
    elif sandbox.backend() is None:
        log(
            f"sandbox: {sandbox.no_sandbox_reason()} — coding agents and "
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
    operators.update(memory_passes)  # the coding agent calls the memory makes (claim distillation)
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
            raise RuntimeError("pi coding agent does not implement preflight")
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


class Search:
    """One search, open in this process: everything the engine sets up around
    a `Harness` (memory, store, journal, status + heartbeat, agents, evaluator,
    the climber's loop), held so the search can be run to the end in one call
    (`finish`, what `execute_search` does) or driven a step at a time
    (`begin`, then `propose` / `run` / `step`) and settled exactly once.

        search = hillclimb.api.start("fitness-landscape", climber=climber, agent="toy")
        action = search.propose()       # what the policy wants next; nothing runs
        outcome = search.run(action)    # run it, commit it, let the policy observe it
        outcome = search.step()         # both in one call
        search.finish()                 # the climber's own loop runs the rest
        result = search.close()         # the SearchOutcome `run` returns

    `hillclimb.Climber.start` is this, spelled on the climber. Stepping is
    serial and for a policy climber; a loop climber (gepa) owns its control
    flow and can only `finish`."""

    def __init__(
        self,
        config: Config,
        problem: ProblemSpec,
        search_dir: Path,
        budget: BudgetManager,
        log: Log = print,
        seed_from: Path | None = None,
        knowledge_context: str | None = None,
        target: str = "",
        interrupt_raises: bool = False,
    ):
        self.config = config
        self.problem = problem
        self.search_dir = search_dir
        # Ctrl-C from Python (`hillclimb.run`, `Climber.search`): the search is
        # settled `stopped` (resumable), then the KeyboardInterrupt goes on, so
        # a script or a notebook stops instead of starting its next search.
        # The CLI keeps it: it prints how to resume and exits 2
        self.interrupt_raises = interrupt_raises
        self.run_dir = search_dir.parents[1]
        self.budget = budget
        self.log = log
        self.seed_from = seed_from
        self.knowledge_context = knowledge_context
        self.target = target
        self.outcome: SearchOutcome | None = None  # set when the search is settled
        self.harness = None
        self.loop = None
        self._stepping = False
        self._handler = None  # the SIGTERM handler this search installed
        self._previous_handler = None

    # --- setup ---

    def open(self) -> SearchOutcome | None:
        """Set the search up, in the order the engine always has. Returns None
        when it is ready, or the `stopped` outcome when a stop arrived first.
        A setup that fails settles the search `failed` and raises: nothing is
        left running behind it (the heartbeat, the store)."""
        from hillclimb.harness.glue import build_memory
        from hillclimb.modules.memory.base import MemoryEnv

        log, search_dir = self.log, self.search_dir
        # the memory this search runs under: its climber's, unless the user
        # switched learning off
        memory = build_memory(self.config, search_dir)
        if self.config.learning.enabled and not memory.enabled:
            self.config = self.config.model_copy(deep=True)
            self.config.learning.enabled = False  # this search neither reads nor writes memory
            log("memory: none (this climber runs without cross-search memory)")
        memory.bind(MemoryEnv(
            config=self.config, problem=self.problem, search_dir=search_dir, target=self.target, log=log,
        ))
        self.memory = memory
        self.store = open_store(self.config)
        self.key = key_for(search_dir)
        self.store.clear_stale_stops(self.key)
        self.journal = Journal(self.store.journal(self.key))
        self.status = StatusWriter(
            lambda s: self.store.write_status(self.key, s),
            SearchStatus(
                search_id=search_dir.name,
                run_id=self.run_dir.name,
                state="running",
                pid=os.getpid(),
            ),
            budget=self.budget,
        )
        self.status.start_heartbeat()
        # SIGTERM before the harness exists just raises; once it does, the
        # handler latches it closed first (installed below)
        self._install_handler(_raise_stop_requested)
        try:
            self._build()
        except (StopRequested, KeyboardInterrupt) as exc:
            outcome = self._settle("stopped", exc)
            self._reraise_interrupt(exc)
            return outcome
        except Exception as exc:
            self._settle("failed", exc)
            raise
        self._install_handler(self._stop_on_signal)
        return None

    def _build(self) -> None:
        """Agents, preflights, evaluator, the climber's loop, the harness."""
        import threading

        from hillclimb.harness.slots import MachineSlots

        config, problem, search_dir, log = self.config, self.problem, self.search_dir, self.log
        memory, store, key, journal, status = self.memory, self.store, self.key, self.journal, self.status
        abort = threading.Event()
        from hillclimb.harness.procs import CHILDREN_FILE, track_children

        # every coding agent and verifier this engine starts, while it lives:
        # what `resume`/`stop`/`kill` stop if this process is killed outright
        track_children(search_dir / CHILDREN_FILE)
        knowledge = resolve_knowledge_dir(config)
        if knowledge is not None and config.hillclimb_dir is not None and knowledge.parent == config.hillclimb_dir:
            from hillclimb.project import ensure_owned_dir

            ensure_owned_dir(knowledge)  # the folder's own knowledge/, marked for `reset`
        # what the search is handed before it starts: prior experience for the
        # prompts (an externally supplied section stands in for the memory's
        # own), a reference solution, and what memory learned about how to start
        retrieved = memory.retrieve(context=self.knowledge_context)
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

        _preflight_sandbox(config, log)
        _preflight_pi_routes(config, search_dir, router, agents, log, memory_passes=memory.agent_passes())
        from hillclimb.project import machine_cache_dir

        machine_max = config.concurrency.effective_machine_max_agents()
        slots = MachineSlots(machine_cache_dir() / "agent-slots", machine_max) if machine_max > 0 else None
        self.evaluator = evaluator = build_evaluator(config, problem, search_dir, journal, status=status, log=log)
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
        self.loop = build_loop(config, priors=priors, log=log, search_dir=search_dir)
        self.harness = Harness(
            problem=problem,
            config=config,
            journal=journal,
            agent=agent_obj,
            executor=evaluator.executor,
            budget=self.budget,
            search_dir=search_dir,
            log=log,
            evaluator=evaluator,
            status=status,
            slots=slots,
            abort=abort,
            seed_solution=self.seed_from,
            memory=memory,
            retrieved=retrieved,
            router=router,
            agents=agents,
            drain_commands=lambda: store.drain_commands(key),
            operators=build_operators(config, search_dir),
            prompts_dir=climber.prompts_dir,
            tuner=build_tuner(config, search_dir),
        )

    # --- SIGTERM: one handler while the search is open, the previous one after ---

    def _install_handler(self, handler) -> None:
        try:  # signal handlers are main-thread-only; embedded callers skip them
            previous = signal.signal(signal.SIGTERM, handler)
        except ValueError:
            return
        if self._handler is None:
            self._previous_handler = previous
        self._handler = handler

    def _restore_handler(self) -> None:
        if self._handler is None:
            return
        try:
            if signal.getsignal(signal.SIGTERM) == self._handler:  # not ours any more: leave it
                signal.signal(signal.SIGTERM, self._previous_handler or signal.SIG_DFL)
        except ValueError:
            pass
        self._handler = None

    def _stop_on_signal(self, signum, frame):
        reason = f"signal {signal.Signals(signum).name}"
        self.harness.request_stop(reason)
        raise StopRequested(reason)

    # --- settling: every way a search ends goes through here, once ---

    def _so_far(self) -> Candidate | None:
        """What a search that did not finish would ship if it ended here."""
        return self.journal.selected_candidate(self.problem.higher_is_better, self.config.holdout.selection)

    def _finalize(self, state: str, last_error: str | None = None) -> None:
        from hillclimb.harness.procs import track_children

        track_children(None)  # settled: nothing of this search runs any more
        self.status.finalize(state, last_error=last_error)
        self.store.close()

    def _settle(self, state: str, exc: BaseException) -> SearchOutcome:
        """End the search as `parked`, `stopped` or `failed` because of `exc`."""
        if state == "failed":
            error = f"{type(exc).__name__}: {exc}"
            self._finalize(state, last_error=error[:500])
        elif state == "parked":
            error = str(exc)
            self._finalize(state, last_error=error[:500])
        else:
            error = str(exc) or None
            self._finalize(state, last_error=str(exc)[:500] or None)
        self._restore_handler()
        self.outcome = SearchOutcome(self.run_dir, self.search_dir, self._so_far(), state, error=error, config=self.config)
        _closed(self)
        return self.outcome

    def _reraise_interrupt(self, exc: BaseException) -> None:
        """A Ctrl-C from Python goes on once the search is recorded `stopped`."""
        if isinstance(exc, KeyboardInterrupt) and self.interrupt_raises:
            self.log(f"stopped (Ctrl-C). Resume with: hillclimb resume {search_ref(self.search_dir)}")
            raise exc

    def _end(self, body: Callable[[], Candidate | None]) -> SearchOutcome:
        """Run `body` — whatever is left of the search; it returns what ships
        — then settle: the host's holdout, the final status, and what a
        finished search leaves behind."""
        config, problem, search_dir, journal, log = self.config, self.problem, self.search_dir, self.journal, self.log
        try:
            selected = body()
            selected = _finish_holdout(config, problem, search_dir, journal, self.evaluator, selected, log)
        except ParkedSearch as exc:
            return self._settle("parked", exc)
        except (StopRequested, KeyboardInterrupt) as exc:
            outcome = self._settle("stopped", exc)
            self._reraise_interrupt(exc)
            return outcome
        except Exception as exc:
            self._settle("failed", exc)
            raise
        self._finalize("done")
        self.outcome = SearchOutcome(self.run_dir, search_dir, selected, "done", config=config)
        _closed(self)
        try:
            if problem.emflow_problem and selected is not None:
                _official_verify(config, problem, search_dir, journal, selected, log)
            if problem.mlebench_comp_id and selected is not None:
                _mlebench_grade(config, problem, search_dir, selected, log)
            # what the finished search leaves for the next ones (and its own card,
            # beside its artifacts, under any memory)
            self.memory.record(journal, budget_s=self.budget.total_s, cost_usd=self.harness.total_cost_usd())
        finally:
            self._restore_handler()
        return self.outcome

    def finish(self) -> SearchOutcome:
        """Run the climber's own loop until the search is over, and settle it.
        After steps taken by hand, the loop carries on from the journal."""
        if self.outcome is not None:
            return self.outcome
        self.budget.resume()
        return self._end(lambda: self.harness.execute(self.loop))

    def close(self) -> SearchOutcome:
        """Settle the search where it stands and return its outcome: `done`
        when a budget ended it, `parked` or `stopped` when a park or a stop
        did, and `stopped` — resumable with `hillclimb resume` — when it is
        closed by hand with budget left. Safe to call again."""
        if self.outcome is not None:
            return self.outcome

        def body() -> Candidate | None:
            self.harness.wait(timeout=0)  # a tick: a stop queued since the last step counts
            self.harness.raise_latched()
            if self.harness.closed_reason is None:
                raise StopRequested("closed by hand with budget left")
            return self._so_far()

        return self._end(body)

    def __enter__(self) -> Search:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- stepping ---

    def begin(self) -> None:
        """Make the search ready to be stepped: write its floor (the baseline,
        the seed), then stop the clock — from here the budget is spent only
        while a step runs, never while a person reads what one did."""
        if self.outcome is not None:
            return
        self._stepping = True
        try:
            self.harness.start()
        except (StopRequested, KeyboardInterrupt) as exc:
            self._settle("stopped", exc)
            self._reraise_interrupt(exc)
            return
        except Exception as exc:
            self._settle("failed", exc)
            raise
        self.budget.pause()

    def _policy_loop(self, what: str):
        """The loop, when a person may step it: an open search of a policy climber."""
        from hillclimb.harness.loop import PolicyLoop

        if self.outcome is not None:
            raise RuntimeError(f"{what}: this search is closed ({self.outcome.state})")
        if not isinstance(self.loop, PolicyLoop):
            raise RuntimeError(
                f"{what}: a {type(self.loop).__name__} owns its control flow and cannot be stepped. "
                "Call finish() to let it run"
            )
        return self.loop

    def select(self):
        """π_sel alone: the node(s) the selector would start the next attempt
        from, as a `Selection`; None for a root step (or a closed search)."""
        loop = self._policy_loop("select()")
        self.harness.wait(timeout=0)  # a tick: control commands, the cost ceiling
        if not self.harness.open:
            return None
        loop.catch_up(self.harness)
        return loop.select(self.harness.view())

    def propose(self) -> Action | None:
        """What the climber would do next, without doing it: the selector
        picks the node(s), the policy names the operator. None when the
        policy holds or the search is closed (`closed_reason` says which)."""
        loop = self._policy_loop("propose()")
        self.harness.wait(timeout=0)  # a tick: control commands, the cost ceiling
        if not self.harness.open:
            return None
        loop.catch_up(self.harness)
        return loop.propose(self.harness.view())

    def run(self, action: Action) -> Outcome:
        """Run one action — the policy's proposal or your own — to its end:
        the attempt is made, scored, committed, and the policy observes the
        result. An action the harness cannot take (an unknown target, a
        closed search) comes back as a `rejected` Outcome saying why."""
        from hillclimb.harness.loop import HarnessClosed, Outcome, Ticket
        from hillclimb.modules.policies.base import Action

        if not isinstance(action, Action):
            raise TypeError(
                f"run() takes an Action, e.g. Action('improve', target_id='c002'), not {action!r} "
                "(hillclimb.run(problem, ...) is what runs a whole search)"
            )
        loop = self._policy_loop("run()")
        for candidate_id in (action.target_id, *action.inspiration_ids):
            if candidate_id and candidate_id not in self.journal.candidates:  # a typo, said plainly
                reason = f"no candidate {candidate_id} in this search"
                return Outcome(ticket=Ticket(id="", action=action, rejected=reason), kind="rejected", candidate=None)
        self.harness.clear_strikes()  # a person trying things is not a policy stuck on the impossible
        self.budget.resume()
        try:
            outcome = self.harness.run(action)
        except HarnessClosed as exc:
            return Outcome(ticket=Ticket(id="", action=action, rejected=str(exc)), kind="rejected", candidate=None)
        except StopRequested:
            raise  # a signal: the harness is latched closed, close() settles the search `stopped`
        except Exception as exc:  # the engine broke mid-attempt: the search is over, like any engine crash
            self._settle("failed", exc)
            raise
        finally:
            if self.outcome is None:
                self.budget.pause()
        loop.observe(self.harness, outcome.candidate)
        return outcome

    def step(self) -> Outcome | None:
        """One move of the climber: `run(propose())`. None when there was
        nothing to run — the policy holds, or the search is closed."""
        action = self.propose()
        return self.run(action) if action is not None else None

    # --- reading an open search ---

    @property
    def state(self) -> SearchState:
        """What the policy sees: the journal (holdout-blind), what is in
        flight, the budget left, the metric's direction."""
        return self.harness.view()

    @property
    def candidates(self) -> list[Candidate]:
        """Every candidate so far, as the policy sees them."""
        return list(self.state.journal.candidates.values())

    @property
    def best(self) -> Candidate | None:
        """The best scored candidate so far, by validation score."""
        return self.state.journal.best_candidate(self.problem.higher_is_better)

    def source(self, candidate_id: str) -> str | None:
        """A candidate's `solution.py`."""
        return self.harness.source(candidate_id)

    @property
    def closed_reason(self) -> str | None:
        """Why no new work can start (a spent budget, a stop, a park, a
        settled search); None while the search is open."""
        if self.outcome is not None:
            return self.outcome.error or self.outcome.state
        return self.harness.closed_reason

    @property
    def is_open(self) -> bool:
        return self.closed_reason is None

    @property
    def ref(self) -> str:
        return search_ref(self.search_dir)

    @property
    def result(self) -> SearchOutcome:
        """The search as a `SearchOutcome`: its outcome once settled, a live
        reading of it (state `running`) until then."""
        if self.outcome is not None:
            return self.outcome
        return SearchOutcome(self.run_dir, self.search_dir, self._so_far(), "running", config=self.config)

    def __repr__(self) -> str:
        where = self.outcome.state if self.outcome is not None else (self.closed_reason or "open")
        return f"Search({self.ref!r}, {where})"


# the search a person is stepping in this process, if any: status records carry
# the engine's pid, so two open at once would be one engine to every viewer
_STEPPING: Search | None = None


def _closed(search: Search) -> None:
    global _STEPPING
    if _STEPPING is search:
        _STEPPING = None


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
    search = Search(
        config, problem, search_dir, budget, log,
        seed_from=seed_from, knowledge_context=knowledge_context, target=target,
    )
    return search.open() or search.finish()


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


# printed at the start of a search run from Python (`run`, `start`), where
# nothing else says the live views exist; `hints=False` leaves it out
FOLLOW_HINT = "follow it live in another terminal: hillclimb watch · hillclimb tree · hillclimb chart"


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
    spec_set: Sequence[str] = (),
    log: Log = print,
    hint: bool = False,
    interrupt_raises: bool = False,
) -> SearchOutcome:
    """Resolve a single-problem target, create the Run/Search dirs, and run
    the engine to completion. Suites are a CLI concern (parallel processes);
    this API runs exactly one search. `seed_from` scores an incumbent
    solution as the floor candidate a re-search must beat."""
    search = _new_search(
        target, budget_s=budget_s, name=name, run_id=run_id, run_name=run_name, config=config,
        agent=agent, model=model, holdout=holdout, seed_from=seed_from,
        knowledge_context=knowledge_context, spec_set=spec_set, log=log, hint=hint,
        interrupt_raises=interrupt_raises,
    )
    return search.open() or search.finish()


def sdk_config(**overrides: Any) -> Config:
    """The config a Python entry point runs with. A folder that is not a
    hillclimb dir yet is made one, as `hillclimb problem get` does, so a
    notebook or script works in a fresh folder; the CLI asks first instead."""
    from hillclimb.project import HillclimbDirNotFound

    try:
        return Config.load(**overrides)
    except HillclimbDirNotFound:
        from hillclimb.project import scaffold_hillclimb_dir, scaffold_target

        # beside a project's own problems/ or runs/, it goes in hillclimb/
        folder = scaffold_hillclimb_dir(scaffold_target(Path.cwd())[0])
        print(f"hillclimb: made {folder} a hillclimb dir (hillclimb.yaml, problems/, runs/)")
        return Config.load(**overrides)


def _new_search(
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
    spec_set: Sequence[str] = (),
    log: Log = print,
    hint: bool = False,
    interrupt_raises: bool = False,
) -> Search:
    """A new search on `target`, created (run dir, spec, search dir, climber
    snapshot) and not yet opened."""
    config = config or sdk_config(agent=agent, model=model)
    if agent:
        config.agent = agent
    if model:
        config.model = model
    if not holdout:
        config.holdout.enabled = False
    problem = load_problem(target, config)
    run_dir_is_new = run_id is None
    climber_block = config.climber_block()  # before anything is written: no climber, no run folder
    total_s = resolve_budget(budget_s, config)  # ...and no budget, no run folder either
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
    seed_path = Path(seed_from) if seed_from else None
    if run_dir_is_new:
        write_run_spec(run_dir, [spec_entry(
            target, budget=total_s, agent=config.agent, model=config.model,
            climber=climber_block, parallel_agents=config.concurrency.parallel_agents,
            n_replicates=config.evaluation.n_replicates, seed_from=seed_path, set=spec_set,
        )])
    search_dir = create_search(config, problem, run_dir, run_id, total_s, seed_from=seed_path)
    log(
        f"Search {search_ref(search_dir)} (problem={problem.problem_id}, "
        f"agent={config.agent}, model={config.model}, budget={total_s}s)"
    )
    if hint:
        log(FOLLOW_HINT)
    return Search(
        config,
        problem,
        search_dir,
        BudgetManager(total_s, config.budget.stop_margin_s),
        log,
        seed_from=seed_path,
        knowledge_context=knowledge_context,
        target=target,
        interrupt_raises=interrupt_raises,
    )


# -- fleets: N detached engines under one run -----------------------------------
#
# `run --parallel-searches` and the hosted platform: independent searches on one
# problem, each its own engine process, sharing discoveries through the run
# dir's live knowledge cards. Processes, not threads: status.json carries the
# engine's pid, so two searches in one process would be indistinguishable to
# every viewer.


# --- the Python front door --------------------------------------------------------------


def _not_while_importing(what: str) -> None:
    from hillclimb.modules import refs

    if refs.importing():
        raise RuntimeError(
            f"hillclimb.{what}() was called while a climber file was being imported — a search "
            "loads the file your classes live in, which runs it again. Put the call under "
            "`if __name__ == \"__main__\":`"
        )


def _with_climber(config: Config, climber: Any) -> None:
    """Make `climber` — a name, a block, a `ClimberSpec` or a composed
    `Climber` — the config's climber. File refs written in Python resolve
    from the working directory."""
    from hillclimb.climber import Climber, as_spec

    if isinstance(climber, Climber):
        config.climber = climber.spec
        config._live_climber = None if climber.portable else climber
    else:
        config.climber = as_spec(climber).anchored(Path.cwd())
        config._live_climber = None


def _target(problem: Any, config: Config) -> str:
    """What `load_problem` takes: a `Problem` is made to exist first (saved
    when defined here, copied in when bundled), a string is passed on."""
    from hillclimb.problem import Problem

    return problem.resolve(config) if isinstance(problem, Problem) else str(problem)


def run(
    problem: str | Problem,
    *,
    climber: Any = None,
    budget: Budget | str | int | None = None,
    agent: str | None = None,
    model: str | None = None,
    config: Config | None = None,
    holdout: bool = True,
    learning: bool = True,
    max_evaluations: int | None = None,
    seed_from: Path | str | None = None,
    name: str | None = None,
    log: Log = print,
    hints: bool = True,
) -> SearchOutcome:
    """Run one search, here, to completion — the Python counterpart of
    `hillclimb run PROBLEM --no-detach`.

        import hillclimb as hc

        outcome = hc.run("heilbronn-11", climber=hc.Climber(operator_policy=hc.policies.Greedy(num_drafts=5)),
                         budget="10m")
        outcome.selected.val_score

    `problem` is a problem id, a path, a provider target or a `Problem`.
    `climber` is a .py file, a block, or a composed `hillclimb.Climber`
    (None: the folder's `climber:` block). `budget` is a `Budget` — time,
    evaluations, tokens, cost — or just `"10m"` / `"2h"` / seconds for the
    clock (None: the folder's run defaults in runs/config.yaml; with none
    there either it raises NoBudget); `max_evaluations=N` is short for
    `Budget(evaluations=N)`. `learning=False` keeps the search out of the
    folder's knowledge, both ways. The folder's hillclimb.yaml supplies
    everything else unless a `config` is given."""
    _not_while_importing("run")
    config = config.model_copy(deep=True) if config is not None else sdk_config(agent=agent, model=model)
    if climber is not None:
        _with_climber(config, climber)
    limits = _limits(budget, max_evaluations)
    return run_search(
        _target(problem, config),
        budget_s=limits.seconds,
        name=name, config=config, agent=agent, model=model, holdout=holdout,
        seed_from=seed_from, spec_set=_front_door_set(config, learning, limits), log=log, hint=hints,
        interrupt_raises=True,
    )


def _limits(budget: Any, max_evaluations: int | None) -> Budget:
    """The Budget a front-door call means: `budget=` as given, with the
    `max_evaluations=` shorthand laid over it."""
    from dataclasses import replace

    from hillclimb.harness.budget import Budget

    limits = Budget.of(budget)
    return replace(limits, evaluations=max_evaluations) if max_evaluations is not None else limits


def start(
    problem: str | Problem,
    *,
    climber: Any = None,
    budget: Budget | str | int | None = None,
    agent: str | None = None,
    model: str | None = None,
    config: Config | None = None,
    holdout: bool = True,
    learning: bool = True,
    max_evaluations: int | None = None,
    seed_from: Path | str | None = None,
    name: str | None = None,
    log: Log = print,
    hints: bool = True,
) -> Search:
    """Open one search, here, to drive a step at a time — `run` with the loop
    in your hands. Takes what `run` takes and returns the open `Search`
    (`propose` / `run` / `step` / `finish` / `close`). It is an ordinary
    search: `hillclimb watch` shows it, and one closed with budget left can
    be resumed with `hillclimb resume`. One at a time per process.

        search = hc.Climber(operator_policy="greedy").start("fitness-landscape", agent="toy")
    """
    global _STEPPING
    _not_while_importing("start")
    if _STEPPING is not None and _STEPPING.outcome is None:
        raise RuntimeError(
            f"search {_STEPPING.ref} is still open in this process: close() it before starting another"
        )
    config = config.model_copy(deep=True) if config is not None else sdk_config(agent=agent, model=model)
    if climber is not None:
        _with_climber(config, climber)
    limits = _limits(budget, max_evaluations)
    search = _new_search(
        _target(problem, config),
        budget_s=limits.seconds,
        name=name, config=config, agent=agent, model=model, holdout=holdout,
        seed_from=seed_from, spec_set=_front_door_set(config, learning, limits), log=log, hint=hints,
        interrupt_raises=True,
    )
    _STEPPING = search
    if search.open() is None:
        search.begin()
    return search


def _front_door_set(config: Config, learning: bool, limits: Budget) -> list[str]:
    """Apply what `run` takes beside the config, and return it as the `set`
    pairs the run's spec records, so the spec reruns the same search."""
    pairs = []
    if not learning:
        config.learning.enabled = False
        pairs.append("learning.enabled=false")
    return pairs + limits.apply(config)


def run_spec(path: Path | str, *, config: Config | None = None, log: Log = print) -> list[SearchOutcome]:
    """Run every search of a run spec, here, one after the other — the
    Python counterpart of `hillclimb run SPEC` (which detaches one engine per
    entry). One run holds them all and carries its own `spec.yaml`."""
    from hillclimb.climber import as_spec
    from hillclimb.problem import load_suite, suite_problem_targets

    _not_while_importing("run_spec")
    base = config if config is not None else sdk_config()
    suite = load_suite(path, base)
    targets = suite_problem_targets(suite, base)
    run_id = new_run_id(suite.suite_id)
    problem_ids = list(dict.fromkeys(load_problem(target, base).problem_id for target in targets))
    # every entry's budget (its own, else the folder's run defaults) before the run exists
    budgets = [resolve_budget(entry.budget, base) for entry in suite.problems]
    run_dir = create_run(base, RunMeta(
        run_id=run_id, name=suite.suite_id, kind="suite", target=str(path), problem_ids=problem_ids,
    ))
    configs, entries = [], []
    for entry, target in zip(suite.problems, targets):
        entry_config = base.model_copy(deep=True)
        entry_config.hillclimb_dir = base.hillclimb_dir
        if entry.agent:
            entry_config.agent = entry.agent
        if entry.model:
            entry_config.model = entry.model
        if entry.climber is not None:
            entry_config.climber = as_spec(entry.climber)
        if entry.parallel_agents is not None:
            entry_config.concurrency.parallel_agents = entry.parallel_agents
        if entry.n_replicates is not None:
            entry_config.evaluation.n_replicates = entry.n_replicates
        entry_config.apply_overrides(parse_set_overrides(entry.set))
        seed = Path(entry.seed_from) if entry.seed_from else None
        if seed is not None and not seed.is_absolute():
            seed = (suite.suite_path.parent / seed).resolve()
        configs.append((entry_config, seed))
        entries.append(spec_entry(
            target, name=entry.name, budget=entry.budget, agent=entry_config.agent, model=entry_config.model,
            climber=entry_config.climber_block(), parallel_agents=entry.parallel_agents,
            n_replicates=entry.n_replicates, seed_from=seed, set=entry.set,
        ))
    write_run_spec(run_dir, entries, source=path)
    return [
        run_search(
            target, budget_s=total_s,
            run_id=run_id, run_name=suite.suite_id, config=entry_config, seed_from=seed, log=log,
        )
        for (entry_config, seed), total_s, target in zip(configs, budgets, targets)
    ]


def child_launch_context(config: Config) -> tuple[Path, dict[str, str]]:
    """(cwd, env) for a child `hillclimb run`: rooted at the hillclimb dir
    with HILLCLIMB_DIR pinned, so the child never has to search."""
    root = config.hillclimb_dir or Path.cwd()
    # the command launching it checked the coding agents' logins already
    # (`connect.ensure_agent_ready`): a detached engine has no terminal to ask at
    from hillclimb.connect import CHECKED_ENV

    return root, {**os.environ, "HILLCLIMB_DIR": str(root), CHECKED_ENV: "1"}


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
    name (one .py file, a folder) as `--climber`, a block as a `--set
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
    runs (one .py file, a block or a `ClimberSpec`), and the
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

    # what an engine's log shows once its search is really under way: its
    # first operator started (`[m:ss left] draft …`). The floor and an
    # incumbent seed come before it, and either can still fail
    STARTED_MARKERS = (" left] ",)

    def startup_failures(self, *, wait_s: float = 10.0) -> list[tuple[Path, int]]:
        """Engines that died while starting, as (log path, exit code). Waits
        until every engine has either got its search going (its log shows it)
        or exited, at most `wait_s`. A detached launch must not report an
        engine running in the background when it died at once — a budget
        that does not parse, an unknown agent, a missing extra, a seed file
        that is not there all end the engine in its first second."""
        deadline = time.monotonic() + wait_s
        pending = list(zip(self.procs, self.log_paths))
        while pending and time.monotonic() < deadline:
            still = []
            for proc, log_path in pending:
                if proc.poll() is not None:
                    continue
                try:
                    text = log_path.read_text(errors="replace")
                except OSError:
                    text = ""
                if not any(marker in text for marker in self.STARTED_MARKERS):
                    still.append((proc, log_path))
            pending = still
            if pending:
                time.sleep(0.2)
        return [
            (log_path, proc.returncode)
            for proc, log_path in zip(self.procs, self.log_paths)
            if proc.poll() is not None and proc.returncode not in (0,)
        ]

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
    from hillclimb.agents import agent_names, is_builtin

    fleet_agent = agent or config.agent
    if fleet_agent in agent_names() and not is_builtin(fleet_agent):
        raise ValueError(
            f"agent {fleet_agent!r} was registered in this process (register_agent); a detached "
            "engine cannot know it. Run it here with hillclimb.run(...) instead"
        )
    from hillclimb.agents import require_agent_clis

    # every coding agent the engines will call, checked here: a detached
    # engine would only find a missing CLI after its first operator calls
    require_agent_clis([fleet_agent, *(route.agent for route in config.routing.values() if route.agent)])
    problem = load_problem(target, config)
    ensure_runtime_venv(config, problem.runtime, log=log, requirements=problem.requirements_file)
    name = run_name or problem.problem_id
    if not budget:  # settled before the run exists: every engine is handed it
        budget = f"{resolve_budget(None, config)}s"
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
