# GEPA × hillclimb implementation plan

*Paths as written at the time: `search_strategy.py`, `policy.py`, `evaluation.py` and friends moved in the 2026-09 package-layout refactor — see `docs/package-layout-plan.md`.*


This document is a self-contained implementation brief for a Codex coding agent. Work in the
`hillclimb` repository, preserve unrelated local changes, implement the MVP described here, and
do not broaden the scope without recording why.

## Task for Codex

Add GEPA as an optional, engine-level optimizer for hillclimb. Users must be able to run:

```bash
uv sync --extra gepa
uv run hillclimb run PATH_TO_PROBLEM \
  --policy gepa \
  --seed-from PATH_TO_SEED_SOLUTION
```

GEPA should generate and select candidate `solution.py` text while hillclimb remains the source
of truth for candidate directories, agent execution, verifier calls, validation trials, budgets,
controls, the append-only journal, status, best/selected candidates, and private holdout scoring.

The runtime mutation agent should use hillclimb's existing agent/router abstraction. This makes
Claude Code usable by configuring the normal hillclimb agent or a `routing.gepa` override; do
not add a second Claude SDK integration inside the GEPA adapter.

## Definition of done

The work is complete only when all of the following are true:

- `search.policy: gepa` and `hillclimb run ... --policy gepa` dispatch to a dedicated GEPA search
  runner without pretending that GEPA satisfies the read-only `SearchPolicy` contract.
- A seeded GEPA run proposes edited `solution.py` candidates through an existing hillclimb agent
  agent, evaluates them with the existing verifier contract, and journals every evaluation.
- Higher-is-better and lower-is-better metrics both produce correct GEPA fitness values.
- Invalid proposals receive useful reflective feedback and a finite dominated fitness rather than
  crashing the optimization.
- Validation reports and bounded validation logs can reach GEPA as actionable side information.
  Hidden holdout scores, reports, paths, and logs never reach GEPA, its proposer prompts, or its
  checkpoint state.
- On normal completion, hillclimb applies its existing top-k holdout and selection semantics and
  returns the selected candidate.
- Wall-clock, cost, stop, park, abort, status heartbeat, and resume behavior remain functional.
- An interrupted run resumes from both hillclimb's journal and GEPA's run directory without
  duplicating candidate IDs or silently restarting the search.
- Greedy and OpenEvolve behavior remains unchanged, and the full test suite passes.
- A live, sequential three-arm experiment completes three Greedy searches, three OpenEvolve
  searches, and three GEPA searches from the identical executable seed, wall-clock budget,
  verifier settings, Claude Code agent, and model. All nine searches must finish successfully
  and their results must be recorded before the implementation task is considered complete.
- The README explains installation, configuration, limitations, resume behavior, and the
  distinction between the OpenEvolve policy and the GEPA engine.

## Important repository state rule

This working tree may already contain unrelated modified and untracked files. Before editing:

```bash
git status --short
```

Do not reset, discard, reformat, or overwrite unrelated work. Read the current versions of every
file before patching because the repository may have moved since this plan was written. Keep GEPA
changes isolated and review the final diff by path.

## Why GEPA is not a normal `SearchPolicy`

The current `SearchPolicy` seam in `src/hillclimb/policy.py` is deliberately read-only. A policy
chooses an `Action`—operator, target, inspirations, complexity, route, and metadata—then
`GreedySearcher` creates the workspace and asks a hillclimb agent to draft or edit the actual
candidate.

GEPA is a full optimizer. Its faithful loop includes candidate text, evaluation feedback,
reflective mutation, parent/Pareto selection, and optionally merge. Registering GEPA only in
`src/hillclimb/policies/__init__.py` would either violate the policy contract or reduce GEPA to a
GEPA-inspired parent selector. Do not do that.

Keep the public configuration surface compatible—`search.policy: gepa`—but dispatch it before
`get_policy()` is called:

```text
execute_search
  └─ build_search_strategy(...)
       ├─ policy == "gepa"  -> GEPASearcher
       └─ otherwise         -> GreedySearcher(get_policy(...))
```

Use a small protocol for the two methods the API needs:

```python
class SearchStrategy(Protocol):
    def run(self) -> Candidate | None: ...
    def total_cost_usd(self) -> float: ...
```

The factory can live in `src/hillclimb/search_strategy.py`. Avoid renaming or moving
`GreedySearcher` in the first integration; that would create unnecessary merge risk.

## Ownership model

| Concern | Owner | Required behavior |
|---|---|---|
| Candidate text | GEPA | Candidate is initially `{"solution.py": source}`. |
| Parent/Pareto selection | GEPA | Use upstream GEPA behavior and checkpointing. |
| Reflection and mutation | GEPA + hillclimb agent | A custom GEPA proposer invokes the routed agent in a scratch workspace. |
| Problem/data access | hillclimb | Link or copy with the same isolation rules as existing candidates. |
| Candidate IDs/directories | hillclimb | Continue canonical `cNNN` IDs and normal search-tree layout. |
| Validation execution | hillclimb | Use `Executor` and the trusted result-file contract. |
| Trial aggregation/noise | hillclimb | Respect `search.n_trials` and `search.trial_mode`. |
| Raw metric direction | hillclimb | Journal raw scores; negate only at the GEPA boundary when lower is better. |
| Journal/best/status | hillclimb | Preserve append-only replay and existing UIs. |
| Wall/cost budget and controls | hillclimb | Stop before starting work that cannot fit and honor stop/park/abort. |
| Hidden holdout | hillclimb only | Run after optimization and never disclose it to GEPA. |
| Resume | both | Reconcile GEPA checkpoint state with hillclimb journal metadata. |

## MVP scope

Implement these constraints explicitly rather than leaving ambiguous partial support:

- Single-task GEPA optimization. The verifier-defined hillclimb problem is the task; do not add a
  dataset/generalization API in this change.
- One mutable component: `solution.py`.
- A seed is required. Accept `--seed-from`, or use an executable code baseline if the problem
  already provides one. If neither exists, fail before optimization with an actionable message
  telling the user to pass `--seed-from`.
- Serial GEPA proposal/evaluation. Reject `search.parallel_agents > 1` for GEPA with a clear
  unsupported-in-MVP message rather than silently ignoring it, and explicitly set GEPA's own
  `EngineConfig.parallel=False` because upstream currently defaults it to true.
- GEPA merge disabled. Reject `use_merge: true` in the MVP. Mutation lineage is supported; merge
  lineage can be added after the single-parent bridge is proven.
- Preserve the current journal schema. Use existing operators such as `seed` and `improve`; put
  GEPA-specific identity and lineage in `policy_meta`.
- Do not change greedy or OpenEvolve semantics.

Non-goals for this pull request are multi-file evolution, GEPA dataset adapters, merge,
parallel proposals, seedless drafting, replacing existing policies, a new UI, and changing the
journal schema.

## Public configuration

Keep using `SearchConfig.policy` and `SearchConfig.policy_params`, because `SearchMeta`, CLI
resume, suite/fleet propagation, and experiment overrides already persist those fields.

Example:

```yaml
agent: claude-code
model: claude-sonnet-4-5

search:
  policy: gepa
  parallel_agents: 1
  n_trials: 1
  policy_params:
    max_metric_calls: 50
    max_candidate_proposals: 50
    reflection_minibatch_size: 1
    candidate_selection_strategy: pareto
    frontier_type: objective
    cache_evaluation: true
    use_merge: false
    failure_fitness: -1.0e100
    seed: 0

routing:
  gepa:
    agent: claude-code
    model: claude-sonnet-4-5
```

Treat model identifiers in documentation as examples; use identifiers supported by the installed
agent. `Router.resolve("gepa")` already falls back through `routing.default` to the global
agent/model/auth settings.

Create an integration-specific Pydantic model, for example `GEPAParams`, instead of spreading
untyped `dict.get()` calls through the runner. Forbid unknown keys so misspelled budget or privacy
settings fail early. Suggested MVP fields and defaults:

```python
class GEPAParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_metric_calls: int = Field(default=50, gt=0)
    max_candidate_proposals: int | None = Field(default=None, gt=0)
    reflection_minibatch_size: int = Field(default=1, gt=0)
    candidate_selection_strategy: str = "pareto"
    frontier_type: str = "objective"
    cache_evaluation: bool = True
    use_merge: Literal[False] = False
    failure_fitness: float = -1.0e100
    seed: int = 0
```

Validate enum-like values against the installed GEPA release. Do not pass arbitrary policy
parameters through to GEPA. Hillclimb's wall-clock budget remains authoritative even when GEPA's
metric/proposal caps are configured.

## Optional dependency and import behavior

In `pyproject.toml`, add an extra and the matching development dependency. At implementation time,
verify GEPA's latest compatible public API and version from the upstream package rather than
blindly copying this example constraint:

```toml
[project.optional-dependencies]
gepa = [
    "gepa>=0.1.4,<0.2",
]

[dependency-groups]
dev = [
    # existing entries...
    "gepa>=0.1.4,<0.2",
]
```

Keep imports lazy. A greedy installation must not import GEPA. Selecting `--policy gepa` without
the extra should produce one concise error:

```text
the GEPA optimizer needs the `gepa` package: pip install 'hillclimb[gepa]'
```

## Proposed module layout

Use a narrow integration package so upstream-specific code does not leak into generic search:

```text
src/hillclimb/
  search_strategy.py                  # protocol + dispatch factory
  evaluation.py                     # shared trial/report helpers extracted from search.py
  integrations/
    gepa/
      __init__.py                   # lazy availability check and public builder
      config.py                     # GEPAParams validation
      proposer.py                   # routed agentic custom candidate proposer
      evaluator.py                  # source -> hillclimb candidate -> GEPA feedback
      searcher.py                   # GEPASearcher orchestration
tests/
  test_gepa_config.py
  test_gepa_proposer.py
  test_gepa_searcher.py
```

If the clean implementation needs fewer files, combine small modules. Do not create a generic
framework before a second engine requires it.

## Detailed implementation phases

### Phase 0 — preflight and upstream API verification

1. Read `CLAUDE.md` and any newly added `AGENTS.md` in scope.
2. Inspect `git status --short`; preserve all unrelated work.
3. Read the current versions of `api.py`, `search.py`, `policy.py`, `executor.py`, `journal.py`,
   `candidate.py`, `baseline.py`, `control.py`, `routing.py`, `status.py`, `config.py`, and the
   agent base classes.
4. Install/sync the optional GEPA dependency in the development environment.
5. Prefer the public single-task `gepa.optimize_anything.optimize_anything` frontend. Inspect its
   installed signature plus `GEPAConfig`, `EngineConfig`, `ReflectionConfig`, custom candidate
   proposer, stop callbacks, `run_dir`, resume behavior, and result type. The expected current
   high-level contract is:

   ```python
   evaluator(candidate: dict[str, str]) -> tuple[float, dict]

   proposer(candidate, reflective_dataset, components_to_update, metadata=None)
       -> dict[str, str]
   ```

   The lower-level `gepa.optimize`/`GEPAAdapter` API remains a fallback only if the installed
   `optimize_anything` frontend cannot support the custom proposer or required resume callbacks.
   Do not build a custom adapter merely to reproduce behavior already provided by the public
   high-level evaluator contract.
6. Add a small compatibility test around every upstream signature hillclimb relies on. Prefer
   keyword arguments. If the released API differs, adapt the design locally while preserving the
   ownership and privacy invariants in this plan.

### Phase 1 — extract shared validation helpers safely

GEPA and `GreedySearcher` must interpret verifier results identically. Avoid copying the current
private trial/report logic into a second engine.

1. Extract only the reusable validation functions from `GreedySearcher` into
   `src/hillclimb/evaluation.py`:

   - execute one validation trial;
   - read and trust `eval_result.json` only when the top-level split is `validation`;
   - stamp report provenance according to `problem.report_trusted`;
   - run/aggregate `n_trials` using the configured mode;
   - construct `Trial` records consistently.

2. Make current `GreedySearcher` methods thin delegates or direct callers of those helpers.
3. Keep candidate preparation, operator scheduling, commit, holdout, and selection in
   `GreedySearcher`; do not perform a broad refactor.
4. Run the existing executor, report-trust, multi-trial, search, and holdout tests before adding
   GEPA. This phase must be behavior-preserving.

### Phase 2 — runner dispatch and configuration

1. Add the `SearchStrategy` protocol and `build_search_strategy(...)` factory.
2. Move only construction branching out of `execute_search`; keep API finalization, official
   verification/grading, knowledge distillation, and status finalization shared.
3. For `policy != "gepa"`, construct `GreedySearcher` exactly as today and call `get_policy()`.
4. For `policy == "gepa"`, lazily import and construct `GEPASearcher`; do not call `get_policy()`.
5. Pass the same dependencies already created by `execute_search`: journal, executor, budget,
   status, slots/abort where relevant, router, agent pool, holdout scorer, command drain,
   knowledge context, reference solution, and seed path.
6. Parse `GEPAParams` and enforce MVP constraints before creating any candidate.
7. Keep `SearchMeta.policy` and `policy_params` unchanged so existing resume logic continues to
   restore the user's selection.

### Phase 3 — agentic GEPA proposer

Implement a custom GEPA candidate proposer instead of a second provider-specific reflection API.

For each proposal:

1. Receive the parent candidate, reflective dataset, requested components, and GEPA metadata.
2. Require `components_to_update == ["solution.py"]` or the equivalent set. Reject unexpected
   component names.
3. Resolve `Router.resolve("gepa")`, acquire the agent from `AgentPool`, and respect the
   remaining wall and cost budgets before launching it.
4. Create a scratch directory under:

   ```text
   SEARCH_DIR/gepa/proposals/ITERATION_ID/
   ```

   Use a collision-safe suffix if the upstream iteration ID is absent. GEPA's `run_dir` itself
   should be `SEARCH_DIR/gepa/state` so proposal workspaces do not collide with checkpoints.
5. Materialize the parent `solution.py`, the normal problem/data links, and a compact
   `feedback.json`. Do not put holdout artifacts or links in this workspace.
6. Build a mutation prompt containing:

   - the objective and metric direction;
   - constraints and expected submission interface;
   - an instruction to inspect the current code and linked problem/data files;
   - compact actionable validation feedback from the reflective dataset;
   - optional knowledge context/reference-solution note already selected by hillclimb;
   - an explicit instruction to edit `solution.py` in place;
   - an explicit instruction not to invent, read, or optimize against holdout data;
   - the remaining time budget.

7. Invoke the existing `Agent` with an `OperatorRequest` pointed at the scratch
   directory. Use the agent's normal authentication, abort event, model, timeout, cost, and token
   accounting.
8. On success, read the edited `solution.py`, ensure it is non-empty and remains inside the
   workspace, and return `{"solution.py": source}` to GEPA.
9. Record proposal metadata in a thread-safe bridge keyed by a stable source hash:

   - GEPA iteration ID and parent iteration ID;
   - agent/model/session/cost/tokens/duration;
   - scratch directory;
   - parent source hash.

10. If the agent fails without producing usable source, raise a typed proposer error that the
    runner translates into a controlled failed proposal or search failure according to upstream
    GEPA's supported callback semantics. Never return the unchanged parent while claiming a new
    proposal.

Do not require final assistant prose from Claude/Codex to drive the loop—the source file is the
proposal. Therefore an `OperatorResult.output_text` field is not necessary for the MVP unless the
installed GEPA API proves it is required.

### Phase 4 — hillclimb-backed GEPA evaluator

Implement an idempotent bridge from GEPA candidate text to a canonical hillclimb candidate.

#### Content identity and lineage

- Hash normalized `solution.py` content with a stable algorithm.
- Maintain `source_hash -> Candidate` and GEPA iteration/source mappings.
- On first evaluation, allocate the next journal ID and create the normal `cNNN` candidate
  directory.
- Copy the proposed source into that canonical directory and attach the proposal's agent info.
- Use operator `improve` for generated proposals and existing `seed`/baseline semantics for the
  initial candidate.
- Put at least these values in `policy_meta`:

  ```json
  {
    "optimizer": "gepa",
    "gepa_iteration_id": 12,
    "gepa_parent_iteration_id": 7,
    "source_hash": "...",
    "gepa_fitness": 0.123
  }
  ```

- Resolve the parent iteration to a journal candidate ID. If lineage cannot be resolved on a
  resumed run, fail clearly instead of attaching the candidate to an arbitrary parent.
- If GEPA evaluates identical source again, return the cached evaluation and do not create a new
  candidate ID or repeat the verifier call.

#### Validation and score mapping

1. Append the existing `candidate_created` journal event before validation.
2. Register the candidate in `StatusWriter.current`, using phase `exec`.
3. Execute validation with the shared helpers from Phase 1 and respect `n_trials`, trial mode,
   timeout, runtime isolation, and trusted-report rules.
4. Commit the candidate as `ok` or `buggy`, update raw validation best according to the existing
   improvement/accept-band semantics, append `candidate_result`, update router/bandit accounting if
   applicable, and refresh status counts/best/cost.
5. Return a scalar GEPA fitness that is always maximized:

   ```python
   fitness = candidate.val_score if problem.higher_is_better else -candidate.val_score
   ```

6. For missing/invalid/non-finite results, keep the hillclimb candidate `buggy` and return the
   configured finite `failure_fitness`. Assert that the failure value cannot outrank any valid
   candidate seen in the run; if necessary track a dynamic dominated value rather than relying
   only on the static default.
7. Journal raw metric values only. Store transformed GEPA fitness in `policy_meta` or GEPA state,
   not in `Candidate.val_score`.

#### Actionable side information (ASI)

Return structured, serializable feedback suitable for GEPA reflection. Include:

- candidate ID and validity;
- raw validation score, metric name, direction, and transformed fitness;
- verifier metrics;
- compact trusted validation report;
- return code, timeout, duration, and submission validity;
- bounded tails of validation stdout/stderr and tracebacks;
- trial-level scores when multiple trials are used;
- a short diagnosis prompt asking for a concrete source change.

Cap every textual field and the total serialized feedback size. Reuse existing tail/compaction
helpers where possible. Scrub secrets and environment details before persisting feedback.

ASI must never include `holdout_score`, holdout reports, holdout stdout/stderr, the hidden data
path, official verification results, MLE-bench grading, or any value derived from them. Add an
explicit regression test that recursively searches the object and serialized proposer prompt for
forbidden holdout keys and sentinel values.

### Phase 5 — `GEPASearcher` lifecycle

`GEPASearcher.run()` should follow this order:

1. Replay the journal and reconcile any pending candidates with the same recovery rules used by
   the existing searcher. Rebuild source-hash, lineage, agent-cost, and iteration maps from
   candidate files plus `policy_meta`.
2. Write/recover the declared or scored baseline using the existing baseline helpers. Do not
   create a second baseline on resume.
3. Resolve the seed:

   - if `seed_from` is provided, materialize and validation-score it as a normal `seed` candidate;
   - otherwise use an executable baseline `solution.py` if one exists;
   - otherwise stop before invoking GEPA with the required-seed error.

4. Preload the evaluator cache with the scored seed so GEPA's initial evaluation reuses it.
5. Construct GEPA with the public high-level API. The intended shape is:

   ```python
   from gepa.optimize_anything import (
       EngineConfig,
       GEPAConfig,
       ReflectionConfig,
       optimize_anything,
   )

   gepa_config = GEPAConfig(
       engine=EngineConfig(
           run_dir=str(search_dir / "gepa" / "state"),
           seed=params.seed,
           max_metric_calls=params.max_metric_calls,
           max_candidate_proposals=params.max_candidate_proposals,
           candidate_selection_strategy=params.candidate_selection_strategy,
           frontier_type=params.frontier_type,
           parallel=False,
           cache_evaluation=params.cache_evaluation,
       ),
       reflection=ReflectionConfig(
           reflection_minibatch_size=params.reflection_minibatch_size,
           module_selector="all",
           reflection_lm=None,
           custom_candidate_proposer=proposer,
       ),
       merge=None,
       refiner=None,
       stop_callbacks=hillclimb_stopper,
       callbacks=callbacks,
   )

   result = optimize_anything(
       seed_candidate={"solution.py": seed_source},
       evaluator=evaluator,
       objective=objective,
       background=background,
       config=gepa_config,
   )
   ```

   Adapt names to the verified installed 0.1.x API. In particular, confirm whether
   `reflection_lm=None` is accepted when `custom_candidate_proposer` is set. Never fall back to
   GEPA's default provider-backed reflection model because that would bypass hillclimb's routing,
   credentials, abort handling, and cost accounting.
6. The construction must preserve these settings:

   - candidate `{"solution.py": seed_source}`;
   - the hillclimb-backed evaluator;
   - the custom proposer;
   - configured metric/proposal limits;
   - `run_dir=SEARCH_DIR/gepa/state`;
   - merge disabled;
   - deterministic seed where supported;
   - a stop callback that consults hillclimb controls and budgets.

   Do not pass hillclimb's hidden holdout as GEPA's `test_set`. Although upstream treats `test_set`
   as hidden from its engine, hillclimb already owns a stricter post-search holdout lifecycle and
   must remain the only component that reads those artifacts.
7. Let GEPA drive proposal/evaluation until it finishes or a hillclimb stop condition fires.
8. On normal completion, perform the hidden holdout finalization described below, sync selection,
   and return the selected candidate.
9. On park/stop, propagate the existing `ParkedSearch`/`StopRequested` exceptions so
   `execute_search` writes the correct terminal state. Do not run final holdout on a parked search;
   resume must continue the optimizer first.
10. `total_cost_usd()` must sum proposal-agent costs once per evaluated candidate and survive
   replay. Do not charge cached reevaluations twice.

### Phase 6 — budget, controls, abort, and resume

Hillclimb remains the outer authority.

#### Before every proposal

- Drain control commands.
- Apply prune commands where meaningful; GEPA's own population state must not be corrupted. If
  pruning cannot safely remove an upstream candidate in the MVP, document that prune affects
  hillclimb display/selection only.
- Raise the normal park or stop exception when commanded.
- Check `BudgetManager.should_stop()` and remaining operator time.
- Check the configured USD cost ceiling using journaled agent costs.
- Pass the shared abort event to the agent and machine-slot acquisition.

#### Between proposal/evaluation iterations

- Refresh status counts, cost, best, selected, and current entries.
- Use a GEPA stop callback if the installed API supports it. If callbacks only run at iteration
  boundaries, document that control latency is bounded by the current agent/verifier call; abort
  still handles an in-flight agent process.
- Do not use a background thread that mutates the journal.

#### Resume consistency

- Keep GEPA checkpoints below `SEARCH_DIR/gepa/state`.
- Rebuild hillclimb maps from append-only journal events and candidate source hashes.
- Require GEPA iteration/parent metadata on all GEPA-generated candidates.
- Verify the checkpoint's seed/problem/config identity against the search metadata. A changed seed,
  metric direction, component list, or privacy-relevant option is a hard resume error.
- Reconcile a candidate journaled before a crash but lacking a result using existing pending
  recovery semantics.
- Ensure `Journal.next_id()` remains the only source of new canonical IDs.
- Never delete a checkpoint automatically. If checkpoint and journal are irreconcilable, report
  the exact mismatch and stop.

### Phase 7 — holdout and final selection

Do not call holdout during GEPA optimization, even if a candidate becomes the validation best.

On normal completion:

1. Rank valid candidates by raw validation score using `problem.higher_is_better`.
2. Apply the existing holdout `top_k` rule (`top_k <= 0` means the existing all-candidate behavior).
3. Score holdout through the existing scorer in canonical candidate directories.
4. Persist holdout values only in hillclimb candidate/journal artifacts.
5. If a holdout execution fails, record the failure consistently and continue down the validation
   ranking when needed to obtain the configured number of valid holdout candidates.
6. Call the existing `resync_best`/selection logic so rank-blend, holdout-first, or validation
   selection behaves exactly like other search modes.
7. Return the selected candidate. Official EMFlow verification, MLE-bench grading, and knowledge
   distillation remain in `execute_search` after the runner returns.

Add a code-level guard: the object passed to GEPA and the GEPA run directory must be created before
holdout and must never be rewritten with post-hoc holdout fields.

### Phase 8 — documentation and experiment support

Update the README and relevant CLI help:

- installation with `hillclimb[gepa]`;
- a minimal command and YAML example;
- the seed requirement and one-component MVP;
- how to select Claude Code through the normal agent/routing configuration;
- resume/checkpoint location;
- validation-versus-holdout privacy guarantee;
- GEPA limitations in the MVP;
- the architecture difference:
  - greedy: hillclimb chooses parent and asks an agent to mutate;
  - OpenEvolve policy: OpenEvolve supplies population selection/inspirations while hillclimb's
    agent still performs mutation;
  - GEPA engine: GEPA owns reflective mutation and Pareto search while hillclimb evaluates and
    records candidates.

Ensure suite/fleet/experiment configuration accepts `search.policy: gepa` without special casing
beyond runner dispatch. An experiment should be able to compare policies using ordinary config
overrides.

## Test plan

Use fake agents, fake executors, and a monkeypatched/fake GEPA driver for most tests. No test in
the default suite may contact a model provider or require credentials.

### Unit tests

- GEPA params accept documented fields and reject unknown, invalid, parallel, or merge settings.
- Missing optional dependency produces the exact actionable install message.
- Runner factory selects `GEPASearcher` only for `policy == "gepa"`; all existing policies take the
  unchanged path.
- Proposer materializes parent source and feedback, resolves `routing.gepa`, invokes the fake
  agent, and returns the edited source.
- Proposer rejects missing, empty, escaped, or unchanged output as specified.
- Proposal bridge records agent/model/cost/tokens and GEPA lineage by source hash.
- Evaluator converts higher-is-better and lower-is-better raw scores correctly.
- Invalid verifier output returns dominated finite fitness plus useful stderr/report feedback.
- Duplicate source content reuses the existing evaluation and candidate ID.
- Multi-trial aggregation matches greedy behavior.
- Validation report trust/provenance matches existing behavior.
- ASI and proposer prompts are size-bounded and contain no holdout sentinel.
- Journal replay reconstructs hash/iteration/parent/cost maps.
- Incompatible journal/checkpoint state fails loudly.
- Stop, park, wall budget, cost ceiling, and abort propagate through existing exception paths.
- `total_cost_usd()` is stable across cache hits and resume.

### End-to-end tests with fakes

- Start from a seed, have a fake mutation agent improve source twice, have fake GEPA evaluate the
  candidates, and verify canonical directories, parent IDs, events, status, best, and selection.
- Repeat with a lower-is-better problem.
- Produce a buggy candidate followed by a recovery and assert the reflection feedback contains the
  validation failure.
- Complete a run with top-k holdout and assert holdout occurs only after the last GEPA evaluation.
- Resume after one journaled GEPA iteration and assert no duplicate baseline, seed, proposal, or
  candidate ID.
- Exercise `hillclimb run --policy gepa` and `hillclimb resume` metadata restoration.
- Assert greedy and OpenEvolve factory/e2e tests still pass unchanged.

### Optional upstream integration test

Mark a small real-GEPA, fake-agent test with `pytest.importorskip("gepa")`. It should exercise the
installed callback signatures without making external model calls. Keep it deterministic and
fast.

### Validation commands

Run targeted tests while iterating, then the full checks:

```bash
uv sync --extra gepa
uv run pytest tests/test_gepa_config.py tests/test_gepa_proposer.py tests/test_gepa_searcher.py -q
uv run pytest tests/test_search.py tests/test_api.py tests/test_cli.py tests/test_openevolve_policy.py -q
uv run pytest -q
uv run ruff check src tests
```

If this repository's current contributor instructions specify different commands, follow them and
record the exact commands/results in the final handoff. Do not claim tests passed unless they were
actually run.

## Manual smoke test

Use the bundled circle-packing problem or another deterministic demo with an existing executable
seed. Keep the budget and GEPA caps small:

```bash
uv run hillclimb run src/hillclimb/demo/circle-packing \
  --policy gepa \
  --seed-from PATH_TO_WORKING_SOLUTION \
  --budget 20m \
  --set search.parallel_agents=1 \
  --set search.policy_params.max_metric_calls=3 \
  --set search.policy_params.max_candidate_proposals=2
```

Adapt CLI option spelling to the current CLI if necessary. Inspect:

- the journal event order;
- `tree`/`watch` visibility;
- canonical candidate source and reports;
- `SEARCH_DIR/gepa/state` resume files;
- proposal workspaces;
- `best` and selected pointers;
- the absence of holdout values in all GEPA prompts, feedback, and state.

## Phase 9 — required three-arm live acceptance experiment

This phase is a release gate, not optional follow-up work. Codex must run hillclimb with all three
search strategies and record the results before declaring the task complete:

1. Greedy;
2. OpenEvolve;
3. GEPA.

These are three **search strategies/optimizers**, not three operator agents. All arms must use the
same real Claude Code operator agent and the same verified Claude model so the experiment varies
only the search strategy.

### Add shared-seed support to experiments

The current experiment YAML carries config overrides but cannot pass the CLI-only `--seed-from`
setting. A fair comparison requires identical executable source, not merely the same baseline
score. Extend experiment orchestration as part of this change:

1. Add an optional top-level `seed_from` field to `ExperimentSpec`.
2. Resolve a relative seed path against the experiment YAML's parent directory and validate it
   before creating a run.
3. Pass the resolved absolute path as `--seed-from PATH` to every child search, in both sequential
   and parallel launch modes.
4. Show the resolved seed in `--dry-run` output.
5. Preserve the existing behavior when `seed_from` is absent.
6. Add tests for relative/absolute resolution, missing seed errors, dry-run display, and child
   arguments in both schedules.
7. Record or verify the seed source hash for each search so the acceptance report can prove every
   arm started identically. Prefer metadata/provenance over relying only on filenames.

Add a small, deterministic, valid circle-packing seed at:

```text
hillclimb/experiments/seeds/circle-packing.py
```

It should emit a valid 26-row `submission.csv`, finish quickly, and avoid randomness. Score it with
the normal verifier before the experiment. Do not seed any arm with a previously optimized winner;
the purpose is to compare improvement from a neutral, reproducible starting point.

### Commit an explicit experiment spec

Add `hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml`. Use this shape, replacing
`VERIFIED_CLAUDE_MODEL_ID` with an identifier successfully exercised through the installed
`claude-code` agent:

```yaml
name: gepa-vs-openevolve-vs-greedy
problems: [circle-packing]
seed_from: seeds/circle-packing.py
repeats: 3
budget: 15m
schedule: sequential
noise_floor: 0.01

defaults:
  agent: claude-code
  model: VERIFIED_CLAUDE_MODEL_ID
  learning.enabled: false
  search.parallel_agents: 1
  search.n_trials: 1
  holdout.enabled: true
  holdout.top_k: 5

arms:
  greedy:
    search.policy: greedy
    search.policy_params: {}

  openevolve:
    search.policy: openevolve
    search.policy_params:
      random_seed: 42

  gepa:
    search.policy: gepa
    search.policy_params:
      max_metric_calls: 100
      max_candidate_proposals: 100
      reflection_minibatch_size: 1
      candidate_selection_strategy: pareto
      frontier_type: objective
      cache_evaluation: true
      use_merge: false
      failure_fitness: -1.0e100
      seed: 42
```

The 15-minute budget is per search: three arms × three repeats produces nine searches and up to 135
minutes of sequential search time, plus setup and reporting. The hillclimb wall clock is the primary
fairness constraint. GEPA's metric/proposal caps are safety ceilings; record whether either ceiling
binds. Greedy and OpenEvolve do not currently expose an equivalent proposal cap, so do not claim
evaluation-count parity when only wall time is held equal.

Keep the schedule sequential to avoid CPU/runtime contention and provider concurrency differences.
Disable learning in every arm so earlier repeats cannot transfer distilled knowledge to later arms.
The arm order must remain repeat-major and round-robin as implemented by `experiment.expand()`.

### Preflight and execution commands

The live experiment requires installed optional dependencies, working Claude Code authentication,
sufficient provider quota, and at least the full nine-search wall/cost allowance. Dummy/fake
agents satisfy automated tests but do not satisfy this acceptance phase.

Run:

```bash
uv sync --extra openevolve --extra gepa

uv run hillclimb verify circle-packing \
  --solution hillclimb/experiments/seeds/circle-packing.py \
  --repeat 3

uv run hillclimb experiment run \
  hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml \
  --dry-run

uv run hillclimb experiment run \
  hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml \
  --sequential

uv run hillclimb experiment report \
  hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml \
  --control greedy
```

Before the paid run, the dry run must print exactly three arms × one problem × three repeats = nine
searches, show a shared seed path, and show the same agent/model/budget/trial/learning settings for
all arms. Correct the configuration before spending model quota if any of these differ.

If the cloud environment lacks Claude credentials, quota, optional dependencies, or the authorized
cost budget, this phase is blocked. Do not silently substitute Codex, a dummy agent, fewer arms,
shorter runs, or synthetic results, and do not mark the task complete. Report the exact missing
precondition and the already completed implementation/test work.

### Required result artifact

Create `docs/gepa-three-arm-results.md` after the live runs. It must contain:

- repository commit and dirty-worktree note;
- GEPA, OpenEvolve, hillclimb, Python, and platform versions;
- experiment-spec hash and seed-source hash;
- resolved Claude agent/model and non-secret authentication mode;
- start/end timestamps and the exact commands used;
- one row per search with arm, repeat, run/search reference, terminal state, seed score, selected
  candidate ID, selected validation/holdout score, candidate/verifier-call count, invalid-candidate
  count, wall time, model tokens, and model cost where the agent reports them;
- per-arm mean, median, spread, paired gap to Greedy, and repeat wins from `experiment report`;
- whether GEPA hit a metric/proposal cap or any arm hit wall/cost limits;
- any retries, resumes, failures, or deviations, without deleting the original failed run records;
- a concise interpretation that treats gaps within the configured noise floor as inconclusive.

Circle packing currently has no hidden holdout split, so its experiment report legitimately falls
back to validation score. Keep the GEPA holdout-leakage unit/end-to-end tests as a separate required
privacy gate; do not invent a circle-packing holdout merely for this benchmark.

### Three-arm experiment acceptance criteria

All of the following must pass:

- `--dry-run` expands to exactly nine jobs: three repeats for each named arm.
- The same seed file and source hash are used by all nine searches.
- The same Claude Code agent, Claude model, 15-minute budget, trial settings, verifier, metric
  direction, learning setting, and serial operator setting are used by all arms.
- Every search reaches terminal state `done`; parked, stopped, failed, crashed, or still-running
  searches must be resumed or rerun and documented before completion.
- The experiment report contains `n=3` for Greedy, OpenEvolve, and GEPA.
- Every search has a scored seed, at least one evaluated post-seed candidate, and a non-null selected
  candidate/score.
- GEPA candidates appear in the normal journal/tree/status views with correct lineage and agent
  accounting.
- GEPA state, prompts, and ASI contain no holdout fields or sentinel values.
- The exact run references and complete results are recorded in `docs/gepa-three-arm-results.md`.
- No minimum winning score is required: successful execution and honest measurement are the gate.
  Do not tune the configuration after viewing results to make GEPA, OpenEvolve, or Greedy win.
- No claim of superiority is made from one repeat or from a gap inside the noise floor.

## Risks and safeguards

| Risk | Safeguard |
|---|---|
| GEPA API changes | Lazy adapter, compatibility tests, pinned compatible range, verify installed signatures first. |
| Holdout leakage | Post-optimization-only holdout, ASI allow-list, sentinel regression tests, separate state directory. |
| Duplicate candidates on resume | Stable source hash, journal replay, checkpoint reconciliation, idempotent evaluator. |
| Metric direction inversion | Raw score in journal, one explicit max-fitness transform, tests in both directions. |
| Invalid code breaks search | Canonical buggy candidate + finite dominated fitness + actionable logs. |
| Agent cost is lost | Attach proposer `AgentInfo` to the evaluated candidate and replay its cost. |
| Scope explosion | One task, one file, one parent, serial execution, required seed, no merge. |
| Existing behavior regresses | Small dispatch factory, shared validation helpers, existing suite before/after. |
| Live acceptance cannot run in cloud | Treat missing Claude auth/quota/cost authority as a blocker; never replace it with fake evidence. |
| Dirty worktree collisions | Read before edit, patch narrowly, never reset, review diff by path. |

## Suggested commit sequence

Keep commits independently testable if the environment permits commits:

1. `refactor: share validation evaluation helpers`
2. `feat: add optional GEPA runner dispatch and config`
3. `feat: bridge GEPA proposals to hillclimb agents`
4. `feat: evaluate and resume GEPA candidates`
5. `feat: finalize GEPA runs with private holdout selection`
6. `test: cover GEPA lifecycle privacy and direction`
7. `docs: document GEPA optimizer`
8. `experiment: run and report three-arm acceptance matrix`

Do not commit unrelated pre-existing changes. If the working tree cannot support clean commits,
leave the changes uncommitted and report exactly which paths belong to this task.

## Final acceptance checklist

- [ ] Optional extra installs and missing-extra error is actionable.
- [ ] `--policy gepa` dispatches at engine level, not through `_POLICIES`.
- [ ] Seed requirement is enforced before model spend.
- [ ] Claude Code can be selected through `agent` or `routing.gepa`.
- [ ] GEPA proposes `solution.py`; hillclimb creates/evaluates canonical candidates.
- [ ] Raw score direction and GEPA max-fitness mapping are tested.
- [ ] Invalid candidates produce finite dominated fitness and reflective feedback.
- [ ] Candidate metadata contains reproducible GEPA lineage and agent accounting.
- [ ] GEPA inputs/state have no holdout data.
- [ ] Normal completion applies existing top-k holdout and selection.
- [ ] Stop, park, abort, wall/cost budgets, heartbeat, and resume work.
- [ ] Greedy and OpenEvolve behavior is unchanged.
- [ ] Targeted and full tests pass.
- [ ] Ruff/lint checks pass or pre-existing failures are documented precisely.
- [ ] README and CLI help describe the feature and MVP limits.
- [ ] Experiment YAML supports one shared seed and the launcher passes it to every arm.
- [ ] The committed three-arm dry run expands to nine comparable jobs.
- [ ] Three Greedy, three OpenEvolve, and three GEPA live searches all finish in state `done` using
      the same Claude agent/model, seed, budget, and verifier settings.
- [ ] Each of the three arms reports `n=3`, a selected score for every repeat, and at least one
      evaluated post-seed proposal per search.
- [ ] `docs/gepa-three-arm-results.md` records all nine run references, measurements, environment,
      hashes, commands, failures/retries, and an honest noise-aware interpretation.
- [ ] Final response lists changed files, design deviations, tests run, and remaining risks.

## Codex Cloud kickoff prompt

Use this prompt with the repository and this plan available to the cloud task:

```text
Implement the GEPA × hillclimb MVP exactly as specified in
docs/gepa-integration-plan.md.

Start by reading all repository instructions and inspecting the current dirty-worktree-aware
state. Verify the installed GEPA API before coding. Preserve unrelated changes. Work through the
phases in order, run targeted tests after each risky boundary, then run the full suite and lint.

Do not expose holdout data to GEPA. Do not register GEPA as a normal SearchPolicy. Keep the MVP to
one solution.py component, serial execution, a required executable seed, and no merge. Use
hillclimb's existing routed agent so Claude Code can be the mutation agent.

Do not finish after automated tests. Implement shared experiment seed support, then complete the
required three-arm live acceptance experiment: three Greedy, three OpenEvolve, and three GEPA
searches using the same seed, Claude agent/model, verifier settings, and 15-minute budget. All
nine searches must reach state done and be recorded in docs/gepa-three-arm-results.md. If real
Claude credentials, quota, or authorized cost are unavailable, report that as a concrete blocker;
do not substitute fake runs or mark the task complete.

Continue until the acceptance checklist is satisfied or you encounter a concrete blocker that
cannot be resolved from the repository or upstream official documentation. In the final report,
list changed files, commands and results, deviations from the plan, and any unresolved risks.
```

For a cloud task, make sure the relevant branch/commit and this plan exist in the remote repository
or attach/paste the plan into the task. A cloud agent checks out the selected remote branch/commit;
uncommitted files that exist only on the local computer are not automatically available.

## Primary references

- GEPA repository and README: <https://github.com/gepa-ai/gepa>
- GEPA `optimize_anything` API: <https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/>
- GEPA core optimizer and custom proposer API: <https://gepa-ai.github.io/gepa/api/core/optimize/>
- GEPA adapter guide: <https://gepa-ai.github.io/gepa/guides/adapters/>
- GEPA guide for Claude Code as proposer: <https://gepa-ai.github.io/gepa/guides/claude-cli-as-proposer/>
- Codex cloud overview: <https://learn.chatgpt.com/docs/cloud>
- Codex cloud environments: <https://learn.chatgpt.com/docs/environments/cloud-environment>

When upstream documentation and this plan disagree on an API detail, follow the installed,
officially documented API and record the deviation. Preserve the architectural invariants,
privacy boundary, and acceptance criteria above.
