# The optimizer-host abstraction

*Paths as written at the time: `search_strategy.py`, `policy.py`, `evaluation.py` and friends moved in the 2026-09 package-layout refactor — see `docs/package-layout-plan.md`.*


This document is the architecture plan that generalizes what
`docs/gepa-integration-plan.md` builds for GEPA into hillclimb's durable
integration contract. It is a design brief, not an implementation task list:
each phase below becomes its own scoped change (the GEPA MVP is phase 1 and
already has its own brief).

## Strategic premise

Hillclimb will not out-research the optimizer field. Open LLM-driven
evolution engines are appearing faster than anyone can evaluate them —
OpenEvolve, ShinkaEvolve, CodeEvolve, GigaEvo, GEPA, Darwinian Evolver, with
more monthly. Hillclimb's durable assets are the things none of those
projects want to own:

- the verifier contract (a problem *is* its verifier, holdout privacy,
  median-of-trials noise handling);
- agent execution infrastructure (routed agents, machine slots, budgets,
  cost accounting, resume);
- the append-only record and the TUIs that render it (`watch`, `tree`,
  `chart`, `surface`, `similarity`);
- the knowledge economy.

So the abstraction is **hillclimb as an optimizer host**: external engines
own their search brains; hillclimb offers a small set of services and demands
in return only that every candidate flows through `evaluate` and every
lineage lands in the journal. Engines compete; the host contract stays.

## Survey: what varies across the engine space

The contract is derived from the union of observed engine shapes, so new
engines fit without contract changes:

| Axis | Range observed |
|---|---|
| Archive state | single champion (greedy) · tree (AIDE) · islands (FunSearch, CodeEvolve) · MAP-Elites (OpenEvolve, CodeEvolve) · Pareto-per-instance (GEPA) |
| Parent selection input | 1 parent · parent + top-k inspirations (AlphaEvolve-style) · 2+ parents for merge/crossover |
| Mutation mechanism | raw LLM call with engine-built prompt (FunSearch, OpenEvolve) · reflective LLM call over feedback (GEPA) · full coding agent (hillclimb greedy, GEPA custom proposer) |
| Fitness shape | scalar · per-instance vector (GEPA Pareto) · behavior descriptors (MAP-Elites) · rich text feedback (reflective engines) |
| Model usage | single model · bandit/weighted ensembles (ShinkaEvolve, CodeEvolve) |
| Concurrency | serial · heavily async-parallel (AlphaEvolve-class) |

What is invariant: every engine consumes exactly four things from its
environment — *evaluate this candidate*, *give me LLM or agent output*,
*tell me when to stop*, *persist my state*. Those four, plus the record that
feeds the UIs, are the five host services.

## The two integration tiers

Nothing here replaces the existing seams; it names them and adds one.

1. **`SearchPolicy`** (`policy.py`, registry in `policies/__init__.py`) —
   a read-only *brain inside hillclimb's loop*: `propose()`/`observe()` over
   a `PolicyInput`, replay-deterministic. Correct tier for libraries that are
   archives/selectors (OpenEvolve's database today; pyribs later). Unchanged.
2. **`SearchStrategy`** (from the GEPA plan: `search_strategy.py`, protocol
   `run() -> Candidate | None` + `total_cost_usd()`) — an *engine that owns
   its loop*, dispatched before `get_policy()` is called. `GreedySearcher`
   is the first runner; `GEPASearcher` the second. All future full engines
   (ShinkaEvolve, CodeEvolve, …) are runners built on the host services.

Rule of thumb: if the library's entry point is "call my `optimize()`/`run()`
and hand me callbacks", it is a runner; if it can answer "what next?" as a
pure function of journal state, it is a policy.

## The five host services

Engines never touch these subsystems directly; a thin per-engine adapter
consumes them through one facade object (working name `OptimizerHost`,
`src/hillclimb/host.py`), constructed by `execute_search` from the objects it
already creates: journal, executor, budget, status, router, agent pool,
slots, abort event, command drain.

### 1. `evaluate(source, *, context) -> EvalResult`

Candidate text in, canonical candidate out. The host:

- hashes normalized source for idempotency (identical source returns the
  cached `EvalResult`, no new candidate id, no verifier call);
- allocates the next `cNNN` id via `Journal.next_candidate_id()`, creates the
  normal `candidate_dir`, appends `candidate_created`;
- runs `search.n_trials` trials through the shared helpers extracted into
  `evaluation.py` (GEPA plan phase 1), honoring `trial_mode`, timeouts,
  report trust;
- commits `ok`/`buggy`, appends `candidate_result`, updates status/best.

The return type is designed once for the union of engine needs:

```python
@dataclass(frozen=True)
class EvalResult:
    candidate_id: str
    score: float | None        # raw val_score, journal direction (never negated)
    valid: bool
    instance_scores: dict[str, float]   # per-instance breakdown (see extension)
    features: dict[str, float]          # Candidate.metrics — behavior descriptors
    feedback: str              # bounded, holdout-scrubbed reflection text (ASI)
    trials: tuple[TrialSummary, ...]
    cost_usd: float
```

A float serves greedy; `instance_scores` serves GEPA's Pareto frontier and
any per-instance engine; `features` serves every MAP-Elites engine (the
verifier contract already journals auxiliary numeric keys as
`Trial.metrics`); `feedback` serves every reflective engine. Direction
handling stays the GEPA-plan rule: raw scores in the journal, exactly one
maximizing transform at the engine boundary, inside the adapter.

**Privacy invariant (host-level, not per-engine):** `EvalResult`, proposer
prompts, and engine state may never contain `holdout_score`, holdout
reports/paths/logs, official verification, or anything derived from them.
The holdout-sentinel regression test from the GEPA plan moves up to test the
host service so every engine inherits it.

### 2. `propose(parent, *, inspirations=(), instructions="", components=("solution.py",)) -> Proposal`

The agentic mutation service, generalizing the GEPA plan's phase-3 proposer:
scratch dir under `SEARCH_DIR/<engine>/proposals/`, parent source plus
inspiration candidates' sources materialized, normal problem/data links, one
`OperatorRequest` through `Router.resolve(<engine>)` and `AgentPool`,
slot-gated and cost-accounted, edited components read back and validated
(non-empty, inside the scratch dir, actually changed).

Two commitments the survey forces:

- **N-ary.** `inspirations` is a first-class argument — AlphaEvolve-style
  prompts and merge/crossover need it. Greedy's single parent is the
  degenerate case, and the plumbing already exists: `Action.inspiration_ids`
  and the ensemble operator's copy-into-candidate-dir behavior.
- **Optional.** Engines that build their own prompts (FunSearch-style) skip
  `propose` entirely and use `complete` instead. The host never insists on
  owning mutation.

Returned `Proposal` carries the new source dict plus `AgentInfo`
(model/cost/tokens/session) so the eventual candidate's accounting survives
journal replay.

### 3. `complete(prompt, *, purpose, route=None) -> Completion`

A routed raw-LLM endpoint. Easy to miss and strategically load-bearing:
FunSearch/OpenEvolve-class engines call an LLM API directly from their own
prompt-building code. Without a host endpoint, every engine embeds its own
provider client — losing cost accounting, subscription auth (the
`ANTHROPIC_API_KEY` trap), abort wiring, and the UCB1 router. With it,
ShinkaEvolve's model-ensemble bandit and hillclimb's router become the same
machinery: the engine's "which model?" question is answered by
`Router.resolve(purpose)` with a `models:` pool.

Cost/audit: completions that don't yield candidates still cost money. They
are journaled as a new `event: "llm_call"` audit line (replay already skips
unknown event kinds — `Journal._replay` keeps only `CANDIDATE_EVENTS`), and
their cost feeds `total_cost_usd()` and the status record.

Implementation is a thin wrapper over a completion-capable agent; it does
NOT reuse the agentic `Agent` path (no candidate dir, no turns).

### 4. Control plane

Unchanged from the GEPA plan, promoted to a host guarantee: before every
proposal/completion the adapter calls `host.checkpoint()`, which drains the
command queue, raises `ParkedSearch`/`StopRequested`, and checks
`BudgetManager.should_stop()` plus the cost ceiling. Engines that accept
stop callbacks get `host.should_stop` handed in; engines that don't are
bounded by checkpoint calls at their iteration boundaries, with the abort
event covering in-flight agent/verifier processes. The hillclimb wall clock
and cost ceiling are always authoritative over engine-internal caps.

### 5. Record & state

- **Journal as lingua franca.** Engine candidates use existing operators
  (`seed`, `improve`, `ensemble`) with engine identity and lineage in
  `policy_meta` (`{"optimizer": "<engine>", ...iteration ids, source_hash,
  transformed fitness}`). This single rule is what keeps every TUI working
  for every engine with zero view changes.
- **Checkpoint dir.** `SEARCH_DIR/<engine>/state` for engine-owned
  checkpoints; `SEARCH_DIR/<engine>/proposals/` for scratch dirs. Resume
  reconciles engine checkpoints against journal replay exactly per the GEPA
  plan's phase 6 rules (source-hash maps, hard error on irreconcilable
  state, `Journal.next_candidate_id()` as the only id source, never delete a
  checkpoint automatically).

## The per-instance verifier contract extension

Today `$HILLCLIMB_RESULT` is `{"score": …}` plus auxiliary numeric keys that
become `Trial.metrics` (features, never scores). The extension adds one
reserved key:

```json
{"score": 0.042, "instances": {"zone1": 0.038, "zone2": 0.051, "zone3": 0.037}}
```

- `instances` maps stable instance keys to per-instance scores in the same
  metric and direction as `score`. Keys must be identical across candidates
  of a search (Pareto comparison is per-key); the engine treats a changed
  key set as an error.
- Journaled as a new `Trial.instance_scores: dict[str, float]` field
  (defaults empty — old journals replay unchanged); aggregated per-key by
  median into a `Candidate.instance_scores` property, mirroring
  `Candidate.metrics`.
- `score` remains authoritative for greedy acceptance, accept bands, best/
  selection, and every existing view. Verifiers that don't emit `instances`
  change nothing; engines that need instances and don't get them fall back
  to the single-instance degenerate case (and say so in status/logs).
- Providers opt in individually: emflow (per-zone/per-task losses) and
  mlebench (per-fold or metric components) each decide what an instance is.

This is the one contract change that turns Pareto/QD engines from "wired up"
into "actually better than greedy" — without it, GEPA's instance frontier
collapses to best-average.

## What stays engine-owned (deliberately not abstracted)

Archive structure, islands, MAP-Elites cells, Pareto bookkeeping, parent
selection, novelty filtering, prompt construction, merge logic — everything
engines compete on. No hillclimb concept normalizes across them; the
engine's brain is a black box behind the adapter. Corollary: no plugin
megaframework, no engine SDK, no registry of engine capabilities beyond a
small `EngineTraits` record (`supports_parallel`, `needs_instances`,
`uses_propose`, `uses_complete`) used only to fail configs early with honest
messages.

## Concurrency rule

`evaluate` must be safe to call from engine worker threads from day one:
journal appends, status writes, and id allocation happen under the searcher
state lock (the same discipline `SearchPolicy` already documents). GEPA's
MVP being serial is a config constraint (`EngineTraits.supports_parallel =
False`), never an architectural assumption — AlphaEvolve-class engines are
aggressively parallel and the slot/`parallel_agents` infrastructure
already exists to meter them.

## Phasing

1. **GEPA MVP** (`docs/gepa-integration-plan.md`, in flight). Ships
   `SearchStrategy` dispatch and `evaluation.py`. One adjustment to that plan:
   shape the shared evaluation helpers' return as `EvalResult` now, even
   though GEPA only consumes score + feedback — it is the one interface
   every later engine touches.
2. **Host facade extraction.** After GEPA lands, factor its
   evaluator/proposer/control bridge into `host.py` (`OptimizerHost`) with
   the holdout-sentinel test at host level. Pure refactor; `GEPASearcher`
   becomes the first consumer. Do not build this before GEPA proves the
   bridge — a facade with one consumer is speculation.
3. **Per-instance extension.** Verifier contract + `Trial.instance_scores`
   + `EvalResult.instance_scores`; wire GEPA's instance-Pareto frontier to
   it; emflow provider emits per-zone instances first. Includes a
   per-instance heatmap view (candidates × instances) — a plotui screen no
   optimizer library ships, and the point of owning the UI layer.
4. **`complete()` + second engine.** ShinkaEvolve is the chosen third
   runner: its sample-efficiency thesis (novelty rejection, adaptive parent
   sampling, bandit LLM ensemble) matters most when every evaluation spawns
   a paid agent, and it exercises `complete()` and the router-as-ensemble
   wiring. This phase is what earns the generalization — adapt the facade to
   the second real consumer rather than predicting it.
5. **Parallel engines.** Lift the serial constraint for engines that want
   it; evaluate/propose already thread-safe from phase 2, so this is
   scheduling and slot arithmetic, not redesign.
6. **Meta-level (exploratory).** Point a landed engine (GEPA) at
   hillclimb's own operator prompt templates, with experiment outcomes as
   the metric — the DSPy-shaped payoff. Requires nothing new from the host
   contract; the experiment framework and credit economy are the harness.

Each phase ends green: full suite passes, greedy/OpenEvolve behavior
byte-identical, and (for engine phases) a three-arm-style live experiment on
the bundled circle-packing problem recorded under `docs/`, following the
GEPA plan's acceptance pattern. The experiment rig doubles as the standing
evaluation for engine newcomers (CodeEvolve, GigaEvo, Darwinian Evolver stay
on a watchlist until a benchmark says they earn an adapter).

## Sanity check: the roster against the contract

| Engine | evaluate | propose | complete | notes |
|---|---|---|---|---|
| greedy (native) | ✓ | ✓ (1 parent; ensemble n-ary) | – | trivial runner; stays `GreedySearcher` |
| GEPA | ✓ (+feedback, +instances) | ✓ (custom proposer) | – | phase 1/3 |
| OpenEvolve-as-engine | ✓ (+features) | – | ✓ | policy-tier integration already exists; full engine only if ever needed |
| ShinkaEvolve | ✓ (+feedback) | – | ✓ (model pool → router bandit) | phase 4 |
| CodeEvolve | ✓ (+features) | ✓ (inspiration crossover) | ✓ | watchlist |
| AIDE | – | – | – | competitor/benchmark arm, not an integration |

No engine on the list needs a sixth service — evidence the contract is the
right cut.

## Non-goals

- Replacing `SearchPolicy` or changing greedy/OpenEvolve semantics.
- A generic engine SDK, plugin discovery, or capability negotiation beyond
  `EngineTraits`.
- Normalizing engine archives into a hillclimb concept.
- Multi-objective scoring (distinct from per-instance; revisit only when an
  engine demands it).
- Changing the journal schema beyond `Trial.instance_scores` and the
  `llm_call` audit event.
