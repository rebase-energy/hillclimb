# Changelog

## 0.6.0 — 2026-09-30

The climber becomes something you compose. It is one block of config,
defined where the run is defined, built from prebuilt or your own modules —
a policy, a selector, operators, a tuner, a memory — each named the same
way and each with a base class; the same blocks compose in Python
(`hillclimb.Climber`, `hillclimb.run`). The SDK's names were tidied without
aliases: a climber written for 0.5 needs the edits listed under "Renamed"
and "Migrating from 0.5". Run folders are not affected — every recorded
search still loads, and resumes.

### Added
- **A sandbox, on by default.** Agents and verifier runs are confined by the
  operating system: `sandbox-exec` on macOS, bubblewrap on Linux (`sudo apt
  install bubblewrap`). They write only to their candidate's folder, the
  temp dirs and the ML caches, and cannot read `~/.ssh`, cloud logins, `.env`
  files, the journal or the holdout runs. A search does not start when the
  sandbox cannot, and says how to fix it; `sandbox: off` in `hillclimb.yaml`
  (or `HILLCLIMB_SANDBOX=off`) runs without. Windows has no sandbox and runs
  unsandboxed with a warning. Codex keeps its own sandbox.
  See [docs/sandbox.md](docs/sandbox.md).
- **`hillclimb sandbox check`** runs a script that behaves like a hostile
  solution inside the sandbox and lists what happened to every attempt:
  writes outside its folder, reads of your keys, connections out, a signal
  to a process outside. Exit code 1 when one got through.
- **Agents without internet.** `allow_internet_for_agents: false` in
  `hillclimb.yaml` leaves the agents nothing but their model provider: their
  traffic goes through a proxy in the engine that refuses every other host.
  Claude Code loses web search, web fetch and MCP servers, codex loses web
  search, and the draft prompt stops asking for web research. The default
  stays `true`.

### Changed
- **`allow_network` is `allow_internet_during_solution`** in `problem.yaml`,
  so it cannot be mistaken for the agents' internet. The old key still loads.
  The sandbox now enforces it: a verifier has no network unless its problem
  sets it to `true`, so a verifier that downloads something itself needs it.
- **A study's setups are experiments.** What `hillclimb experiment run`
  compares is a *study*, and each named setup in it is an *experiment* (it
  was an *arm*). A spec lists them under `experiments:`; `arms:` still
  loads. `hillclimb run` tags a search with `--study S --experiment E` (it
  was `--experiment S --arm E`), and a mixed fleet's per-setup settings are
  `--experiment-set NAME:KEY=VALUE` (`--arm-set` still works). `search.yaml`
  records `study`, `experiment` and `experiment_overrides`; runs written
  before the rename read the same. `experiment report --json` names the
  study `study` and lists `experiments` (each with an `experiment` key). In
  the Python API, `create_search`, `fleet_argv`, `run_fleet`, `mixed_fleet`
  and `FleetEngine` take the new names, and `load_experiment` /
  `ExperimentSpec` are `load_study` / `StudySpec`.

### The climber is a block in the run config
A climber is no longer a separate `climber.yaml` referenced by name: it is
DEFINED where the run is defined, as one block. The same block is an entry's
`climber:` in a run spec, the spec's own top-level `climber:` (the default
for its entries), and `climber:` in `hillclimb.yaml` (the folder's default).

```yaml
# run.yaml
problems:
  - target: heilbronn-11
    budget: 30m
    climber:
      policy: greedy                  # or `loop: gepa`
      params: {num_drafts: 5}
      operators: [draft, debug, improve, crossover.py:Crossover]
      operator_params: {draft: {retrieval: false}}
      tuner: optuna
      memory: files
      prompts: prompts/
```

- **Every module is named the same three ways**: a registry name (`greedy`,
  `optuna`), a `.py` file (`mine.py` or `mine.py:Class`), or
  `package.module:Class`. Tuners took registry names only before. File refs
  are relative to the file the block is written in.
- **A bare name is a preset**: `climber: greedy | openevolve | gepa`, or one
  `.py` file. `--climber NAME` is the same on the command line. Defaults
  live on the classes, so `{policy: greedy}` and `{loop: gepa}` are complete.
- **A run records what it ran.** `runs/<id>/spec.yaml` carries every
  search's full block, and `hillclimb run runs/<id>/spec.yaml` runs it again.
- **Precedence is per block.** A spec entry's block replaces the spec's
  default, which replaces the folder's, which replaces the user-level one —
  whole, never merged, so one policy's params cannot reach another. `--set
  climber.params.k=v` then edits the block that was chosen; `--set
  climber.operators.draft.retrieval=false` reaches one operator's params.
- **A search resumes as exactly the climber it started as.** Its snapshot
  (`searches/<id>/climber/`) holds the block, the local files it reaches and
  its prompts; `resume` restores all of it (operators, memory and graph
  module were taken from the live config before). A climber's identity
  (`climber_sha256`) is its block, the bytes of every local file it reaches —
  files only reached by a relative import included — and its prompts.
- **`hillclimb climber show [NAME]`** prints a climber as a paste-able block.
  `climber new NAME --from greedy` copies a policy's source into
  `climbers/NAME.py` and prints the block that runs it.
- **Experiments** name or define a climber with `climber:` (a preset, a
  file, or the block); `climber.<field>` overrides edit it.
- **A selector slot: `select:`.** A policy decides the kind of move; its
  selector picks WHICH candidate to expand and what rides along
  (inspirations, a paragraph of prompt context, a note in the journal).
  `best` (the default) is what makes greedy greedy; `map-elites` is
  OpenEvolve's MAP-Elites archive. Its settings are `select_params`. Write
  your own as a `Selector` subclass in a file: `select: mine.py`.
- **`openevolve` is a composition, not a policy of its own**: the preset is
  `{policy: greedy, select: map-elites, params: {ensemble: false,
  tune_budget: 0}}`. The same schedule runs over any selector, and any
  policy can take `map-elites`.
- **`Policy` is a base class** (it was a protocol): list your knobs in
  `DEFAULTS`, implement `propose`, and `param()`, `resolved_params()`,
  `debuggable_tip()`, `prospective_branches()`, `draft_complexity()`,
  `top_distinct()` and `self.selector` are there. A class with just
  `propose` and `observe` still runs.
- **A param the policy does not have is an error**, before anything is
  spent: `climber.params: greedy has no param 'num_draft' (it has: ...)`.
  It was silently ignored.
- **Memory is a module the block names**, like every other slot: `memory:
  files | none`, a `.py` file or `package.module:Class`, with its behaviour
  in `memory_params` (for `files`: `max_cards`, `live`, `claims`,
  `graph_retrieval`, `credit`, `playbooks`, `skills`, `complexity_prior`, and
  the `graph` module that indexes it). A `Memory` subclass takes part in a
  search through five steps — `bind`, `retrieve`, `live`, `publish`,
  `record` — each of which defaults to doing nothing. `learning.enabled:
  false` and `--no-learning` stay the user's switch over any climber;
  `learning.dir`, `learning.tool` and `learning.claims_timeout_s` stay in
  hillclimb.yaml.
- **`Tuner` is a base class** too, and a tuner can be a file
  (`tuner: anneal.py`).
- **Compose a climber in Python.** The building blocks are importable by
  name — `hillclimb.policies`, `.selectors`, `.operators`, `.tuners`,
  `.memory`, `.loops` — and take their params by keyword:

  ```python
  import hillclimb as hc

  climber = hc.Climber(
      policy=hc.policies.Greedy(num_drafts=3),
      select=hc.selectors.MapElites(num_islands=2),
      operators=[hc.operators.Draft(retrieval=False), hc.operators.Debug(), MyCrossover],
      tuner="optuna",
      memory=hc.memory.FilesMemory(max_cards=1),
  )
  hc.run("heilbronn-11", climber=climber, budget="10m")   # one search, in this process
  hc.run_spec("run.yaml")                                 # every entry of a spec
  climber.write("climber.yaml")                           # the same climber, as its block
  ```

  A composed climber IS the block: your own classes are written down by
  registry name, `module:Class`, or `their_file.py:Class`. A class that
  exists only in the running process (a notebook cell) still runs with
  `hc.run`, but such a climber cannot be written down, resumed or handed to a
  detached engine, and says so (`NotPortableError`).

**Migrating from 0.5**
- A directory climber (`climbers/mine/climber.yaml`) is no longer a
  reference: `hillclimb climber show climbers/mine` prints it as the block to
  paste into your run config.
- `climber: {ref: NAME, ...}` in `hillclimb.yaml`, `climber.ref` in `--set`
  and experiment specs, and the `operators: {draft: {...}}` overlay still
  load: they read as the block they meant.
- In a block, `description`, `similarity` and `holdout_timing` are refused
  with what to do instead (a loop declares `holdout_timing` on its class).
- MAP-Elites' settings (`num_islands`, `feature_dimensions`,
  `num_inspirations`, `random_seed`, ...) are `climber.select_params` now;
  among `climber.params` they are refused with that advice. A 0.5 block
  `climber: {ref: openevolve, params: {...}}` is sorted into the two on read.
- `learning.max_cards`, `.live`, `.claims`, `.graph_retrieval`, `.credit`,
  `.playbooks`, `.skills` and `.complexity_prior` are `climber.memory_params.*`,
  and `climber.graph` is `climber.memory_params.graph`. The old keys still
  load, in config files and in `--set`.
- `RandomTuner` and `OptunaTuner` are `RandomSearch` and `Optuna`.
- `DraftOperator`, `DebugOperator`, `ImproveOperator`, `EnsembleOperator` and
  `GepaReflectOperator` are `Draft`, `Debug`, `Improve`, `Ensemble` and
  `GepaReflect`. Class paths recorded in run folders are mapped.
- `GreedyPolicy` is `Greedy` (`hillclimb.modules.policies.greedy`); its
  `complexity_start` constructor argument is the param `complexity_start`.
  `OpenEvolvePolicy` is gone as a class to subclass (it survives only so
  pre-0.6 searches resume).
- Searches started before 0.6 still load in every view and resume from
  their snapshot. `search.yaml` is schema v4: `climber` (its label),
  `climber_spec` (the block), `climber_sha256`; v2 and v3 records are mapped
  on read.

### Renamed, without aliases
A climber written for 0.5 needs these edits before it loads; `hillclimb.sdk`
raises an `ImportError` that names the new spelling. Run records are not
affected: old journals and `search.yaml` files read as before.

| 0.5 | 0.6 |
|---|---|
| `SearchPolicy` | `Policy` |
| `SearchLoop` | `Loop` |
| `PolicyInput` | `SearchState` |
| `PolicyJournal` | `JournalView` |
| `Preparation` | `Attempt` |
| `Action.policy_meta`, `Candidate.policy_meta` | `climber_meta` (the journal key `policy_meta` is read as before) |
| `OperatorRequest`, `OperatorResult` (agents) | `AgentRequest`, `AgentResult` |
| `hillclimb.integrations.{emflow,mlebench,einsteinarena}` | `hillclimb.providers.…` |
| `hillclimb.integrations.gepa` | `hillclimb.climbers.gepa` (refs recorded in run folders are mapped) |
| `--policy`, `hillclimb policy check` (hidden since 0.4) | `--climber`, `hillclimb climber check` |
| `modules.policies.get_policy`, `load_policy_file`, `policy_label`, `policy_base_dir` | `climber.load_climber`, `climber_label`, `climber_base_dir` |

### Fixed
- **A resumed `openevolve` search has the database the live one had.** The
  MAP-Elites database is now rebuilt from the journal alone (scored
  candidates in journal order), so it no longer depends on the order results
  landed in, on a tune trial moving a score already binned, or on a resume
  showing every candidate the finished journal.
- **`hillclimb climber check` has a `resume` finding** that catches exactly
  that class of bug: a policy that watched the journal grow must propose what
  one shown the finished journal proposes. The check also knows a climber's
  own operators now (it reported them as unknown), and `inject`.
- **A tuner seed set by the climber seeds the tuner.** Only a seed the user
  overlaid in `climber.tuner_params` was used.
- **The claim-distill pass records what it spent**: a `memory_agent_call`
  audit line in the journal. It stays outside the search's budget.

## 0.5.0 — 2026-09-28

### Added
- **No bash needed on Windows.** `hillclimb problem get` writes the verifier
  for the machine that fetches the problem: `verifier.sh` on macOS and Linux,
  as before, and on Windows a `verifier.py` with the same steps in Python. The
  engine runs it with its own interpreter. A problem with only a `verifier.sh`
  still runs on Windows through Git for Windows' bash.
- **Multidimensional knapsack ladder:** `mknap-100-5` and `mknap-250-10`,
  Chu & Beasley's OR-Library instances, with the best feasible values as
  reference lines.
- `best-known/` (in the repo): the best construction we have for each example
  problem, with its value, provenance and reference; `check.py` re-scores
  every file.

### Changed
- **The agent is the agent; the operator is the move.** The coding agent that
  runs the operators (draft, debug, improve, ensemble) is `--agent claude-code
  | codex | pi | dummy` and `agent:` / `agent_auth:` in config.yaml and the
  `routing:` block; it was `--backend`. The agents a search keeps busy at once
  are `--parallel-agents` and `concurrency.parallel_agents` (with
  `concurrency.machine_max_agents`); they were counted as operators. The
  package is `hillclimb.agents`, the protocol `Agent`. Every old spelling
  still loads: the flags as aliases, config keys and suite specs on read,
  `search.yaml` and journals written before the rename.
- **The hillclimb dir is the folder holding `hillclimb.yaml`.** `hillclimb
  init [DIR]` makes the current folder (or DIR) one in place; `problems/`,
  `runs/`, `knowledge/`, `climbers/` and `experiments/` sit beside the file
  instead of under a nested `hillclimb/` folder. `init` writes `.gitignore`
  rules that commit each run's record and ignore its bulk.
- `hillclimb --help` lists the core commands; `hillclimb --help --all` lists
  every one.
- CPU accounting: a trial's `cpu_s` counts the solution run (the starter
  verifiers no longer `exec` their scorer, which dropped it on macOS), agent
  calls journal their own CPU, and a group killed at a timeout adds what its
  running descendants had burned.

## 0.4.0 — 2026-09-28

The method is now a **climber**: a shareable bundle (a search policy or loop,
operators, prompts, a tuner) that the fixed **harness** runs. Everything a user
touches changed spelling once, in this release; old spellings are mapped on
load and say where they went.

### Added
- **Pluggable knowledge graph.** `graph:` in a climber manifest (or
  `climber.graph` in config.yaml) names the module that builds
  `knowledge/graph.json`, ranks the claims a search is shown and answers
  `hillclimb knowledge query`: `knowledge-graph` (the built-in, the default),
  or a `file.py` / `package.module:Class` subclassing `hillclimb.sdk.GraphModule`.
  `graph.json` records its builder, so switching (or editing a file module)
  rebuilds it. `hillclimb climber check` resolves `graph:` too.
- **Climbers.** `hillclimb run <problem> --climber greedy | openevolve | gepa |
  hillclimb/climbers/<name> | file.py`. `hillclimb climber list` shows what is
  available, `hillclimb climber new <name> --from greedy` copies a climber
  (manifest, policy source, prompts) into `hillclimb/climbers/<name>/` for
  editing, `hillclimb climber check` replays recorded journals through it
  before any budget is spent. A search snapshots its climber into
  `searches/<id>/climber/` and resumes from that copy.
- **Starter problems.** A bundled catalog of construction problems in the
  heilbronn shape — a CSV of numbers, an exact verifier, no data, no holdout,
  zero noise — with the best known value beside each one in
  `hillclimb problem list`: circle packing (26, 32), Heilbronn triangles
  (11, 14, 17, convex 13), low-autocorrelation binary sequences (40, 60),
  Tammes and Thomson sphere problems, AlphaEvolve's autocorrelation
  inequalities, an 11-dimensional kissing configuration, Golomb rulers.
  Every family is stamped by a generator under `problems/make_*.py`.
- **Budget in every dimension:** `budget.max_evaluations`, `budget.max_tokens`
  beside the clock and `budget.max_cost_usd`.
- `hillclimb verify` scores a problem whose floor is a set of files
  (`baseline_files`) as the engine does.
- `hillclimb.sdk`: the one import a climber needs.
- **Native Windows.** `pip install hillclimb` works on Windows (plotui 0.5.1
  ships a Windows wheel). File locks, process groups, `hillclimb ps`/`stop`/
  `reset`, venvs and directory links have Windows equivalents
  (`harness/oscompat.py`); `.sh` verifiers run through Git for Windows' bash.

### Changed
- climber.yaml / config.yaml: `memory: knowledge-graph` → `memory: files`
  (the memory is the YAML under `hillclimb/knowledge/`; the graph is a derived
  index over it). The old spelling still loads; `hillclimb climber check`
  points it out.
- config.yaml: `search.policy`/`search.policy_params` → one `climber:` block
  (`climber: greedy` or `climber: {ref, params, tuner, memory}`);
  `search.n_replicates`/`noise_k`/`min_improvement` → `evaluation:`;
  `search.parallel_operators`/`machine_max_operators` → `concurrency:`;
  `ensemble.*` and `search.num_drafts` → `climber.params`. Old keys still load.
- `hillclimb policy check` → `hillclimb climber check`; `--policy` → `--climber`
  (the old flags work and print a one-line note).
- Experiment specs use the new keys (`climber.ref`, `climber.params`,
  `concurrency.parallel_operators`, `evaluation.n_replicates`).
- Run folders record `climber`, `climber_sha256`, `climber_manifest`,
  `climber_params`, `hillclimb_version` (SearchMeta schema 3; v2 folders keep
  loading).
- The harness refuses an invalid action before anything exists (a debug on a
  passing candidate, say); three refusals in a row end the search.
- The greedy climber ranks ensemble inputs on validation scores only
  (policies are holdout-blind).

### Removed
- The hidden 50-candidate cap (it was unreachable from config; set
  `budget.max_evaluations`).
- The engine tier: GEPA runs on the same harness seam as every other climber.
- The image-rendering `hillclimb tree` (the live TUI of the same name stays).
