# Changelog

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
