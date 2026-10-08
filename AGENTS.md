# hillclimb

Instructions for coding agents working in this repository. Codex and pi read
this file directly; Claude Code reads it through `CLAUDE.md` (`@AGENTS.md`),
so it is the one place to keep them, written for any agent.

Hillclimbing on verifier-defined problems: a search engine that spawns
headless coding agents (Claude Code, Codex, pi) as operators
(draft/debug/improve/ensemble), scores each candidate through the problem's
verifier, and keeps the best solution per search.

Hierarchy: **Run** (one invocation, `runs/<run-id>/`) → **Search** (one engine
process, `searches/<search-id>/`) → **Candidate** (immutable code artifact) →
**Trial** (one parameter set of that code) → **Replicate** (one seeded
execution). A trial's score is the MEDIAN of its replicates; a candidate's
score is its BEST trial (`Candidate.stamp_best_trial` sets `Trial.is_best`,
the only place direction enters the aggregate; `val_score`/`metrics`/
`report`/`holdout_score` all follow the best trial). Candidates without a
declared parameter space have exactly one trial (`params={}`); pre-split
journals fold their flat `trials` list into one trial on load. The problem
is an *attribute* of a search, not a
level: a run may hold searches on different problems (MLE-bench) or several
on one (`--parallel-searches`). Search ids are `<problem-id>`, then `<problem-id>-2`, `-3`
(atomic mkdir allocation); `search.yaml` carries `problem_key` — the
canonical cross-run identity (`emflow://…`, `mlebench://…`, or the local
problem id; backfilled on read like hillclimb-go's `EffectiveProblemKey`) —
and every problem-scoped view (chart, best-ever, knowledge) groups on it.

A problem **is its verifier**: `problems/<id>/verifier.sh` is the scoring process
the engine starts. It drives `solution.py` and writes the score to
`$HILLCLIMB_RESULT` (a `{"score": …}` object or a bare number; other numeric
keys are journaled as `Replicate.metrics` — feature dimensions for policies,
never a score); exit 0 means valid. A replicate the engine kills under a
timeout clamped by the search's *remaining budget* is journaled `abandoned`
("cut off at the budget wall"), never `buggy`: verifier crashes/contract
failures and unit-test timeouts make a candidate buggy and thus a debug target.
Optional `unit_tests` in `problem.yaml` declares a framework-neutral
root + argv (`{python}`, `{solution}`, `{tests}`). The suite is frozen once per
run/problem, copied visibly but non-authoritatively into candidates, and run
once per trial after the first verifier replicate establishes runnability.
Candidate/trial verdicts are `passing` (verifier + tests), `failing` (verifier
ran and the suite completed with failing tests), and `buggy` (execution/contract failure); old
`ok` journal values load as `passing`. Only passing trials rank or ship, while
failing and buggy candidates are debug targets. `holdout: true` in
`problem.yaml` makes the engine run the same script
with `--holdout` in a directory coding agents never see. Holdout scoring is the
HOST's, never a search strategy's: `CandidateEvaluator` (`harness/evaluation.py`)
scores the hidden split as part of `run_trial` — a trial that fails it is
not-ok, like a verifier crash — and `api.build_evaluator` decides WHEN
(`holdout_timing`): `inline` per candidate as it lands (greedy; `watch`
shows holdout live; `holdout.top_k` gates the spend, floors are never gated),
`after` once `run()` has returned (`api._finish_holdout`, for engines whose
state must never see a holdout value — the `_ENGINES` entry declares it). Providers (`emflow://`,
`mlebench://`) supply their own argv for the same contract. Never read a score
off stdout — coding agent code shares that stream.

A problem may optionally ship `interface.py` (Gym-spaces-style Python objects
from `hillclimb.spaces`, picked up by default like `contract.md`): its
`describe()` renders into the contract prompt, `check()` gives verifiers and
coding agents located format violations, `sample()` writes a format-valid artifact,
and `hillclimb verify` lints the baseline's output against it. The engine
never runs the check itself — a PYTHONPATH shim
(`runtime.ensure_interface_shim`, wired always-on in `api.build_executor`)
just makes `from hillclimb import spaces` importable inside the runtime venvs
so the verifier or coding agent can call it voluntarily. `spaces.py` must stay
self-contained (stdlib top-level imports only; it is copied verbatim into the
shim).

**Package map** (`docs/package-layout-plan.md`; `tests/test_layout.py` enforces the
import directions): `hillclimb.harness` is the fixed core — `core.py` plus everything
that scores, records or spends (evaluation, executor, journal, candidate, store, run,
budget, slots, control) and `glue.py`, config → climber → loop; `hillclimb.modules` is
everything a climber is built from, one subpackage per kind with its contract in
`base.py` (`policies/`, `selectors/`, `operators/`, `tuners/`, `memory/`,
`similarity/`), implementations importing only `hillclimb.sdk`, plus `refs.py` (the ONE
resolver of a module's name) and `spec.py` (`ClimberSpec`, the `climber:` block — light
enough for `config.py` to import); `hillclimb.tui` is every terminal view and its layout
(never imported by the harness or the modules); `hillclimb.cli` is one module per
command group with `common.py` for what commands share (reached as `common.x()` so
one patch covers every command) and `__main__.py` for the engine children. The flat
top level is the public surface only: `api` (incl. `run` / `run_spec` / `start` and the
`Search` session), `results` (`SearchOutcome`, `open_search`), `config`,
`problem`, `project`, `benchmark_providers`, `climber`, `experiment`, `connect`,
`spaces` (byte-copied into runtime venvs, so it stays), the six facades `policies`,
`selectors`, `operators`, `tuners`, `memory`, `loops` (lazy single modules:
`hillclimb.policies.Greedy`; never packages — `_moved.py` owns the
`hillclimb.policies.` prefix of pre-move refs), `sdk/`, `catalog.py` + `scaffold/`, `agents/`,
`providers/` (emflow, mlebench, einsteinarena), `prompts/`, `runtime/`. The package
holds NO climber: the repo-root `climbers/` is the catalog (`greedy/policy.py` = `Best` +
`Greedy`, `openevolve/policy.py` = `MapElites` + `Greedy`, each ONE self-contained file
holding both policies with every decision and default written out; `gepa/`, a folder
that brings a loop), `climber get` copies a folder byte for byte and
`tests/test_catalog.py` keeps the two `Greedy` copies from drifting; `src/hillclimb/
climbers/` is an empty docstring package. `modules/policies/` and `modules/selectors/`
hold only the bases, which decide nothing. `_moved.py` maps pre-move module prefixes
(`MOVED`) and renamed classes (`RENAMED`, incl. the 0.6-0.8 `modules.policies.greedy:Greedy`
/ `modules.selectors.best:Best` paths) wherever a `module:Class` ref is imported.

- The hillclimb dir: any folder holding a `hillclimb.yaml` (the config and
  the marker), looked for in the CWD only (or the CWD's `hillclimb/` subfolder) — no upward search, so a hillclimb checkout beside a project is never taken for it; problems/, runs/, knowledge/,
  climbers/, experiments/ and store.sqlite sit beside it and relative config
  paths resolve against it. `init [DIR]` makes the CWD (or DIR) one and
  refuses a folder with its own problems/ or runs/; `reset` deletes only
  those hillclimb-owned entries (`common.owned_paths`), never the folder.
  This repo's root is its own hillclimb dir.
- Every run carries its spec: `api.write_run_spec` writes `runs/<run-id>/spec.yaml`
  (one `SuiteEntry` per search as resolved — `spec_entry`; entries carry the FULL
  `climber:` block and `set`) from every launch path (foreground run, suite, fleet,
  experiment, `api.run` / `run_spec`), so `hillclimb run <run_dir>/spec.yaml` reruns
  it; `init` writes no `specs/` any more and its `.gitignore` rules
  (`common.INIT_GITIGNORE`) commit a run's record and ignore its bulk (candidates/,
  logs/, control/, best/ except solution.py + params.json, store.sqlite,
  knowledge/graph.json, .env)
- Directory vocabulary: every level is `<level>_dir` — `run_dir`,
  `search_dir`, `candidate_dir` (`searches/<id>/candidates/<cid>/`, where the
  coding agent works), `trial_dir` (`candidates/<cid>/trials/t<i>/`, holds the
  trial's `params.json`), `replicate_dir` (`…/t<i>/replicates/r<j>/`, the
  verifier's cwd; r0 of the best trial is hoisted to the candidate root).
  The word "workspace" is banned (`tests/test_vocabulary.py`
  enforces it); old journals/status files that still carry a `workspace` key
  are mapped to `candidate_dir` on load.
  Machine-scoped state (shared venvs, emflow cache, coding agent slots) lives in
  `~/.cache/hillclimb/`.
- Tests: `uv run pytest`
- New problem: `hillclimb problem new <id>` (scaffold from `scaffold/problem/`, a two-step
  run.py + score.py problem) or copy a bundled one (`hillclimb problem get <problem>`); check a
  verifier with `hillclimb verify <problem> --repeat 5` (the spread it prints
  is the noise floor — improvements below it are not real)
- Noisy metrics: a trial's score is the MEDIAN of its replicates;
  `evaluation.n_replicates` + `noise_k`/`min_improvement` set an accept band so
  the search cannot climb noise (the floor is the within-trial replicate
  spread — spread across parameter sets is signal), and
  `concurrency.parallel_replicates: 1` (one at a time) is mandatory when the
  metric measures the machine (time/throughput/memory) — replicates running
  at once measure each other; 0 (the default) runs all of a trial's at once.
  Seeds are never tuned. `n_trials`/`trial_mode` and `replicate_mode:
  parallel | serial` (= 0 | 1, `config.LEGACY_VALUES`) are accepted as
  legacy spellings
- Concurrency: `concurrency.parallel_agents` per search, `concurrency.machine_max_agents`
  across the machine (flock slots in `~/.cache/hillclimb/agent-slots/`, default
  `min(8, cores-2)`); verifier, holdout and coding agent envs are capped at
  `concurrency.solution_cpus` cores (default 1): `$HILLCLIMB_CPUS` plus the
  `executor.SINGLE_THREAD_ENV` thread variables (`executor.single_threaded`,
  parent values win for the latter). Nothing enforces it for processes a
  solution starts itself; `Replicate.cpus` journals the allotment and
  `Replicate.oversubscribed` (cpu_s / duration_s > 1.5 x it, runs >= 5 s)
  flags a run that used more (`cpu 6.9/1` in watch); `cli/run.
  _warn_oversubscribed` warns at launch. CPU accounting:
  `harness/procs.py` (`Reaper`) reaps every child the harness spawns — verifier
  runs and coding agent calls — through `os.wait4`, sampling live descendants with `ps`
  before a group kill; `AgentResult.cpu_s` → `AgentInfo.cpu_s` is the coding agent
  call's local CPU and the chart's cost fold adds it to the trials'. Starter
  verifiers call their scorer plainly, never `exec` it (macOS drops the shell's
  child CPU at an exec; `tests/test_verifier_scripts.py`). `hillclimb ps` lists the
  engine process trees; `stop --all` reaps engines whose hillclimb dir was deleted; `reset` kills only the engines pinned to this folder's hillclimb dir, then deletes the dir
- Operator agents: `claude-code`, `codex`, `pi`, `dummy` and `toy` (`agents/toy.py`:
  scripted moves by attempt KIND on `fitness-landscape`, deterministic per
  `(seed, candidate, kind)`, a `toy: step=<float>` prompt line sets the stride,
  two declared params so tune jobs run, refuses non-operator calls such as
  `distill`). `agents.register_agent(name, factory)` (`hc.register_agent`) adds
  one for THIS process only: `run_fleet` refuses a registered agent, like a
  non-portable climber. Pi supports
  `routing.<op>.sampling` (numeric provider fields), with action → operator →
  default precedence and candidate-journal persistence. Sampling on other
  coding agents fails validation. `pi.models_file` adds custom/local providers;
  copied configs live under isolated `~/.cache/hillclimb/pi-home/<auth>/`
  (content-hashed subdir for custom models). Pi startup is offline, search
  startup preflights each model/sampling route, and provider errors are read
  from JSON `stopReason` even on exit 0. Debug children use `--fork` to keep
  history while binding tools to the child's cwd; `--session` restores the
  parent's cwd and must not be used across candidates.
- Sandbox (`harness/sandbox.py`, stdlib-only, `docs/sandbox.md`): on by
  default (`sandbox: off` / `sandbox.enabled`, `$HILLCLIMB_SANDBOX=off`; the
  test suite runs with it off, `tests/test_sandbox.py` switches it on). A
  `SandboxPolicy` (write / write_prefix / deny_read / network open|none|proxy)
  is enforced by `launch(argv, policy)`: sandbox-exec on macOS, bwrap on Linux,
  nothing on Windows (`backend()` None → unsandboxed + warning). `backend()`
  probes once and raises `SandboxUnavailable` with the fix; `cli/common.
  require_sandbox` and `api._preflight_sandbox` refuse before anything spends.
  macOS allows NO sandbox inside a sandbox, so a policy wraps the whole
  process ONCE: claude-code and pi are wrapped (`_sandboxed` adds the
  candidate dir + their own state), codex never (its own sandbox), a
  meta-problem's verifier never. Verifier/unit-test/holdout runs get
  `verifier_policy` through `run_logged(sandbox=…)`; network is the problem's
  `allow_internet_during_solution` (`allow_network` is the legacy key), now
  enforced. `allow_internet_for_agents: false` → `AgentRequest.
  allow_internet=False` → network `proxy`: `AllowlistProxy` (a thread of the
  engine, CONNECT only) lets the coding agent's `MODEL_HOSTS` through; on Linux the
  netns reaches it through `sandbox.py bridge` (localhost:3128 → unix socket).
  It needs the sandbox; the draft drops its research cue
  (`OperatorContext.agent_internet`). Claude Code's own `sandbox` setting is
  NOT used: under bypassPermissions it let curl through. `hillclimb sandbox
  check` (`harness/sandbox_check.py` + the stdlib-only `sandbox_probe.py` it
  runs INSIDE the verifier and the no-internet coding agent policy) is the proof a
  user can run; the docs site's Sandbox page shows its output
- Model per agent: `config.model` (default `sonnet`) is Claude's vocabulary; the
  codex coding agent's `native_model` omits `--model` for a Claude alias/id so the Codex
  CLI's own default answers (journaled as `codex-default`; the connect ping says
  so too) — OpenRouter routes always pass the id
- Coding agent billing: `agent_auth` picks who pays — `subscription` (the Claude or
  ChatGPT login), `api-key`, or `openrouter`, which points codex or pi at
  OpenRouter (`wire_api: responses`; the key comes from the environment or a
  `.env` beside hillclimb.yaml) and bills OpenRouter credits instead. Every codex
  call runs under an isolated `CODEX_HOME` in
  `~/.cache/hillclimb/codex-home/<auth>/`, so personal `~/.codex` settings
  change neither a search's results nor its token bill; a provider 402 parks
  the search as `out_of_credits`, and `harness/pricing.py` fills `cost_usd` from
  OpenRouter's catalogue so `budget.max_cost_usd` applies. `hillclimb connect`
  (`connect.py`) is where a credential is checked on purpose instead of at the
  first spawn: every probe runs through the SAME env builders the coding agents use
  (`subscription_env`/`codex_env`/`pi_env`), so an inherited `ANTHROPIC_API_KEY`
  shadowing a subscription shows up in the table; a bare `connect` is the
  status of all four targets (`claude`/`codex`/`pi` are coding agents and own their
  login, `openrouter` is a billing route), `connect <agent>` runs that login,
  materializes the isolated home, pings the route with one tool-free call
  (`connect.ping`) and pins `agent`/`agent_auth` with a line-level edit of
  a config.yaml that keeps its comments — the USER level
  (`~/.config/hillclimb/config.yaml`, so `connect` precedes `init`; a folder's
  hillclimb.yaml overrides it, `--local` writes there) and only when no coding agent is
  pinned yet, unless `--default`. Keys live in a `.env` (user-level beside the
  user config, read under the folder's own by `Config.load`), never in `Config`.
  Connection states: `logged-out` / `logged-in` / `ready` (`no-key` /
  `key-set` / `ready` for openrouter) — `ready` = the login works AND
  `mark_connected` left `connected.json` in `record_dir` (the ping's scratch dir
  under the machine cache; codex/pi also need their staged home); `Status.ok`
  is the login, `Status.connected` the `ready` state.
  Before a command uses coding agents (`run`, `resume`, `experiment run`,
  `paper add`), `cli/common.ensure_agents_ready` → `connect.ensure_agent_ready`
  pings each (default + routes) unless one passed within `FRESH_S` (stamp:
  `connected.json`); a dead subscription login (`agents.base.login_expired`)
  asks "Log in again now?" at a TTY and runs `run_relogin` in the operator
  home, else `AgentLoginError` stops the command. Engines the launcher starts
  carry `HILLCLIMB_AGENT_CHECKED=1` (`api.child_launch_context`) and skip it;
  so does the test suite (conftest). A dead login mid-search is
  `error_kind="login_expired"` and parks at once.
  `hillclimb disconnect <agent>` is the mirror on hillclimb's side only:
  `unpin_config_defaults` comments the pin out in place, `remove_staged` drops the
  cache homes, `remove_env_key` the key. hillclimb may START a coding agent's login it
  needs, it NEVER logs a coding agent out — the account is the person's
- DataStore (`harness/store.py`): the one read/write path for a search's records —
  run/search metadata, the append-only journal (`Journal(store.journal(key))`,
  append order is the replay contract), the status record, and the stop/prune
  command queue. Backends: `FileDataStore` (default; `runs/` as today) and
  `SqliteDataStore` (`store.backend: sqlite` → `store.sqlite`, WAL,
  multi-process, writes no yaml). `open_store(config)` picks it; `resolve_search`/
  `latest_search`/`running_searches` replace dir walking; `SearchRecord.state`
  is derived at read time (`status.derive_state`, pid + heartbeat). `key_for(search_dir)`
  is `(run_id, search_id)`; `SearchMeta.search_uid` is the global id. Views and
  commands never open `journal.jsonl`/`status.json` directly — only the file
  backend does. `hillclimb store sync` imports the folder into another backend
- Tunable parameters (`spaces.py` contract + `harness/params.py` engine view +
  `modules/tuners/`): a coding agent may declare numeric knobs in
  `params.json` next to `solution.py` (flat `name -> {type: float|int|
  categorical, low/high|choices, log, step, default}`; the runtime reads
  values through `spaces.params()`, which follows `$HILLCLIMB_PARAMS` to the
  trial dir's copy — the root copy carries defaults, a trial copy adds a
  `value` per param; `spaces.py` stays stdlib-only, it is byte-copied into
  the runtime shim). A valid declaration sets `Candidate.tunable`; a
  malformed one is scored on the solution's own defaults with
  `params_error` set, never a crash. WHEN to tune is the policy's decision:
  greedy proposes `Action(operator="tune", target_id=cid)` (params
  `tune_budget`/`tune_gate`/`tune_parallel`/`tune_burst`, all derived from
  the journal + in-flight refs) and the harness runs a *tune job* — no
  coding agent, no machine slot, one new Trial on the EXISTING candidate on a deep
  copy, merged in `_commit_tune` under the state lock, re-journaled (replay
  keeps the last record; `tune_started`/`tune_discarded` audit lines), holdout
  only for a trial that became the candidate's best. WHICH values come from
  the tuner seam (`climber.tuner: random | optuna`, `climber.tuner_params`,
  extra `hillclimb[optuna]`): `ask(space, history, higher_is_better, seed)`
  is a pure function of the candidate's trials + pending sets, so no study
  state survives a call and resume is free. Children of a tunable parent
  inherit its best trial's values as their defaults (`prompts/params_cue.md`
  tells the coding agent); `best/params.json` ships the selected candidate's best
  trial values. Seeds are never tuned.
- Similarity scores (`modules/similarity/`): pluggable "how alike are two
  solutions" measures. A `SimilarityScore` subclass implements
  `represent(solution)` (once per solution; None = no row) and optionally
  `compare(a, b)` (higher = more alike, 1.0 = same; the default is cosine for
  vectors/`{feature: weight}` dicts, Jaccard for sets); `represent_many` is
  the batch hook, `SimilarityUnavailable` aborts the whole score, any other
  exception leaves one solution unrepresented. Named like policies: registry
  (`solution-card` — an LLM writes a domain-free method card, cards are
  embedded, cosine; `api-calls` — imports + alias-resolved library calls +
  `method=` strings, rename-invariant; `code-tokens` — the map's structural
  Jaccard), a `.py` file (`SIMILARITY_SCORE = cls` or exactly one subclass),
  `module:Class`, or `register_score`. `cache = True` persists JSON
  representations in `~/.cache/hillclimb/similarity/` keyed by name +
  `version` + params + `cache_key` (file bytes) — never in a run.
  `solution-card` caches cards and vectors separately and calls OpenRouter
  directly (`agents/openrouter.py`, `OPENROUTER_API_KEY`); its card noise is ~0.02.
  `similarity.scores` (config) lists the defaults for `hillclimb similarity
  scores [search] [-s name] [-c ids] [-f file…] [--explain] [--json]`
- Vocabulary (0.6; clean-break renames, persisted records mapped on read):
  **Harness** = the fixed core, **Climber** = the block of exchangeable
  modules, **SelectorPolicy** (π_sel) = which candidate the next attempt starts
  from, **OperatorPolicy** (π_op) = which operator to apply to it, **Loop** =
  control flow, plus **Operator**, **Tuner**,
  **Memory**, **SimilarityScore**. A **coding agent** is the CLI an operator
  calls (Claude Code, Codex, pi): the climber is itself an agent, so prose
  (docs, help text) says "coding agent" in headings, definitions and a
  section's first mention, while identifiers keep the short form (`agent:`,
  `--agent`, `AgentRequest`, `hillclimb.agents`). An operator has a `kind`
  (`create | repair | refine | combine`); `role` is only what a climber
  plays in a search (`solver | improver`). `SearchState` is what a policy reads,
  `JournalView` its journal, `Attempt` what an operator returns,
  `AgentRequest`/`AgentResult` a coding agent call, `climber_meta` a policy's note
  on a candidate (journal key `policy_meta` mapped by
  `Candidate._legacy_policy_meta_key`). `hillclimb.sdk` is the contracts
  import a climber's file needs (a lazy facade; it raises an ImportError
  naming the new spelling for each renamed name); `tests/test_sdk_imports.py`
  AST-scans the bundled modules (`CLIMBER_MODULES`), `tests/test_vocabulary.py`
  bans the pre-0.6 names. `tests/test_prompt_golden.py` pins every prompt
  byte (greedy scenarios, openevolve, GEPA proposer;
  `HILLCLIMB_UPDATE_GOLDENS=1` regenerates — review the diff)
- The climber is a BLOCK in the run config (`modules/spec.py` `ClimberSpec`,
  `hillclimb/climber.py`; `docs/climbers.md`). There is no `climber.yaml`
  to point at and no reference/overlay split: `climber:` DEFINES it —
  `operator_policy` xor `loop` (neither = refused), `params`, `selector_policy` +
  `selector_params` (0.7 names; `policy`, `select`, `select_params` still
  load), `operators` (names/refs, each optionally `- draft:
  {retrieval: true}`) + `operator_params` (by operator name), `tuner` +
  `tuner_params`, `memory` + `memory_params`, `prompts`, `name` (a label,
  not identity); `routing`, `description`, `similarity`, `holdout_timing`
  are refused with advice (`routing` is the user's). The SAME block is a
  run-spec entry's `climber:`, the spec's top-level `climber:` (default for
  its entries) and `runs/config.yaml`'s (`Config.climber`, the folder default).
  Precedence: layers REPLACE the block whole (runs/config.yaml < spec
  default < entry < `--climber NAME`; hillclimb.yaml and the user config
  refuse a `climber:`, `config._check_general_level`); `--set climber.<field>` and
  experiment overrides then EDIT the chosen block (`climber=<name|block>`
  replaces it; `climber.operator_policy`/`climber.loop` drop each other;
  `climber.operators.<name>.<k>` addresses `operator_params`). There are NO
  presets and NO default: a block names its `operator_policy:` or `loop:`
  (`spec._one_brain` refuses a block of knobs alone; `Config.climber` is None until
  the folder names one, `climber_block()` / `as_spec(None)` raise `NoClimber` with
  the fetch hint, `cli/run.py` refuses before detaching, `api._new_search` before
  writing a run folder). A bare string is one `.py` file (its one policy or Loop,
  plus the Operator subclasses in it, and its own `SelectorPolicy` subclass —
  `Climber._own_selector`, which a subclass of a catalog class inherits from the
  file its base is written in; a file that BUILDS a `Climber(...)` is that whole
  climber instead, `spec.composed_block`: its classes written `file.py:Class`,
  imported only when the source calls `Climber(`) or a folder holding `policy.py`
  (`spec.folder_block`). Defaults live on the CLASSES (`OperatorPolicy.DEFAULTS`,
  a loop's `operators` / `holdout_timing`). 0.5 shapes still load: `climber:
  {ref: X, ...}`, `operators` as a mapping, `graph:`, `--set climber.ref=X`, and
  `learning.<behaviour flag>` (→ `memory_params`); the pre-0.9 registry names
  (`greedy`, `best`, `openevolve`, `map-elites`, `gepa`) and the 0.6–0.8 module
  paths resolve ONLY for a record (`refs.resolve_ref(legacy=True)` via
  `catalog.RECORDED` / `RECORDED_MODULES` / `RECORDED_PRESETS`; set by
  `load_snapshot`, `_load_climber_dir`, the no-snapshot resume and
  `SearchMeta._from_older_schemas`; the block keeps its recorded spelling so
  `climber_sha256` stands) — a new config naming them is told
  `hillclimb climber get <name>`
- The catalog (`catalog.py`, stdlib-only at the top; `hatch_build.py`): the engine
  ships NO problem and NO climber — the repo-root `problems/` and `climbers/` ARE
  the catalog, bundled into the wheel under `hillclimb/_catalog/` by the hatch hook
  (per file: `catalog.PROBLEM_IDS` for problems, every `climbers/*/policy.py` folder;
  nothing for an editable install) and found by `catalog.root()` (the bundle, else
  this checkout). `install_problem` / `install_climber` copy an entry (never
  overwriting; a climber folder without `prompts/` gets one generated,
  `climber.write_prompts_folder`), `catalog.climber(name)` / `module(name)` are the
  Python SDK's route (`hc.catalog.module("greedy").Greedy`). `scaffold/` holds
  `problem new`'s template and the Windows verifier. `tests/test_catalog.py` holds
  the wheel's file set, the import rule for every catalog `.py`, and the two
  shipped `Greedy` copies / the `Best`-`MapElites` schedules to each other
- Climber folders (`hillclimb climber get <name>`, `cli/climber.py` +
  `catalog.install_climber`): `climbers/<name>/` = the catalog folder as it is —
  `policy.py` (selector policy + operator policy, every default a `DEFAULTS` entry,
  and the committed `Climber(selector_policy=Best(), operator_policy=Greedy(),
  operators=[Draft(), …], tuner=RandomSearch(), memory=FilesMemory(), name=…)` that
  makes the file the whole climber — no `prompts=`: `prompts/` beside ANY climber
  file is its prompts dir unless the block or `Climber(prompts_dir=…)` says
  otherwise (`spec._with_default_prompts`, applied where a file ref becomes a
  block); NO climber.yaml — the folder is a way of naming that file:
  `spec.folder_block` falls back to `<dir>/policy.py`, `load_climber` too, and `get`
  pins `climber: climbers/<name>/policy.py`; `--name N` rewrites the file's
  `name=`; gepa is a folder of several files with its own `prompts/gepa_reflect.md`;
  a pre-0.9 folder's `climber.yaml` still loads) and `prompts/` (the templates its operators render — `Operator.templates`
  declares them — plus `README.md` from `climber.prompts_guide`, whose token table
  is `operators.builtin.TOKEN_GUIDE`; `tests/test_climber_get.py` holds both to the
  templates). A folder holding `climber.yaml` is a first-class ref
  (`spec.folder_block`: its refs come back prefixed with the folder, `load_climber`
  reads it with the folder as base; a `ClimberSpec` validated from a string looks a
  relative folder up under `context={"base_dir": …}`, the hillclimb dir for
  `runs/config.yaml`, `--set climber=` and `--climber`). `get` pins `climber:
  climbers/<name>/policy.py` in runs/config.yaml (`cli/climber.pin_climber`: a scalar line is
  replaced, the commented `init` line uncommented, an active block left alone); `get
  gepa` copies the loop folder like any; the copy is its own identity
- Module refs (`modules/refs.py`): every slot is named the same three ways —
  a registry name, `file.py[:Class]` (relative to the file the block is
  written in; anchored absolute by `ClimberSpec.anchored` at each boundary),
  or `package.module:Class`. `resolve_ref(ref, kind)`; `KINDS` = policy,
  select, loop, operator, tuner, memory, graph, similarity (each with its
  registry, module attribute — `POLICY`, `SELECTOR`, … — and base class or
  duck-typed methods; a kind's `home` package registers its built-ins on
  import). `FileScope` imports a climber's local files as ONE synthetic
  package rooted at their common dir and named by the digest of their bytes
  (relative imports followed by `source_closure`, so a file only reached by
  import is part of identity). `ClimberLoadError` lives here. `construct`
  passes only the kwargs a constructor takes; `**knobs` is the by-keyword
  form for people (`Greedy(num_drafts=3)`) and never takes what the loader
  offers
- `Climber` (`hillclimb/climber.py`): `resolve_climber(block, base_dir)` /
  `Climber.from_spec` — or composed in Python, `Climber(selector_policy=…, operator_policy=…,
  operators=[…], tuner=…, memory=…)` (keywords in the order a step runs
  them; the positional first argument is a policy, never a name) from
  names, classes or instances (an
  instance = its class + `.params`; written the most portable way:
  registry name > `module:Class` > `its_file.py:Class` > `live:Name`, the
  last making it not `portable`: `to_spec()`/resume/fleets raise
  `NotPortableError`, `api.run` still runs it via `Config._live_climber`).
  `.brain` (lazy), `.selector()`, `.operator_set()` (per-search
  `OperatorSet`; an operator the climber did not list is refused),
  `.tuner()`, `.memory()`, `.graph_module()`, `.build_loop(priors=…)`
  (a param the policy's `DEFAULTS` lacks is a `ClimberLoadError`;
  `priors` — what memory learned, e.g. `complexity_start` — sit UNDER the
  block's params and are recorded in `SearchMeta.memory_priors` at first
  run), `.holdout_timing` (the loop class's). `sha256` (`identity`) = the
  block without `name` + the FileScope digest + `tree_sha256(prompts)`,
  independent of where files are. `load_climber(str)` = one file | a folder holding `policy.py` |
  a pre-0.6 directory (read as the block it is — `hillclimb climber show
  <dir>` is the migration). `harness.glue.search_climber/build_loop/
  build_operators/build_tuner/build_memory/holdout_timing` are the glue
  (snapshots memoized per file+mtime)
- Run folders record the climber (`harness/run.py`, `SCHEMA_VERSION = 4`):
  `SearchMeta.climber` (its label), `climber_sha256`, `climber_spec` (the
  block as launched), `climber_ref` (how a pre-0.6 search named it),
  `memory_priors`, `climber_portable`, `hillclimb_version`. `create_search`
  resolves every module and builds the loop BEFORE allocating a dir (a bad
  climber costs nothing), `climber.snapshot_climber` writes
  `<search_dir>/climber/` = `climber.yaml` (`snapshot: 2`, the block) +
  `files/` + `prompts/` and checks the snapshot reproduces the identity; the
  engine — and `resume`, which restores the WHOLE block — load THAT
  (`glue.search_climber(config, search_dir)`), so editing live files never
  changes a started search. `load_snapshot` still reads what 0.4/0.5 left
  (a manifest, one file; frozen fixtures in
  `tests/fixtures/legacy_snapshots/`). v2/v3 records stay readable in every
  store backend: ONE before-validator (`SearchMeta._from_older_schemas`)
  folds `policy*` and `climber_manifest`/`climber_params`/`tuner*` into the
  block; `_load_meta` accepts `READABLE_SCHEMA_VERSIONS = (2, 3, 4)`. A child
  engine gets a block as `--set climber=<json>` ahead of the other pairs
  (`api.climber_argv`), a name as `--climber`
- Harness + loop (`harness/core.py`, `harness/loop.py`): `Harness` is the fixed core
  (candidate dirs, coding agent calls, trials, the journal's single writer, `best/`,
  accept band, budgets, control queue, crash recovery, holdout) and knows no
  policy. A `Loop.run(harness)` reaches it only through
  `view()`/`capacity`/`inflight`/`open`/`closed_reason` (pure reads) and
  `submit(action) -> Ticket` / `wait(timeout) -> [Outcome]` / `run(action) ->
  Outcome` / `source(cid)`; `Harness.execute(loop)` writes baseline + seed,
  runs the loop, commits whatever it left in flight and raises
  ParkedSearch/StopRequested for `api.execute_search` to map. Results are
  COMMITTED INSIDE `wait()`/`run()` on the loop's thread (same atomicity as
  the old scheduler); the cycle is tick → fill → consume exactly as before
  (`wait` ticks AFTER its commit, `execute` ticks once up front — the golden
  event sequences pin this). Stop/park NEVER raise into loop code: `_tick`
  latches them (`_close`), `open`/`capacity` go False/0, in-flight work still
  commits, and later `submit`/`run` raise `HarnessClosed` — so a loop (or a
  library like gepa) that swallows exceptions cannot spend more. The one
  exception is the CLOCK, the only state that moves off the loop's thread:
  out-of-budget at `submit`/`run` is a quiet `rejected` Ticket, not an error
  and not a strike. An invalid action is refused before anything exists
  (`Ticket.rejected` + an `action_rejected` audit line); 3 refusals in a row
  with nothing in flight raise `ClimberError`. `wait()` with nothing in
  flight is a 1 s tick, so a holding policy does not end a search, and the
  hard deadline is re-checked on every poll. `PolicyLoop` is the built-in
  loop (fill free slots with `policy.propose`, `observe` every outcome,
  `catch_up` replays the journal once per candidate — the resume contract).
  EVERY climber runs as `Harness.execute(loop)` (`api.execute_search`):
  `api.Search` holds what the engine sets up around a harness (`open()`:
  memory, store, journal, status + heartbeat, agents, preflights, evaluator,
  loop, harness; ONE SIGTERM handler, the previous restored) and settles a
  search exactly once (`_end` / `_settle`: holdout, final status, official
  verify, `memory.record`; a failing setup ends `failed`, nothing left
  running). `execute_search` = `Search(...).open() or search.finish()`.
  Stepping (`api.start` → `Search`; `Climber.start/propose/run/step/finish/
  close` forward to it, `Climber._search` never touches spec or identity):
  `begin()` = `Harness.start()` (baseline + seed, idempotent, what `execute`
  runs first) then `budget.pause()` — the clock runs only inside a step;
  `propose()` ticks, `catch_up`s and asks the policy; `run(action)` =
  `clear_strikes()` + `harness.run` + `loop.observe` (a closed harness or an
  unknown id is a `rejected` Outcome, never a raise); `close()` reads
  `raise_latched()`: park → `parked`, stop → `stopped`, a spent budget →
  `done` with the full tail, closed by hand with budget left → `stopped`
  (resumable). Serial, policy climbers only (a loop climber can only
  `finish()`), one open stepped search per process (`api._STEPPING`: status
  carries the pid). `tests/test_stepping.py` steps the golden scenarios to
  the recorded sequences. `Harness.run` abandons its own job on an
  interrupt.
  `harness.glue.build_loop(config)` returns `PolicyLoop(policy)` or, for
  `gepa`, its own `GepaLoop` — there is no engine tier, no `_ENGINES`, no
  `SearchStrategy`. `holdout_timing(config)` is the user's `holdout.timing`
  (`inline | after`), tightened to `after` for gepa. Harness-native,
  coding-agent-free actions: `tune` and `inject` (`Action(INJECT_ACTION,
  args={"source": text}, target_id=parent)` scores a text the loop already
  has; `--seed-from` runs through the same path as operator `seed`; the text
  is never journaled, its `Candidate.solution_sha256` — `candidate.source_hash`,
  newline-normalized — is, on every scored candidate). `Attempt.texts`
  writes extra files from text; `Attempt.require_change` turns a coding agent
  that hands the parent back into Outcome `unchanged` (abandoned, never
  scored, no evaluation spent). `Outcome.result` is an `EvalResult` projected
  from the holdout-blind copy. `Harness.request_stop(reason)` closes it from a
  signal handler BEFORE the handler raises, so a swallowed StopRequested
  cannot keep spending. Tests that poke at harness and policy together use
  `tests/harness_factory.SearchRig` (a `Harness` subclass with a policy
  attached: serial `run_operator`, `decide`, the greedy `_ensemble_*`
  helpers) or, for harness-only tests, `make_harness` + `harness.run(Action(...))`
- Budget dimensions: the budget is the USER's in every dimension — the clock
  (`budget.total_s`), `budget.max_evaluations`, `budget.max_tokens`,
  `budget.max_cost_usd` (0 = no limit). An *evaluation* is a verifier trial
  the climber caused: one per scored attempt plus one per tune trial; the
  baseline's and the seed's first trial are the harness's floor and free.
  `budget.journal_spend(journal) -> Spend` derives evaluations/tokens/cost
  from the journal on every read, so resume needs no counter. Evaluations
  and tokens END the search like the clock (no new work, in-flight lands,
  state `done`); cost still PARKS (hosted-credit semantics). Work in flight
  has its evaluation reserved, so the cap is never overshot. A climber only
  ever reads what is left: `BudgetView.evaluations_remaining /
  tokens_remaining / cost_remaining_usd` (None = no limit); `BudgetStatus`
  carries spent/limit for `hillclimb status`. There is NO hidden candidate
  cap any more (the old `max_candidates=50` default was unreachable from
  config); `Harness(max_candidates=…)` survives only as a test knob
- Operators (`modules/operators/`): HOW one attempt is made. An `Operator` subclass
  sets `name` + `kind` (`create | repair | refine | combine`) and implements
  `prepare(ctx) -> Attempt(prompt, copy_parent, inherit_params,
  copy_inspirations, fork_session, files)`; it never touches disk, journal or
  coding agent. The harness (`search._prepare` → `_prepare_attempt`, pure, so a
  refusal leaves no dir/journal/spend) checks `valid_target`, executes the
  preparation, and fills `{{contract}}` itself — appending the contract when
  a prompt has no token, so an operator cannot drop it (`search._contract` is
  harness-owned). `OperatorContext` is holdout-blind (masked target,
  inspirations, journal); harness-side answers (`render`, `live_experience`,
  `failure_reason`, `report_section`) take candidate ids. `Candidate.kind` is
  journaled (backfilled from the operator on load, None for an unknown
  operator; `role` up to 0.6 — the journal key and an `Operator` subclass's
  attribute still load, `Candidate._legacy_role_key` /
  `Operator.__init_subclass__`) and `journal.drafts()`/`debug_chain()`,
  `debug_depth` and the dummy coding agent key on KIND, never on the operator's
  name; the registry
  (`operators.get_operator/operator_names/operator_kind/register_operator`) is the
  vocabulary `climber check` and the pi route preflight derive from.
  `baseline`/`seed`/`tune` stay harness-native (no prompt). Inspiration files
  are named by `sdk.inspiration_filename(i)`, never a literal
- Policies and selectors (`modules/policies/`, `modules/selectors/`): ONE
  STEP IS TWO DECISIONS IN A FIXED ORDER (the RSI framework's π_sel then
  π_op; `PolicyLoop.propose(view)` = `selector.schedule(view, busy)` then
  `policy.propose(view, selection)`; `api.Search.select()/propose()` the
  same for stepping). The BASES DECIDE NOTHING (`SelectorPolicy.schedule/select`
  and `OperatorPolicy.propose` raise, naming the reference): every decision is
  written in the shipped climber file. `Best` (`climbers/greedy/policy.py`) OWNS THE
  SCHEDULE: `schedule` = failing tip (`debuggable_tip`) → combine window
  (`should_combine`, `combine_candidates` = `sdk.top_distinct(..., skip_kind=
  "combine")`, `Selection(combine=True)`) → None while `prospective_branches <
  num_drafts` (a root step) → `select(state, busy)`; `MapElites` carries the same
  schedule (its `ensemble` default False) over OpenEvolve's database; a selector of
  one's own subclasses `Best` to replace `select` (`pick`, its name for one day, is
  aliased by `__init_subclass__`, which also blanks an undeclared `name`) or writes
  its own `schedule`. The schedule knobs (`num_drafts`, `debug`, `max_debug_depth`,
  `ensemble*`) are `Best.DEFAULTS`, set as `selector_params` (`DEFAULTS` merged
  over the MRO, params held live like a policy's). A one-file climber's own
  `SelectorPolicy` subclass (or `SELECTOR = …`) is its selector when the block names
  none (`Climber._own_selector` / `_selector_ref`); `build_loop` routes a knob the
  selector class declares from `params` to it, so `--set climber.params.<knob>` (a
  meta tune trial) reaches either half. `spec.SCHEDULE_KNOBS` +
  `schedule_to_selector` move them out of `params` wherever a block is read
  (`ClimberSpec` before-validator, `block()`/`canonical()` for identity,
  `Climber.build_loop` for overlays, `config.current_setting` for `--set
  climber.params.<knob>`, v2 records). `OperatorPolicy` is π_op: `propose(state,
  selection) -> Action | None` (None = hold; `selection` None = root step),
  the base is the plain mapping (draft / debug / ensemble with a drain hold /
  improve via `draft_action` + `expand_action(state, selection, operator=)`),
  `DEFAULTS` merged over the MRO, `param()`, `resolved_params()`,
  `self.selector`, `draft_complexity`; a schedule knob passed to a policy is a
  TypeError naming the selector. `Greedy` adds TUNE of the CHOSEN candidate
  (`tune_now(state, candidate)`: budget, gate vs best, headroom, parallel,
  burst) ahead of improve; `greedy.py` must stay byte-identical to
  `problems/meta-heilbronn/greedy.py`. A `SelectorPolicy` (`sync(state)`,
  `select(state, busy=) -> Selection(target_id, inspiration_ids,
  prompt_context, meta, combine)`, `creation_meta`) picks the parent: `best`
  (greedy's ranking + busy-target rule) and `map-elites` (OpenEvolve's
  database; its state is a function of the JOURNAL — scored candidates in
  journal order, rebuilt when a result lands out of order or a tune trial
  moves a binned score — and it seeds/restores the global RNG around every
  OpenEvolve call). `policies/compat.py` holds `OpenEvolvePolicy` only so
  pre-0.6 snapshots resume. A policy is holdout-blind: `SearchState` wraps
  its journal in `journal.JournalView` (`Candidate.holdout_blind()` copies,
  writes raise), so `holdout.selection` decides what ships and never what a
  policy expands; it carries NO config (`journal`, `inflight`, `budget`,
  `higher_is_better`, `accept_band`). The exploration process is ONE dict:
  the block's `params`. `hillclimb climber check [SPEC]`
  (`modules/policies/check.py`, pure: no coding agent, verifier or writes) resolves
  every module, then replays the store's journals plus an empty one and
  reports contract breaches (stall on empty journal, `replay` /
  `idempotent` / `resume` divergence — the last compares a policy that
  watched the journal grow with one shown the finished journal — dangling
  or wrong-status targets, an operator the climber lacks, journal/file
  writes, shared-instance factory, dirty prompts); `--smoke` adds a
  dummy-coding-agent search; a loop climber is out of scope (exit 2)
- Memory (`modules/memory/`): a module the block names (`memory: files |
  none | file | module:Class`, `memory_params`). `Memory` (base.py) = five
  no-op-by-default steps — `bind(env)`, `retrieve() -> Retrieved(text,
  reference, reference_note, priors)`, `live()`, `publish(journal, …)`,
  `record(journal, …)` — plus `agent_passes()` (routes to preflight) and
  `graph_module()`. `FilesMemory` (files.py) is the knowledge/ directory
  (its `DEFAULTS`: `max_cards`, `live`, `complexity_prior`, `claims`,
  `graph_retrieval`, `credit`, `playbooks`, `skills`, `graph`); `NoMemory`
  keeps nothing (a search still gets its own card). The harness takes
  `memory=` + `retrieved=`. The USER keeps `learning.enabled` (the switch
  over any climber; `--no-learning`), `learning.dir`, `learning.tool`,
  `learning.claims_timeout_s`
- Prompts: `prompts:` in the block names a dir that shadows built-in
  OPERATOR templates by name (`render(..., _override=dir)`, bound per
  search); `Climber.lint_prompts()` refuses harness-owned templates
  (`contract_*`, the clauses, the knowledge passes) and unknown tokens, and
  `create_search` refuses to start on a finding
- GEPA (`climbers/gepa/`, extra `hillclimb[gepa]`): a climber that
  brings its own `Loop`. gepa drives proposal order, Pareto selection
  and its checkpoint; everything that costs or counts is `harness.run(...)`:
  a reflective mutation is one `gepa-reflect` attempt (`operator.py`, a real
  operator with template `prompts/gepa_reflect.md`, `require_change`, the
  feedback written beside the parent's solution as `feedback.json`; routed
  as `routing.gepa-reflect`), any other text gepa evaluates (a merge) is an
  `inject`, and results are cached by `source_hash` so gepa's later
  `evaluate(text)` of its own proposal is a lookup. The seed is whatever the
  harness already scored (`--seed-from` → operator `seed`, else the problem's
  baseline candidate — never evaluated twice). `GepaScoring` (`evaluator.py`)
  holds the cache, the maximizing fitness, the instance-key rule (checked on
  gepa's EVALUATE path, where exceptions propagate out of `optimize()` — the
  proposer's are swallowed) and the allow-list ASI. Failed rounds are
  ordinary abandoned candidates with their cost journaled; three in a row
  (`agent_failed | no_solution | unchanged | crashed`) end the search as
  `ParkedSearch`. Budgets in every dimension, the cost ceiling and stop/park
  bind gepa through the harness (`should_stop = not harness.open`). State:
  `SEARCH_DIR/loop/state/` + `loop/identity.json` (`harness.state_dir`);
  resume rebuilds the cache from `view().journal` + `harness.source()` and
  refuses a candidate dir whose text no longer matches its
  `solution_sha256`. `driver.py` is the only module importing gepa
  (`skip_perfect_score=False` is mandatory there — the upstream default
  silently disables mutation for unbounded scores); the default suite drives
  `GepaLoop` through `tests/gepa_fakes.py` (`make_gepa`), and
  `tests/test_gepa_compat.py` runs the real library. Shared trial execution
  lives in `harness/evaluation.py` (`CandidateEvaluator` is journal-free by
  construction; `EvalResult` is the projection loops consume). A verifier may
  write a reserved `instances` key next to `score` (per-instance breakdown,
  stable keys → `Replicate.instance_scores`, median-aggregated) — GEPA's
  Pareto frontier and future QD loops consume it; circle-packing is the
  reference producer, and the emflow eval runner emits one instance per
  scored origin (`<asof>/<zone>`, GEFCom2014's task x zone); a candidate may
  miss keys (failed instances), never introduce new ones
- Meta-problem kit (`meta.py`, `cli/meta.py`; built, UNDOCUMENTED until its
  own launch — nothing in README/docs names it): a problem with
  `solution_kind: climber` in `problem.yaml` is a META-problem whose
  `solution.py` is a ONE-FILE climber (policy + optional `Operator`
  subclasses with inline prompts + param defaults; `load_climber` already
  takes one `.py`), so the candidate contract is unchanged (`$HILLCLIMB_SOLUTION`,
  `solution_sha256`, inject, `best/`). The kind selects `contract_climber`
  (+ `params_cue_climber`, both harness-owned) and `create_search` derives
  `SearchMeta.role` (`solver` | `improver`, default solver so every old
  record loads; `watch` labels only improvers). The role is the PROBLEM's
  doing, never a key of the block — the same climber runs at either level. Its
  verifier is `hillclimb grade` (hidden; `meta evaluate` is its old spelling) on
  `$HILLCLIMB_ENGINE_PYTHON` (new verifier env key: the engine's interpreter;
  `api.build_executor` also exports `$HILLCLIMB_DIR`): reads `grade.yaml`
  (`meta.GradeSpec`: inner `problems`, `budget` per inner search, `repeats`,
  optional `floor`/`target`, `score`, `aggregate`, `outer_holdout`), writes a nested hillclimb dir under the replicate dir
  (`meta.nested_config`: the user's coding agent/model/routing/problems, own runs,
  files store, learning off), measures each inner floor ONCE
  (`score_floor`, like `verify` — a search's files-only c000 is unscored),
  runs one inner `hillclimb run --climber <solution.py>` per problem × repeat
  serially (scrubbing the outer verifier's `HILLCLIMB_*` keys; a trial's
  `$HILLCLIMB_PARAMS` values ride as `--set climber.params.k=v`, so the
  tuner seam tunes policy knobs), and scores each inner search with the
  spec's `score:` — default **gap closed** = per problem
  `(best − floor)/(target − floor)` direction-aware, clamped at 0, target =
  best `chart_baselines` value; also `solved`, `raw`, or the user's own
  `file.py:function` (beside the spec) / `module:function`, called as
  `f(best, floor, target, higher_is_better)` — then the median over repeats
  and the spec's `aggregate:` over problems (`mean` default, `median`,
  `min`, or a function of the list); each instance on the `instances` key,
  spend as `inner_*` metrics. Both belong to the grade, never to the
  climber. The OUTER HOLDOUT: on the holdout split (`--split`, default
  `$HILLCLIMB_SPLIT`; the engine sets it when the meta-problem has `holdout:
  true`) the grade is measured on the spec's `outer_holdout` problems — the
  ones the search on the meta-problem never climbs on — and a spec
  without them is refused. Split vocabulary: validation = what a search
  climbs, holdout = hidden from that search, outer holdout = hidden from
  the search on the meta-problem too. An inner
  run that exits non-zero fails the verifier (→ buggy → debug target); a
  spec the outer `budget.exec_timeout_s` cannot fit is refused before
  spending. `hillclimb meta check` = import allow-list (`hillclimb.sdk`,
  `hillclimb.spaces`, the six facades, stdlib; `check_climber_source`, the v1 permissions
  rule) + `climber check`. Reference meta-problem `problems/meta-heilbronn/`
  (repo only, NOT in the bundled catalog; baseline `greedy.py` byte-identical
  to `climbers/greedy/policy.py` — both policies, so a candidate can change where
  attempts start — `tests/test_meta.py` + `tests/test_catalog.py` enforce).
  Deferred: directory-shaped candidates, parallel inner runs, replay/ReplayHarness
- Studies (`experiment.py`, run by `hillclimb experiment run|report`): a spec
  (`experiments/<name>.yaml`) is problems × named experiments (dotted config
  overrides, `Config.apply_overrides`; `arms:` is the legacy key) × repeats;
  searches are tagged in `SearchMeta` (`study`, `experiment`, `repeat`,
  `experiment_overrides`; `SearchMeta._legacy_arm_tags` maps records written
  with the old `experiment`/`arm`/`arm_overrides` names) and the report groups
  on those tags through the store — the first experiment is the control, gaps
  are paired by repeat and judged against the spec's `noise_floor`. Sequential
  schedule (repeat-major, experiments round-robin) is mandatory when an
  experiment touches shared state (memory);
  otherwise `schedule: parallel` + `max_concurrent: N` (or
  `--max-concurrent N`) — unbounded parallel starts every search at once and
  the ones past the machine's coding agent slots burn their budget in
  `waiting-slot`. `--run-id R --first-repeat K` appends repeats to a finished
  run. `SearchMeta.seed_sha256` records the seed each search started from.
  `experiment report --json` (`experiment.summaries_to_dict`) emits the experiments,
  paired gaps and a `verdict` per comparison (`better`/`worse`/`tie`/
  `within-noise`/`unknown`) so a meta-verifier reads a score, not a table.
  `problems/make_heilbronn.py` stamps the heilbronn difficulty ladder
  (11/14/17; committed dirs must match the generator — `tests/test_heilbronn_ladder.py`)
- CLI voice (`cli/common.py`): every sentence a command speaks goes through `say`/
  `warn`/`fail` with the theme's markup (`[head]` lead, `[path]` paths/refs/ids,
  `[cmd]` commands, `[note]` dim explanations, `[ok]/[warn]/[bad]` verdicts; every
  dynamic value through `_m()`); a foreground engine's log lines go through
  `engine_log` (`engine_line` marks the clock, `word:` leads, candidate ids and
  trouble by shape). Data a script pipes — `--json`, diffs, file bodies, stdout
  tails — stays on `typer.echo`, byte-exact. `say_no_hillclimb_dir` is the one
  rendering of the missing-dir hint
- `hillclimb run <problem>` DETACHES by default (one-search fleet through
  `_run_problem_fleet`/`api.run_fleet`, summary + `hillclimb watch` hint); a suite's or
  study's child (`--run-id`), a study's experiment, and `--no-detach` run in-process;
  the in-terminal log is a clock gutter (`common.engine_log`/`split_engine_line`, clock
  from `BudgetManager.clock_str`)
- CLI: `uv run hillclimb --help` (engine); live TUIs (each App's `TITLE` is `hillclimb <command>`): `top` (`tui/top.py`: the control pane — ONE table, each engine heading its process tree via `psview.engine_lines`; `s`/`g` stop the row's engine through the queue, `k` kills the row's process tree (`orphans.kill_process_tree`) or, on an engine row, the engine; no jumps into watch; the data is `tui/machine.py` (`scan`: engines plus `job_kind` foreground `verify`/`grade`/`run` processes, anchored on the program, their dir from `HILLCLIMB_DIR` else their cwd, by the same no-upward-search rule), Textual-free and shared with `ps`, whose boxed nvidia-smi frame is `tui/psview.py` — compute only (`scan(read_searches=False)`, one box, each engine heading its tree), rich only, measured to fit width and, under `ps --watch`, height; roles from `harness/orphans.classify`; each engine's search read through ITS hillclimb dir via `Config.load(start=…)`), `watch` (coding agents; `watch candidates` jumps to a search),
  every screen's way back is `keys.back_binding` (esc or b, footer `esc/b back`), so
  `b` is never anything else — the tree views select the best with `*`;
  `chart` (best score vs time; ONE chart per plain problem — `chart_index` groups plain runs by problem, a study per run — opening on the plain climb; with several runs `ChartScreen.view` cycles (`v`) climb → runs → compare; in `runs` the runs colour the one climb's dots (`ClimbEvent.run`, `build_climb_plot(run_colors=)` from `RUN_PALETTE`, legend per run; labels `run_labels`: `#n HH:MM` or the run's own name; `TIE_RTOL` keeps a float-precision tie from counting as a new best); `]`/`[` cycle all → run k → all (`ChartScreen.run_focus`: the others fade to grey), `compare` = `run_curves`, one line per run, `end_dots`; `d` follows the focused run; a bare `chart` on a folder with
  several charts opens `ChartPickerScreen` first — enter opens,
  esc pops back; `watch` pushes the same `ChartScreen` with `c` via
  `watch.push_chart`; `chart_index` is the pure row builder; a study
  anchor confines the chart to its own run (`chart_run_scope`) so two runs
  of one study never overlay each other's `experiment rN`, while a plain
  problem still folds every run into one climb;
  `--detail`/`d` overlays one search's
  exploration tree on the curve), `tree` (one search's tree, drawn like the
  Darwin Gödel Machine's archive tree: candidate number inside each circle,
  fill = score on a cyan ramp (hollow = never scored), ring = fate ladder in no
  ramp hue — white = expanded, none = scored, red = failed — star = best, the
  best's parent chain in cyan. `tui/tree.py` is the pure layout + fates,
  `tui/treeview.py` the plotui base screen with a face-on locked camera
  (`TreePlotWidget`, `TreeScreen`, also used by `watch`'s tree panel),
  `tui/treedraw.py` the pure encoding + one Graph3d trace using plotui's
  `set_graph_borders`/`set_graph_labels`/`"star"` (labels are drawn by plotui
  inside the mark only where they fit); `fit_radius` sizes marks from the
  closest projected pair so circles never overlap, and `tui/treedrawview.py`
  rebuilds on every zoom/reset/resize to apply it, subclassing the base
  widget/screen through `TreePlotWidget`'s
  `_build_plot`/`_label_nodes`/`_legend_spans`/`_legend_entry_at`/
  `_flat_to_id`/`_place_labels` hooks; the ring ladder is plotui's own
  legend box with host rows (`treedraw.legend_entries` → `Plot.set_legend_entries`,
  top-left, node-style swatches so expanded/scored share a fill and differ
  only by the white ring; clicks resolve via `legend_entry_hit`, keys 1-5),
  the score ramp stays a text overlay),
  `treeclimb` (the DGM two-panel figure: the `tree` tree on the left, the climb
  chart on the right — `tui/treeclimb.py` pure: scored nodes at (candidate
  number, score) where the number is the circle number (`treedraw.node_number`, else
  creation order), best-so-far walked in NUMBER order (same final
  best as `tree.accepted`, intermediate steps may differ from `chart`'s
  landing order), the best's parent chain as a thick line, a cursor at
  the scrub tick's candidate, axes pinned to the live tree; `tui/treeclimbview.py`
  subclasses `TreeDrawScreen`, keeps its ids so scrubbing/detail/n-p are
  inherited, and re-shows the chart from the same scrubbed tree in
  `_apply_view`. Two plots on one screen need two Kitty image-id pairs:
  plotui's `PlotWidget(image_slot=n)` (the chart takes slot 1;
  `PlotWidget.kitty_cleanup()` names every slot taken) and distinct
  placement ids (iTerm2 keys placements by `p=` alone, so plotui places
  each frame as `p=<image id>`); the chart's legend is a text overlay in
  the empty corner (top-left rising, bottom-left falling), hotkeys 6-9
  after the tree legend's 1-5, and plotui's `Plot.keep_out` (set by the
  widget from its overlay spans) keeps the hover readout off it), `surface` (one search's candidates on the problem's
  3D terrain — needs the problem to ship `landscape.py` (`elevation(x, y)` +
  `grid(n)`, picked up by default like `contract.md`) and journal each
  candidate's position as `surface_metrics` keys (default x/y) in
  `Replicate.metrics`; `tui/surface.py` pure layer, `tui/surfaceview.py` the free-orbit
  screen — start the camera at negative pitch, plotui's default views a
  surface from underneath; no landscape = prints why and returns;
  `problems/fitness-landscape/` is the reference problem), `plot` (NOT a TUI: a problem's optional `plot.py` is plain matplotlib, `plot(solution_dir, ax) -> caption`, run by `runtime/plot_solution.py` in the problem's runtime venv like the verifier — `cli/common.show_solution_plot` adds matplotlib there on first use, saves a PNG and opens it with the system viewer; reads output files, never reruns solution.py; `summit --plot`; heilbronn/circle-packing plots come from their generators), `similarity` — two views of one search's candidates,
  same inputs, nothing stored; a bare `similarity` is `similarity reference` (the cube with named axes, the site's figure; `DefaultCommandGroup.default_command`): `similarity map` (`v` from the cube; `tui/similarity_map.py`
  pure layer, `tui/similarity_mapview.py` screen) embeds every candidate by
  pairwise distance (behavioral / structural / blend, `m` cycles) with
  classical MDS, Procrustes-aligned to the previous layout so a live search
  grows in place, drawn as one `add_graph3d` with lineage edges + a gold
  best-so-far `add_line3d`, click dims outside a lineage, `space` replays
  growth, `v` swaps to the cube; `similarity reference` (`v` swaps back;
  `SimilarityBase`/`RunScopeMixin` in `tui/similarityview.py` are shared) shows one
  search's candidates as a 3D scatter at behavioral/structural/lineage
  distance from a reference — the seed (else baseline) by default, `c`
  toggles the current champion — coloured by score rank; distances are
  derived at render time from existing artifacts (the problem's optional
  `fingerprint.py` — `fingerprint(candidate_dir) -> vector`, picked up by
  default like `landscape.py`, for outputs with equivalences the flat file
  misses — else submission.csv or the best trial's r0 evaluator report, solution.py
  tokens, parent chains) and NEVER stored; a study's experiment opens the
  **run scope** instead (`build_run_similarity`: every search of the problem
  in the run, each measured from its own copy of the shared seed, ids
  `<search>/<cid>`, coloured by experiment with the chart's palette, `--single`
  opts out); `tui/similarity.py` pure layer with fingerprint caches,
  `tui/similarityview.py` the screens; no usable reference = prints why and
  returns), `graph` (knowledge graph)
- Catalog problems: the repo-root `problems/` is the ONE copy (`catalog.PROBLEM_IDS`
  lists what ships; `meta-heilbronn`, `bin-packing`, the `make_*.py` generators and
  a developer's own problems never leave the repo); circle-packing is the lean
  first-run edition (`requirements.txt`, a 60 s budget); `fitness-landscape` ships
  beyond the starter set: the terrain the `toy` agent walks
- Python SDK surface (`hillclimb/__init__.py`, lazy): `hc.run(...,
  max_evaluations=, learning=)` (both recorded as `set` pairs in the run's
  spec), `hc.Action`, `hc.register_agent`, `hc.open_search(ref)`.
  `results.SearchOutcome` reads through the store and never writes (its
  journal's backend refuses appends): `best`, `candidates`, `history`
  (`tui.tree.accepted_lineage`), `spend` (`journal_spend` + `Spend.seconds`
  from the status record), `solution`, `params`, `source(cid)`
  (`candidate.read_solution`, shared with `Harness.source`), `to_frame()`;
  a finished search is read once, a `running` one on every access.
  `examples/` (repo only, not in the wheel) runs on `fitness-landscape` +
  `toy`; every script keeps its entry under `if __name__ == "__main__":`
  and `tests/test_examples.py` runs each `main(evaluations=4)` in a fresh
  hillclimb dir
- Native Windows (`harness/oscompat.py`, the ONE place POSIX assumptions
  meet Windows; `tests/test_oscompat.py`): msvcrt locks, CREATE_NEW_PROCESS_GROUP
  + `taskkill /T /F` as the group kill, psutil (Windows-only dep) for liveness
  and the `ps` listings — `os.kill(pid, 0)` KILLS on Windows, never call it —
  `Scripts/python.exe` venvs, junctions when symlinks need privileges, `.cmd`
  coding agents resolved through PATHEXT, forward-slash `$HILLCLIMB_*` paths, and a
  UTF-8-mode relaunch of the CLI (`ensure_utf8_mode`). `problem get` writes the
  verifier for the fetching OS: verifier.py on Windows (the problem's own, else
  `scaffold/windows_verifier.py`), verifier.sh elsewhere; `problem.windows_edition`
  picks the `.py` sibling at load time; a `.sh`-only problem runs through Git
  Bash. `tests/test_windows_verifier.py` holds every catalog verifier.sh
  without its own verifier.py to the standard shape and both editions to one
  score. `.github/workflows/quickstart.yml` walks the quickstart on
  ubuntu/macos/windows
- Driving runs from chat: the `hillclimb` skill (`.claude/skills/hillclimb/SKILL.md`, which
  Claude Code loads as a skill; any other agent can read it as instructions)

## Run-state rules

The engine process is the **single writer** of search state — the journal,
status and `best/` of `runs/<run-id>/searches/<search-id>/`, whichever
DataStore backend holds the records. Never edit those files (or rows)
directly — control a search through `hillclimb stop|kill|prune|resume`, which
route through the store's command queue when the engine is live. The journal
is append-only; replay keeps the last record per candidate. What stays on disk
in every backend: `candidates/`, `best/`, coding agent streams/logs, `injected_claims.json`.

`knowledge/graph.json` is a **derived index** (gitignored) rebuilt
deterministically from the knowledge YAML (cards, entities.yaml,
concepts.yaml, credit/, consolidated.yaml, papers/) — never hand-edit it;
`hillclimb knowledge rebuild` regenerates it. The YAML files are the source
of truth and are git-versioned; `knowledge/credit/` holds append-only
per-search outcome events (one file per search — never merge or rewrite
them). `knowledge/playbooks/`, `knowledge/consolidated.yaml`, and
`knowledge/skills/` are consolidation/harvest outputs — regenerate via
`hillclimb knowledge consolidate` rather than hand-editing (playbook edits
are legitimate but land as reviewed git diffs). `knowledge/papers/` holds
paper-derived claims (`hillclimb paper add <pdf> [--problem <target>]`, one
sonnet coding agent pass per PDF, content-hash cached); paper claims ride the same
retrieval/credit economy as search claims, wired to `paper:` graph nodes. Schema v2: nodes carry both
`pos` (2D, consumed by hillclimb-go) and `pos3` (3D, the plotui viewer) —
keep `pos` byte-stable when touching layout code.

## plotui dependency

The knowledge-graph viewer renders through `plotui` (Rust core via maturin),
a core dependency that `uv sync` takes from PyPI, so a fresh clone syncs
without a sibling checkout. Co-developing plotui: `uv pip install -e
../plotui` with `UV_NO_SYNC=1` set (a plain `uv run` syncs the PyPI wheel
back), and reinstall after editing its Rust source (uv won't notice `.rs`
changes on its own); `CONTRIBUTING.md` has the steps. Never override PlotWidget's Textual `on_*` handlers in
subclasses — Textual dispatches them per MRO class (both run); hook the
`apply_zoom/apply_rotate/apply_pan/apply_reset/on_click_at` primitives
instead. Direct mode (iTerm2) double-buffers frames across two Kitty image
ids, so anything that hides or covers a plot must emit
`PlotWidget.kitty_cleanup()` (both ids), never `Plot.kitty_cleanup()` alone.
3D scatter views that want the gridded back walls build their plot with `theme.boxed_plot()` (plotui `set_box_grid`); the similarity views follow the hillclimb.sh figure (`similarityview.SCORE_RGB` = the site's ramp, `fate_shape`: disc built on / circle left / dot failed, `legend_markup` under the plot, `set_axis_titles3d` on the reference cube, `START_CAMERA` = the site's composition) — change the site and the terminal together; smooth dragging (`glide`), the zoom-following grab (`drag_rotate`) and the drag-time resolution cap (`interactive_max_pixels`) are `PlotWidget` settings — tune them there, never with special cases in a hillclimb view; a test that drags a 3D plot and reads the camera calls `widget.flush_glide()` first.
Chart traces are always named; `show_legend=False` flips plotui's
`legend_visible` instead so the hover readout keeps the series names.
