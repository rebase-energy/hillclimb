# hillclimb

Hillclimbing on verifier-defined problems: a greedy search engine that spawns
headless Claude Code agents as operators (draft/debug/improve/ensemble),
scores each candidate through the problem's verifier, and keeps the best
solution per search.

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
with `--holdout` in a directory agents never see. Holdout scoring is the
HOST's, never a search strategy's: `CandidateEvaluator` (`harness/evaluation.py`)
scores the hidden split as part of `run_trial` — a trial that fails it is
not-ok, like a verifier crash — and `api.build_evaluator` decides WHEN
(`holdout_timing`): `inline` per candidate as it lands (greedy; `watch`
shows holdout live; `holdout.top_k` gates the spend, floors are never gated),
`after` once `run()` has returned (`api._finish_holdout`, for engines whose
state must never see a holdout value — the `_ENGINES` entry declares it). Providers (`emflow://`,
`mlebench://`) supply their own argv for the same contract. Never read a score
off stdout — agent code shares that stream.

A problem may optionally ship `interface.py` (Gym-spaces-style Python objects
from `hillclimb.spaces`, picked up by default like `contract.md`): its
`describe()` renders into the contract prompt, `check()` gives verifiers and
agents located format violations, `sample()` writes a format-valid artifact,
and `hillclimb verify` lints the baseline's output against it. The engine
never runs the check itself — a PYTHONPATH shim
(`runtime.ensure_interface_shim`, wired always-on in `api.build_executor`)
just makes `from hillclimb import spaces` importable inside the runtime venvs
so the verifier or agent can call it voluntarily. `spaces.py` must stay
self-contained (stdlib top-level imports only; it is copied verbatim into the
shim).

**Package map** (`docs/package-layout-plan.md`; `tests/test_layout.py` enforces the
import directions): `hillclimb.harness` is the fixed core — `core.py` plus everything
that scores, records or spends (evaluation, executor, journal, candidate, store, run,
budget, slots, control) and `glue.py`, config → climber → loop; `hillclimb.modules` is
everything a climber exchanges, one subpackage per kind with its contract in `base.py`
(`policies/`, `operators/`, `tuners/`, `similarity/`, `memory/`), implementations
importing only `hillclimb.sdk`; `hillclimb.tui` is every terminal view and its layout
(never imported by the harness or the modules); `hillclimb.cli` is one module per
command group with `common.py` for what commands share (reached as `common.x()` so
one patch covers every command) and `__main__.py` for the engine children. The flat
top level is the public surface only: `api`, `config`, `problem`, `project`,
`benchmark_providers`, `climber`, `experiment`, `connect`, `spaces` (byte-copied into
runtime venvs, so it stays), `sdk/`, `demo/`, `agents/`, `integrations/`, `prompts/`,
`runtime/`, `climbers/`. `_moved.py` maps pre-move `module:Class` refs at the two
places they are imported.

- The hillclimb dir: any folder holding a `hillclimb.yaml` (the config and
  the marker), found by upward search; problems/, runs/, knowledge/,
  climbers/, experiments/ and store.sqlite sit beside it and relative config
  paths resolve against it. `init [DIR]` makes the CWD (or DIR) one and
  refuses a folder with its own problems/ or runs/; `reset` deletes only
  those hillclimb-owned entries (`common.owned_paths`), never the folder.
  This repo's root is its own hillclimb dir.
- Every run carries its spec: `api.write_run_spec` writes `runs/<run-id>/spec.yaml`
  (one `SuiteEntry` per search as resolved — `spec_entry`; entries carry `climber`
  and `set` too) from every launch path (foreground run, suite, fleet, experiment),
  so `hillclimb run <run_dir>/spec.yaml` reruns it; `init` writes no `specs/` any
  more and its `.gitignore` rules (`common.INIT_GITIGNORE`) commit a run's record
  and ignore its bulk (candidates/, logs/, control/, best/ except solution.py +
  params.json, store.sqlite, knowledge/graph.json, .env)
- Directory vocabulary: every level is `<level>_dir` — `run_dir`,
  `search_dir`, `candidate_dir` (`searches/<id>/candidates/<cid>/`, where the
  agent works), `trial_dir` (`candidates/<cid>/trials/t<i>/`, holds the
  trial's `params.json`), `replicate_dir` (`…/t<i>/replicates/r<j>/`, the
  verifier's cwd; r0 of the best trial is hoisted to the candidate root).
  The word "workspace" is banned (`tests/test_vocabulary.py`
  enforces it); old journals/status files that still carry a `workspace` key
  are mapped to `candidate_dir` on load.
  Machine-scoped state (shared venvs, emflow cache, agent slots) lives in
  `~/.cache/hillclimb/`.
- Tests: `uv run pytest`
- New problem: copy a bundled one (`hillclimb problem get <problem>`); check a
  verifier with `hillclimb verify <problem> --repeat 5` (the spread it prints
  is the noise floor — improvements below it are not real)
- Noisy metrics: a trial's score is the MEDIAN of its replicates;
  `evaluation.n_replicates` + `noise_k`/`min_improvement` set an accept band so
  the search cannot climb noise (the floor is the within-trial replicate
  spread — spread across parameter sets is signal), and `replicate_mode:
  serial` is mandatory when the metric measures the machine
  (time/throughput/memory) — parallel replicates measure each other. Seeds
  are never tuned. `n_trials`/`trial_mode` are accepted as legacy spellings
- Concurrency: `concurrency.parallel_agents` per search, `concurrency.machine_max_agents`
  across the machine (flock slots in `~/.cache/hillclimb/agent-slots/`, default
  `min(8, cores-2)`); verifier and agent envs are single-threaded
  (`executor.SINGLE_THREAD_ENV`, parent values win). CPU accounting:
  `harness/procs.py` (`Reaper`) reaps every child the harness spawns — verifier
  runs and agent calls — through `os.wait4`, sampling live descendants with `ps`
  before a group kill; `OperatorResult.cpu_s` → `AgentInfo.cpu_s` is the agent
  call's local CPU and the chart's cost fold adds it to the trials'. Starter
  verifiers call their scorer plainly, never `exec` it (macOS drops the shell's
  child CPU at an exec; `tests/test_verifier_scripts.py`). `hillclimb ps` lists the
  engine process trees; `stop --all` reaps engines whose hillclimb dir was deleted; `reset` kills only the engines pinned to this folder's hillclimb dir, then deletes the dir
- Operator agents: `claude-code`, `codex`, `pi` and `dummy`. Pi supports
  `routing.<op>.sampling` (numeric provider fields), with action → operator →
  default precedence and candidate-journal persistence. Sampling on other
  agents fails validation. `pi.models_file` adds custom/local providers;
  copied configs live under isolated `~/.cache/hillclimb/pi-home/<auth>/`
  (content-hashed subdir for custom models). Pi startup is offline, search
  startup preflights each model/sampling route, and provider errors are read
  from JSON `stopReason` even on exit 0. Debug children use `--fork` to keep
  history while binding tools to the child's cwd; `--session` restores the
  parent's cwd and must not be used across candidates.
- Model per agent: `config.model` (default `sonnet`) is Claude's vocabulary; the
  codex agent's `native_model` omits `--model` for a Claude alias/id so the Codex
  CLI's own default answers (journaled as `codex-default`; the connect ping says
  so too) — OpenRouter routes always pass the id
- Agent billing: `agent_auth` picks who pays — `subscription` (the Claude or
  ChatGPT login), `api-key`, or `openrouter`, which points codex or pi at
  OpenRouter (`wire_api: responses`; the key comes from the environment or a
  `.env` beside hillclimb.yaml) and bills OpenRouter credits instead. Every codex
  call runs under an isolated `CODEX_HOME` in
  `~/.cache/hillclimb/codex-home/<auth>/`, so personal `~/.codex` settings
  change neither a search's results nor its token bill; a provider 402 parks
  the search as `out_of_credits`, and `harness/pricing.py` fills `cost_usd` from
  OpenRouter's catalogue so `budget.max_cost_usd` applies. `hillclimb connect`
  (`connect.py`) is where a credential is checked on purpose instead of at the
  first spawn: every probe runs through the SAME env builders the agents use
  (`subscription_env`/`codex_env`/`pi_env`), so an inherited `ANTHROPIC_API_KEY`
  shadowing a subscription shows up in the table; a bare `connect` is the
  status of all four targets (`claude`/`codex`/`pi` are agents and own their
  login, `openrouter` is a billing route), `connect <target>` runs that login,
  materializes the isolated home, pings the route with one tool-free call
  (`connect.ping`) and pins `agent`/`agent_auth` with a line-level edit of
  a config.yaml that keeps its comments — the USER level
  (`~/.config/hillclimb/config.yaml`, so `connect` precedes `init`; a folder's
  hillclimb.yaml overrides it, `--local` writes there) and only when no agent is
  pinned yet, unless `--default`. Keys live in a `.env` (user-level beside the
  user config, read under the folder's own by `Config.load`), never in `Config`.
  Connection states: `logged-out` / `logged-in` / `ready` (`no-key` /
  `key-set` / `ready` for openrouter) — `ready` = the login works AND
  `mark_connected` left `connected.json` in `record_dir` (the ping's scratch dir
  under the machine cache; codex/pi also need their staged home); `Status.ok`
  is the login, `Status.connected` the `ready` state.
  `hillclimb disconnect <target>` is the mirror on hillclimb's side only:
  `unpin_config_defaults` comments the pin out in place, `remove_staged` drops the
  cache homes, `remove_env_key` the key. hillclimb may START an agent's login it
  needs, it NEVER logs an agent out — the account is the person's
- DataStore (`harness/store.py`): the one read/write path for a search's records —
  run/search metadata, the append-only journal (`Journal(store.journal(key))`,
  append order is the replay contract), the status record, and the stop/prune
  command queue. Agents: `FileDataStore` (default; `runs/` as today) and
  `SqliteDataStore` (`store.backend: sqlite` → `store.sqlite`, WAL,
  multi-process, writes no yaml). `open_store(config)` picks it; `resolve_search`/
  `latest_search`/`running_searches` replace dir walking; `SearchRecord.state`
  is derived at read time (`status.derive_state`, pid + heartbeat). `key_for(search_dir)`
  is `(run_id, search_id)`; `SearchMeta.search_uid` is the global id. Views and
  commands never open `journal.jsonl`/`status.json` directly — only the file
  agent does. `hillclimb store sync` imports the folder into another agent
- Tunable parameters (`spaces.py` contract + `harness/params.py` engine view +
  `modules/tuners/`): an agent may declare numeric knobs in
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
  agent, no machine slot, one new Trial on the EXISTING candidate on a deep
  copy, merged in `_commit_tune` under the state lock, re-journaled (replay
  keeps the last record; `tune_started`/`tune_discarded` audit lines), holdout
  only for a trial that became the candidate's best. WHICH values come from
  the tuner seam (`climber.tuner: random | optuna`, `climber.tuner_params`,
  extra `hillclimb[optuna]`): `ask(space, history, higher_is_better, seed)`
  is a pure function of the candidate's trials + pending sets, so no study
  state survives a call and resume is free. Children of a tunable parent
  inherit its best trial's values as their defaults (`prompts/params_cue.md`
  tells the agent); `best/params.json` ships the selected candidate's best
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
- Harness + climber restructure (in progress on branch `harness-climber`;
  vocabulary: **Harness** = the fixed core, **Climber** = the shareable
  bundle of exchangeable modules, **SearchPolicy** = the pure what-next
  decision, **SearchLoop** = control flow, plus **Operator**, **Memory**,
  **Tuner**, **SimilarityScore**). `hillclimb.sdk` is the one import a
  climber needs — a lazy facade, so modules it re-exports may import it back;
  `tests/test_sdk_imports.py` AST-scans the bundled climber modules
  (`CLIMBER_MODULES`) and carries a shrinking `ALLOWED` list of exceptions,
  each naming the phase that removes it. `tests/test_prompt_golden.py` pins
  every prompt byte (greedy scenarios, openevolve, GEPA proposer;
  `HILLCLIMB_UPDATE_GOLDENS=1` regenerates — review the diff)
- Climbers (`climber.py`, bundled manifests in `src/hillclimb/climbers/<name>/climber.yaml`):
  the shareable unit. `load_climber(ref, base_dir)` resolves a bundled name
  (`greedy | openevolve | gepa`), a directory holding `climber.yaml`, or ONE
  `.py` file (a one-file climber: the single SearchPolicy — duck-typed
  `propose`+`observe`, or `POLICY = …` — or SearchLoop subclass it defines,
  plus any `Operator` subclasses in it); every failure is a
  `ClimberLoadError` naming the file and the fix. `ClimberManifest`
  (`extra="forbid"`): `name`, exactly one of `policy` | `loop`, `params`,
  `operators` (built-in names or `file.py:Class` / `module:Class`, each
  optionally `- draft: {retrieval: true}`), `memory: files | none`
  (`knowledge-graph` is the pre-0.4 spelling, mapped on read, never rewritten
  on disk — the snapshot's hash is the climber's identity), `graph`
  (the GraphModule over the memory: `knowledge-graph`, a `file.py` in the
  climber dir, or `module:Class`; `modules/memory/base.py` is the contract,
  `graphs.py` the resolver, `glue.build_graph_module` the one place consumers
  ask; graph.json records its `builder` and is rebuilt on a mismatch),
  `tuner`/`tuner_params`, `similarity`, `prompts` (a dir that shadows built-in
  OPERATOR templates by name), `holdout_timing: after`; `routing` is RESERVED
  and refused (the model is the user's choice). Module refs inside a manifest
  are `file.py[:Class]` relative to the climber dir (imported as one
  digest-named package, so files may import each other and versions coexist)
  or `package.module:Class`. `Climber.build_loop(params=<user overlay>,
  complexity_start, parallelism, log)` constructs the policy/loop with
  whichever of those kwargs its signature accepts; `operator_set()` returns a
  per-search `OperatorSet` (the harness's `operators=` — an operator the
  climber did not list is refused; the global `operators._OPERATORS` is only
  the built-in catalogue); `prompts_dir` is bound to the search
  (`render(..., _override=dir)`), and `lint_prompts()` refuses harness-owned
  templates (`contract_*`, the clauses, the knowledge passes) and unknown
  tokens. `sha256` = `tree_sha256(root)` (no `__pycache__`/dotfiles) or the
  one file's hash. `climber.ref` (config; `--climber` on the CLI) IS the climber
  ref: `harness.glue.search_climber/build_loop/build_operators/
  holdout_timing` are the glue (`_user_params` lays only what the user
  actually set over the manifest's params)
- Run folders record the climber (`harness/run.py`, `SCHEMA_VERSION = 3`):
  `SearchMeta.climber` (the ref as written), `climber_sha256`,
  `climber_manifest` (as loaded), `climber_params` (the USER's overlay),
  `hillclimb_version`, `tuner`/`tuner_params` (the user's override; None =
  the manifest's). `create_search` loads the climber BEFORE allocating a dir
  (an unloadable one costs nothing) and `climber.snapshot_climber` copies its
  files into `<search_dir>/climber/`; the engine — and a resume — load THAT
  (`harness.glue.search_climber(config, search_dir)` → `load_snapshot`),
  so editing the live dir never changes a started search. `resume` notes a
  changed live hash, and refuses only when the climber is gone AND there is
  no snapshot. v2 records stay readable in every store backend:
  `SearchMeta._from_v2` (a before-validator) maps `policy*` → `climber*` and
  drops `templates_*`; `_load_meta` accepts `READABLE_SCHEMA_VERSIONS = (2, 3)`
  and hides anything else. `harness.glue.build_tuner` wires the
  manifest's tuner (user's `climber.tuner` wins) into the Harness
- Harness + loop (`harness/core.py`, `harness/loop.py`): `Harness` is the fixed core
  (candidate dirs, agent calls, trials, the journal's single writer, `best/`,
  accept band, budgets, control queue, crash recovery, holdout) and knows no
  policy. A `SearchLoop.run(harness)` reaches it only through
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
  `harness.glue.build_loop(config)` returns `PolicyLoop(policy)` or, for
  `gepa`, its own `GepaLoop` — there is no engine tier, no `_ENGINES`, no
  `SearchStrategy`. `holdout_timing(config)` is the user's `holdout.timing`
  (`inline | after`), tightened to `after` for gepa. Harness-native,
  agent-free actions: `tune` and `inject` (`Action(INJECT_ACTION,
  args={"source": text}, target_id=parent)` scores a text the loop already
  has; `--seed-from` runs through the same path as operator `seed`; the text
  is never journaled, its `Candidate.solution_sha256` — `candidate.source_hash`,
  newline-normalized — is, on every scored candidate). `Preparation.texts`
  writes extra files from text; `Preparation.require_change` turns an agent
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
  sets `name` + `role` (`create | repair | refine | combine`) and implements
  `prepare(ctx) -> Preparation(prompt, copy_parent, inherit_params,
  copy_inspirations, fork_session, files)`; it never touches disk, journal or
  agent. The harness (`search._prepare` → `_prepare_attempt`, pure, so a
  refusal leaves no dir/journal/spend) checks `valid_target`, executes the
  preparation, and fills `{{contract}}` itself — appending the contract when
  a prompt has no token, so an operator cannot drop it (`search._contract` is
  harness-owned). `OperatorContext` is holdout-blind (masked target,
  inspirations, journal); harness-side answers (`render`, `live_experience`,
  `failure_reason`, `report_section`) take candidate ids. `Candidate.role` is
  journaled (backfilled from the operator on load, None for an unknown
  operator) and `journal.drafts()`/`debug_chain()`, `debug_depth` and the
  dummy agent key on ROLE, never on the operator's name; the registry
  (`operators.get_operator/operator_names/role_of/register_operator`) is the
  vocabulary `climber check` and the pi route preflight derive from.
  `baseline`/`seed`/`tune` stay harness-native (no prompt). Inspiration files
  are named by `sdk.inspiration_filename(i)`, never a literal
- Search policies (`modules/policies/`): `greedy` (default) and `openevolve`
  (OpenEvolve's MAP-Elites database as the what-next brain; optional extra,
  `climber.params` pass through to its `DatabaseConfig`). A policy
  owns only `propose`/`observe` and is holdout-blind: `PolicyInput` wraps
  whatever journal it is given in `journal.PolicyJournal` (snapshot of
  `Candidate.holdout_blind()` copies — no holdout fields, no `is_selected`,
  no `path`, writes raise), and `observe()` receives the candidate from that
  view, so `holdout.selection` decides what ships and never what a policy
  expands. It must stay replay-deterministic — the
  openevolve policy seeds/restores the global RNG around every OpenEvolve call
  because that library samples via the `random` module. The exploration
  process is ONE dict: a policy's knobs arrive through its constructor's
  `params` and nothing else — `PolicyInput` carries NO config (it has
  `journal`, `inflight`, `budget`, `higher_is_better`, `accept_band`; a
  policy agrees with the harness on "better" via `view.accept_band`).
  `GreedyPolicy.DEFAULTS` lists every greedy knob (`num_drafts`,
  `max_debug_depth`, `ensemble*`, `tune_*`); `param(name)` =
  `params.get(name, DEFAULTS[name])`, `resolved_params()` the resolved dict.
  Until a climber manifest carries params, `policies.ConfigBackedParams(config)`
  (harness-side glue, a live Mapping: `search.policy_params` over the old
  `search.num_drafts` / `search.max_debug_depth` / `ensemble.*` blocks) is
  what `build_loop`, `climber check` and `SearchRig` hand the policy.
  File policies: a `climber.ref` ending in `.py` is loaded from that path
  (`policies.load_policy_file`; relative to the folder holding the
  hillclimb dir via `policy_base_dir(config)`, the `runs_dir` anchor); the
  file exposes `POLICY` (class or `(params, *, complexity_start)` factory)
  or exactly one class with `propose`+`observe`. `SearchMeta.policy` keeps
  the path as written, `policy_sha256` its hash (`create_search` fails
  before allocating a dir if the file is unreadable; `resume` warns on a
  changed hash). Arm/display names use `policy_label` (the file stem).
  `hillclimb climber check` (`modules/policies/check.py`, pure: no agent, verifier or
  writes) is the cheap pre-verifier for an edited process: replays the
  store's recorded journals plus an empty one through the policy and
  reports contract breaches (stall on empty journal, replay/idempotence
  divergence, dangling or wrong-status targets, journal/file writes,
  shared-instance factory, dirty prompt overrides); `--smoke` adds a
  dummy-agent search; engines in `_ENGINES` are out of scope (exit 2)
- Prompt overrides: `paths.prompts_dir` (default `<hillclimb dir>/prompts/`)
  shadows package templates by name (`prompts/render.py`: `set_override_dir`
  is activated once per engine process in `api.execute_search`, refusing to
  start on `lint_overrides` findings — an override may drop `{{tokens}}`,
  never add unknown ones); `templates_digest` hashes the effective set into
  `SearchMeta.templates_sha256` + `templates_overridden` at `create_search`.
  Tests that activate an override dir must restore the previous setting
- GEPA (`integrations/gepa/`, extra `hillclimb[gepa]`): a climber that
  brings its own `SearchLoop`. gepa drives proposal order, Pareto selection
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
  doing, never a manifest key — the same bundle runs at either level. Its
  verifier is `hillclimb meta evaluate` (hidden group) on
  `$HILLCLIMB_ENGINE_PYTHON` (new verifier env key: the engine's interpreter;
  `api.build_executor` also exports `$HILLCLIMB_DIR`): reads `meta.yaml`
  (inner `problems`, `budget` per inner search, `repeats`, optional
  `floor`/`target`), writes a nested hillclimb dir under the replicate dir
  (`meta.nested_config`: the user's agent/model/routing/problems, own runs,
  files store, learning off), measures each inner floor ONCE
  (`score_floor`, like `verify` — a search's files-only c000 is unscored),
  runs one inner `hillclimb run --climber <solution.py>` per problem × repeat
  serially (scrubbing the outer verifier's `HILLCLIMB_*` keys; a trial's
  `$HILLCLIMB_PARAMS` values ride as `--set climber.params.k=v`, so the
  tuner seam tunes policy knobs), and scores **gap closed** = per problem
  `(best − floor)/(target − floor)` direction-aware, clamped at 0, target =
  best `chart_baselines` value; median over repeats, mean over problems,
  each instance on the `instances` key, spend as `inner_*` metrics. An inner
  run that exits non-zero fails the verifier (→ buggy → debug target); a
  spec the outer `budget.exec_timeout_s` cannot fit is refused before
  spending. `hillclimb meta check` = import allow-list (`hillclimb.sdk`,
  `hillclimb.spaces`, stdlib; `check_climber_source`, the v1 permissions
  rule) + `climber check`. Reference meta-problem `problems/meta-heilbronn/`
  (repo only, NOT in the bundled catalog; baseline `greedy.py` byte-identical
  to `modules/policies/greedy.py`, `tests/test_meta.py` enforces).
  Deferred: directory-shaped candidates, parallel inner runs, replay/ReplayHarness
- Experiments (`experiment.py`): a spec (`experiments/<name>.yaml`)
  is problems × named arms (dotted config overrides, `Config.apply_overrides`)
  × repeats; searches are tagged in `SearchMeta` (`experiment`, `arm`,
  `repeat`, `arm_overrides`) and the report groups on those tags through the
  store — the first arm is the control, gaps are paired by repeat and judged
  against the spec's `noise_floor`. Sequential schedule (repeat-major, arms
  round-robin) is mandatory when an arm touches shared state (memory);
  otherwise `schedule: parallel` + `max_concurrent: N` (or
  `--max-concurrent N`) — unbounded parallel starts every search at once and
  the ones past the machine's operator slots burn their budget in
  `waiting-slot`. `--run-id R --first-repeat K` appends repeats to a finished
  run. `SearchMeta.seed_sha256` records the seed each search started from.
  `experiment report --json` (`experiment.summaries_to_dict`) emits the arms,
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
  experiment's child (`--run-id`), an experiment arm, and `--no-detach` run in-process;
  the in-terminal log is a clock gutter (`common.engine_log`/`split_engine_line`, clock
  from `BudgetManager.clock_str`)
- CLI: `uv run hillclimb --help` (engine); live TUIs: `watch` (agents; `watch candidates` jumps to a search),
  every screen's way back is `keys.back_binding` (esc or b, footer `esc/b back`), so
  `b` is never anything else — the tree views select the best with `*`;
  `chart` (best score vs time per search; a bare `chart` on a folder with
  several run×problem pairs opens `ChartPickerScreen` first — enter opens,
  esc pops back; `watch` pushes the same `ChartScreen` with `c` via
  `watch.push_chart`; `chart_index` is the pure row builder; an experiment
  anchor confines the chart to its own run (`chart_run_scope`) so two runs
  of one experiment never overlay each other's `arm rN`, while a plain
  problem still folds every run into one climb;
  `--detail`/`d` overlays one search's
  exploration tree on the curve), `tree` (one search's exploration tree —
  `tui/tree.py` is the pure layout + fates, `tui/treeview.py` the plotui screen with a
  face-on locked camera), `tree2` (the same layout drawn like the Darwin
  Gödel Machine's archive tree: candidate number inside each circle, fill =
  score on a viridis ramp (hollow = never scored), ring = fate ladder in no
  viridis hue — white = expanded, none = scored, red = failed — star = best,
  the best's parent chain bold — `tui/tree2.py` pure encoding + one Graph3d
  trace using plotui's `set_graph_borders`/`set_graph_labels`/`"star"`
  (labels are drawn by plotui inside the mark only where they fit);
  `fit_radius` sizes marks from the closest projected pair so circles never
  overlap, and `tui/tree2view.py` rebuilds on every zoom/reset/resize to apply
  it, subclassing the `tree` widget/screen through `TreePlotWidget`'s
  `_build_plot`/`_label_nodes`/`_legend_spans`/`_legend_entry_at`/
  `_flat_to_id`/`_place_labels` hooks; the ring ladder is plotui's own
  legend box with host rows (`tree2.legend_entries` → `Plot.set_legend_entries`,
  top-left, node-style swatches so expanded/scored share a fill and differ
  only by the white ring; clicks resolve via `legend_entry_hit`, keys 1-5),
  the score ramp stays a text overlay; a candidate for replacing `tree`),
  `archive` (the DGM two-panel figure: `tree2` on the left, the progress
  chart on the right — `tui/archive.py` pure: scored nodes at (candidate
  number, score) where the number is the circle number (`tree2.node_number`, else
  creation order), best-so-far walked in NUMBER order (same final
  best as `tree.accepted`, intermediate steps may differ from `chart`'s
  landing order), the best's parent chain as a thick line, a cursor at
  the scrub tick's candidate, axes pinned to the live tree; `tui/archiveview.py`
  subclasses `Tree2Screen`, keeps its ids so scrubbing/detail/n-p are
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
  `problems/fitness-landscape/` is the reference problem), `similarity` — two views of one search's candidates,
  same inputs, nothing stored: `similarity map` (default; `tui/similarity_map.py`
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
  tokens, parent chains) and NEVER stored; an experiment arm opens the
  **run scope** instead (`build_run_similarity`: every search of the problem
  in the run, each measured from its own copy of the shared seed, ids
  `<search>/<cid>`, coloured by arm with the chart's palette, `--single`
  opts out); `tui/similarity.py` pure layer with fingerprint caches,
  `tui/similarityview.py` the screens; no usable reference = prints why and
  returns), `graph` (knowledge graph)
- Bundled problems in `src/hillclimb/demo/` (package data for `problem get`;
  copies of `problems/`, circle-packing with a lean `requirements.txt`); keep
  the two in sync
- Native Windows (`harness/oscompat.py`, the ONE place POSIX assumptions
  meet Windows; `tests/test_oscompat.py`): msvcrt locks, CREATE_NEW_PROCESS_GROUP
  + `taskkill /T /F` as the group kill, psutil (Windows-only dep) for liveness
  and the `ps` listings — `os.kill(pid, 0)` KILLS on Windows, never call it —
  `Scripts/python.exe` venvs, junctions when symlinks need privileges, `.cmd`
  agents resolved through PATHEXT, forward-slash `$HILLCLIMB_*` paths, and a
  UTF-8-mode relaunch of the CLI (`ensure_utf8_mode`). `problem get` writes the
  verifier for the fetching OS: verifier.py on Windows (the problem's own, else
  `demo/windows_verifier.py`), verifier.sh elsewhere; `problem.windows_edition`
  picks the `.py` sibling at load time; a `.sh`-only problem runs through Git
  Bash. `tests/test_windows_verifier.py` holds every bundled verifier.sh
  without its own verifier.py to the standard shape and both editions to one
  score. `.github/workflows/quickstart.yml` walks the quickstart on
  ubuntu/macos/windows
- Driving runs from chat: use the `hillclimb` skill (`.claude/skills/hillclimb/SKILL.md`)

## Run-state rules

The engine process is the **single writer** of search state — the journal,
status and `best/` of `runs/<run-id>/searches/<search-id>/`, whichever
DataStore agent holds the records. Never edit those files (or rows)
directly — control a search through `hillclimb stop|kill|prune|resume`, which
route through the store's command queue when the engine is live. The journal
is append-only; replay keeps the last record per candidate. What stays on disk
in every backend: `candidates/`, `best/`, agent streams/logs, `injected_claims.json`.

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
sonnet agent pass per PDF, content-hash cached); paper claims ride the same
retrieval/credit economy as search claims, wired to `paper:` graph nodes. Schema v2: nodes carry both
`pos` (2D, consumed by hillclimb-go) and `pos3` (3D, the plotui viewer) —
keep `pos` byte-stable when touching layout code.

## plotui dependency

The knowledge-graph viewer renders through `plotui`, an editable local path
dep (`../plotui`, Rust core via maturin) in the `tui` extra and dev group —
`uv sync` builds it and needs a Rust toolchain. After editing plotui's Rust
source, run `uv sync --reinstall-package plotui` here (uv won't notice `.rs`
changes on its own). Never override PlotWidget's Textual `on_*` handlers in
subclasses — Textual dispatches them per MRO class (both run); hook the
`apply_zoom/apply_rotate/apply_pan/apply_reset/on_click_at` primitives
instead. Direct mode (iTerm2) double-buffers frames across two Kitty image
ids, so anything that hides or covers a plot must emit
`PlotWidget.kitty_cleanup()` (both ids), never `Plot.kitty_cleanup()` alone.
Chart traces are always named; `show_legend=False` flips plotui's
`legend_visible` instead so the hover readout keeps the series names.
