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
on one (the demo). Search ids are `<problem-id>`, then `<problem-id>-2`, `-3`
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
HOST's, never a search strategy's: `CandidateEvaluator` (`evaluation.py`)
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

- The hillclimb dir: all data lives in a `hillclimb/` folder (config.yaml
  marker, problems/, specs/, runs/) found by upward search; this repo overrides
  runs/problems to its legacy top-level dirs in `hillclimb/config.yaml`.
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
- New problem: `hillclimb init` scaffolds `problems/example/`; check a
  verifier with `hillclimb verify <problem> --repeat 5` (the spread it prints
  is the noise floor — improvements below it are not real)
- Noisy metrics: a trial's score is the MEDIAN of its replicates;
  `search.n_replicates` + `noise_k`/`min_improvement` set an accept band so
  the search cannot climb noise (the floor is the within-trial replicate
  spread — spread across parameter sets is signal), and `replicate_mode:
  serial` is mandatory when the metric measures the machine
  (time/throughput/memory) — parallel replicates measure each other. Seeds
  are never tuned. `n_trials`/`trial_mode` are accepted as legacy spellings
- Concurrency: `search.parallel_operators` per search, `search.machine_max_operators`
  across the machine (flock slots in `~/.cache/hillclimb/agent-slots/`, default
  `min(8, cores-2)`); verifier and agent envs are single-threaded
  (`executor.SINGLE_THREAD_ENV`, parent values win). `hillclimb ps` lists the
  engine process trees; `stop --all` reaps engines whose hillclimb dir was deleted; `reset` kills only the engines pinned to this folder's hillclimb dir, then deletes the dir
- Operator backends: `claude-code`, `codex`, `pi` and `dummy`. Pi supports
  `routing.<op>.sampling` (numeric provider fields), with action → operator →
  default precedence and candidate-journal persistence. Sampling on other
  backends fails validation. `pi.models_file` adds custom/local providers;
  copied configs live under isolated `~/.cache/hillclimb/pi-home/<auth>/`
  (content-hashed subdir for custom models). Pi startup is offline, search
  startup preflights each model/sampling route, and provider errors are read
  from JSON `stopReason` even on exit 0. Debug children use `--fork` to keep
  history while binding tools to the child's cwd; `--session` restores the
  parent's cwd and must not be used across candidates.
- Agent billing: `backend_auth` picks who pays — `subscription` (the Claude or
  ChatGPT login), `api-key`, or `openrouter`, which points codex or pi at
  OpenRouter (`wire_api: responses`; the key comes from the environment or a
  `.env` beside config.yaml) and bills OpenRouter credits instead. Every codex
  call runs under an isolated `CODEX_HOME` in
  `~/.cache/hillclimb/codex-home/<auth>/`, so personal `~/.codex` settings
  change neither a search's results nor its token bill; a provider 402 parks
  the search as `out_of_credits`, and `pricing.py` fills `cost_usd` from
  OpenRouter's catalogue so `budget.max_cost_usd` applies. `hillclimb connect`
  (`connect.py`) is where a credential is checked on purpose instead of at the
  first spawn: every probe runs through the SAME env builders the backends use
  (`subscription_env`/`codex_env`/`pi_env`), so an inherited `ANTHROPIC_API_KEY`
  shadowing a subscription shows up in the table; a bare `connect` is the
  status of all four targets (`claude`/`codex`/`pi` are backends and own their
  login, `openrouter` is a billing route), `connect <target>` runs that login,
  materializes the isolated home, pings the route with one tool-free call
  (`connect.ping`) and pins `backend`/`backend_auth` with a line-level edit of
  config.yaml that keeps its comments — and only when no backend is pinned yet,
  unless `--default`. Keys live in the `.env`, never in `Config`
- DataStore (`store.py`): the one read/write path for a search's records —
  run/search metadata, the append-only journal (`Journal(store.journal(key))`,
  append order is the replay contract), the status record, and the stop/prune
  command queue. Backends: `FileDataStore` (default; `runs/` as today) and
  `SqliteDataStore` (`store.backend: sqlite` → `hillclimb/store.sqlite`, WAL,
  multi-process, writes no yaml). `open_store(config)` picks it; `resolve_search`/
  `latest_search`/`running_searches` replace dir walking; `SearchRecord.state`
  is derived at read time (`status.derive_state`, pid + heartbeat). `key_for(search_dir)`
  is `(run_id, search_id)`; `SearchMeta.search_uid` is the global id. Views and
  commands never open `journal.jsonl`/`status.json` directly — only the file
  backend does. `hillclimb store sync` imports the folder into another backend
- Tunable parameters (`spaces.py` contract + `params.py` engine view +
  `tuner.py`/`tuners/`): an agent may declare numeric knobs in
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
  the tuner seam (`search.tuner: random | optuna`, `search.tuner_params`,
  extra `hillclimb[optuna]`): `ask(space, history, higher_is_better, seed)`
  is a pure function of the candidate's trials + pending sets, so no study
  state survives a call and resume is free. Children of a tunable parent
  inherit its best trial's values as their defaults (`prompts/params_cue.md`
  tells the agent); `best/params.json` ships the selected candidate's best
  trial values. Seeds are never tuned.
- Similarity scores (`similarity_scores/`): pluggable "how alike are two
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
  directly (`openrouter.py`, `OPENROUTER_API_KEY`); its card noise is ~0.02.
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
- Harness + loop (`harness/core.py`, `loop.py`): `Harness` is the fixed core
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
  `search_strategy.build_loop(config)` returns `PolicyLoop(policy)` or, for
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
- Operators (`operators/`): HOW one attempt is made. An `Operator` subclass
  sets `name` + `role` (`create | repair | refine | combine`) and implements
  `prepare(ctx) -> Preparation(prompt, copy_parent, inherit_params,
  copy_inspirations, fork_session, files)`; it never touches disk, journal or
  backend. The harness (`search._prepare` → `_prepare_attempt`, pure, so a
  refusal leaves no dir/journal/spend) checks `valid_target`, executes the
  preparation, and fills `{{contract}}` itself — appending the contract when
  a prompt has no token, so an operator cannot drop it (`search._contract` is
  harness-owned). `OperatorContext` is holdout-blind (masked target,
  inspirations, journal); harness-side answers (`render`, `live_experience`,
  `failure_reason`, `report_section`) take candidate ids. `Candidate.role` is
  journaled (backfilled from the operator on load, None for an unknown
  operator) and `journal.drafts()`/`debug_chain()`, `debug_depth` and the
  dummy backend key on ROLE, never on the operator's name; the registry
  (`operators.get_operator/operator_names/role_of/register_operator`) is the
  vocabulary `policy check` and the pi route preflight derive from.
  `baseline`/`seed`/`tune` stay harness-native (no prompt). Inspiration files
  are named by `sdk.inspiration_filename(i)`, never a literal
- Search policies (`policies/`): `greedy` (default) and `openevolve`
  (OpenEvolve's MAP-Elites database as the what-next brain; optional extra,
  `search.policy_params` pass through to its `DatabaseConfig`). A policy
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
  what `build_loop`, `policy check` and `SearchRig` hand the policy.
  File policies: `search.policy` ending in `.py` is loaded from that path
  (`policies.load_policy_file`; relative to the folder holding the
  hillclimb dir via `policy_base_dir(config)`, the `runs_dir` anchor); the
  file exposes `POLICY` (class or `(params, *, complexity_start)` factory)
  or exactly one class with `propose`+`observe`. `SearchMeta.policy` keeps
  the path as written, `policy_sha256` its hash (`create_search` fails
  before allocating a dir if the file is unreadable; `resume` warns on a
  changed hash). Arm/display names use `policy_label` (the file stem).
  `hillclimb policy check` (`policy_check.py`, pure: no agent, verifier or
  writes) is the cheap pre-verifier for an edited process: replays the
  store's recorded journals plus an empty one through the policy and
  reports contract breaches (stall on empty journal, replay/idempotence
  divergence, dangling or wrong-status targets, journal/file writes,
  shared-instance factory, dirty prompt overrides); `--smoke` adds a
  dummy-backend search; engines in `_ENGINES` are out of scope (exit 2)
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
  lives in `evaluation.py` (`CandidateEvaluator` is journal-free by
  construction; `EvalResult` is the projection loops consume). A verifier may
  write a reserved `instances` key next to `score` (per-instance breakdown,
  stable keys → `Replicate.instance_scores`, median-aggregated) — GEPA's
  Pareto frontier and future QD loops consume it; circle-packing is the
  reference producer, and the emflow eval runner emits one instance per
  scored origin (`<asof>/<zone>`, GEFCom2014's task x zone); a candidate may
  miss keys (failed instances), never introduce new ones
- Experiments (`experiment.py`): a spec (`hillclimb/experiments/<name>.yaml`)
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
- CLI: `uv run hillclimb --help` (engine); live TUIs: `watch` (agents; `watch candidates` jumps to a search),
  `chart` (best score vs time per search; a bare `chart` on a folder with
  several run×problem pairs opens `ChartPickerScreen` first — enter opens,
  esc pops back; `watch` pushes the same `ChartScreen` with `c` via
  `watch.push_chart`; `chart_index` is the pure row builder; an experiment
  anchor confines the chart to its own run (`chart_run_scope`) so two runs
  of one experiment never overlay each other's `arm rN`, while a plain
  problem still folds every run into one climb;
  `--detail`/`d` overlays one search's
  exploration tree on the curve), `tree` (one search's exploration tree —
  `tree.py` is the pure layout + fates, `treeview.py` the plotui screen with a
  face-on locked camera), `tree2` (the same layout drawn like the Darwin
  Gödel Machine's archive tree: candidate number inside each circle, fill =
  score on a viridis ramp (hollow = never scored), ring = fate ladder in no
  viridis hue — white = expanded, none = scored, red = failed — star = best,
  the best's parent chain bold — `tree2.py` pure encoding + one Graph3d
  trace using plotui's `set_graph_borders`/`set_graph_labels`/`"star"`
  (labels are drawn by plotui inside the mark only where they fit);
  `fit_radius` sizes marks from the closest projected pair so circles never
  overlap, and `tree2view.py` rebuilds on every zoom/reset/resize to apply
  it, subclassing the `tree` widget/screen through `TreePlotWidget`'s
  `_build_plot`/`_label_nodes`/`_legend_spans`/`_legend_entry_at`/
  `_flat_to_id`/`_place_labels` hooks; the ring ladder is plotui's own
  legend box with host rows (`tree2.legend_entries` → `Plot.set_legend_entries`,
  top-left, node-style swatches so expanded/scored share a fill and differ
  only by the white ring; clicks resolve via `legend_entry_hit`, keys 1-5),
  the score ramp stays a text overlay; a candidate for replacing `tree`),
  `archive` (the DGM two-panel figure: `tree2` on the left, the progress
  chart on the right — `archive.py` pure: scored nodes at (candidate
  number, score) where the number is the circle number (`tree2.node_number`, else
  creation order), best-so-far walked in NUMBER order (same final
  best as `tree.accepted`, intermediate steps may differ from `chart`'s
  landing order), the best's parent chain as a thick line, a cursor at
  the scrub tick's candidate, axes pinned to the live tree; `archiveview.py`
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
  `Replicate.metrics`; `surface.py` pure layer, `surfaceview.py` the free-orbit
  screen — start the camera at negative pitch, plotui's default views a
  surface from underneath; no landscape = prints why and returns;
  `problems/fitness-landscape/` is the reference problem), `similarity` — two views of one search's candidates,
  same inputs, nothing stored: `similarity map` (default; `similarity_map.py`
  pure layer, `similarity_mapview.py` screen) embeds every candidate by
  pairwise distance (behavioral / structural / blend, `m` cycles) with
  classical MDS, Procrustes-aligned to the previous layout so a live search
  grows in place, drawn as one `add_graph3d` with lineage edges + a gold
  best-so-far `add_line3d`, click dims outside a lineage, `space` replays
  growth, `v` swaps to the cube; `similarity reference` (`v` swaps back;
  `SimilarityBase`/`RunScopeMixin` in `similarityview.py` are shared) shows one
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
  opts out); `similarity.py` pure layer with fingerprint caches,
  `similarityview.py` the screens; no usable reference = prints why and
  returns), `graph` (knowledge graph)
- `hillclimb demo`: zero-setup demo (N parallel detached `hillclimb run`s, `stop --all` ends it) — bundled circle-packing problem in
  `src/hillclimb/demo/` (package data, a copy of `problems/circle-packing`
  with a lean `requirements.txt`); keep the two in sync
- Driving runs from chat: use the `hillclimb` skill (`.claude/skills/hillclimb/SKILL.md`)

## Run-state rules

The engine process is the **single writer** of search state — the journal,
status and `best/` of `runs/<run-id>/searches/<search-id>/`, whichever
DataStore backend holds the records. Never edit those files (or rows)
directly — control a search through `hillclimb stop|kill|prune|resume`, which
route through the store's command queue when the engine is live. The journal
is append-only; replay keeps the last record per candidate. What stays on disk
in every backend: `candidates/`, `best/`, agent streams/logs, `injected_claims.json`.

`hillclimb/knowledge/graph.json` is a **derived index** (gitignored) rebuilt
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
