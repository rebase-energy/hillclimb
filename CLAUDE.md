# hillclimb

Hillclimbing on verifier-defined problems: a greedy search engine that spawns
headless Claude Code agents as operators (draft/debug/improve/ensemble),
scores each candidate through the problem's verifier, and keeps the best
solution per search.

Hierarchy: **Run** (one invocation, `runs/<run-id>/`) → **Search** (one engine
process, `searches/<search-id>/`) → **Candidate** (immutable code artifact) →
**Trial** (one execution). The problem is an *attribute* of a search, not a
level: a run may hold searches on different problems (MLE-bench) or several
on one (the demo). Search ids are `<problem-id>`, then `<problem-id>-2`, `-3`
(atomic mkdir allocation); `search.yaml` carries `problem_key` — the
canonical cross-run identity (`emflow://…`, `mlebench://…`, or the local
problem id; backfilled on read like hillclimb-go's `EffectiveProblemKey`) —
and every problem-scoped view (chart, best-ever, knowledge) groups on it.

A problem **is its verifier**: `problems/<id>/verifier.sh` is the only process
the engine starts. It drives `solution.py` and writes the score to
`$HILLCLIMB_RESULT` (a `{"score": …}` object or a bare number; other numeric
keys are journaled as `Trial.metrics` — feature dimensions for policies,
never a score); exit 0 means valid. A trial the engine kills under a
timeout clamped by the search's *remaining budget* is journaled `abandoned`
("cut off at the budget wall"), never `buggy`: only a non-zero exit or the
problem's own `exec_timeout_s` makes a candidate buggy and thus a debug
target (`watch` colours the candidates cell red/yellow/green on
buggy/abandoned/clean). `holdout: true` in `problem.yaml` makes the engine run the same script
with `--holdout` in a directory agents never see. Providers (`emflow://`,
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
  agent works), `trial_dir`. The word "workspace" is banned (`tests/test_vocabulary.py`
  enforces it); old journals/status files that still carry a `workspace` key
  are mapped to `candidate_dir` on load.
  Machine-scoped state (shared venvs, emflow cache, agent slots) lives in
  `~/.cache/hillclimb/`.
- Tests: `uv run pytest`
- New problem: `hillclimb init` scaffolds `problems/example/`; check a
  verifier with `hillclimb verify <problem> --repeat 5` (the spread it prints
  is the noise floor — improvements below it are not real)
- Noisy metrics: a candidate's score is the MEDIAN of its trials;
  `search.n_trials` + `noise_k`/`min_improvement` set an accept band so the
  search cannot climb noise, and `trial_mode: serial` is mandatory when the
  metric measures the machine (time/throughput/memory) — parallel trials
  measure each other
- Concurrency: `search.parallel_operators` per search, `search.machine_max_operators`
  across the machine (flock slots in `~/.cache/hillclimb/agent-slots/`, default
  `min(8, cores-2)`); verifier and agent envs are single-threaded
  (`executor.SINGLE_THREAD_ENV`, parent values win). `hillclimb ps` lists the
  engine process trees; `stop --all` reaps engines whose hillclimb dir was deleted; `reset` kills only the engines pinned to this folder's hillclimb dir, then deletes the dir
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
- Search policies (`policies/`): `greedy` (default) and `openevolve`
  (OpenEvolve's MAP-Elites database as the what-next brain; optional extra,
  `search.policy_params` pass through to its `DatabaseConfig`). A policy
  owns only `propose`/`observe`; it must stay replay-deterministic — the
  openevolve policy seeds/restores the global RNG around every OpenEvolve call
  because that library samples via the `random` module
- Search engines (`search_runner.py`, architecture in
  `docs/optimizer-host-plan.md`): optimizers that own their whole loop
  dispatch as a `SearchRunner` via `_ENGINES` before `get_policy()` is ever
  called — `gepa` (`integrations/gepa/`, extra `hillclimb[gepa]`) is the
  first: a routed hillclimb agent is its mutation proposer
  (`SEARCH_DIR/gepa/proposals/`), every evaluation is a canonical journaled
  candidate (`policy_meta.optimizer: gepa`), checkpoints in
  `SEARCH_DIR/gepa/state`, holdout only after the optimizer finishes and
  never visible to it. `driver.py` is the only module importing gepa
  (`skip_perfect_score=False` is mandatory there — the upstream default
  silently disables mutation for unbounded scores); the default suite drives
  `GEPASearcher` through `tests/gepa_fakes.py`. Shared trial execution lives
  in `evaluation.py` (`CandidateEvaluator` is journal-free by construction;
  `EvalResult` is the projection engines consume). A verifier may write a
  reserved `instances` key next to `score` (per-instance breakdown, stable
  keys → `Trial.instance_scores`, median-aggregated) — GEPA's Pareto
  frontier and future QD engines consume it; circle-packing is the
  reference producer, and the emflow eval runner emits one instance per
  scored origin (`<asof>/<zone>`, GEFCom2014's task x zone); a candidate
  may miss keys (failed instances), never introduce new ones
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
  face-on locked camera), `surface` (one search's candidates on the problem's
  3D terrain — needs the problem to ship `landscape.py` (`elevation(x, y)` +
  `grid(n)`, picked up by default like `contract.md`) and journal each
  candidate's position as `surface_metrics` keys (default x/y) in
  `Trial.metrics`; `surface.py` pure layer, `surfaceview.py` the free-orbit
  screen — start the camera at negative pitch, plotui's default views a
  surface from underneath; no landscape = prints why and returns;
  `problems/fitness-landscape/` is the reference problem), `similarity` (one
  search's candidates as a 3D scatter at behavioral/structural/lineage
  distance from a reference — the seed (else baseline) by default, `c`
  toggles the current champion — coloured by score rank; distances are
  derived at render time from existing artifacts (the problem's optional
  `fingerprint.py` — `fingerprint(candidate_dir) -> vector`, picked up by
  default like `landscape.py`, for outputs with equivalences the flat file
  misses — else submission.csv or trial-0's evaluator report, solution.py
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
