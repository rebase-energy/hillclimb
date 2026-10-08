# Changelog

## Unreleased

### Changed
- **The engine ships no problem and no climber: one catalog, fetched with `get`.** hillclimb is
  the harness and the contracts; what to climb and how to climb it are the user's. What the
  package ships is a *catalog* of examples — the repository's own `problems/` and `climbers/`
  folders, bundled into the wheel as package data (`hillclimb/_catalog/`, by `hatch_build.py`)
  and read in place from a checkout — and two commands that copy from it: `hillclimb problem get
  <id>` (as before) and `hillclimb climber get <name>` (`greedy`, `openevolve`, `gepa`). The
  second copy of the problems under `src/hillclimb/demo/` is gone; `problems/` is the one copy.
- **No default climber.** `hillclimb run` on a folder that names none refuses — in the terminal,
  before anything is written — and says how: `hillclimb climber get greedy`, or `--climber
  <file.py>`. `hillclimb init` → `problem get` → `climber get` → `run`. `hc.run(...)` in a fresh
  folder says the same (`hc.catalog.climber('greedy')` in Python). A `--set climber.params.*`
  with no climber is an error too; `--set climber.operator_policy=…` starts the block.
- **The whole climber is Python you can read.** `climber get greedy` copies the catalog's
  `policy.py` byte for byte: the selector policy (`Best`, its `schedule` written out: a failing
  tip first, the ensemble window, roots until `num_drafts`, then the best), the operator policy
  (`Greedy`: draft, debug, ensemble, a tune trial, improve), every default as a `DEFAULTS` entry,
  and at the end the `Climber(...)` that wires them to the operators, tuner and memory — no
  config file (`climber: climbers/greedy/policy.py`; the folder names the same file). A
  `prompts/` beside any climber file is its prompts dir without being named. `SelectorPolicy`
  and `OperatorPolicy` are plumbing now (knobs, `param`, `resolved_params`, `observe`); their
  `schedule`/`propose` raise and name the reference. A policy of your own subclasses
  `hc.catalog.module("greedy").Greedy` / `.Best` and inherits the selector written beside its
  base. `climber new mine --from greedy` copies the folder. gepa is a catalog folder of several
  files (`climbers/gepa/`), loaded through `FileScope` like any climber.
- **Old names survive only for records.** A search that recorded `greedy`, `best`,
  `openevolve`, `map-elites`, `gepa` or the 0.6–0.8 module paths resumes and replays — its
  record resolves them to the catalog's files, with the same `climber_sha256` — while a new
  config or Python call naming them is refused with the fetch command. `hillclimb.policies` /
  `hillclimb.selectors` keep only the contracts (`OperatorPolicy`, `SelectorPolicy`).
- `FileScope`: files far apart on disk (a user's policy beside a catalog class installed in
  site-packages) are one package each, rooted where they are, never one rooted at `/`; a
  one-package scope hashes and numbers its files exactly as before, so every recorded identity
  stands.
- `Climber(prompts=...)` is `Climber(prompts_dir=...)`; the old keyword is refused by name.

## 0.8.0 — 2026-10-07

### Added
- `--agent toy` is listed in `run --help` and `smoke --help` beside `dummy`.
- **`hillclimb climber get greedy`: the default climber as a folder you can read.** Copies a
  preset out as `climbers/greedy/` — `climber.yaml` with every default spelled out, `policy.py`
  (which operator makes the next attempt) and `prompts/` with the six templates its operators
  render, the words the coding agents get — and makes it this folder's climber, so an edited
  template is what the next `hillclimb run` climbs with. `prompts/README.md` says how a prompt is
  made: when each template is used (the schedule, numbers filled in) and what the harness fills
  into every `{{token}}` — the problem, the parent candidate, earlier attempts, memory — and
  when a token is empty; a test holds that guide to the templates. A folder holding
  `climber.yaml` is now a first-class way of naming a climber (`climber: climbers/greedy`,
  `--climber climbers/greedy`), with its own identity; `climber list` shows folders and marks
  the default by identity, so a preset and a folder copied from it never both wear the star.
  Built-in operators declare the templates they render (`Operator.templates`).
  `hillclimb init` now makes an empty `climbers/` beside `problems/` and `runs/`, so the
  folder `climber get` writes into is there from the start.
- **A CPU allotment per solution run: `--solution-cpus C` (`concurrency.solution_cpus`,
  default 1).** Every run of a solution — the verifier's, the holdout's, and the coding
  agents' own test runs — gets `$HILLCLIMB_CPUS` = C, and the math libraries' thread pools
  are capped to match (until now they were capped at 1, and holdout runs not at all). The
  contract prompt tells the coding agent to size any process or thread pool from it, never
  from `os.cpu_count()`.
- **Oversubscribed runs are marked.** A replicate journals its allotment as `cpus`; a run
  whose CPU time outpaces its wall time by more than 1.5 × that allotment is flagged
  (`Replicate.oversubscribed`), and `hillclimb watch` shows it beside the candidate:
  `cpu 6.9/1`. A heilbronn solution that starts `multiprocessing.Pool(os.cpu_count())`
  next to three parallel coding agents was the case: its scores measured the machine's
  load.
- **`--parallel-replicates P` (`concurrency.parallel_replicates`) replaces `replicate_mode`.** How
  many of a trial's `n_replicates` seeded runs execute at once is a count like the other
  parallelism levels: 0 = all (the default, what `parallel` was), 1 = one after another (what
  `serial` was; required when the metric measures the machine), 2 = batches of two, which the
  mode could not say. `replicate_mode: parallel | serial` and the 0.3 `trial_mode` still load, in a
  config file and in `--set`, as 0 | 1.
- **`hillclimb run` warns when the parallelism outruns the machine**: searches × agents ×
  replicates at once × solution CPUs against `os.cpu_count()`.
- `docs/concepts.md` has a Parallelism section: the four levels and what sets each.

## 0.7.1 — 2026-10-05

### Fixed
- **`experiment report` judges a gap against the spread across repeats too.** The verdict used
  only the spec's `noise_floor` — the verifier's noise, 0 for a deterministic one — so a gap
  inside the control's own repeat-to-repeat spread was called "better beyond noise". The
  threshold is now the larger of the noise floor and either experiment's spread across repeats,
  and the report says which (`within noise (0.131, the spread across repeats)`); `--json`
  carries `threshold` and `threshold_source` per comparison.
- **A search where no attempt passed says so.** `done` with every attempt buggy or failing only
  means the budget ran out: the run summary, `hillclimb status` and `climber check --smoke` now
  warn, name a candidate to `show`, and under `--agent dummy` explain that its canned solution
  knows nothing of the problem.
- **`uv sync` works in a fresh clone.** plotui comes from PyPI instead of a `../plotui` path
  source; `CONTRIBUTING.md` (new) has the setup, the goldens and how to co-develop plotui.
- **`hillclimb similarity` opens again.** Its app stored the run scope as `self.run`,
  shadowing Textual's `App.run()`, so every launch from the CLI died with "'NoneType' object
  is not callable" (tests drove the screens directly and never noticed).
- **`hillclimb similarity` draws a problem whose baseline is a sample file.** heilbronn's
  baseline is a `sample_submission.csv` with no `solution.py`, and both views refused with
  "baseline c000 has no readable solution.py". They now anchor on the earliest candidate
  that can be compared (a champion asked for with `c` is still never swapped).
- **A run without a score says why.** `verify` and the run log name the problem instead of
  "no score" or a bare exit code: "the score is not a number: 'great'", "the score is NaN",
  "no score written to $HILLCLIMB_RESULT", and for a verifier that failed, the last line it
  wrote to stderr (`c004 buggy — verifier exited 1: FileNotFoundError: answer.json`). Each
  replicate journals it as `error`.
- **The Python API works in a fresh folder.** `Problem(...)`, `climber.search(...)` and
  `hc.run(...)` outside a hillclimb dir make the current folder one (as `hillclimb problem
  get` does) and say so, instead of raising `HillclimbDirNotFound`.
- **Bundled problem files no longer say "edit the template in problems/make_….py"**, a
  generator that is in the repository, not in your copy.
- **`hillclimb verify` works on every bundled problem.** `circle-packing` and
  `fitness-landscape` declare their floor as a number, so `verify` had no solution to run
  and exited 1 ("ships no baseline"); both now name their `sample_submission.csv` as
  `baseline_files`, which `verify` scores (0.5 and 2.94927, the declared values).
- **An expired Claude login is said plainly and fixed by `connect`.** When hillclimb's
  operator login (its own Claude Code home) had an OAuth session that expired or was
  revoked, a search failed three coding agent calls and parked with a summary cut off at
  "Failed to authenticate: OA…", and `hillclimb connect claude` reported the ping failure
  without logging in again (it only did so when logged out). Now the first such call parks
  the search with "Claude login expired — run `hillclimb connect claude`, then `hillclimb
  resume`", and `connect claude` recognises the dead session and logs in again (`claude
  auth logout`, then the login, in the operator home only) as `connect codex` already did.
- **A search stops when a short budget is spent.** A budget under 10 s has a stop margin
  of 0, and the remaining time never goes below 0, so `0 < 0` never fired: a `--budget 5s`
  run kept starting candidates at "0:00 left" (644 of them in one test) when the budget ran
  out before the first verify finished.
- **A mistake in a problem folder is one line, not a traceback**: a missing `metric:` or
  `higher_is_better:`, a missing description, a verifier that is not executable, and
  `run`/`verify` on a problem not fetched yet, which now names `hillclimb problem get <id>`
  (or `problem new` for a name that is not bundled).
- **`higher_is_better` must be a real `true`/`false`.** A quoted `"false"` loaded as `True`
  and climbed the wrong way; it is now an error. A key problem.yaml does not know (a typo
  like `higher_is_beter`) is warned about, with the key it most likely meant.
- **A search whose numeric baseline (`baseline: 0`) is never beaten says so**, instead of
  pointing at a `best/solution.py` that does not exist.
- **A missing coding agent CLI is found before the search detaches**: `hillclimb run` with
  no `claude` on PATH stops with the install command, rather than parking three failed
  operator calls later.

### Added
- **A dead Claude or Codex login is renewed before a command runs into it.** `run`,
  `resume`, `experiment run` and `paper add` check each coding agent they will use with
  one tiny ping (skipped when one passed within the hour); an expired or revoked
  subscription login asks "Log in again now?" at the terminal, logs in again in the
  operator home and carries on — no `hillclimb connect` by hand. Without a terminal, or
  answered no, the command stops before anything is spent, with the fix.
- **`hillclimb init --datastore sqlite`** writes the `store:` block that keeps a folder's
  records of runs (runs, searches, journals, status) in one `store.sqlite` from the first
  run on; the default (`files`) carries the choice commented out in `hillclimb.yaml`.

### Changed
- **`hillclimb similarity` opens the reference cube**, the figure hillclimb.sh and the docs
  show, with its axes named in the plot: `behaviour →`, `code →`, `lineage →`. It opened
  the map before, whose three directions are a layout rather than metrics and so carry no
  names; the map is `v` away, or `hillclimb similarity map`.
- **The similarity views look like the figure on hillclimb.sh, and turn smoothly.** Small
  marks whose shape is the candidate's fate (● built on, ○ left, · failed) and whose
  colour is its score on the site's ramp; the best a gold diamond, the origin an open
  diamond in its corner; in `similarity reference` the best's lineage is a white line
  from the corner out and the axes are named (`behaviour →`, `code →`, `lineage →`); a
  legend row sits under the plot. The cube's three back walls are quietly gridded with
  the front open and its frame is brighter where it is nearer; a drag turns it at a steady
  60 frames a second, following the pointer at any zoom. (plotui: `set_box_grid`,
  `set_axis_titles3d`, the `circle` mark, glide and grab — needs the plotui release after
  0.6.0.)
- **Every human-facing listing speaks the CLI's theme.** `knowledge graph --stats` (and
  `knowledge rebuild`) show the graph as a headline and a node/edge table, largest first;
  `store searches` is a table with the state coloured; `climber check` marks each check
  `ok`/`FAIL`; `experiment report` renders its Markdown tables on a terminal (still plain
  Markdown when piped); `similarity scores` prints its matrix as a table, the closer a pair
  the brighter; `verify`'s runtime-setup lines are themed too. JSON, YAML, diffs and the
  prompt texts the knowledge commands preview stay plain, for scripts and coding agents.
- **`hillclimb resume` detaches, like `hillclimb run`**: the search continues in a background
  engine and the terminal comes back with where its log is and how to follow it.
  `--no-detach` keeps it in the terminal (what `resume` did by default before).
- **The default runtime is lean: numpy, scipy, pandas, scikit-learn.** The first verify of
  a bundled problem (or a `problem new` one) built a 790 MB venv with torch, xgboost and
  lightgbm, which no example uses. A problem that needs them ships its own
  `requirements.txt`; MLE-bench competitions get them through `runtime/requirements-ml.txt`.
  Building the runtime prints one line, not uv's download log (its tail on failure).
- **The README is shorter and renders on PyPI.** It opens with what hillclimb is for (a
  testbed for autoresearch algorithms), lists the five climber modules, ends `Quickstart`
  with `hillclimb summit`, and its "Learn more" grid follows the docs. Images and file
  links are absolute, so the PyPI page shows the logo and mountain instead of broken images.

## 0.7.0 — 2026-10-03

### Fixed
- **`hillclimb summit` copies `params.json`**, the tuned values a tunable solution reads
  through `spaces.params()`. Without it the copied `solution.py` crashed on its first
  `P['…']` (`KeyError`).
- **`hillclimb verify --solution` brings the `params.json` beside the solution**, so a
  summited solution scores with its tuned values instead of crashing.
- **A failed `verify` run shows its last lines of output** (the traceback) where it
  printed a log path inside a temp folder that was already deleted.

### Added
- **`hillclimb problem new <id>` starts a problem of your own.** It writes a small
  two-step problem that already runs (number partitioning: `run.py` calls the solution's
  `partition`, `score.py` checks the answer and writes the score, plus `instances.py`,
  `description.md`, `contract.md` and `baseline.py`) for you to edit into yours, making
  the folder a hillclimb dir first if it is not one. It never overwrites a folder. `init`'s
  next steps and the missing-verifier error point to it.
- **`hillclimb plot` draws a solution, and `hillclimb summit --plot` copies the best and
  draws it.** A problem may ship `plot.py` — plain matplotlib, `plot(solution_dir, ax)`
  drawing the solution's output files and returning an optional caption — picked up by
  default like `interface.py`. It runs in the problem's runtime venv like the verifier
  (matplotlib is added there on first use), saves a PNG (`solution.png` beside a summited
  solution) and opens it; it reads what the solution wrote and never reruns it.
  `heilbronn-11/14/17` draw their numbered points with one smallest triangle shaded (the
  caption counts the ties), `circle-packing` and `circle-packing-32` their circles shaded
  by radius. A problem without `plot.py` says so.
- **A climber put together in Python runs from the CLI.** A `.py` file that builds a
  `Climber(selector_policy=…, operator_policy=…, operators=[…], tuner=…, memory=…)` is
  that whole climber wherever a climber file is accepted: `hillclimb run <problem>
  --climber climbers/my_climber.py`, a run spec, an experiment's setup, `climber show`.
  Its own classes are recorded as `my_climber.py:Class`; a file that builds two names one
  as `my_climber.py:climber`. A file that builds none is still one operator policy.
- **`hillclimb top`**: the control pane for every hillclimb process on the machine, from
  any folder — `ps`, live, in one table where each engine heads its own process tree.
  Keys act on the highlighted row: `s` / `g` stop its engine (now / gracefully) through
  the same command queue as `hillclimb stop`, `k` kills that process and what it started
  (on an engine row the whole engine, like `hillclimb kill`), `o` / `r` order the engines.
- **`hillclimb chart` is one chart per problem, and can tell its runs apart.** The picker
  listed every run of a problem as a row of its own and opened the same folded climb
  whichever you chose; it now lists problems (a study's problem still once per run) and
  opens directly when there is one. The chart still opens on the plain climb — one
  staircase across every run. With several runs, `v` cycles to the same climb with each
  run's dots in its own colour, named in the legend (`#2 15:08`: number and start time, or
  the run's own name) and the line saying which run set the best, then to a comparison,
  one line per run from its own first candidate. `]` / `[` focus one run (the others fade
  to grey, `d` overlays that run's tree). A tie to float precision keeps the credit with
  the run that got there first. The line over the chart wraps between its items instead
  of being cut off at the terminal's edge, with what the climb cost on a dimmed line of its own;
  `tree` and `treeclimb` wrap theirs the same way (`tui/lines.py`).
- Every TUI names its command in the header: `hillclimb watch`, `hillclimb top`,
  `hillclimb chart`, `hillclimb tree`, … where they all read `hillclimb`.
- **`hillclimb ps` is drawn like nvidia-smi** and is about compute only: one box sized to
  the terminal — the machine at a glance (engines, coding agents against the slot cap,
  cpu, memory) above one process table where each engine heads its own tree, its
  row naming the problem and hillclimb dir — where it printed free lines that wrapped
  past the screen. Processes are labelled by role: a coding agent's own tool calls (its
  Bash shell trying `python solution.py`) are `tool` and the servers it loads from your
  Claude config are `mcp`, where `ps` used to call them `verifier` / `child`; what a
  verifier runs is `solution`. Commands are cut to what tells them apart (`claude ·
  sonnet`, `$ python solution.py`, `python c003/trials/t0/…/solution.py`); columns that
  do not fit are dropped, nothing wraps. `hillclimb ps --watch` (`-w`, interval `-n`)
  redraws it in place until ctrl+c and keeps the frame on one screen (MCP servers
  folded, rows shared between engines). Search progress stays in `watch`. A hillclimb
  command computing in a terminal — `verify`, `grade`, `run --no-detach` — gets a block of
  its own too (it is compute hillclimb started, engine or not); `top` lists and ends them.

### Changed
- `hillclimb experiment report` called a gap of exactly 0 "worse" when no noise floor
  was known; it is a tie, as `--json` already said.
- **`hillclimb tree2` is `hillclimb tree`, and the old `tree` is gone.** The tree is
  the Darwin Gödel Machine drawing — candidate numbers in the circles, score as the
  fill, fate as the ring, the best a star — which was `tree2`; the fate-only tree it
  grew out of is removed (`watch` keeps that encoding for its tree panel). No alias:
  `tree2` is not a command.
- **`hillclimb archive` is `hillclimb treeclimb`**: the tree and the climb chart of
  one search, side by side — the name is the two panels. `archive` is not a command
  any more; nothing is kept under the old name. The tree (`tree`, and the left panel
  here) takes the look the docs
  settled on: score is one cyan ramp, dark teal to the accent, instead of viridis; the
  best's lineage is drawn in that cyan, thinner; the star is 1.3× a disc (was 1.4);
  ensemble-input edges are no longer drawn (ensembles are not part of the first
  release — the data still carries them). The chart's lineage stays white beside its
  cyan staircase. Rings are as before: white = expanded, none = scored, red = failed.
- **The two decisions are named after the RSI framework: `selector_policy` (π_sel)
  and `operator_policy` (π_op).** In a block they were `select:` and `policy:`,
  the selector's knobs `select_params:`; they are `selector_policy:`,
  `operator_policy:` and `selector_params:` now, and the old spellings still
  load — in a block, as `--set climber.select=…`/`climber.policy=…`/
  `climber.select_params.*`, in a snapshot and in a search's record — the new
  spelling winning where both are given. What is written out (`climber show`,
  a snapshot, `climber.write()`, `to_spec().block()`) uses the new keys. A
  climber's identity is unchanged by the rename, so a search started before
  it resumes without a "climber changed" note. The classes are
  `SelectorPolicy` and `OperatorPolicy` (`hillclimb.sdk`, `hillclimb.selectors`,
  `hillclimb.policies`); `Selector` and `Policy` remain as aliases for one
  release. In Python: `Climber(selector_policy=…, operator_policy=…)`;
  `select=`, `policy=` and `select_params=` still work. The file-ref markers
  `SELECTOR = …` / `POLICY = …` and `refs.register("policy", …)` are unchanged.
- **One step is two decisions, in a fixed order: the selector picks the node, the
  policy picks the operator.** This is the structure of the RSI framework
  (π_sel, then π_op) and the loop now enforces it: `PolicyLoop` asks
  `selector.schedule(state, busy)` first and hands what it chose to
  `policy.propose(state, selection)`. The selector's answer is the node(s) the next
  attempt starts from: a failing tip (repair comes first), None (a root step), one
  scored candidate, or the top candidates with `combine=True`. The schedule — `num_drafts`,
  `debug`, `max_debug_depth`, `ensemble`, `ensemble_reserve_fraction`, `ensemble_top_k`,
  `ensemble_max_attempts` — therefore moved from the policy to the `Selector` base class
  (its `schedule` wraps a subclass's `select`: `Best.select`, `MapElites.select`), and its knobs
  are `select_params`. A block that still writes them under `params`, a `--set
  climber.params.num_drafts=…`, a v2/v3 record and a pre-0.6 snapshot all load: the
  knobs land in `select_params`. In Python it is `Best(num_drafts=3)`;
  `Greedy(num_drafts=3)` says so. `Policy.propose(state, selection)` is the new
  signature (`None` = a root step); the base `propose` is the plain mapping (draft,
  debug, ensemble, improve) and `Greedy` adds tune, for the CHOSEN candidate only
  (`tune_now`; the old scan for any tunable candidate in the gate is gone). `Policy`
  keeps `draft_action`, `expand_action(state, selection, operator=)`, `draft_complexity`;
  a selector of your own implements `select` (the hook was briefly called `pick`; that
  spelling still loads) and leaves `schedule` alone. The `openevolve` preset is
  `select_params: {ensemble: false}` + `params: {tune_budget: 0}`.
- **`Climber`'s keywords are in the order a step runs them**: `select`, `policy`,
  `operators`, `tuner`, `memory` (then `loop`, `params`, `select_params`, `prompts`,
  `name`). A positional first argument is still a preset name, or a policy.
- `climber.select()` is the selector's answer alone (a `Selection`, or None for a root
  step); `climber.propose()` is the whole decision.

### Added
- **The sklearn shape.** `Problem` is what a climber searches, `Budget` is what the
  search may spend (`Budget(wall_clock="2h", evaluations=200, tokens=..., cost_usd=...)`; a
  plain `"10m"` still works), and `climber.search(problem, budget=...)` is the whole
  search in one call, with the result on the climber afterwards:
  `climber.best`, `.selected`, `.candidates`, `.history`, `.spend`, `.solution`,
  `.params`, `.to_frame()` and `.result` (the `SearchOutcome`). `Problem("heilbronn-11")`
  is a problem that exists (the folder's, else a bundled one copied in);
  `Problem(name, score=my_function, description=..., output=...)` defines one from a
  Python scoring function and writes its folder under `problems/` the first time it is
  searched (a generated `verify.py` imports the function with the engine's interpreter).
  `hillclimb.run` takes a `Problem` and a `Budget` too.
- **Step a search by hand.** `climber.start(problem, ...)` opens one search
  and leaves the loop to you: `climber.propose()` is what the policy would do
  next (nothing runs), `climber.run(action)` runs it or an `hc.Action` of your
  own, `climber.step()` is both, `climber.state` is what the policy sees,
  `climber.finish()` hands the rest to the climber and `climber.close()`
  settles it. The clock runs only while a step does. It is an ordinary
  search (`watch`, `chart`, `status` show it); closed with budget left it is
  `stopped` and `hillclimb resume` continues it. One at a time per process; a
  `loop:` climber can only `finish()`. Underneath is `hillclimb.api.Search`
  (`hillclimb.api.start`).
- **A search's result is readable.** What `hc.run` returns (`SearchOutcome`)
  has `.best`, `.candidates`, `.history` (every rise of the best-so-far),
  `.spend` (evaluations, tokens, cost, seconds), `.solution`, `.params`,
  `.source(candidate_id)`, `.journal` (read-only), `.meta`, `.status` and
  `.to_frame()` (one pandas row per candidate). `hc.open_search(ref)` reads
  any earlier search, the latest without a ref, and follows a running one.
- **A free agent whose scores move.** `--agent toy` is a scripted stand-in on
  the bundled `fitness-landscape` problem (a solution is one point on a
  terrain): drafts are random points, improves step from the parent, and it
  declares two parameters so tune trials run. `hc.register_agent(name, cls)`
  adds a scripted agent of your own, for searches run in that process.
- **`hc.run(..., max_evaluations=, learning=)`**, both recorded in the run's
  `spec.yaml`. `hc.Action` is exported at the root.
- **`examples/`**: seven scripts (search and read, step by step, your own policy,
  operator, selector and agent, a problem of your own), on Claude Code by default
  and on the scripted agent with `toy` on the command line, which is how the test
  suite runs them.
- `fitness-landscape` is a bundled problem (`hillclimb problem get
  fitness-landscape`) with its own lean runtime.

### Changed
- A search that parks or stops returns what it would ship so far as
  `outcome.selected` (it was `None`).
- A search whose setup fails (an unknown agent, a climber that does not lint)
  is recorded `failed` with the reason, instead of being left `running` until
  its process exits; a stop during setup returns a `stopped` outcome.
- `hillclimb resume` keeps a search that started with learning off out of the
  knowledge.
- **An operator's `role` is its `kind`.** What an operator's candidates are
  (`create | repair | refine | combine`) is `Operator.kind`, `Candidate.kind`
  and the `kind` key in the journal, so `role` means one thing: what a
  climber plays in a search (`solver | improver`). An operator written with
  `role = ...` still loads, and so does every recorded journal. In the SDK,
  `ROLES` is `OPERATOR_KINDS`.
- **The CLI an operator calls is a "coding agent"** in every message, help
  screen and doc (Claude Code, Codex, pi), since a climber is itself an
  agent. Names you type are unchanged: `--agent`, `agent:`, `agent_auth`,
  `parallel_agents`.

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
