# hillclimb

Auto-hillclimbing for verifier-defined problems. A greedy search engine spawns headless coding
agents (Claude Code) as operators — **draft** new solutions, **debug** failures,
**improve** the best one, **ensemble** at the end — executes every candidate in a
sandboxed venv, scores it against a hidden holdout, and keeps the best submission
in `runs/<run-id>/searches/<search-id>/best/`.

The design is deliberately three-layered:

1. **`hillclimb` CLI** — the headless engine. Scriptable, plain output, meaningful
   exit codes. All run state lives on disk.
2. **Your interactive agent** (Claude Code) is the front door: the repo ships a
   skill (`.claude/skills/hillclimb/SKILL.md`) that teaches it to start, monitor,
   and control runs in the background while you chat.
3. **`hillclimb watch`** — a live TUI you keep open beside the agent:
   runs → searches → candidate trees, with an on-demand candidate
   detail panel for notes, scores, lineage, output, and the timestamped operator stream when present.
   Drag the divider or use `+` / `-` to resize the detail panel.

## Try it in three commands

```bash
pip install hillclimb
claude login                   # agents run through the Claude Code CLI and bill your subscription
hillclimb problem list         # see every problem bundled with hillclimb
hillclimb problem get circle-packing   # 1. the problem: a folder you can read — the verifier IS the problem
hillclimb run circle-packing --budget 10m --parallel-searches 3 --parallel-operators 3   # 2. climb
```

`hillclimb problem get` creates a `hillclimb/` dir in the current folder and copies
the bundled circle-packing problem into `hillclimb/problems/` (with a declared
one-circle baseline, sum of radii 0.5, scored at t=0) — open `verifier.sh`
and `verify.py` before running anything, that is the whole interface.
`hillclimb run` then starts **three 10-minute searches in parallel, in the
background**, all in one run so they share discoveries as they go — your
prompt comes straight back. Parallelism has two levels: `--parallel-searches`
is how many independent searches climb the problem (each its own engine
process), `--parallel-operators` how many agents each search keeps busy at
once (one candidate each). `hillclimb demo` is the same thing as one command.
While they climb:

```bash
hillclimb watch candidates   # one search's candidates: drafting, debugging, improving
hillclimb watch              # all the searches side by side
hillclimb chart              # the hillclimb: best score so far across every search, every candidate a dot
hillclimb graph              # the knowledge graph growing as searches finish
hillclimb tree               # one search's exploration tree: expanded vs discontinued lineages
hillclimb chart --detail     # the curve with that tree drawn on it (every scored candidate, parent edges)
```

Run `hillclimb problem list` to see the bundled catalog. It currently includes
`circle-packing`, `knapsack`, and `heilbronn-convex-13`. Get one with
`hillclimb problem get <problem>` before running it.

`hillclimb stop --all` ends the demo (the best solutions stay in `runs/`); `hillclimb reset` ends it AND deletes this folder's `hillclimb/` dir — only engines pinned to that dir are killed, never another folder's;
3 searches x 3 operators is 9 agents, capped machine-wide by `search.machine_max_operators`; each search's engine log is
under `hillclimb/runs/<run-id>/logs/`.

## Install (from source)

```bash
uv sync
claude login   # operator calls bill your Claude subscription
```

Optional — shell tab-completion for commands, subcommands, and options:

```bash
hillclimb --install-completion   # writes into your shell config; restart the shell
```

## Quickstart

```bash
uv run hillclimb run problems/circle-packing --budget 10m
uv run hillclimb watch                                  # live dashboard (2nd terminal)
uv run hillclimb status                                 # or: plain-text status of the latest search
```

Try the engine without spending agent calls: `--backend dummy`.

## The hillclimb dir

All hillclimb data lives in one `hillclimb/` folder inside your project, so it
never mingles with the rest of the repo. `hillclimb init` creates it:

```
my-project/
└── hillclimb/
    ├── config.yaml     # defaults, and the marker that makes this the hillclimb dir
    ├── problems/       # problem definitions
    ├── specs/          # committed run specs (versioned run parameters)
    └── runs/           # search artifacts (gitignored by init)
```

Commands work from any subdirectory — the hillclimb dir is found by upward
search for `hillclimb/config.yaml` (like git). Without one, commands error and
point you at `hillclimb init`; `HILLCLIMB_DIR` pins it explicitly.

Config precedence, highest first: CLI flags → the hillclimb dir's
`config.yaml` → user `~/.config/hillclimb/config.yaml` → built-in defaults.

Machine-scoped state is shared across hillclimb dirs under `~/.cache/hillclimb/`
(honors `XDG_CACHE_HOME`; `HILLCLIMB_CACHE_DIR` overrides): solution-runtime
venvs keyed by a hash of their requirements (rebuilt automatically when
requirements change), the emflow problem cache, and the cross-search agent
semaphore. Checkouts predating this layout left `.runtime-venv*/` and `cache/` in the
project dir — safe to delete.

### Run specs: versioned run parameters

The canonical way to run is a committed spec file, so the repo fully describes
its searches (`git log` explains every run). Entries carry per-search
parameters; CLI flags override them for ad-hoc experiments:

```yaml
# hillclimb/specs/gefcom.yaml
problems:
  - target: emflow://gefcom2014:solar
    model: opus
    budget: 2h
    parallel_operators: 3
  - target: emflow://gefcom2014:wind
    budget: 1h
```

```bash
uv run hillclimb run hillclimb/specs/gefcom.yaml            # exactly as committed
uv run hillclimb run hillclimb/specs/gefcom.yaml --model sonnet   # ad-hoc override
```

A spec with a single top-level `target:` (plus the same parameter keys) runs
one search. `run.yaml` records which spec launched the run.

## Concepts

The UI and on-disk metadata use this hierarchy, coarse to fine:

```text
Run
└── Search
    └── Candidate
        └── Trial
```

- **Problem**: reusable definition under `problems/<id>/`.
- **Run**: one invocation of hillclimb. A single-problem run contains one
  search; a suite run contains one search per suite entry.
- **Search**: one search worker (engine process) exploring one problem.
- **Candidate**: an immutable code artifact produced by an operator. Any change
  to the code — however small — is a new candidate with a new id.
- **Trial**: one execution of a candidate with a fixed parameterization
  (params, seed). Today the engine runs exactly one trial per candidate; the
  schema supports several so re-evaluations and parameter tuning can land
  without another migration.

`hillclimb watch` opens on the Runs screen. Metadata carries
`schema_version: 2`; directories from the pre-v2 flat layout are ignored.

## Supported agents

Operators are headless coding-agent processes, one per operator call, behind
the backend seam in `src/hillclimb/backends/`:

| backend | what it is |
|---|---|
| `claude-code` | Claude Code in headless mode — the production backend; bills your Claude subscription |
| `codex` | Codex CLI in non-interactive mode; uses your Codex login by default |
| `dummy` | no model calls: a scripted operator for exercising the engine, TUIs and run layout |
| `fake` | deterministic canned operator for the test suite |

Other agents (Codex, Pi, OpenCode, …) plug in at the same seam: a backend
implements the `OperatorBackend` protocol in `backends/base.py` — take a prompt plus a
working directory, return the agent's JSON result — and is selected with
`--backend <name>`.

## Search policies

*What to try next* is the search engine, and it is a seam of its own:
`src/hillclimb/policy.py` defines it, `src/hillclimb/policies/` holds the
implementations. Everything else — candidate dirs, prompts, agent calls, trials,
holdout, journaling, `best/` — is harness, and a policy never touches it.

| policy | what it does |
|---|---|
| `greedy` | debug the newest buggy tip > ensemble in the final budget window > draft until `num_drafts` branches are scored > improve the best |
| `openevolve` | [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)'s MAP-Elites database decides what to expand: a population kept diverse over feature dimensions, split across islands with migration; parent + inspirations sampled per island (exploration / elite archive / fitness-weighted). Hillclimb's operators do the mutating, the verifier the scoring, and the `debug` rule is kept. `pip install 'hillclimb[openevolve]'` |
| `gepa` | [GEPA](https://github.com/gepa-ai/gepa) owns the whole loop — reflective mutation over evaluation feedback and Pareto selection over the verifier's per-instance scores — while hillclimb evaluates, journals, and holds the private holdout. A full *engine*, not a policy (see below). `pip install 'hillclimb[gepa]'` |

```yaml
# config.yaml — OpenEvolve's quality-diversity search over hillclimb's operators
search:
  policy: openevolve
  policy_params:
    num_islands: 3
    feature_dimensions: [complexity, score]   # built-ins: complexity, diversity, score
    num_inspirations: 2                       # copied in as candidate_<i>.py
```

Any other feature dimension must be a numeric key the verifier writes next
to `score` (see *Trial metrics* below), e.g. `feature_dimensions: [runtime_s,
score]`. Each evolved candidate's `policy_meta` records its island, grid cell
and inspirations in the journal (`hillclimb show <candidate>` prints it);
the watch TUI and knowledge graph don't surface it yet.

A policy is two methods over a read-only `SearchView`:

```python
class SearchPolicy(Protocol):
    name: str
    params: dict   # persisted into SearchMeta, so `resume` restores them

    def propose(self, view: SearchView) -> Action | None: ...
    def observe(self, view: SearchView, candidate: Candidate) -> None: ...
```

`propose` returns one `Action` — an operator (`draft`/`debug`/`improve`/
`ensemble`), the candidate to target, optional `inspiration_ids`, and
optional per-action routing — or `None` to hold the slot until an in-flight
result lands. `observe` is called after every terminal result, and replayed
over every existing candidate when the policy is constructed, which is what
makes `hillclimb resume` work.

Three rules the harness relies on, spelled out in `policy.py`:

- `propose`/`observe` run only on the scheduler thread, under the search's
  state lock. A policy may read candidate dirs; it must never write.
- Every decision must be derivable from replayed journal state — compute it
  from the `SearchView`, or rebuild your caches in `observe`.
- Ensemble-style actions must carry their inputs in `inspiration_ids`; the
  harness copies those solutions into the new candidate dir.

To add one: implement the protocol, register it in the `_POLICIES` dict in
`policies/__init__.py`, and select it with `hillclimb run --policy <name>`
(constructor arguments come from `search.policy_params` in config.yaml).
`policies/greedy.py` is 170 lines and is the reference. Beam search, MCTS,
evolutionary populations, novelty search and bandits over operators all fit
this shape — greedy is just the one that ships.

### The GEPA engine (optional extra)

Some optimizers cannot be reduced to "what next?" — they own proposal,
reflection, and selection themselves. Those integrate one tier up, as a
**search runner** (`src/hillclimb/search_runner.py`): dispatched by the same
`search.policy` name, but handed the full dependency set instead of a
`SearchView`. The architecture is `docs/optimizer-host-plan.md`; GEPA is the
first such engine:

- greedy: hillclimb chooses the parent and asks an agent to mutate;
- `openevolve` policy: OpenEvolve supplies selection/inspirations, hillclimb's
  agent still mutates;
- `gepa` engine: GEPA drives reflective mutation and Pareto search, hillclimb
  evaluates and records.

```bash
uv sync --extra gepa
uv run hillclimb run <problem> --policy gepa --seed-from my_solution.py
```

GEPA's mutations are performed by a routed hillclimb agent (configure
`routing.gepa`, falling back to `routing.default` and the global
backend/model) in scratch dirs under `SEARCH_DIR/gepa/proposals/`; every
evaluation becomes a normal journaled `cNNN` candidate, so `watch`, `tree`,
and `chart` work unchanged (`policy_meta.optimizer == "gepa"` carries the
lineage). GEPA checkpoints under `SEARCH_DIR/gepa/state` and `hillclimb
resume` continues both the journal and the optimizer, with a warm evaluation
cache so replayed proposals cost nothing.

MVP limits: one mutable file (`solution.py`), serial
(`search.parallel_operators: 1`), no merge, and a **required executable
seed** — pass `--seed-from` or ship an executable baseline. Holdout privacy
is strict and one-way: holdout scoring runs only after the optimizer
finishes, and no holdout value ever reaches GEPA's prompts, feedback, or
state (regression-tested with sentinels). When the verifier emits per-instance
scores (below), `frontier_type: instance` (the default) tracks GEPA's Pareto
frontier per instance; without them the frontier degenerates to the aggregate
score.

## emflow problems (optional extra)

With the `emflow` extra installed (`pip install 'hillclimb[emflow]'`),
targets of the form `emflow://<name>` run problems from
[emflow](https://github.com/rebase-energy/emflow)'s registry — agents author
`Predictor` classes (`solution.py` exposing `get_model()`), and the provider
supplies the verifier command: a generic evaluator fits and scores them on the
problem's validation split, with the hidden holdout as a second run of the
same command. A bare package name is a virtual
suite (one search per variant):

```bash
uv run hillclimb run emflow://gefcom2014:solar --budget 2h   # one track
uv run hillclimb run emflow://gefcom2014 --budget 2h         # all four tracks
```

The baseline candidate (`c000`) is the benchmark's reference model evaluated
for real, and a finished search ends with one official emflow Verifier run
(leaderboard row + rank, with `n_trials` recorded for selection honesty).
Programmatic use: `hillclimb.run_search("emflow://gefcom2014:solar",
budget_s=7200)`.

## MLE-bench problems

Targets of the form `mlebench://<competition-id>` run
[MLE-bench](https://github.com/openai/mle-bench) competitions against a local
mle-bench checkout (located via `paths.mlebench_python`; prepare data first
with `mlebench prepare -c <competition-id>` in that venv). Agents see only the
prepared PUBLIC split and climb on their own validation score; when the search
finishes, the engine runs `mlebench grade-sample` exactly once on the selected
candidate and writes the report (score + medal flags) to
`mlebench-grade.json` — the private test set never influences selection.

A split name is a virtual suite, one search per listed competition
(`lite` is an alias for the 22-competition `low` split):

```bash
uv run hillclimb run mlebench://spaceship-titanic --budget 2h   # one competition
uv run hillclimb run mlebench://lite --budget 4h                # MLE-bench Lite
```

## Defining Problems

A problem is a folder, and a problem **is its verifier**. Users define new
problems without changing Python code:

```
problems/my-problem/
├── problem.yaml
├── description.md
├── verifier.sh            # the contract: exit 0 = valid, write the score
└── data/                  # optional runtime inputs
```

Minimum `problem.yaml`:

```yaml
problem_id: my-problem
metric: my-score
higher_is_better: true
description: description.md
time_budget_s: 900
```

Everything else is optional:

```yaml
verifier: verifier.sh            # the default
holdout: true                    # engine also runs `verifier.sh --holdout`
contract: contract.md            # what solution.py must be/do (prompt section)
baseline: 0.5                    # scored at t=0 as the floor candidate; numeric values also become chart lines
chart_baselines:                 # optional named horizontal lines in `hillclimb chart`
  previous best: 0.73
requirements: requirements.txt   # per-problem venv (default: shared csv venv)
data_dir: data
allow_network: false
```

`chart_baselines` accepts any number of `label: score` entries. Each becomes
a named horizontal reference line, in the order written. A numeric `baseline`
automatically adds the line named `baseline`; `chart_baselines` is only needed
for additional references. A file-based baseline is still evaluated as c000,
but cannot become a fixed reference line until its score is known.

### The verifier contract

`verifier.sh` is the only process the engine starts. It drives `solution.py`
itself — run it, import it, shell out to it — and reports the score:

```bash
#!/usr/bin/env bash
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"   # writes ./submission.csv

rm -f "$HILLCLIMB_RESULT"                   # only the scorer may score
exec "$HILLCLIMB_PYTHON" problem/verify.py  # writes $HILLCLIMB_RESULT
```

| | |
|---|---|
| exit 0 | the candidate is valid; non-zero routes it to the `debug` operator |
| `$HILLCLIMB_RESULT` | the score: `{"score": <float>, "report": {...}, <other numeric keys>}`, or a bare number |
| `$HILLCLIMB_PYTHON` | the managed runtime venv's interpreter (bare `python` resolves via PATH: wrong interpreter) |
| `$HILLCLIMB_SOLUTION` | the solution path for this run (trial-dir aware) |
| `$HILLCLIMB_SPLIT` | `validation` or `holdout` |
| `$HILLCLIMB_TRIAL_SEED` | set when the engine runs repeated trials |

The result file is both the score carrier and the completion proof: the engine
deletes it before every run, so a stale file can never fake success, and exit 0
without one is a contract violation rather than a silent zero. Reading the
score from a file rather than stdout is what keeps it honest — agent-authored
code runs inside the verifier and shares its stdout.

The command runs with cwd = the candidate's working dir (`solution.py`, plus
`./problem/` and `./data/` symlinks). Validation runs get a
credential-scrubbed environment; `--holdout` runs in a directory agents never
see, with the full environment (private holdout data may be gated).

```bash
uv run hillclimb verify my-problem --repeat 5   # score it outside a search
```

`hillclimb verify` is the fastest way to check a new verifier, and `--repeat`
reports the spread between identical runs — an improvement smaller than that
is noise, not progress. `problems/bin-packing/` and `problems/circle-packing/`
are the two reference shapes (evaluator-driven, and run-then-score).

### Trial metrics (optional)

Any other **numeric** key in the result object is journaled verbatim as the
trial's `metrics` (`{"score": 12.3, "runtime_s": 0.8, "n_params": 40}`), and
a candidate's metrics are the per-key median of its trials — the same rule
as the score. The engine never ranks on them; they are feature dimensions
for quality-diversity policies (`openevolve`) and context for reports.
Strings, booleans and NaN are dropped silently.

### Trial reports (optional)

`$HILLCLIMB_RESULT` may carry a `report` block alongside the score: any
verifier that writes one gets its breakdown stored on the trial, rendered into improve prompts ("attack the largest contributors"),
and shown by `hillclimb show` and the watch TUI:

```json
{"split": "validation",
 "report": {
   "version": 1,
   "overall": {"score": 12.3, "n_origins": 100, "n_scored": 2400},
   "segment_label": "store",
   "zones": [{"zone": "store-7", "score": 19.9, "n_scored": 240}, ...],
   "horizons": [{"bucket": "13-24h", "score": 14.1, "n": 1200}, ...],
   "quantiles": [{"q": 0.9, "pinball": 4.1, "coverage": 0.95}, ...],
   "worst_origins": [{"asof": "...", "zone": "store-7", "score": 44.0}, ...],
   "residual_bias": {"mean_error": -1.2, "mean_abs_error": 8.8, "mean_actual": 41.0},
   "report_error": null
 }}
```

All sections are optional; order `zones` (any segmentation — the label is
yours via `segment_label`) worst-first. Producers, by trust:

- **emflow problems** — the evaluator computes the full breakdown (per-zone,
  per-horizon, per-quantile calibration, persistence skill) automatically.
- **directory problems** — the verifier writes the file (see
  `problems/circle-packing/verify.py`); a verifier that discards what the
  solution left behind gives its report evaluator trust.
- **self-reported problems** (MLE-bench) — the number is the agent's own
  claim, so the report is stored and rendered labelled *self-reported*.

Only `"split": "validation"` reports are ever fed back to operators — holdout
evaluations never produce one, by construction. `report.enabled: false` in
config disables prompt injection (data is still recorded).

### Per-instance scores (optional)

A third reserved key, `instances`, carries the breakdown of `score` over the
problem's sub-instances — zones, folds, test cases — in the same metric and
direction, with keys stable across the search:

```json
{"score": 2.158, "instances": {"circle-00": 0.083, "circle-01": 0.083}}
```

They are journaled per trial (`Trial.instance_scores`), aggregated per key by
median like everything else, and consumed by engines whose selection is
per-instance — GEPA keeps a candidate alive if it wins on *any* instance, not
just on average. `problems/circle-packing/verify.py` (one instance per
circle) is the reference producer; verifiers that emit nothing lose nothing.
emflow per-zone and MLE-bench per-fold instances are planned follow-ups.

## Noise: not climbing your own measurement error

A greedy search will happily spend a whole budget chasing a metric that moves
on its own. Three settings decide whether it can:

```yaml
search:
  n_trials: 5            # evaluate each candidate this many times
  trial_mode: serial     # `parallel` (default) | `serial`
  noise_k: 2             # a gain must beat 2x the measured noise floor
  min_improvement: 0.0   # ...or an absolute floor, in metric units
```

Concurrency is bounded machine-wide, not per search: `search.parallel_operators`
is how many operators one search keeps in flight, and
`search.machine_max_operators` (default `min(8, cores - 2)`, `0` = off) caps
the total across every search on the machine — extra operators wait
(`waiting-slot` in `hillclimb watch`). Every verifier and agent process gets
`OMP/OPENBLAS/MKL_NUM_THREADS=1` unless the parent environment sets them, so
N operators cost at most N cores; `hillclimb ps` shows what is actually running.

- **The candidate's score is the MEDIAN of its trials**, so one slow run or
  unlucky seed does not become the number the search ranks on. With
  `n_trials: 1` (the default) it is simply that trial's score.
- **`trial_mode: serial` is required whenever the metric measures the
  machine** — wall-clock time, throughput, memory. Parallel trials share a
  CPU, so they measure each other. For seed variance, parallel is right and
  three times faster.
- **The accept band** is what stops the climb. A candidate becomes the new
  best only if it beats the incumbent by more than
  `max(min_improvement, noise_k x noise_floor)`, where the noise floor is the
  median per-candidate trial spread (MAD) the search has actually observed.
  Both default to `0`, which is the strict comparison. The band also gates the
  routing bandit's reward, so noise cannot train the model router either.
  Rejected near-misses are logged, not hidden:

```
c007 val=0.8123 beats c004 (0.8109) by less than the accept band (0.0042): within noise, not promoted
```

Measure before you tune: `hillclimb verify <problem> --repeat 5` runs the
verifier five times and reports the floor, with the settings to match.

```
5 runs: median 0.9738, spread 0.0822, noise floor (MAD) 0.0104
an improvement smaller than ~0.0208 cannot be told from noise. To stop the search climbing it:
  search:
    n_trials: 5
    noise_k: 2
```

## Operator scaffolds and model routing

Two prompt scaffolds sharpen the default operators (both on by default; the
`operators:` config block gates prompt injection only, so A/B arms record
identical data):

- **Retrieval-augmented draft** (`operators.draft_retrieval`) — the draft
  agent is told to web-search the current state of the art for the problem
  *class* before writing code (methods only — searching for solutions to the
  specific competition is explicitly forbidden).
- **Ablation-guided improve** (`operators.improve_ablation`) — the improve
  agent first attributes the score to the solution's components (fast,
  subsampled ablation runs, focused by the trial report's breakdown), records
  findings in `ablation.md`, then confines its ONE change to the
  highest-leverage component. Later improves of the same solution are handed
  the newest sibling `ablation.md` so components aren't re-measured.

The `routing:` block maps operators to backends/models; giving a route a
`models:` **pool** instead of a scalar turns model choice into a UCB1 bandit
(per operator) that learns which model earns improvements — rewards derive
from journaled results (improved on parent = 1, working-but-flat = 0.25,
buggy = 0), so bandit state rebuilds from journal replay and survives
`resume`:

```yaml
routing:
  improve: {models: [sonnet, opus-4.8]}   # bandit picks per call
  debug: {model: haiku}                   # scalar routes stay scalars
```

## Cross-search memory: the knowledge graph

hillclimb learns across searches, and the memory is file-based and
git-versionable — it lives in your hillclimb dir, under `knowledge/`:

- **Cards** (`knowledge/<family>/*.yaml`) — every finished search distills a
  statistical card (operator stats, top approaches, failure modes; no model
  calls) that future searches on the family receive as a "prior experience"
  prompt section. Concurrent searches in one run also share **live cards**
  mid-flight.
- **Claims** (`learning.claims`, default on) — after distilling the card, one
  cheap agent pass (routing key `distill`, default model haiku) turns the
  search into typed claims: `histgradientboosting helps` on this family,
  with confidence and candidate-id evidence. Claim subjects are canonical
  **entities** (`knowledge/entities.yaml`, alias-deduped) classified
  closed-set into a small curated **concept ontology**
  (`knowledge/concepts.yaml` — tabular / time-series / decision-trees /
  neural-networks / …; the agent may only *propose* additions, which you
  promote by flipping `proposed: false`).
- **Graph** (`knowledge/graph.json`) — a derived index rebuilt
  deterministically from the YAML (never hand-edit; `hillclimb knowledge
  rebuild` regenerates it, and it is gitignored). Every node/edge carries
  `first_seen`, claims gain `superseded_at` when a newer belief displaces
  them, so any historical view is a pure filter.
- **Retrieval** (`learning.graph_retrieval`, default on) — new searches also
  get the top graph-ranked claims: same-family first, then cross-family
  claims that share a concept with the problem.
- **Credit** (`learning.credit`, default on) — injected claims share the
  search's outcome (did it beat the best prior score on the problem?), so
  every claim accumulates a measured track record that adjusts its retrieval
  ranking; chronically failing claims retire. Memory that learns whether
  it's right.
- **Playbooks** (`learning.playbooks`, default on) — `hillclimb knowledge
  consolidate` is the sleep phase: multi-family claims generalize up the
  concept hierarchy, and each concept with enough evidence gets an
  agent-written playbook (`knowledge/playbooks/<concept>.md`, a reviewable
  git diff) that replaces the raw claims block in draft prompts; credit
  flows to the playbook's source claims.
- **Skills** (`learning.skills`, default on) — winning solutions are
  harvested into `knowledge/skills/` (2 best per family) and the best match
  lands in the next search's first draft as `reference_solution.py`: proven
  scaffolds, not prose hints.
- **Query tool** (`operators.knowledge_tool`, default on) — operator agents
  are told they can run `hillclimb knowledge query "<keywords>"` mid-search
  to consult the memory before re-deriving something expensive.
- **Does it help?** — an experiment with a memory-on and a memory-off arm
  (`learning.enabled: false`) answers it on holdout; see *Experiments* below.

Explore it interactively with `hillclimb knowledge graph` (or `g` inside
`hillclimb watch`): a true-3D scene rendered by [plotui](../plotui) (Rust
rasterizer; full-pixel Kitty graphics — kitty, Ghostty, iTerm2 ≥ 3.5, and
WezTerm are supported). Drag rotates, shift-drag pans, scroll zooms — and zoom
doubles as semantic level-of-detail: zoom out and entities fold into concept
supernodes. Every node type has its own marker (rings for problems and
families, triangles for searches, squares for libraries, diamonds for
techniques, open diamonds for claims) — the legend in the top-left corner
is the key, and each entry is a toggle: click it or press its number (1–8)
to hide that type. Click a node for the detail panel (re-click or Enter opens a
search's candidates), scrub through time search by search or change by change (`g` flips the
timeline between one tick per finished search and one per graph change —
per candidate, since claims are stamped with their evidencing candidate's
finish), filter and
color by concept from the sidebar. `?` slides out a panel with every key and
gesture — the footer carries only the few worth a permanent slot. Node
positions come from a 3D spring layout cached in graph.json (`pos3`; the 2D
`pos` stays for hillclimb-go).

![knowledge graph TUI](docs/graph-tui.png)

Design notes and rationale: `docs/memory-graph.md`.

## Local optimization demo suite

These problems are small, local, and require no download, so they are good for
exercising parallel searches in `hillclimb watch`:

| problem | objective |
|---|---|
| `circle-packing` | maximize total radius for 26 circles in a unit square |
| `heilbronn-11` | maximize the smallest triangle area among 11 points |
| `tsp-200` | minimize a 200-city Euclidean TSP tour |
| `labs-60` | minimize length-60 binary autocorrelation energy |

Start the suite, then open the TUI:

```bash
uv run hillclimb run problems/demo-suite.yaml --name "Optimization demo" --budget 10m
uv run hillclimb watch
```

## Commands

Search-addressing commands take `<run-id>/<search-id>`, a bare `<run-id>` (when
the run has a single search), or `latest` (the default).

| command | what it does |
|---|---|
| `problem get [circle-packing\|knapsack\|heilbronn-convex-13]` | copy a bundled problem into `hillclimb/problems/` (creates the `hillclimb/` dir if needed) and list its files (`fetch` is a deprecated alias) |
| `demo [--budget 10m] [--parallel-searches 3] [--parallel-operators 3]` | `problem get` + `run --parallel-searches` in one command |
| `init [dir]` | create the `hillclimb/` dir (config, problems/, specs/, runs/) with an example problem |
| `verify <problem> [--repeat N] [--holdout]` | run a problem's verifier once, outside a search; `--repeat` measures the noise floor |
| `run <target> [--name ...] [--budget 2h] [--backend ...] [--model ...]` | start a run for one problem or a suite YAML |
| `run <problem> --parallel-searches N --parallel-operators M` | N independent searches (detached engines, one run) each running M agents at once |
| `resume [search]` | continue a parked / stopped / crashed search |
| `status [search]` | search state + candidate tree (text) |
| `watch` | live TUI over runs, searches, and candidates |
| `chart` | live chart: best score so far by tested-candidate count across the problem's searches as a staircase, every scored candidate a dot (one line per arm in an experiment) |
| `graph` | the knowledge-graph TUI (same screen as `knowledge graph`) |
| `show [search] <candidate-id>` | everything about one candidate: scores, evaluation breakdown, diff vs parent, output |
| `ps` | every process hillclimb owns on this machine: engines with their agents and verifiers nested; `orphan` marks engines whose hillclimb dir was deleted |
| `stop [search] [--all]` | graceful stop: finish current operator, then park; `--all` also reaps orphaned engines when no hillclimb dir is found |
| `kill [search]` | SIGTERM the engine now (state finalized, resumable) |
| `prune <search> <candidate-id>` | cut a candidate and its subtree from the search |
| `tree [search]` | render the exploration tree to `<search>/tree.png` |
| `smoke [problem]` | one real agent call end-to-end (auth / contract check) |
| `knowledge graph [--stats]` | interactive knowledge-graph TUI (or a text summary) |
| `knowledge rebuild` | force-rebuild the derived `knowledge/graph.json` index |
| `knowledge distill [search] [--backfill]` | run the LLM claims pass on a search / all cards |
| `knowledge consolidate [--dry-run]` | sleep phase: generalize claims + rewrite playbooks |
| `knowledge query "<terms>" [--json]` | read-only memory lookup (also available to agents) |
| `knowledge show <target>` | the prior-experience section a new search would get |
| `run <problem> --set key=value … [--experiment E --arm A]` | any config setting, dotted; tag the search as an experiment arm |
| `experiment run <spec> [--repeats N] [--parallel] [--dry-run]` | every arm × problem × repeat of a spec |
| `experiment report [spec] [--problem X] [--control A] [--noise-floor F]` | compare the arms on holdout |

Exit code `2` from `run`/`resume` means the search parked or was stopped — resume it.

## Experiments: which setup wins?

Every knob — search policy, its params, the model, cross-search memory,
trial count — is a config setting, so "does X help?" is one experiment: a
problem × named *arms* (sets of config overrides) × N repeats. A spec lives
in `hillclimb/experiments/<name>.yaml`:

```yaml
problems: [circle-packing]
repeats: 3
budget: 15m
schedule: sequential        # sequential | parallel
noise_floor: 0.02           # from `hillclimb verify circle-packing --repeat 5`
arms:
  greedy:       {search.policy: greedy}             # first arm = the control
  greedy-nomem: {search.policy: greedy, learning.enabled: false}
  openevolve:   {search.policy: openevolve, search.policy_params: {population_size: 50}}
```

`hillclimb experiment run <name>` launches the matrix: sequentially by
default — repeat by repeat, arms round-robin inside, so shared state such as
the knowledge graph is seen by every arm at the same point (mandatory when an
arm touches memory) — or `--parallel` for stateless comparisons (policy,
model). Each search is a normal `hillclimb run … --experiment <name> --arm
<arm> --set key=value`, tagged in its `search.yaml` (`experiment`, `arm`,
`repeat`, `arm_overrides`), so a search you start by hand with those flags
counts too. `hillclimb experiment report <name>` compares the arms on the
selected candidate's holdout score (val when holdout is off): n / mean /
median / spread, best-of-repeat wins, minutes to best, tokens, and each arm's
paired gap to the control judged against the noise floor — a gap inside it
is reported as "within noise, not a result". `hillclimb chart` colours an
experiment's curves by arm.

A spec may name one shared executable seed — `seed_from: seeds/foo.py`,
resolved against the spec's directory — which rides `--seed-from` into every
child search, so arms are compared from identical source (the dry run prints
the resolved path and its sha256). Mandatory for engines that require a seed
(GEPA); see `hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml` for the
three-strategy comparison this shipped with.

## How runs and searches are laid out

```
runs/
└── <run-id>/                        # one hillclimb invocation
    ├── run.yaml                     # run metadata (schema_version: 2)
    ├── logs/                        # per-search engine logs (suite runs)
    └── searches/<search-id>/        # <problem-id>, then <problem-id>-2, -3 for more on one problem
        ├── search.yaml              # immutable search config (schema_version: 2)
        ├── status.json              # live heartbeat: state, pid, budget, current candidate
        ├── journal.jsonl            # append-only event log — the source of truth
        ├── control/                 # command queue (stop/prune) polled by the engine
        ├── best/                    # current selected submission (+ solution.py)
        └── candidates/<candidate-id>/  # one working dir per operator call
            ├── prompt.md
            ├── agent_stream.jsonl
            ├── solution.py
            └── submission.csv
```

**Single-writer rule:** only the engine process mutates search state. The TUI,
the CLI control commands, and chat agents all send commands through the store's
command queue (`control/` in the file backend), or apply them offline only when
the engine is provably not running. Never edit `journal.jsonl` or `status.json`
by hand.

### The DataStore (`store.backend`)

`run.yaml`, `search.yaml`, `journal.jsonl`, `status.json` and `control/` are
the *file backend's* representation of a search's records. The engine, the
CLI and the TUIs all read and write those records through one abstraction —
the **DataStore** (`src/hillclimb/store.py`) — and `hillclimb/config.yaml`
picks the backend:

```yaml
store:
  backend: files        # default — the folder above; nothing to set up, git-versionable
  # backend: sqlite     # one database file instead: hillclimb/store.sqlite
  # sqlite_path: hillclimb/store.sqlite
```

With `sqlite`, a search dir holds only what has to be files (`candidates/`,
`best/`, logs) and everything else lives in the database — cross-run views
(the chart, `store searches`, experiments) query it instead of walking run dirs,
and N concurrent engines (the demo) write it safely. The single-writer rule
is unchanged: the engine owns a search's records whichever backend holds
them; `stop`/`prune` go through the store's command queue.

`hillclimb store sync` imports the folder's searches into the configured
store (skipping ones it already has) — run it once after switching to
`sqlite` so earlier history shows up. `hillclimb store searches
[--problem KEY]` lists what the store holds.

A new backend implements the `DataStore` protocol: run/search metadata
(upsert), the journal (append-only, returned in append order — policies
replay it), one status record per search, and a consume-once command queue.
Candidate working dirs, `best/`, agent streams/logs, problems, knowledge YAML
and agent slots stay on the local filesystem in every backend — agents and
verifiers need real files.

Search states: `running` (fresh heartbeat + live pid) · `parked` (rate limit;
resume later) · `stopped` (user stop/kill) · `done` · `failed` (see
`last_error`) · `crashed` (derived: stale heartbeat or dead pid) · `unknown`
(no status.json yet).

## Pruning

Prune a branch that is overfitting or wasting budget — from the TUI (`x` on a
candidate), the CLI, or by asking your agent. Pruned candidates keep their status and
scores (shown grayed/struck), but the engine stops building on them and they are
excluded from selection; `best/` repoints immediately if the selected candidate was
pruned. The whole subtree goes with the candidate. The baseline (`c000`) cannot be
pruned.

## Development

```bash
uv run pytest        # test suite (fake backends, no agent calls)
uv run hillclimb smoke   # one real claude call: verifies auth + stream contract
```
