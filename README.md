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
   detail panel for notes, scores, lineage, output, and agent stream when present.
   Drag the divider or use `+` / `-` to resize the detail panel.

## Install

```bash
uv sync
claude login   # operator calls bill your Claude subscription
```

## Quickstart

```bash
uv run hillclimb run problems/circle-packing --budget 10m
uv run hillclimb watch                                  # live dashboard (2nd terminal)
uv run hillclimb status                                 # or: plain-text status of the latest search
```

Try the engine without spending agent calls: `--backend dummy`.

## Workspaces

All hillclimb data lives in one `hillclimb/` folder inside your project, so it
never mingles with the rest of the repo. `hillclimb init` creates it:

```
my-project/
└── hillclimb/
    ├── config.yaml     # workspace defaults + the workspace marker
    ├── problems/       # problem definitions
    ├── specs/          # committed run specs (versioned run parameters)
    └── runs/           # search artifacts (gitignored by init)
```

Commands work from any subdirectory — the workspace is found by upward search
for `hillclimb/config.yaml` (like git). Without one, commands error and point
you at `hillclimb init`; `HILLCLIMB_WORKSPACE` pins the root explicitly.

Config precedence, highest first: CLI flags → workspace `hillclimb/config.yaml`
→ user `~/.config/hillclimb/config.yaml` → built-in defaults.

Machine-scoped state is shared across workspaces under `~/.cache/hillclimb/`
(honors `XDG_CACHE_HOME`; `HILLCLIMB_CACHE_DIR` overrides): solution-runtime
venvs keyed by a hash of their requirements (rebuilt automatically when
requirements change), the emflow problem cache, and the cross-search agent
semaphore. Pre-workspace checkouts left `.runtime-venv*/` and `cache/` in the
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
    parallel_agents: 3
  - target: emflow://gefcom2014:wind
    budget: 1h
```

```bash
uv run hillclimb run hillclimb/specs/gefcom.yaml            # exactly as committed
uv run hillclimb run hillclimb/specs/gefcom.yaml --model sonnet   # ad-hoc override
```

A spec with a single top-level `target:` (plus the same parameter keys) runs
one search. `run.yaml` records which spec launched the run.

## Semantics

The UI and on-disk metadata use this hierarchy, coarse to fine:

```text
Run
└── Search
    └── Candidate
        └── Trial
```

- **Problem**: reusable definition under `problems/<id>/`.
- **Run**: one invocation of hillclimb. A single-problem run contains one
  search; a suite run contains one search per problem.
- **Search**: one search worker (engine process) exploring one problem.
- **Candidate**: an immutable code artifact produced by an operator. Any change
  to the code — however small — is a new candidate with a new id.
- **Trial**: one execution of a candidate with a fixed parameterization
  (params, seed). Today the engine runs exactly one trial per candidate; the
  schema supports several so re-evaluations and parameter tuning can land
  without another migration.

`hillclimb watch` opens on the Runs screen. Metadata carries
`schema_version: 2`; directories from the pre-v2 flat layout are ignored.

## emflow problems (optional extra)

With the `emflow` extra installed (`pip install 'hillclimb[emflow]'`),
targets of the form `emflow://<name>` run problems from
[emflow](https://github.com/rebase-energy/emflow)'s registry — agents author
`Predictor` classes (`solution.py` exposing `get_model()`), a generic
evaluator fits and scores them on the problem's validation split, and the
hidden holdout is a second evaluator run. A bare package name is a virtual
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

A problem is a folder. Users define new problems without changing Python code:

```
problems/my-problem/
├── problem.yaml
├── description.md
├── verify.py
├── sample_submission.csv
└── data/                  # optional runtime inputs
```

Minimum `problem.yaml`:

```yaml
problem_id: my-problem
metric: my-score
lower_is_better: false
description: description.md
sample_submission: sample_submission.csv
verifier: verify.py
time_budget_s: 900
```

Generated `solution.py` writes `submission.csv`. The orchestrator then runs the
problem verifier and parses its final `val_score: <number>` line.

### Evaluator problems (`kind: evaluator`)

When "write a submission CSV and grade it" doesn't fit — the solution is a
module the evaluator drives, a program to benchmark, a policy to simulate —
the problem can own its whole evaluation:

```yaml
problem_id: bin-packing
kind: evaluator
metric: mean-bins
lower_is_better: true
description: description.md
contract: contract.md            # what solution.py must be/do (prompt section)
eval: "{python} problem/evaluate.py"                # the ONLY validation process
holdout_eval: "{python} problem/evaluate.py --holdout"  # optional; hidden dir, full env
requirements: requirements.txt   # optional; per-problem venv (default: shared csv venv)
baseline: baseline.py            # optional; scored at t=0 as the floor candidate
time_budget_s: 900
```

The eval command runs with cwd = the candidate workspace (`solution.py`,
`./problem/` and `./data/` symlinks present), must print `val_score: <float>`
as its final stdout line, and must write `eval_result.json`
(`{"split": "validation", "score": ..., "report": {...}}`) — the completion
proof, and the carrier for the trial report above (evaluator-trusted).
Placeholders: `{python}` → the managed runtime venv's interpreter (always use
it — bare `python` resolves via PATH), `{solution}` → the solution path.
Validation runs get a credential-scrubbed environment and
`HILLCLIMB_TRIAL_SEED`; `holdout_eval` runs in a directory agents never see
with the full environment. `problems/bin-packing/` is the reference example.

### Trial reports (optional)

`eval_result.json` is hillclimb's evaluator report contract: any evaluation
that writes it into the working directory gets its breakdown stored on the
trial, rendered into improve prompts ("attack the largest contributors"),
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
- **problems with a `verifier:`** — the verifier may write the file (see
  `problems/circle-packing/verify.py`); anything the solution itself wrote is
  discarded before the verifier runs, so the report carries evaluator trust.
- **verifier-less problems** — the agent's own script may write it (the
  contract invites this); it is stored and rendered labelled *self-reported*.

Only `"split": "validation"` reports are ever fed back to operators — holdout
evaluations never produce one, by construction. `report.enabled: false` in
config disables prompt injection (data is still recorded).

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
| `run <target> [--name ...] [--budget 2h] [--backend ...] [--model ...]` | start a run for one problem or a suite YAML |
| `resume [search]` | continue a parked / stopped / crashed search |
| `status [search]` | search state + candidate tree (text) |
| `watch` | live TUI over runs, searches, and candidates |
| `stop [search]` | graceful stop: finish current operator, then park |
| `kill [search]` | SIGTERM the engine now (state finalized, resumable) |
| `prune <search> <candidate-id>` | cut a candidate and its subtree from the search |
| `tree [search]` | render the exploration tree to `<search>/tree.png` |
| `smoke [problem]` | one real agent call end-to-end (auth / contract check) |

Exit code `2` from `run`/`resume` means the search parked or was stopped — resume it.

## How runs and searches are laid out

```
runs/
└── <run-id>/                        # one hillclimb invocation
    ├── run.yaml                     # run metadata (schema_version: 2)
    ├── logs/                        # per-search engine logs (suite runs)
    └── searches/<search-id>/        # one search per problem
        ├── search.yaml              # immutable search config (schema_version: 2)
        ├── status.json              # live heartbeat: state, pid, budget, current candidate
        ├── journal.jsonl            # append-only event log — the source of truth
        ├── control/                 # command queue (stop/prune) polled by the engine
        ├── best/                    # current selected submission (+ solution.py)
        └── candidates/<candidate-id>/  # one workspace per operator call
            ├── prompt.md
            ├── agent_stream.jsonl
            ├── solution.py
            └── submission.csv
```

**Single-writer rule:** only the engine process mutates search state. The TUI,
the CLI control commands, and chat agents all send commands through `control/`
(or apply them offline only when the engine is provably not running). Never edit
`journal.jsonl` or `status.json` by hand.

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
