# Problems

A problem is a folder, and a problem is its verifier: how to define one, what the verifier must do, and how not to climb your own measurement error.

## Defining problems

Users define new problems without changing Python code:

```
problems/my-problem/
├── problem.yaml
├── description.md
├── verifier.sh            # the contract: exit 0 = valid, write the score
└── data/                  # optional runtime inputs
```

`hillclimb problem new <problem>` writes a small problem that already runs (two-step:
`run.py` calls the solution's function, `score.py` scores it) for you to edit into your own;
`hillclimb problem get <problem>` copies one of the catalog's into `problems/` instead.

```bash
hillclimb problem new my-problem          # problems/my-problem/, a working example
hillclimb verify my-problem --repeat 3    # scores its baseline; run it after every edit
hillclimb run my-problem --budget 10m     # then climb
```

A problem scores its solutions in one of two shapes: a **verifier** (`verifier.sh`, which runs
the solution and then its scorer; below) or **two steps** (`score:`, and `run:` to call a function
the solution defines), where hillclimb runs the solution and the scorer in a sandbox each so the
scorer can keep data the solution never sees, and no bash is needed
([a scorer of its own](#private-data-a-scorer-of-its-own)).

Minimum `problem.yaml`:

```yaml
metric: my-score
higher_is_better: true             # true or false, unquoted
```

Everything else is optional:

```yaml
problem_id: my-problem           # default: the folder's name
description: description.md      # the default
verifier: verifier.sh            # the default
score: score.py                  # instead of a verifier: the scorer of a two-step problem
run: run.py                      # with score: the problem's own runner for the solution
time_limit_s: 60                 # with score: a solution that runs longer is a failed attempt
private: [hidden]                # with score: files only the scorer may read
holdout: true                    # engine also runs `verifier.sh --holdout`
holdout_inputs: [holdout]        # with score: inputs only the holdout split reads
contract: contract.md            # what solution.py must be/do (prompt section)
baseline: 0.5                    # scored at t=0 as the floor candidate; numeric values also become chart lines
                                 # or a file: `baseline: baseline.py`, scored at t=0 as c000
baseline_files:                  # or files the floor is scored on as they are
  submission.csv: sample_submission.csv
chart_baselines:                 # optional named horizontal lines in `hillclimb chart`
  previous best: 0.73
output_artifacts: [submission.csv]  # files of a solution `best/` and `summit` copy beside solution.py
time_budget_s: 900               # the default `hillclimb run --budget`
requirements: requirements.txt   # per-problem venv (default: a shared one with numpy, scipy, pandas, scikit-learn)
unit_tests:                      # optional correctness gate, frozen at run start
  root: tests
  command: ["{python}", "-m", "pytest", "-q", "{tests}"]
data_dir: data                   # default: data/ if the problem has one
allow_internet_during_solution: false
```

Files picked up by name when the problem folder has them: `description.md` (required),
`contract.md` (the interface `solution.py` must implement, added to every prompt), `interface.py`
(the output format as `hillclimb.spaces` objects: described in the prompt, checked against the
baseline by `hillclimb verify`), `requirements.txt` (with `requirements:`), `plot.py`
(`hillclimb plot`, below), `fingerprint.py` (a solution as a vector, for `hillclimb similarity`)
and `landscape.py` (the terrain `hillclimb surface` draws).

`allow_internet_during_solution` says whether `solution.py` may fetch
external data when the verifier runs it. The contract prompt tells the coding agent
which, and the [sandbox](sandbox.md) enforces it: without it the verifier,
and the solution it runs, have no network. A verifier that downloads
something itself needs it set to `true`. `allow_network` is its old name and
still loads. Whether the coding agents may use the internet while they write the
solution is `allow_internet_for_agents` in `hillclimb.yaml`.

`chart_baselines` accepts any number of `label: score` entries. Each becomes
a named horizontal reference line, in the order written. A numeric `baseline`
automatically adds the line named `baseline`; `chart_baselines` is only needed
for additional references. A file-based baseline is still evaluated as c000,
but cannot become a fixed reference line until its score is known.

### The verifier contract

The verifier is the scoring process the engine starts. It drives `solution.py`
itself — run it, import it, shell out to it — and reports the score:

```bash
#!/usr/bin/env bash
set -euo pipefail

"$HILLCLIMB_PYTHON" "$HILLCLIMB_SOLUTION"   # writes ./submission.csv

rm -f "$HILLCLIMB_RESULT"                   # only the scorer may score
"$HILLCLIMB_PYTHON" problem/verify.py       # writes $HILLCLIMB_RESULT
```

| | |
|---|---|
| exit 0 | the verifier accepted the candidate; declared unit tests must still pass |
| non-zero | the candidate is `buggy` and routes to the `debug` operator |
| `$HILLCLIMB_RESULT` | the score: `{"score": <float>, "report": {...}, <other numeric keys>}`, or a bare number |
| `$HILLCLIMB_PYTHON` | the managed runtime venv's interpreter (bare `python` resolves via PATH: wrong interpreter) |
| `$HILLCLIMB_SOLUTION` | the solution path for this run (replicate-dir aware) |
| `$HILLCLIMB_SPLIT` | `validation` or `holdout` |
| `$HILLCLIMB_REPLICATE_SEED` | set when the engine runs repeated replicates (also exported as the legacy `$HILLCLIMB_TRIAL_SEED`) |

The result file is both the score carrier and the completion proof: the engine
deletes it before every run, so a stale file can never fake success, and exit 0
without one is a contract violation rather than a silent zero. Reading the
score from a file rather than stdout is what keeps it honest — coding-agent-authored
code runs inside the verifier and shares its stdout.

**On Windows** the bundled problems need no bash: `hillclimb problem get`
writes a `verifier.py` there instead of `verifier.sh`, the same steps in
Python (`src/hillclimb/scaffold/windows_verifier.py`, or a problem's own), and the
engine runs it with its own interpreter. A problem that has only a
`verifier.sh` still runs on Windows, through Git for Windows' bash
(`HILLCLIMB_BASH` points at another). A `verifier.py` must run the solution as
a subprocess, never import it: in-process, coding agent code could reach the scorer.

The command runs with cwd = the candidate's working dir (`solution.py`, plus
`./problem/` and `./data/` symlinks). Validation runs get a
credential-scrubbed environment; `--holdout` runs in a directory coding agents never
see, with the full environment (private holdout data may be gated).

```bash
uv run hillclimb verify my-problem --repeat 5   # score it outside a search
```

`hillclimb verify` is the fastest way to check a new verifier, and `--repeat`
reports the spread between identical runs — an improvement smaller than that
is noise, not progress. The bundled `knapsack` and `circle-packing` problems
are the two reference shapes: a runner that calls the solution's function plus
a scorer with holdout inputs, and a verifier that runs the solution then scores
its output.

### Private data: a scorer of its own

A verifier runs `solution.py` inside its own process tree, so anything the verifier can read, the
solution can read too. Hidden labels, holdout instances or a secret seed next to the verifier are
not hidden. To keep data from the solution, give the problem a scorer instead of a verifier, and
name what only the scorer may read:

```yaml title="problems/my-problem/problem.yaml"
score: verify.py        # the scorer, instead of verifier.sh
private: [hidden]       # files or folders only the scorer reads
time_limit_s: 30        # optional: the engine stops a solution that runs longer
holdout: true           # the scorer is then also run with --holdout
```

hillclimb then runs two processes, each in a sandbox of its own:

1. **The solution**, `$HILLCLIMB_PYTHON solution.py`, in the candidate's working dir. It cannot
   read the `private:` paths, and it gets no credentials, not even on the holdout split. Anything
   it leaves running is stopped when it exits, and any result file it wrote is deleted.
2. **The scorer**, `verify.py` (or an argv list), in the same dir. It reads what the solution
   wrote and the private data, and writes the score to `$HILLCLIMB_RESULT`, exactly as a verifier
   would. On the holdout split it is passed `--holdout`.

Coding agents cannot read the `private:` paths either, and they are left out of the file list in
the agent's prompt. A `private:` path may sit inside the problem folder (`hidden`) or anywhere else
(an absolute path). Declaring `private:` without `score:` is an error. `hillclimb.Problem(score=...)`
problems always run in two steps, and take `private=[...]` for the same purpose.

The problem's folder and its data are read-only to agents and solutions in every problem, so a
solution can never rewrite the scorer that judges it.

Two more keys cover the other shapes:

- **`run: run.py`** replaces the plain `solution.py` step with the problem's own runner. It runs
  in the solution's sandbox, so it can import the solution and call a function it defines
  (`select_items(items, capacity)`), and writes what it got back for the scorer. The bundled
  `knapsack` problem works this way.
- **`holdout_inputs: [holdout]`** are inputs of the hidden split, such as the holdout instances or
  their seed. The solution and the scorer read them only when they run on the holdout split. Coding
  agents never read them, and neither does a validation run, whose output the agents see.

### Frozen unit tests (optional)

When `unit_tests` is declared, hillclimb snapshots the test tree and command
before any search worker in the run starts. Coding agents receive a disposable copy at `./unit_tests`,
but evaluation always uses the frozen bundle. Each parameter trial first runs
the verifier, then runs the suite once; score replicates continue only after
the suite passes. `{python}`, `{solution}`, and `{tests}` are available in the
argv command, and the problem's `requirements.txt` must install its runner.

Candidate statuses separate correctness from executability: `passing` means
the verifier and tests succeeded, `failing` means the verifier succeeded and
the suite completed with failing tests, and `buggy` means execution crashed,
timed out, or broke the evaluation contract. Failing and buggy candidates may be debugged, but only passing
candidates can rank, tune, reach holdout, or ship.

### Replicate metrics (optional)

Any other **numeric** key in the result object is journaled verbatim as the
replicate's `metrics` (`{"score": 12.3, "runtime_s": 0.8, "n_params": 40}`);
a trial's metrics are the per-key median of its replicates and a candidate's
are its best trial's — the same rule as the score. The engine never ranks on them; they are feature dimensions
for quality-diversity climbers (`openevolve`) and context for reports.
Strings, booleans and NaN are dropped silently.

### Replicate reports (optional)

`$HILLCLIMB_RESULT` may carry a `report` block alongside the score: any
verifier that writes one gets its breakdown stored on the replicate, rendered into improve prompts ("attack the largest contributors"),
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

- **directory problems** — the verifier writes the file (see
  `problems/circle-packing/verify.py`); a verifier that discards what the
  solution left behind gives its report evaluator trust.
- **self-reported problems** (MLE-bench) — the number is the coding agent's own
  claim, so the report is stored and rendered labelled *self-reported*.

Only `"split": "validation"` reports are ever fed back to operators — holdout
evaluations never produce one, by construction. `report.enabled: false` in
config disables prompt injection (data is still recorded).

### Tunable parameters (optional)

A solution may declare its numeric knobs in `params.json` next to
`solution.py` and read them through `spaces.params()`:

```json
{"restarts": {"type": "int", "low": 1, "high": 64, "log": true, "default": 8},
 "step":     {"type": "float", "low": 1e-4, "high": 0.1, "log": true, "default": 0.01},
 "init":     {"type": "categorical", "choices": ["grid", "random"], "default": "grid"}}
```

```python
from hillclimb import spaces
P = spaces.params()   # {"restarts": 8, "step": 0.01, "init": "grid"} — the trial's values, else the defaults
```

The engine then spends verifier runs, not coding agent turns, on that code: the
climber's search policy proposes *tune* actions on promising candidates, each one a
new **trial** of the same `solution.py` with values from the tuner
(`random` by default, `optuna` with `pip install 'hillclimb[optuna]'`; the
climber's block: `tuner`, `tuner_params`), and
the candidate is scored by its best trial. The
greedy climber's knobs live in its `params` (`climber.params` overrides
them): `tune_budget` (extra
trials per candidate, default 8, `0` disables), `tune_gate` (`band`: within
the accept band of the best; `best`; `always`), `tune_parallel`,
`tune_burst`. Holdout runs with the winning trial's values, `best/` ships
them as `params.json`, and a child improved from a tuned parent starts
from those values as its defaults. A malformed declaration is scored on the
solution's own defaults and reported in `hillclimb show`; seeds are never
tuned — replicate variance is the noise floor, not a knob.

### Per-instance scores (optional)

A third reserved key, `instances`, carries the breakdown of `score` over the
problem's sub-instances — zones, folds, test cases — in the same metric and
direction, with keys stable across the search:

```json
{"score": 2.158, "instances": {"circle-00": 0.083, "circle-01": 0.083}}
```

They are journaled per replicate (`Replicate.instance_scores`), aggregated per key by
median like everything else, and consumed by climbers whose selection is
per-instance — GEPA keeps a candidate alive if it wins on *any* instance, not
just on average. `problems/circle-packing/verify.py` (one instance per
circle) is the reference producer; verifiers that emit nothing lose nothing.
A candidate that leaves an instance unscored simply lacks that key and is
treated as having failed it; MLE-bench per-fold instances are a planned
follow-up.

### A picture of the solution (optional)

A problem whose answer is easier seen than read can ship `plot.py`, picked
up by default like `interface.py` — plain matplotlib. `hillclimb plot` (and
`hillclimb summit --plot`) draws a solution with it, saves a PNG and opens it:

```python
def plot(solution_dir, ax):       # a pathlib.Path, and matplotlib Axes
    rows = read_csv(solution_dir / "submission.csv")
    ax.scatter(xs, ys, label="point")
    ax.set_aspect("equal")
    return "11 points · smallest triangle 0.037037"   # optional caption, under the title
```

It reads what the solution *wrote* — never reruns `solution.py`, so a plot is
instant however long the search behind it ran — and it runs in the problem's
runtime venv, like the verifier, so it can use the problem's own packages;
hillclimb adds matplotlib there the first time a problem is plotted.
`problems/heilbronn-11/plot.py` (the numbered points, one smallest triangle
shaded) and `problems/circle-packing/plot.py` (the circles, shaded by radius)
are the references.

## Noise: not climbing your own measurement error

A greedy search will happily spend a whole budget chasing a metric that moves
on its own. Three settings decide whether it can:

```yaml
evaluation:
  n_replicates: 5        # run each trial (parameter set) this many times
  noise_k: 2             # a gain must beat 2x the measured noise floor
  min_improvement: 0.0   # ...or an absolute floor, in metric units
concurrency:
  parallel_replicates: 1 # of those, how many at once: 0 = all (default), 1 = one after another
```

Concurrency is bounded machine-wide, not per search: `concurrency.parallel_agents`
is how many coding agents one search keeps in flight, and
`concurrency.machine_max_agents` (default `min(8, cores - 2)`, `0` = off) caps
the total across every search on the machine — extra coding agents wait
(`waiting-slot` in `hillclimb watch`). Every verifier and coding agent process gets
`OMP/OPENBLAS/MKL_NUM_THREADS=1` unless the parent environment sets them, so
N coding agents cost at most N cores; `hillclimb ps` shows what is actually running.

- **A trial's score is the MEDIAN of its replicates**, so one slow run or
  unlucky seed does not become the number the search ranks on. With
  `n_replicates: 1` (the default) it is simply that run's score. A candidate
  is scored by its best trial; seeds are never tuned.
- **`concurrency.parallel_replicates: 1` is required whenever the metric
  measures the machine** — wall-clock time, throughput, memory. Replicates
  running at once share a CPU, so they measure each other. For seed
  variance, all at once (`0`, the default) is right and three times faster.
- **The accept band** is what stops the climb. A candidate becomes the new
  best only if it beats the incumbent by more than
  `max(min_improvement, noise_k x noise_floor)`, where the noise floor is the
  median within-trial replicate spread (MAD) the search has actually observed.
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
  evaluation:
    n_replicates: 5
    noise_k: 2
```
