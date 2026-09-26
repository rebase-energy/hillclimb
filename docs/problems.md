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

`hillclimb init` scaffolds a working `problems/example/`; `hillclimb problem
get <problem>` copies a bundled starter problem to edit.

Minimum `problem.yaml`:

```yaml
problem_id: my-problem
metric: my-score
higher_is_better: true
description: description.md
output_artifacts: [submission.csv]  # use [submission.json] for JSON-native tasks
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
unit_tests:                      # optional correctness gate, frozen at run start
  root: tests
  command: ["{python}", "-m", "pytest", "-q", "{tests}"]
data_dir: data
allow_network: false
```

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
exec "$HILLCLIMB_PYTHON" problem/verify.py  # writes $HILLCLIMB_RESULT
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

### Frozen unit tests (optional)

When `unit_tests` is declared, hillclimb snapshots the test tree and command
before any search worker in the run starts. Agents receive a disposable copy at `./unit_tests`,
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

The engine then spends verifier runs, not agent turns, on that code: the
climber's search policy proposes *tune* actions on promising candidates, each one a
new **trial** of the same `solution.py` with values from the tuner
(`random` by default, `optuna` with `pip install 'hillclimb[optuna]'`; the
climber manifest's `tuner`, overridden by `climber.tuner` in config), and
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
emflow problems emit one instance per scored origin, keyed `<asof>/<zone>`
— for GEFCom2014 that is every task x zone of the validation split (solar:
3 tasks x 3 plants = 9 instances) — so a GEPA arm on `emflow://gefcom2014:solar`
keeps a candidate that wins any single task. A candidate that leaves an
origin unscored simply lacks that key and is treated as having failed it;
MLE-bench per-fold instances are a planned follow-up.

## Noise: not climbing your own measurement error

A greedy search will happily spend a whole budget chasing a metric that moves
on its own. Three settings decide whether it can:

```yaml
evaluation:
  n_replicates: 5        # run each trial (parameter set) this many times
  replicate_mode: serial # `parallel` (default) | `serial`
  noise_k: 2             # a gain must beat 2x the measured noise floor
  min_improvement: 0.0   # ...or an absolute floor, in metric units
```

Concurrency is bounded machine-wide, not per search: `concurrency.parallel_operators`
is how many operators one search keeps in flight, and
`concurrency.machine_max_operators` (default `min(8, cores - 2)`, `0` = off) caps
the total across every search on the machine — extra operators wait
(`waiting-slot` in `hillclimb watch`). Every verifier and agent process gets
`OMP/OPENBLAS/MKL_NUM_THREADS=1` unless the parent environment sets them, so
N operators cost at most N cores; `hillclimb ps` shows what is actually running.

- **A trial's score is the MEDIAN of its replicates**, so one slow run or
  unlucky seed does not become the number the search ranks on. With
  `n_replicates: 1` (the default) it is simply that run's score. A candidate
  is scored by its best trial; seeds are never tuned.
- **`replicate_mode: serial` is required whenever the metric measures the
  machine** — wall-clock time, throughput, memory. Parallel replicates share
  a CPU, so they measure each other. For seed variance, parallel is right and
  three times faster.
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
