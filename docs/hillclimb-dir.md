# The hillclimb dir

Where hillclimb keeps its config, problems, specs, climbers and runs, how a run is laid out on disk, and which process may write to it.

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

`hillclimb climber new <name>` adds `hillclimb/climbers/<name>/` beside them
(see [climbers.md](climbers.md)).

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

### The budget is a gate, not a wall

A search stops *starting* operators once its budget is inside the stop
margin; by default (`budget.deadline: graceful`) whatever is still in flight
finishes and is committed, so a search can overrun by up to one operator. The
duration column in `hillclimb watch` keeps counting and says by how much:
`1h 04m 16s (budget: 1h, 4m 16s over)`. Pass `--set budget.deadline=hard` (or
set it in `config.yaml`) to cut in-flight operators off at the deadline
instead; they are journaled `abandoned` ("cut off at the budget deadline").

The clock is one dimension of the budget: `budget.max_evaluations` (verifier
trials the climber may spend), `budget.max_tokens` and `budget.max_cost_usd`
cap the others (`0` = no limit). There is no hidden cap on the number of
candidates.

## How runs and searches are laid out

```
runs/
└── <run-id>/                        # one hillclimb invocation
    ├── run.yaml                     # run metadata (schema_version: 3)
    ├── logs/                        # per-search engine logs (suite runs)
    └── searches/<search-id>/        # <problem-id>, then <problem-id>-2, -3 for more on one problem
        ├── search.yaml              # immutable search config (schema_version: 3)
        ├── status.json              # live heartbeat: state, pid, budget, current candidate
        ├── journal.jsonl            # append-only event log — the source of truth
        ├── control/                 # command queue (stop/prune) polled by the engine
        ├── climber/                 # the climber snapshotted at search start (resume loads this)
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
