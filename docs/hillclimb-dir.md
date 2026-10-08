# The hillclimb dir

Where hillclimb keeps its config, problems, climbers and runs, how a run is laid out on disk (its own spec included), what git tracks, and which process may write to it.

## The hillclimb dir

The hillclimb dir is any folder with a `hillclimb.yaml` in it. `hillclimb
init` makes the current folder one:

```
my-project/
├── hillclimb.yaml      # defaults, and the marker that makes this the hillclimb dir
├── problems/           # problem definitions
├── runs/               # one folder per run: its spec, its records, its artifacts
└── climbers/           # your own climbers, empty until `hillclimb climber get`
```

`hillclimb init DIR` makes `DIR` (created if missing) the hillclimb dir
instead — `hillclimb init hillclimb` keeps everything in a `hillclimb/`
subfolder of a repo. When a folder already has a `problems/`, `runs/`,
`knowledge/` or `climbers/` of its own, `init` does that by itself: hillclimb
keeps to `./hillclimb/`, and every command finds it from anywhere in the
project. `hillclimb problem get` in a folder with no hillclimb dir offers to
make one, by the same rule.

Every folder hillclimb creates carries a hidden `.hillclimb` file. That is
what marks it as hillclimb's: a folder of the same name without one is yours,
and hillclimb never deletes it.

`hillclimb climber get greedy` adds `climbers/greedy/` beside them — the
default climber as a folder to read and edit, its whole policy in
`policy.py` — and `hillclimb climber new <name>` a one-file
`climbers/<name>.py` (see [climbers.md](climbers.md));
learning writes `knowledge/`, experiments live in `experiments/`.

Commands run from the hillclimb dir's root: the folder holding
`hillclimb.yaml`, or a project root whose `hillclimb/` subfolder holds one.
There is no upward search — a folder further up the tree that happens to hold
a `hillclimb.yaml` (a hillclimb checkout beside your project, say) is never
taken for yours. From anywhere else, commands error and point you at
`hillclimb init`; `HILLCLIMB_DIR` pins it explicitly.

Config precedence, highest first: CLI flags → the hillclimb dir's
`hillclimb.yaml` → user `~/.config/hillclimb/config.yaml` → built-in defaults.

`hillclimb reset` stops this dir's engines and deletes what hillclimb made in
it — `hillclimb.yaml`, the sqlite store, and the folders carrying the
`.hillclimb` marker (`problems/`, `runs/`, `knowledge/`, `climbers/`, and a
`hillclimb/` subfolder it created) — and nothing else, since the hillclimb dir
may be your repo's root; `experiments/` with your study specs stays. Without
`--yes` it lists what it will delete and what it keeps, then asks.

Machine-scoped state is shared across hillclimb dirs under `~/.cache/hillclimb/`
(honors `XDG_CACHE_HOME`; `HILLCLIMB_CACHE_DIR` overrides): solution-runtime
venvs keyed by a hash of their requirements (rebuilt automatically when
requirements change), benchmark providers' caches, and the cross-search coding agent
semaphore. Checkouts predating this layout left `.runtime-venv*/` and `cache/` in the
project dir — safe to delete.

### Every run carries its spec

Each run writes `runs/<run-id>/spec.yaml` next to `run.yaml`: one `problems:`
entry per search it launched, with every parameter the launch resolved to —
target, budget, coding agent, model, the climber (its full `climber:` block),
parallelism, replicates, the seed (as an absolute path) and any `--set`
overrides. It is generated from what
actually ran, whether the run came from the CLI, a spec file, a fleet or an
experiment (the header names the file it was launched from). So the recipe
lives with the record and the artifacts, git explains every run, and

```bash
uv run hillclimb run runs/<run-id>/spec.yaml                    # exactly as it ran
uv run hillclimb run runs/<run-id>/spec.yaml --model sonnet     # ad-hoc override
```

runs it again. The same format is a run spec you can write by hand and commit
anywhere in the repo: a `problems:` list of targets (strings) or per-entry
parameter dicts — `target`, `name`, `budget` (`2h` / `30m` / seconds),
`agent`, `model`, `climber`, `parallel_agents`, `n_replicates`,
`seed_from` (relative to the spec file) and `set` (a list of `key=value`
overrides) — or the single-search form with a top-level `target:` plus the
same keys. CLI flags override a spec's values.

`climber` is where the search's climber is **defined**: the block (a policy
or loop, its selector, operators, tuner and memory — see
[climbers.md](climbers.md)), or a .py file / climber folder. A `climber:` at the top of
the spec, beside `problems:`, is the default for entries that name none; an
entry without one in a spec without one uses the folder's, from
`hillclimb.yaml`. File refs in a block are relative to the spec file.

```yaml
climber: {operator_policy: greedy, params: {num_drafts: 5}}      # for every entry below that names none
problems:
  - heilbronn-11
  - target: heilbronn-14
    budget: 1h
    climber: {operator_policy: greedy, selector_policy: map-elites, tuner: optuna}
```

### What git tracks

`hillclimb init` adds rules to the hillclimb dir's `.gitignore` (anchored
with a leading `/`, so they hold whether the dir is a repo's root or a
subfolder) so that the *record* of every run is committed and its *bulk* is
not:

- committed: `run.yaml`, `spec.yaml`, each search's `search.yaml`,
  `journal.jsonl`, `status.json`, `knowledge_card.yaml`, the `climber/`
  snapshot, and `best/solution.py` + `best/params.json` — enough for `git log`
  to explain every run and for `hillclimb chart` to work on a fresh clone;
- ignored: `candidates/` (coding agent streams, replicate outputs, runtime data),
  the run's `logs/`, the `control/` queue, the rest of `best/` (a submission
  can be large), `store.sqlite`, the derived `knowledge/graph.json`, and
  `.env` (keys, never).

A `.gitignore` that already ignores `runs/` as a whole keeps doing so; delete
that line to get the finer rules.

### The budget is a gate, not a wall

A search stops *starting* operators once its budget is inside the stop
margin; by default (`budget.deadline: graceful`) whatever is still in flight
finishes and is committed, so a search can overrun by up to one operator. The
duration column in `hillclimb watch` keeps counting and says by how much:
`1h 04m 16s (budget: 1h, 4m 16s over)`. Pass `--set budget.deadline=hard` (or
set it in `hillclimb.yaml`) to cut in-flight operators off at the deadline
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
    ├── spec.yaml                    # the run's own spec: rerun it with `hillclimb run <this file>`
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
the **DataStore** (`src/hillclimb/harness/store.py`) — and `hillclimb.yaml`
picks the backend:

```yaml
store:
  backend: files        # default — the folder above; nothing to set up, git-versionable
  # backend: sqlite     # one database file instead: store.sqlite
  # sqlite_path: store.sqlite
```

`hillclimb init --datastore sqlite` writes that block for a new folder, so the
database holds its records from the first run on.

With `sqlite`, a search dir holds only what has to be files (`candidates/`,
`best/`, logs) and everything else lives in the database — cross-run views
(the chart, `store searches`, experiments) query it instead of walking run dirs,
and N concurrent engines (`--parallel-searches`) write it safely. The single-writer rule
is unchanged: the engine owns a search's records whichever backend holds
them; `stop`/`prune` go through the store's command queue.

`hillclimb store sync` imports the folder's searches into the configured
store (skipping ones it already has) — run it once after switching to
`sqlite` so earlier history shows up. `hillclimb store searches
[--problem KEY]` lists what the store holds.

A new backend implements the `DataStore` protocol: run/search metadata
(upsert), the journal (append-only, returned in append order — policies
replay it), one status record per search, and a consume-once command queue.
Candidate working dirs, `best/`, coding agent streams/logs, problems, knowledge YAML
and coding agent slots stay on the local filesystem in every backend — coding agents and
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
