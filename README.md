# rebase-hillclimb

Auto-hillclimbing for verifier-defined problems. A greedy search engine spawns headless coding
agents (Claude Code) as operators — **draft** new solutions, **debug** failures,
**improve** the best one, **ensemble** at the end — executes every candidate in a
sandboxed venv, scores it against a hidden holdout, and keeps the best submission
in `runs/<id>/best/`.

The design is deliberately three-layered:

1. **`hillclimb` CLI** — the headless engine. Scriptable, plain output, meaningful
   exit codes. All run state lives on disk.
2. **Your interactive agent** (Claude Code) is the front door: the repo ships a
   skill (`.claude/skills/hillclimb/SKILL.md`) that teaches it to start, monitor,
   and control runs in the background while you chat.
3. **`hillclimb watch`** — a live TUI you keep open beside the agent:
   experiments → problem runs → candidate trees, with an on-demand candidate
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
uv run hillclimb status                                 # or: plain-text status of the latest run
```

Try the engine without spending agent calls: `--backend dummy`.

## Semantics

The UI and on-disk metadata use this hierarchy:

```text
Experiment
└── Problem Run
    └── Candidate tree
```

- **Problem**: reusable definition under `problems/<id>/`.
- **Experiment**: one user-started session. A single-problem run still creates an
  experiment containing one problem run.
- **Problem Run**: one search process for one problem.
- **Candidate**: one evaluated solution in a problem run's tree.

`hillclimb watch` opens on the Experiments screen. Existing older runs that lack
experiment metadata are grouped under a synthetic `Legacy` experiment.

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

## Local optimization demo suite

These problems are small, local, and require no download, so they are good for
exercising parallel runs in `hillclimb watch`:

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

| command | what it does |
|---|---|
| `run <target> [--name ...] [--budget 2h] [--backend ...] [--model ...]` | start an experiment for one problem or a suite YAML |
| `resume [run-id]` | continue a parked / stopped / crashed run |
| `status [run-id]` | run state + candidate tree (text) |
| `watch` | live TUI over experiments, problem runs, and candidates |
| `stop [run-id]` | graceful stop: finish current operator, then park |
| `kill [run-id]` | SIGTERM the engine now (state finalized, resumable) |
| `prune <run-id> <node-id>` | cut a candidate and its subtree from the search |
| `tree [run-id]` | render the exploration tree to `runs/<id>/tree.png` |
| `smoke [problem]` | one real agent call end-to-end (auth / contract check) |

Exit code `2` from `run`/`resume` means the run parked or was stopped — resume it.

## How experiments and runs are laid out

```
runs/
├── experiments/<experiment-id>.yaml # experiment metadata used by the TUI
└── <run-id>/                       # one problem run
    ├── run.yaml                    # immutable run config
    ├── status.json                 # live heartbeat: state, pid, budget, current candidate
    ├── journal.jsonl               # append-only event log — the source of truth
    ├── control/                    # command queue (stop/prune) polled by the engine
    ├── best/                       # current selected submission (+ solution.py)
    └── nodes/<node-id>/            # one candidate workspace per operator call
        ├── prompt.md
        ├── agent_stream.jsonl
        ├── solution.py
        └── submission.csv
```

**Single-writer rule:** only the engine process mutates run state. The TUI, the
CLI control commands, and chat agents all send commands through `control/` (or
apply them offline only when the engine is provably not running). Never edit
`journal.jsonl` or `status.json` by hand.

Run states: `running` (fresh heartbeat + live pid) · `parked` (rate limit; resume
later) · `stopped` (user stop/kill) · `done` · `failed` (see `last_error`) ·
`crashed` (derived: stale heartbeat or dead pid) · `unknown` (legacy run).

## Pruning

Prune a branch that is overfitting or wasting budget — from the TUI (`x` on a
candidate), the CLI, or by asking your agent. Pruned candidates keep their status and
scores (shown grayed/struck), but the engine stops building on them and they are
excluded from selection; `best/` repoints immediately if the selected candidate was
pruned. The whole subtree goes with the candidate. The baseline (`n000`) cannot be
pruned.

## Development

```bash
uv run pytest        # test suite (fake backends, no agent calls)
uv run hillclimb smoke   # one real claude call: verifies auth + stream contract
```
