# Package layout: `hillclimb.tui`, `hillclimb.modules`, `hillclimb.harness`

Status: DONE 2026-09-26, all five phases, on the `layout` branch (one commit each).
Pure moves and renames, no behaviour change. `tests/test_layout.py` enforces the rules.

## Why

`src/hillclimb/` is one flat package of seventy modules. The TUIs and their
layout helpers are about 14,000 of its 39,000 lines and sit beside the seams
a researcher needs: `tree.py` next to `treeview.py`, `chart.py`, `watch.py`,
`archive.py`. Nothing in the core imports a view, so the separation exists in
dependency; it does not exist in layout, and layout is what a person opening
the folder sees. `cli.py` is 3,750 lines and is the first file they open.

## Target

```
src/hillclimb/
  __init__.py            public API: run_search, Config, register_benchmark_provider
  sdk/                   the one import a module needs (unchanged)
  spaces.py              stays top-level: byte-copied into runtime venvs as hillclimb/spaces.py
  config.py problem.py api.py project.py benchmark_providers.py   the public surface, unchanged

  harness/               the fixed core — one process, one journal writer, no policy
    core.py              Harness (unchanged)
    loop.py              SearchLoop / PolicyLoop  (the control-flow contract the harness runs)
    evaluation.py executor.py unit_tests.py baseline.py   scoring
    candidate.py journal.py store.py run.py status.py     records
    budget.py slots.py control.py orphans.py quota.py pricing.py routing.py   spend + concurrency
    dirs.py params.py direction.py grading.py
    glue.py              today's search_strategy.py: config -> climber -> loop/operators/tuner

  modules/               everything a climber exchanges; implementations import only hillclimb.sdk
    policies/  base.py (SearchPolicy, Action, PolicyInput — today's policy.py)  greedy.py  openevolve.py  check.py (policy_check.py)
    operators/ base.py  builtin.py
    tuners/    base.py (today's tuner.py)  random_search.py  optuna.py
    similarity/ base.py  builtin.py  compute.py  solution_card.py  solution_card.md
    memory/    knowledge.py claims.py credit.py consolidate.py graph.py papers.py skills.py
               (the built-in Memory; the ABC comes with the Memory phase, not this one)
    # later, same shape: archives/  feedback/  routers/

  climbers/              bundled manifests (unchanged; module refs updated)
  backends/ integrations/ prompts/ runtime/ demo/   unchanged

  tui/                   every terminal view and the pure layout it draws
    watch.py chart.py graphview.py ganttview.py gantt.py
    tree.py treeview.py tree2.py tree2view.py archive.py archiveview.py
    surface.py surfaceview.py similarity.py similarityview.py similarity_map.py similarity_mapview.py
    header.py theme.py keys.py intro.py
    (viz.py is deleted: its only caller was the image `tree` command, already removed)

  cli/                   `hillclimb` split by command group
    __init__.py          the typer app, `main`, the banner, HillclimbGroup, `say`/`legend`/`next_steps`
    common.py            load_config, parse_budget, _parse_set, open_search — the helpers commands share
    run.py problem.py climber.py connect.py views.py store.py experiment.py knowledge.py control.py
```

What stays top-level on purpose: `spaces.py` (the runtime shim copies it
verbatim and imports it as `hillclimb.spaces`), the public surface
(`config`, `problem`, `api`, `project`, `benchmark_providers`), and `demo/`
(package data; the wheel and `problem get` address it by that name).

## Import-direction rules (enforced by `tests/test_layout.py`, written first)

- `modules/*` implementations import `hillclimb.sdk` and nothing else from
  hillclimb (today's `test_sdk_imports.py`, re-pointed; its `ALLOWED` list
  only shrinks).
- `harness/*` and `modules/*` never import `tui` or `cli`.
- `tui/*` never imports `cli`.
- `sdk` re-exports lazily from `harness/` and `modules/*/base.py`; nothing
  imports `sdk` eagerly at module top level except module implementations.

## What breaks, and the one compatibility table

Nothing is public yet, so no import shims for old paths. Two things persist
across the move and need care:

1. **Module refs in run folders.** A search snapshot's `climber.yaml` and
   `SearchMeta.climber_manifest` carry refs like
   `hillclimb.policies.greedy:GreedyPolicy`. `climber._resolve` gets a
   `MOVED` prefix table (`hillclimb.policies.` → `hillclimb.modules.policies.`,
   `hillclimb.integrations.gepa.` unchanged, …) exactly like
   `config.LEGACY_SETTINGS`, so every recorded search still resumes. One
   test resumes a v3 folder with an old ref.
2. **Test monkeypatches on `cli`.** `tests/test_cli.py` patches
   `cli.load_config`, `cli.run_fleet` and friends on the `hillclimb.cli`
   module. After the split, command modules must reach those names through
   `cli.common` at call time (`common.load_config()`, not
   `from .common import load_config`), so a patch on `common` reaches every
   command. This is the one design constraint the split imposes; tests then
   patch `hillclimb.cli.common`.

Not affected: the journal (JSON, no module paths), similarity caches (keyed
by score name + version), prompt goldens (bytes of templates, no paths),
the store backends, run-folder layout, every config key, every CLI flag.

## Phases — one commit each, suite green at every step

| # | Move | Risk | Size | Notes |
|---|---|---|---|---|
| 0 | `tests/test_layout.py` with the rules above, passing vacuously | none | small | The rules exist before the moves |
| 1 | `tui/` | low | 19 files, ~80 import lines, mostly `cli.py` + tests | Biggest readability win; safe before launch |
| 2 | `modules/` (+ `MOVED` table, manifests, sdk map, `test_sdk_imports` paths) | medium | 4 packages + 4 contract files + memory's 7 | The rename researchers will notice; docs reference these paths |
| 3 | `harness/` | medium | ~22 files, imports across the whole tree | Mechanical; `git mv` keeps history |
| 4 | `cli/` split | high (tests) | 3,750 lines → ~9 files | Last; needs the `common` discipline; optional for launch |
| 5 | docs: README paths, `docs/development.md` package map, CLAUDE.md map, the docs site | low | prose | Every phase updates the lines it touches; this is the sweep |

Each phase: `git mv` the files, fix imports (`ruff --select F401,F811` plus
the suite finds the rest), run the full suite with
`timeout 900 uv run pytest -q -p no:cacheprovider`, commit.

## Order and timing

Phase 1 before launch if time allows: low risk, and it is the change a
first-time reader of the repo benefits from most. Phases 2–5 after launch,
in order, because phase 2 changes the paths the README and the docs site
name, and phase 4 is the only one that reshapes tests rather than
re-pointing them.

## Estimates

Phase 1 one to two hours; phase 2 two to three; phase 3 two; phase 4 three
to four; phase 5 one. About a day and a half of focused work, spread over
the commits above.
