# Development

Running the test suite and the one real coding agent call that checks the stream contract.

```bash
uv sync                  # editable install, Python 3.12+ (plotui from PyPI; CONTRIBUTING.md covers a local checkout)
uv run pytest            # test suite (fake coding agents, no coding agent calls)
uv run hillclimb smoke   # one real claude call: verifies auth + stream contract
```

## Package map

```
src/hillclimb/
  harness/   the fixed core: core.py, loop.py, glue.py (config -> climber -> loop),
             evaluation, executor, unit_tests, baseline, candidate, journal, store, run,
             status, budget, slots, control, orphans, quota, pricing, routing, dirs, params
  modules/   what a climber exchanges — policies/, operators/, tuners/, similarity/, memory/;
             each kind's contract is its base.py; implementations import only hillclimb.sdk
  tui/       every terminal view and the layout it draws; never imported by harness/ or modules/
  cli/       one module per command group; common.py is what commands share (call it as
             common.x(), so a test's patch reaches every command); __main__.py for the engine children
  sdk/       the one import a climber needs
  api.py config.py problem.py project.py benchmark_providers.py climber.py experiment.py connect.py
             the public surface; spaces.py stays here because the runtime shim byte-copies it
  demo/ agents/ providers/ prompts/ runtime/ climbers/
```

The import directions — harness and modules never import the tui or the cli, the tui
never imports the cli, module implementations import only the sdk, the package inits
under harness/ and modules/ import nothing — are pinned by `tests/test_layout.py`.
A `module:Class` ref written before the 2026-09 layout move (a search snapshot, a
`similarity:` config entry) still resolves through `hillclimb/_moved.py`.
