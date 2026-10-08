# Contributing to hillclimb

Thanks for taking a look. Bug reports, problems, climbers and fixes are all welcome.

<!-- TODO(Sebastian): a sentence or two on who maintains hillclimb at Rebase Energy and why -->

## Where to talk

- **Bugs and feature requests:** [GitHub issues](https://github.com/rebase-energy/hillclimb/issues).
  For a search that went wrong, include `hillclimb --version`, your OS, the coding agent
  (`--agent`), and what `hillclimb status <run-id>/<search-id>` shows.
- **Questions, results, ideas:** [the Discord](https://hillclimb.sh/discord).
- **Contact:** <!-- TODO(Sebastian): name + email -->

For anything bigger than a small fix, open an issue first so we can agree on the shape
before you write the code.

## Set up a clone

You need [uv](https://docs.astral.sh/uv/). It fetches Python 3.12 itself if your system
Python is older.

```bash
git clone https://github.com/rebase-energy/hillclimb
cd hillclimb
uv sync                 # dev environment, Python 3.12+
uv run pytest           # the test suite: fake coding agents, no LLM calls, no cost
uv run hillclimb --help
```

The suite runs with the sandbox off. `tests/test_sandbox.py` switches it on for itself.
`uv run hillclimb smoke` makes one real Claude call, to check auth and the stream contract.

To try a change end to end without spending anything, run a search with an agent that
needs no LLM:

```bash
uv run hillclimb run fitness-landscape --agent toy --climber climbers/greedy/policy.py \
    --budget 1m --no-detach
```

### Working on plotui at the same time

The terminal charts render through [plotui](https://pypi.org/project/plotui/),
which `uv sync` installs from PyPI. To work against a local checkout instead (it needs a
Rust toolchain):

```bash
uv pip install -e ../plotui
export UV_NO_SYNC=1     # otherwise `uv run` puts the PyPI wheel back
```

After you edit plotui's Rust source, run `uv pip install -e ../plotui` again: uv does not
notice `.rs` changes on its own.

## Before you open a pull request

- `uv run pytest` passes.
- Golden files: the CLI help screens (`tests/golden/help/`) and every prompt byte
  (`tests/golden/prompts/`) are pinned. If you meant to change them, regenerate with
  `HILLCLIMB_UPDATE_GOLDEN=1 uv run pytest tests/test_cli_help.py` (help) or
  `HILLCLIMB_UPDATE_GOLDENS=1 uv run pytest tests/test_prompt_golden.py` (prompts),
  and read the diff.
- New behaviour comes with a test, and a line in `CHANGELOG.md` under the unreleased
  version.

## How the code is laid out

[`AGENTS.md`](AGENTS.md) is the architecture reference. Coding agents read it, and it is
written for people too. Start with its package map. A few rules the tests enforce:

- `hillclimb.harness` (the fixed core) and `hillclimb.modules` (what a climber is built
  from) never import the TUI or the CLI (`tests/test_layout.py`).
- A climber module imports only `hillclimb.sdk` (`tests/test_sdk_imports.py`).
- The engine process is the only writer of a search's journal, status and `best/`. Never
  edit run state by hand: use `hillclimb stop | kill | prune | resume`.
- Some old names are banned (`tests/test_vocabulary.py`). The test tells you the new one.

## Adding a problem or a climber

- **A problem** is a folder with a verifier: `hillclimb problem new <id>` scaffolds one,
  and `hillclimb verify <id> --repeat 5` checks it and prints its noise floor. See
  [docs/problems.md](docs/problems.md). Problems in the repo-root `problems/` ship in the
  wheel only when they are listed in `catalog.PROBLEM_IDS`.
- **A climber** is a folder under `climbers/` holding a `policy.py`. See
  [docs/climbers.md](docs/climbers.md). `hillclimb climber check <path>` replays recorded
  journals against it without calling a coding agent.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
