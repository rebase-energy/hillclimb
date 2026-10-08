<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rebase-energy/hillclimb/main/docs/assets/hillclimb-logo.svg">
  <img alt="hillclimb" src="https://raw.githubusercontent.com/rebase-energy/hillclimb/main/docs/assets/hillclimb-logo-light.svg" width="620">
</picture>

**Hillclimbing on verifiable rewards.**<br>
Use Claude Code and Codex to autonomously search for python programs that optimize a score you define.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rebase-energy/hillclimb/main/docs/assets/hillclimb-mountain-dark.png">
  <img alt="An ASCII mountain: one climber on the summit under a gold star, another stuck on a lower peak" src="https://raw.githubusercontent.com/rebase-energy/hillclimb/main/docs/assets/hillclimb-mountain-light.png" width="560">
</picture>

[Website](https://hillclimb.sh?utm_medium=readme) ·
[Docs](https://docs.hillclimb.sh?utm_medium=readme) ·
[Quickstart](https://docs.hillclimb.sh/quickstart?utm_medium=readme) ·
[Python SDK](https://docs.hillclimb.sh/python?utm_medium=readme) ·
[CLI reference](https://docs.hillclimb.sh/cli?utm_medium=readme) ·
[Blog](https://docs.hillclimb.sh/blogposts?utm_medium=readme) ·
[Discord](https://hillclimb.sh/discord?utm_medium=readme)

[![PyPI](https://img.shields.io/pypi/v/hillclimb.svg)](https://pypi.org/project/hillclimb/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/hillclimb/)
[![Quickstart](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml/badge.svg)](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rebase-energy/hillclimb/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/discord-join%20chat-5865F2.svg?logo=discord&logoColor=white)](https://hillclimb.sh/discord?utm_medium=readme)

</div>

---

`hillclimb` provides a modular harness, tooling and testbed for exploring and
benchmarking autoresearch algorithms. The aim is to provide a tool to discover better
autoresearch methods, rather than build a single best autoresearcher.

You provide a verifier script, `verifier.sh`, and (optionally) a starting
solution, `solution.py`. `hillclimb` then spends a compute budget running a
search policy — a [climber](https://docs.hillclimb.sh/python/climber?utm_medium=readme) — that
decides which version to build on next and dispatches headless coding agents
to write, debug and improve it. Each version is scored through your verifier,
and `hillclimb` keeps the one that scores best.

The harness is fixed. The **climber** — what to try next and how each attempt
is prompted — is a block you can swap, edit and share, so two methods can be
compared on the same problem under the same budget.

<img alt="hillclimb watch: the candidate tree growing as coding agents draft, improve and ensemble solutions" src="https://raw.githubusercontent.com/rebase-energy/hillclimb/main/docs/assets/hillclimb-watch.gif" width="100%">

## Quickstart

```bash
uv tool install hillclimb                  # or: pip install hillclimb (Python 3.12+)
hillclimb connect claude                   # or codex, pi, openrouter; add --agent dummy to any run for no LLM at all

hillclimb init                             # hillclimb.yaml, problems/, runs/, climbers/ in this folder
hillclimb problem get heilbronn-convex-13  # the verifier is the problem; description.md is the brief
hillclimb climber get greedy               # the climber, as Python you can read and edit (climbers/greedy/policy.py)
hillclimb verify heilbronn-convex-13       # score the floor solution

hillclimb run heilbronn-convex-13 --budget 10m
hillclimb watch                            # live coding agents and scores (also: chart, tree)
hillclimb stop --all
hillclimb summit                           # copy the best solution.py into this folder
```

Every search also keeps its best in `runs/<run-id>/searches/<search-id>/best/`.
The [walkthrough](https://docs.hillclimb.sh/walkthrough?utm_medium=readme) goes through each step.

## Climbers

A climber is five modules, run in this order at every step of a search. The
engine ships none of the first two: the catalog's climbers (`greedy`,
`openevolve`, `gepa`) are Python files `hillclimb climber get` copies into your
folder, to read and edit; the other three are built in or a file of your own.

| Module | Decides | Built in / in the catalog |
| --- | --- | --- |
| `selector_policy` | which candidate the next attempt builds on | `Best`, `MapElites` (catalog) |
| `operator_policy` | which operator to apply to it | `Greedy` (catalog) |
| `operators` | how one attempt is made: the prompt the coding agent gets | `draft`, `debug`, `improve`, `ensemble` |
| `tuner` | which parameter values to try on a candidate | `random`, `optuna` |
| `memory` | what carries over from one search to the next | `files`, `none` |

Together they are one block of config, in a run spec or as the folder's default
in `hillclimb.yaml`:

```yaml
climber:
  selector_policy: climbers/openevolve/policy.py:MapElites   # which candidate to build on next
  operator_policy: climbers/greedy/policy.py:Greedy           # which operator to use on it
  operators: [draft, debug, improve, crossover.py:Crossover]
  tuner: optuna
  memory: files
  params: {num_drafts: 5}
```

Every slot takes a `.py` file (`file.py:Class`), `package.module:Class`, or a
built-in's name. A fetched climber is one file that builds the whole block
(`climber: climbers/greedy/policy.py`). Or compose in Python:

```python
from hillclimb import Budget, Climber, Problem, catalog

greedy, openevolve = catalog.module("greedy"), catalog.module("openevolve")
problem = Problem("heilbronn-convex-13")
budget = Budget(wall_clock="10m", evaluations=40)
climber = Climber(selector_policy=openevolve.MapElites(num_drafts=5), operator_policy=greedy.Greedy())

climber.search(problem, budget=budget)
climber.best, climber.history, climber.to_frame()
```

`MapElites` and the `openevolve` climber need `pip install 'hillclimb[openevolve]'`,
`tuner: optuna` needs `'hillclimb[optuna]'` and `gepa` needs `'hillclimb[gepa]'`.

`climber.start(...)` opens the same search to drive by hand, one `climber.step()` at a
time, and `Problem(name, score=my_function, ...)` defines a problem from a scoring function.
Runnable scripts are in [`examples/`](https://github.com/rebase-energy/hillclimb/tree/main/examples).

Compare two head to head:

```bash
hillclimb climber get openevolve --no-default
hillclimb run heilbronn-convex-13 --climber climbers/greedy/policy.py --climber climbers/openevolve/policy.py   # one search each; --parallel-searches 3 runs three of each
hillclimb experiment report <run-id>
```

## Learn more

<table>
<tr>
<td width="50%" valign="top">

### [Walkthrough →](https://docs.hillclimb.sh/walkthrough?utm_medium=readme)

From first run to following and resuming a search.

</td>
<td width="50%" valign="top">

### [Problems →](https://docs.hillclimb.sh/problems?utm_medium=readme)

Define your own problem and write its verifier.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Climbers →](https://docs.hillclimb.sh/climbers?utm_medium=readme)

The modules a climber is built from.

</td>
<td width="50%" valign="top">

### [Experiments →](https://docs.hillclimb.sh/experiments?utm_medium=readme)

Compare climbers on the same problem.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Example problems →](https://docs.hillclimb.sh/examples?utm_medium=readme)

Every example problem and its best known value.

</td>
<td width="50%" valign="top">

### [CLI reference →](https://docs.hillclimb.sh/cli?utm_medium=readme)

Every `hillclimb` command and its flags.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Python SDK →](https://docs.hillclimb.sh/python?utm_medium=readme)

Build and run searches from Python.

</td>
<td width="50%" valign="top">

### [Blogposts →](https://docs.hillclimb.sh/blogposts?utm_medium=readme)

Blog posts on hillclimbing.

</td>
</tr>
</table>

Questions, results, ideas: [join the Discord](https://hillclimb.sh/discord?utm_medium=readme).

## License

[MIT](https://github.com/rebase-energy/hillclimb/blob/main/LICENSE)
