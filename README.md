<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hillclimb-logo.svg">
  <img alt="hillclimb" src="docs/assets/hillclimb-logo-light.svg" width="620">
</picture>

**Hillclimbing on verifiable rewards.**<br>
Use Claude Code and Codex to autonomously search for python programs that optimize a score you define.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hillclimb-mountain-dark.png">
  <img alt="An ASCII mountain: one climber on the summit under a gold star, another stuck on a lower peak" src="docs/assets/hillclimb-mountain-light.png" width="560">
</picture>

[Website](https://hillclimb.sh) ·
[Docs](https://docs.hillclimb.sh) ·
[Quickstart](https://docs.hillclimb.sh/quickstart) ·
[Python SDK](https://docs.hillclimb.sh/python) ·
[CLI reference](https://docs.hillclimb.sh/cli) ·
[Blog](https://docs.hillclimb.sh/blogposts) ·
[Discord](https://discord.gg/DDKh9UtSH)

[![PyPI](https://img.shields.io/pypi/v/hillclimb.svg)](https://pypi.org/project/hillclimb/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/hillclimb/)
[![Quickstart](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml/badge.svg)](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-join%20chat-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/DDKh9UtSH)

</div>

---

`hillclimb` provides a modular harness, tooling and testbed for exploring and
benchmarking autoresearch algorithms. The aim is to provide a tool to discover better
autoresearch methods, rather than build a single best autoresearcher.

You provide a verifier script, `verifier.sh`, and (optionally) a starting
solution, `solution.py`. `hillclimb` then spends a compute budget running a
search policy — a [climber](https://docs.hillclimb.sh/python/climber) — that
decides which version to build on next and dispatches headless coding agents
to write, debug and improve it. Each version is scored through your verifier,
and `hillclimb` keeps the one that scores best.

The harness is fixed. The **climber** — what to try next and how each attempt
is prompted — is a block you can swap, edit and share, so two methods can be
compared on the same problem under the same budget.

## Quickstart

```bash
pip install hillclimb                      # or: uv tool install hillclimb
hillclimb connect claude                   # or codex, pi, openrouter; add --agent dummy to any run for no LLM at all

hillclimb init                             # hillclimb.yaml, problems/, runs/ in this folder
hillclimb problem get heilbronn-convex-13  # the verifier is the problem; description.md is the brief
hillclimb verify heilbronn-convex-13       # score the floor solution

hillclimb run heilbronn-convex-13 --budget 10m
hillclimb watch                            # live coding agents and scores (also: chart, tree)
hillclimb stop --all
hillclimb summit                           # copy the best solution.py into this folder
```

Every search also keeps its best in `runs/<run-id>/searches/<search-id>/best/`.
The [walkthrough](https://docs.hillclimb.sh/walkthrough) goes through each step.

## Climbers

A climber is five modules, run in this order at every step of a search. Each
one is a built-in or a Python file you can edit.

| Module | Decides | Built in |
| --- | --- | --- |
| `selector_policy` | which candidate the next attempt builds on | `best`, `map-elites` |
| `operator_policy` | which operator to apply to it | `greedy` |
| `operators` | how one attempt is made: the prompt the coding agent gets | `draft`, `debug`, `improve`, `ensemble` |
| `tuner` | which parameter values to try on a candidate | `random`, `optuna` |
| `memory` | what carries over from one search to the next | `files`, `none` |

Together they are one block of config, in a run spec or as the folder's default
in `hillclimb.yaml`:

```yaml
climber:
  selector_policy: map-elites     # which candidate to build on next
  operator_policy: greedy         # which operator to use on it
  operators: [draft, debug, improve, crossover.py:Crossover]
  tuner: optuna
  memory: files
  params: {num_drafts: 5}
```

Every slot takes a prebuilt module, a `.py` file of your own, or
`package.module:Class`. Presets: `greedy`, `openevolve`, `gepa`. Or compose in
Python:

```python
from hillclimb import Budget, Climber, Problem
from hillclimb.policies import Greedy
from hillclimb.selectors import MapElites

problem = Problem("heilbronn-convex-13")
budget = Budget(wall_clock="10m", evaluations=40)
climber = Climber(selector_policy=MapElites(num_drafts=5), operator_policy=Greedy())

climber.search(problem, budget=budget)
climber.best, climber.history, climber.to_frame()
```

`MapElites` and the `openevolve` preset need `pip install 'hillclimb[openevolve]'`,
`tuner: optuna` needs `'hillclimb[optuna]'` and `gepa` needs `'hillclimb[gepa]'`.

`climber.start(...)` opens the same search to drive by hand, one `climber.step()` at a
time, and `Problem(name, score=my_function, ...)` defines a problem from a scoring function.
Runnable scripts are in [`examples/`](examples/).

Compare two head to head:

```bash
hillclimb run heilbronn-convex-13 --climber greedy --climber openevolve   # one search each; --parallel-searches 3 runs three of each
hillclimb experiment report <run-id>
```

## Learn more

<table>
<tr>
<td width="50%" valign="top">

### [Guide →](https://docs.hillclimb.sh/walkthrough)

From first run to following and resuming a search.

</td>
<td width="50%" valign="top">

### [Problems →](https://docs.hillclimb.sh/problems)

Define your own problem and write its verifier.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Climbers →](https://docs.hillclimb.sh/climbers)

The modules a climber is built from.

</td>
<td width="50%" valign="top">

### [Experiments →](https://docs.hillclimb.sh/experiments)

Compare climbers on the same problem.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [Python SDK →](https://docs.hillclimb.sh/python)

Build and run searches from Python.

</td>
<td width="50%" valign="top">

### [Example problems →](https://docs.hillclimb.sh/examples)

Every example problem and its best known value.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### [CLI reference →](https://docs.hillclimb.sh/cli)

Every `hillclimb` command and its flags.

</td>
<td width="50%" valign="top">

### [Blogposts →](https://docs.hillclimb.sh/blogposts)

From Cauchy's gradient descent to LLMs searching over programs.

</td>
</tr>
</table>

Questions, results, ideas: [join the Discord](https://discord.gg/DDKh9UtSH).

## License

[MIT](LICENSE)
