<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hillclimb-logo-dark.svg">
  <img alt="hillclimb" src="docs/assets/hillclimb-logo-light.svg" width="520">
</picture>

**Let coding agents climb a score you define.**

[Website](https://hillclimb.sh) ·
[Quickstart](#quickstart) ·
[Problems](docs/problems.md) ·
[Climbers](docs/climbers.md) ·
[Commands](docs/commands.md) ·
[Changelog](CHANGELOG.md)

[![PyPI](https://img.shields.io/pypi/v/hillclimb.svg)](https://pypi.org/project/hillclimb/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://pypi.org/project/hillclimb/)
[![Quickstart](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml/badge.svg)](https://github.com/rebase-energy/hillclimb/actions/workflows/quickstart.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Give hillclimb a **problem** — a folder whose verifier scores a `solution.py` —
and a budget. It runs Claude Code, Codex or pi as operators that draft, debug
and improve solutions, scores every candidate through your verifier, and keeps
the best.

The harness is fixed. The **climber** — what to try next and how each attempt
is prompted — is a block you can swap, edit and share, so two methods can be
compared on the same problem under the same budget.

## Quickstart

```bash
pip install hillclimb
hillclimb connect claude               # or codex, pi; add --agent dummy to any run for no LLM at all

hillclimb init                         # hillclimb.yaml, problems/, runs/ in this folder
hillclimb problem get heilbronn-11     # the verifier is the problem; description.md is the brief
hillclimb verify heilbronn-11 --repeat 3

hillclimb run heilbronn-11 --budget 10m
hillclimb watch                        # live agents and scores (also: chart, tree)
hillclimb stop --all
```

The best solution lands in `runs/<run-id>/searches/heilbronn-11/best/`.

## Climbers

A climber is one block of config, in a run spec or as the folder's default in
`hillclimb.yaml`:

```yaml
climber:
  policy: greedy                  # what to try next
  select: map-elites              # which candidate to expand
  operators: [draft, debug, improve, crossover.py:Crossover]
  tuner: optuna
  params: {num_drafts: 5}
```

Every slot takes a prebuilt module, a `.py` file of your own, or
`package.module:Class`. Presets: `greedy`, `openevolve`, `gepa`. Or compose in
Python:

```python
import hillclimb as hc

climber = hc.Climber(policy=hc.policies.Greedy(num_drafts=5), select=hc.selectors.MapElites())
hc.run("heilbronn-11", climber=climber, budget="10m")
```

Compare two head to head:

```bash
hillclimb run heilbronn-11 --climber greedy --climber openevolve --parallel-searches 2
hillclimb experiment report <run-id>
```

## Example problems

Exact, noise-free construction problems, each with its best known value as the
target line. `hillclimb problem list` shows the full catalog.

| Family | Instances | Score |
|---|---|---|
| Circle packing | `circle-packing`, `circle-packing-32` | sum of radii ↑ |
| Heilbronn triangles | `heilbronn-11`, `-14`, `-17`, `heilbronn-convex-13` | smallest triangle ↑ |
| Low-autocorrelation sequences | `labs-40`, `labs-60` | sidelobe energy ↓ |
| Tammes / Thomson | `tammes-30`, `-50`, `thomson-50`, `-100` | min angle ↑ / energy ↓ |
| Autocorrelation inequalities | `autocorr-1`, `autocorr-3`, `erdos-overlap` | the constant ↓ |
| Kissing configuration | `kissing-11` | points ↑ |
| Golomb rulers | `golomb-20`, `golomb-27` | length ↓ |
| TSP / knapsack | `tsp-200`, `mknap-100-5`, `mknap-250-10` | tour ↓ / value ↑ |

Bring your own by copying one and editing `verify.py`. Kaggle (MLE-bench),
energy forecasting (emflow) and Einstein Arena come in through
[providers](docs/providers.md).

## Docs

[Problems](docs/problems.md) ·
[Climbers](docs/climbers.md) ·
[Agents](docs/agents.md) ·
[Sandbox](docs/sandbox.md) ·
[Experiments](docs/experiments.md) ·
[Providers](docs/providers.md) ·
[Operators & memory](docs/operators-and-memory.md) ·
[The hillclimb dir](docs/hillclimb-dir.md) ·
[Commands](docs/commands.md) ·
[Development](docs/development.md)

## License

[MIT](LICENSE)
