# hillclimb

Hillclimbing on verifier-defined problems. You give it a **problem** — a folder
whose `verifier.sh` scores a `solution.py` — and a budget. It spawns coding
agents (Claude Code, Codex or pi) as operators that draft, debug and improve
solutions, scores every candidate through your verifier, and keeps the best.
The harness is fixed. The **climber** — what to try next and how each attempt
is prompted — is a bundle you can swap, edit and share, so two methods can be
compared on the same problem under the same budget.

## Get started

Eight commands, ten minutes. Each step says what you should see.

**1. Install and connect an agent.**

```bash
pip install hillclimb
hillclimb connect claude       # logs in; operator calls bill your Claude subscription
```

`connect codex` and `connect pi` do the same for the other agents. No agent
yet? Add `--agent dummy` to any `run` below: it climbs with no LLM at all.

**2. Make a hillclimb dir, get a problem, read it.**

```bash
hillclimb init                         # this folder: hillclimb.yaml, problems/, runs/
hillclimb problem list                 # the bundled example problems, with the best known value
hillclimb problem get heilbronn-11
```

Everything hillclimb writes lives beside `hillclimb.yaml` (`hillclimb init
hillclimb` keeps it all in a subfolder instead). The problem lands in
`problems/heilbronn-11/`. Open `verify.py`: the
verifier *is* the problem. Everything the agents will be told is in
`description.md`.

**3. Check the verifier.**

```bash
hillclimb verify heilbronn-11 --repeat 3
```

Scores the problem's floor three times and prints the spread. Example
problems are exact, so the spread is 0 and any improvement is real. On a
noisy problem of your own, this number is the noise floor the search must beat.

**4. Climb.**

```bash
hillclimb run heilbronn-11 --budget 10m
```

The search starts as a detached engine and the terminal comes straight back
with the run id: a baseline, then drafts, debugs and improves, each scored as it
lands. (`--no-detach` keeps it in this terminal; Ctrl-C then stops it.)

**5. Watch it.**

```bash
hillclimb watch      # every agent, what it is doing, its candidate's score
hillclimb chart      # best score so far against time, every candidate a dot
hillclimb tree       # the exploration tree: what was expanded, what was left
```

The result is `runs/<run-id>/searches/heilbronn-11/best/`:
`solution.py` and the `submission.csv` it wrote.

**6. Stop everything.**

```bash
hillclimb stop --all
```

**7. Next.**

- **Your own problem.** Copy a bundled problem and edit `verify.py`.
  See [docs/problems.md](docs/problems.md).
- **Another climber.** `hillclimb run heilbronn-11 --climber openevolve`.
  `hillclimb climber show openevolve` prints it as the block it is;
  `hillclimb climber new mine --from greedy` copies greedy's source into
  `climbers/mine.py` for editing. See
  [docs/climbers.md](docs/climbers.md).
- **Compare two.** `hillclimb run heilbronn-11 --climber greedy --climber
  openevolve --parallel-searches 2`, then `hillclimb experiment report <run-id>`.
  See [docs/experiments.md](docs/experiments.md).

## Example problems

Construction problems in one shape: the submission is a small CSV of numbers,
the verifier checks the constraints and computes the score exactly, there is
no dataset, no holdout split and no noise. Each ships with the best known
value as a reference line on the chart, and each family is a ladder, so a
ten-minute run visibly climbs on the small instance and an hour does not
saturate the large one.

| Family | Instances | Score |
|---|---|---|
| Circle packing (AlphaEvolve's benchmark) | `circle-packing` (26), `circle-packing-32` | sum of radii, maximize |
| Heilbronn triangles | `heilbronn-11`, `-14`, `-17`, `heilbronn-convex-13` | smallest triangle area, maximize |
| Low-autocorrelation binary sequences | `labs-40`, `labs-60` | sidelobe energy, minimize |
| Tammes (points on a sphere) | `tammes-30`, `tammes-50` | minimum angle, maximize |
| Thomson (charges on a sphere) | `thomson-50`, `thomson-100` | Coulomb energy, minimize |
| Autocorrelation inequalities (AlphaEvolve) | `autocorr-1`, `autocorr-3`, `erdos-overlap` | the inequality's constant |
| Kissing configuration in dimension 11 | `kissing-11` | number of points, maximize |
| Golomb rulers | `golomb-20`, `golomb-27` | ruler length, minimize |
| Travelling salesman on a fixed instance | `tsp-200` | tour length, minimize |
| Multidimensional knapsack (Chu & Beasley instances) | `mknap-100-5`, `mknap-250-10` | total value, maximize |

`hillclimb problem list` prints the catalog with the best known value and
who found it. Problems that need data, a hidden split or a provider (Kaggle
competitions through MLE-bench, energy forecasting through emflow) are
documented in [docs/providers.md](docs/providers.md).

## Climbers

A climber is one block of config, defined where the run is defined — in a
run spec, or in `hillclimb.yaml` as the folder's default:

```yaml
climber:
  policy: greedy                  # what to try next (or `loop:` for the whole control flow)
  select: map-elites              # which candidate it expands
  operators: [draft, debug, improve, crossover.py:Crossover]
  tuner: optuna
  memory: files
  params: {num_drafts: 5}
```

Every slot names a prebuilt module, a `.py` file of your own, or
`package.module:Class`. Three presets stand for whole blocks: `greedy`
(debug failing tips, draft a few branches, improve the best), `openevolve`
(the same schedule over a MAP-Elites archive) and `gepa` (reflective Pareto
search, brings its own loop). The same building blocks compose in Python:

```python
import hillclimb as hc

climber = hc.Climber(policy=hc.policies.Greedy(num_drafts=5), select=hc.selectors.MapElites(), tuner="optuna")
hc.run("heilbronn-11", climber=climber, budget="10m")
```

A run records the block it ran, a search snapshots its climber (so editing
the live files never changes a running search), and `hillclimb climber
check` replays recorded journals through an edited climber before an agent
hour is spent on it.

## What is where

| Path | What it is |
|---|---|
| `src/hillclimb/harness/` | The fixed core every search runs on: `core.py` (the Harness), the loop, evaluation and the executor, the journal, candidates and the store, budgets, slots and the control queue. Never a research surface. |
| `src/hillclimb/modules/` | What a climber is built from, one subpackage per kind, each with its contract in `base.py`: `policies/` (what to try next), `selectors/` (which candidate to expand), `operators/` (how one attempt is made), `tuners/` (which parameter values), `memory/` (what a search knows from others, and the graph module that indexes it), `similarity/` (how alike two solutions are). `refs.py` resolves a module's name, `spec.py` is the `climber:` block. Implementations import only `hillclimb.sdk`. |
| `src/hillclimb/sdk/` | The one import a climber needs: the contracts and the read-only views of the search. |
| `src/hillclimb/climbers/` | Climber libraries that bring more than one module: `gepa/` (its loop, its operator, its scoring view). |
| `src/hillclimb/{policies,selectors,operators,tuners,memory,loops}.py` | The prebuilt building blocks by name, for composing in Python (`hillclimb.policies.Greedy`): lazy windows onto `modules/`. |
| `src/hillclimb/tui/` | Every terminal view (`watch`, `chart`, `tree`, `archive`, `surface`, `similarity`, `graph`) and the layout it draws. Reads the store, imported by nothing else. |
| `src/hillclimb/cli/` | The `hillclimb` command, one module per command group. |
| `src/hillclimb/agents/` | The agents that write code: Claude Code, Codex, pi, and the dummy and fake agents for tests. |
| `src/hillclimb/providers/` | Problem providers: emflow, MLE-bench, Einstein Arena. |
| `src/hillclimb/prompts/` | The operator prompt templates. A climber may shadow them by name. |
| `src/hillclimb/runtime/` | The managed venv the verifier and the solution run in, and the shim that makes `hillclimb.spaces` importable there. |
| `src/hillclimb/demo/` | The example problems as package data, so `hillclimb problem get` works from a bare install. |
| `src/hillclimb/spaces.py` | The output-format contract a problem's `interface.py` is written in, and the `params.json` contract. Stdlib only, byte-copied into runtime venvs. |
| `src/hillclimb/{api,config,problem,climber,experiment,connect}.py` | The public surface: run a search (`run`, `run_spec`), the config schema, load a problem, compose or resolve a climber, experiments, and connecting an agent. |
| `problems/` | The example problems' source of truth, one `make_<family>.py` generator per family; the bundled copies under `demo/` are stamped from here. |
| `hillclimb.yaml`, `experiments/`, `knowledge/` | This repo is itself a hillclimb dir: its config, experiment specs and seeds, and learning (the graph, cards, credit, playbooks), beside `problems/` and `runs/`. |
| `tests/` | The suite (`uv run pytest`). `test_layout.py` pins which package may import which, `test_sdk_imports.py` that climber code imports only the sdk, `golden/` the prompt bytes and every `--help` screen. |
| `docs/` | The topic docs linked below, and the dated design plans. |

## Docs

- [Problems](docs/problems.md) — the verifier contract, floors, unit tests, tunable parameters, per-instance scores, noise
- [Climbers](docs/climbers.md) — the `climber:` block, presets, your own policy / selector / operator / loop, composing in Python, `climber check`, mixed fleets, GEPA
- [Agents](docs/agents.md) — Claude Code, Codex, pi; `connect`; billing through OpenRouter; sampling
- [The sandbox](docs/sandbox.md) — what agents and solutions can write, read and reach; agents without internet
- [Experiments](docs/experiments.md) — studies of experiments, repeats, matched budgets, the report
- [Providers](docs/providers.md) — emflow, MLE-bench, Einstein Arena
- [Operators and memory](docs/operators-and-memory.md) — operator scaffolds, model routing, the knowledge graph
- [The hillclimb dir](docs/hillclimb-dir.md) — config precedence, run specs, how runs are laid out, the store, pruning
- [Commands](docs/commands.md) — every command, the TUIs, the demo suite
- [Development](docs/development.md)
- [Changelog](CHANGELOG.md)
