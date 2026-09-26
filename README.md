# hillclimb

Hillclimbing on verifier-defined problems. You give it a **problem** — a folder
whose `verifier.sh` scores a `solution.py` — and a budget. It spawns coding
agents (Claude Code, Codex or pi) as operators that draft, debug and improve
solutions, scores every candidate through your verifier, and keeps the best.
The harness is fixed. The **climber** — what to try next and how each attempt
is prompted — is a bundle you can swap, edit and share, so two methods can be
compared on the same problem under the same budget.

## Get started

Seven commands, ten minutes. Each step says what you should see.

**1. Install and connect an agent.**

```bash
pip install hillclimb
hillclimb connect claude       # logs in; operator calls bill your Claude subscription
```

`connect codex` and `connect pi` do the same for the other backends. No agent
yet? Add `--backend dummy` to any `run` below: it climbs with no LLM at all.

**2. Get a problem and read it.**

```bash
hillclimb problem list                 # the bundled starter problems, with the best known value
hillclimb problem get heilbronn-11
```

This creates a `hillclimb/` dir in the current folder and copies the problem
into `hillclimb/problems/heilbronn-11/`. Open `verify.py`: the verifier *is*
the problem. Everything the agents will be told is in `description.md`.

**3. Check the verifier.**

```bash
hillclimb verify heilbronn-11 --repeat 3
```

Scores the problem's floor three times and prints the spread. Starter
problems are exact, so the spread is 0 and any improvement is real. On a
noisy problem of your own, this number is the noise floor the search must beat.

**4. Climb.**

```bash
hillclimb run heilbronn-11 --budget 10m
```

One search in the foreground: a baseline, then drafts, debugs and improves,
each scored as it lands. Ctrl-C stops it; the best solution so far is kept.

**5. Watch it, in a second terminal.**

```bash
hillclimb watch      # every agent, what it is doing, its candidate's score
hillclimb chart      # best score so far against time, every candidate a dot
hillclimb tree       # the exploration tree: what was expanded, what was left
```

The result is `hillclimb/runs/<run-id>/searches/heilbronn-11/best/`:
`solution.py` and the `submission.csv` it wrote.

**6. Stop everything.**

```bash
hillclimb stop --all
```

**7. Next.**

- **Your own problem.** Copy a starter and edit `verify.py`, or `hillclimb init`
  for a blank scaffold. See [docs/problems.md](docs/problems.md).
- **Another climber.** `hillclimb run heilbronn-11 --climber openevolve`.
  `hillclimb climber list` shows the bundled ones; `hillclimb climber new mine
  --from greedy` copies one into `hillclimb/climbers/mine/` for editing. See
  [docs/climbers.md](docs/climbers.md).
- **Compare two.** `hillclimb run heilbronn-11 --climber greedy --climber
  openevolve --parallel-searches 2`, then `hillclimb experiment report <run-id>`.
  See [docs/experiments.md](docs/experiments.md).

## Starter problems

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

`hillclimb problem list` prints the catalog with the best known value and
who found it. Problems that need data, a hidden split or a provider (Kaggle
competitions through MLE-bench, energy forecasting through emflow) are
documented in [docs/providers.md](docs/providers.md).

## Climbers

A climber is a directory with a `climber.yaml` naming a search policy (or a
whole loop), the operators it may use, their prompts and a tuner. Three are
bundled: `greedy` (debug failing tips, draft a few branches, improve the
best), `openevolve` (MAP-Elites over hillclimb's operators) and `gepa`
(reflective Pareto search, brings its own loop). A one-file climber is a
`.py` with one policy class. A search snapshots its climber, so editing the
live copy never changes a running search, and `hillclimb climber check`
replays recorded journals through an edited climber before an agent hour
is spent on it.

## Docs

- [Problems](docs/problems.md) — the verifier contract, floors, unit tests, tunable parameters, per-instance scores, noise
- [Climbers](docs/climbers.md) — bundled climbers, the manifest, one-file climbers, mixed fleets, `climber check`, GEPA
- [Agents](docs/agents.md) — Claude Code, Codex, pi; `connect`; billing through OpenRouter; sampling
- [Experiments](docs/experiments.md) — arms, repeats, matched budgets, the report
- [Providers](docs/providers.md) — emflow, MLE-bench, Einstein Arena
- [Operators and memory](docs/operators-and-memory.md) — operator scaffolds, model routing, the knowledge graph
- [The hillclimb dir](docs/hillclimb-dir.md) — config precedence, run specs, how runs are laid out, the store, pruning
- [Commands](docs/commands.md) — every command, the TUIs, the demo suite
- [Development](docs/development.md)
- [Changelog](CHANGELOG.md)
