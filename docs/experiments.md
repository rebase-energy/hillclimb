# Experiments

Which setup wins: a problem times named arms of config overrides times repeats, judged against the noise floor.

Every knob — the climber, its params, the model, cross-search memory,
replicate count — is a config setting, so "does X help?" is one experiment: a
problem × named *arms* (sets of config overrides) × N repeats. A spec lives
in `hillclimb/experiments/<name>.yaml`:

```yaml
problems: [circle-packing]
repeats: 3
budget: 15m
schedule: sequential        # sequential | parallel
max_concurrent: 8           # parallel only: searches alive at once
noise_floor: 0.02           # from `hillclimb verify circle-packing --repeat 5`
arms:
  greedy:       {climber.ref: greedy}             # first arm = the control
  greedy-nomem: {climber.ref: greedy, learning.enabled: false}
  openevolve:   {climber.ref: openevolve, climber.params: {population_size: 50}}
```

`hillclimb experiment run <name>` launches the matrix: sequentially by
default — repeat by repeat, arms round-robin inside, so shared state such as
the knowledge graph is seen by every arm at the same point (mandatory when an
arm touches memory) — or `--parallel` for stateless comparisons (climber,
model). Parallel without a bound starts every search at once, and past the
machine's operator slots (`concurrency.machine_max_agents`, default
`min(8, cores-2)`) the rest burn their wall clock in `waiting-slot` — so
give it `max_concurrent: N` in the spec (or `--max-concurrent N`): the
launcher starts searches in job order, waits on its children before starting
the next, runs the whole matrix, and prints the report at the end (run it
under `nohup` or in tmux; exit 1 if any child failed, 2 if any parked). To
add repeats to a finished run, `--run-id <run> --first-repeat K --repeats M`
appends repeats K..K+M-1 into the same run, so report and chart keep grouping
as one experiment. Each search is a normal `hillclimb run … --experiment <name> --arm
<arm> --set key=value`, tagged in its `search.yaml` (`experiment`, `arm`,
`repeat`, `arm_overrides`), so a search you start by hand with those flags
counts too — as does a mixed fleet (`--climber A --climber B`, see
[climbers.md](climbers.md)). `hillclimb experiment report <name>` compares the arms on the
selected candidate's holdout score (val when holdout is off): n / mean /
median / spread, best-of-repeat wins, minutes to best, tokens, and each arm's
paired gap to the control judged against the noise floor — a gap inside it
is reported as "within noise, not a result". `hillclimb chart` colours an
experiment's curves by arm.

A spec may name one shared executable seed — `seed_from: seeds/foo.py`,
resolved against the spec's directory — which rides `--seed-from` into every
child search, so arms are compared from identical source (the dry run prints
the resolved path and its sha256). Mandatory for climbers that require a seed
(GEPA); see `hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml` for the
three-climber comparison this shipped with, and
`gepa-vs-openevolve-vs-greedy-heilbronn.yaml` for the same three climbers
across a difficulty ladder (`problems/heilbronn-{11,14,17}`, stamped by
`problems/make_heilbronn.py`; its seed reads N off the problem, so one
`seed_from` serves every level). One `hillclimb chart` per problem (a bare
`hillclimb chart` lists them, `p` cycles them, `c` in `hillclimb watch` opens
the highlighted one) and `hillclimb similarity <run>` for the arm-coloured map (`similarity reference <run>` for the cube).
