# Climbers

A climber is *how* to hillclimb: a policy (or a whole loop), the selector it
expands with, the operators it may use and their prompts, a tuner and a
memory. The harness — candidate dirs, coding agent calls, trials, holdout, the
journal, budgets, `best/` — is fixed and the same for every climber.

A climber is **defined where the run is defined**: it is one block of config.

```yaml
# run.yaml — `hillclimb run run.yaml`
problems:
  - target: heilbronn-11
    budget: 30m
    climber:
      operator_policy: greedy       # π_op: which operator on what the selector policy chose (or `loop:` — the whole control flow)
      params: {num_drafts: 5}
      selector_policy: map-elites   # π_sel: WHICH candidate the next attempt starts from
      selector_params: {num_islands: 3}
      operators: [draft, debug, improve, crossover.py:Crossover]   # HOW an attempt is made
      operator_params: {draft: {retrieval: false}}
      tuner: optuna                 # WHICH parameter values a tunable candidate tries
      tuner_params: {seed: 7}
      memory: files                 # what it knows from other searches, and leaves them
      memory_params: {max_cards: 1}
      prompts: prompts/             # templates that shadow the built-in ones by name
```

Every key is optional. With neither `operator_policy:` nor `loop:` the policy is
`greedy`; everything else defaults to what its class says.

## Where the block goes

The same block in three places, highest first:

| where | what it is |
|---|---|
| an entry of a run spec (`problems: [- {target: ..., climber: {...}}]`) | that search's climber |
| the top of a run spec (`climber:` beside `problems:`) | the default for its entries |
| `hillclimb.yaml` | the folder's default: what `hillclimb run <problem>` uses |

A higher one **replaces** a lower one whole — blocks are never merged, so
one policy's params cannot end up under another policy. Two things then
apply on top of whichever block was chosen:

- `--climber NAME` names a climber outright (a preset or one `.py` file).
- `--set climber.params.num_drafts=5` (and `set:` in a spec entry, and an
  experiment's overrides) edits a field of the block.
  `--set climber.operators.draft.retrieval=false` reaches one operator's params.

A run records what it ran: `runs/<run-id>/spec.yaml` carries every search's
full block, and `hillclimb run runs/<run-id>/spec.yaml` runs it again.

## Naming a module

Every slot names its module the same three ways:

| form | example |
|---|---|
| a registry name | `greedy`, `map-elites`, `optuna`, `files` |
| a file | `mine.py` (the one class of that kind it defines) or `mine.py:Class` |
| an importable class | `mypackage.policies:Annealed` |

A file is relative to the file the block is written in (the spec, or
`hillclimb.yaml`). Files may import each other relatively (`from .helpers
import x`). `hillclimb climber list` shows what is registered.

A bare name instead of a block is a **preset**, or one `.py` file:

| preset | its block |
|---|---|
| `greedy` | `{operator_policy: greedy}` — debug the newest failing tip > ensemble in the final budget window > draft until `num_drafts` branches are scored > tune > improve the best |
| `openevolve` | `{operator_policy: greedy, selector_policy: map-elites, params: {ensemble: false, tune_budget: 0}}` — the same schedule over [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)'s MAP-Elites archive: a population kept diverse over feature dimensions, on islands with migration. `pip install 'hillclimb[openevolve]'` |
| `gepa` | `{loop: gepa}` — [GEPA](https://github.com/gepa-ai/gepa) owns the whole loop (see below). `pip install 'hillclimb[gepa]'` |

`hillclimb climber show openevolve` prints a preset as the block it stands
for, ready to paste and edit.

A `.py` file that builds a whole `Climber(...)` (your own policies next to
built-in operators, tuner and memory) stands for THAT climber:
`--climber climbers/my_climber.py`, a run spec or an experiment's setup run
it, the classes it defines are written as `my_climber.py:Class`, and a file
that builds two names one as `my_climber.py:climber`. A file that builds
none is one operator policy, as before (`spec.composed_block`).

## The slots

One step of a search is two decisions, always in this order: the selector
(π_sel) reads the history and picks the node(s) the next attempt starts
from, or none; then the policy (π_op) reads the same history and that choice
and names the operator. The slots, in the order a step runs them:

| slot | decides | base class | built in |
|---|---|---|---|
| `selector_policy` | which node, or none: a failing tip first, roots until `num_drafts`, the top-k to combine in the final window, else the subclass's `select` | `SelectorPolicy` | `best`, `map-elites` |
| `operator_policy` | which operator on what the selector policy chose (draft, debug, ensemble, improve; greedy adds tune) | `OperatorPolicy` | `greedy` |
| `loop` | the control flow itself (instead of a selector and a policy) | `Loop` | `gepa` |
| `operators` | how one attempt is made: the prompt the coding agent gets | `Operator` | `draft`, `debug`, `improve`, `ensemble` |
| `tuner` | which parameter values a tunable candidate tries | `Tuner` | `random`, `optuna` |
| `memory` | what a search knows from others and leaves for the next | `Memory` | `files`, `none` |

Which coding agent and model run is **not** a climber's: `routing:` in a block is
refused. That is the user's choice, in `hillclimb.yaml`.

A climber's own file imports the contracts from `hillclimb.sdk` and, to
build on a prebuilt block, that block from `hillclimb.policies` and its
siblings — nothing deeper.

### A policy

```python
# climbers/drafts_only.py
from hillclimb.sdk import Action, OperatorPolicy


class DraftsOnly(OperatorPolicy):
    """Never improves: whatever the selector chose, draft (repair a failing tip first)."""

    DEFAULTS = {}                         # its knobs; `params` in the block set them

    def propose(self, state, selection):
        node = state.journal.candidates[selection.target_id] if selection is not None else None
        if node is not None and node.status in ("failing", "buggy"):
            return Action(operator="debug", target_id=node.candidate_id)
        return self.draft_action(state)
```

```bash
hillclimb climber check --climber climbers/drafts_only.py   # before spending budget
hillclimb run circle-packing --climber climbers/drafts_only.py
```

`propose(state, selection)` returns one `Action` — an operator, the
candidate to target, optional `inspiration_ids` — or `None` to hold the slot
until an in-flight result lands. `selection` is what the selector chose:
`None` for a root step, else a `Selection` with the node (`target_id`),
`inspiration_ids`, and `combine=True` when those nodes are the inputs of one
combined candidate. `observe(state, candidate)` is called after every
result, and replayed over the journal when a search starts or resumes.

The base `propose` is the plain mapping (no node → draft, a failing node →
debug, several → ensemble, a scored node → improve), so a policy overrides
only where it differs; `draft_action(state)` and `expand_action(state,
selection, operator=...)` build the usual actions, `param(name)` and
`resolved_params()` read its knobs, `self.selector` is the selector the loop
asks first. A class with just `propose` and `observe` and no base runs too.

Three rules the harness relies on:

- A policy only ever sees a holdout-blind `SearchState`. It may read
  candidate dirs; it must never write.
- Every decision must be derivable from the journal — compute it from
  `state`, or rebuild your caches in `observe`. That is what makes `resume`
  work, and what `climber check` verifies.
- A param the block sets must be one the class declares in `DEFAULTS`: a
  typo is an error before anything is spent.

`greedy.py` (`src/hillclimb/modules/policies/greedy.py`) is the reference.
Subclass it to change one move: `from hillclimb.policies import Greedy`.

### A selector

```python
# oldest.py
from hillclimb.sdk import Selection, SelectorPolicy


class Oldest(SelectorPolicy):
    """Build on the oldest scored candidate."""

    def select(self, state, *, busy=frozenset()):
        scored = state.journal.scored_candidates()
        return Selection(scored[0].candidate_id, prompt_context="Oldest first.") if scored else None
```

```yaml
climber: {operator_policy: greedy, selector_policy: oldest.py}
```

A selector implements `select`: the scored node to build on, with whatever
rides along — inspirations (copied in beside the parent), a paragraph for the
prompt, a note journaled on the new candidate; `None` means a root step (the
policy drafts). The base class's `schedule` wraps it with the order every
selector shares and its knobs are `selector_params`: `num_drafts` (3), `debug`
(True), `max_debug_depth` (3), `ensemble` (True), `ensemble_reserve_fraction`
(0.2), `ensemble_top_k` (3), `ensemble_max_attempts` (2). A block that still
writes those under `params` (every block before 0.7) loads; they land in
`selector_params`. A selector with state keeps it a function of the journal
(`sync(state)`), like a policy.

`map-elites` takes OpenEvolve's `DatabaseConfig` fields as `selector_params`
(`num_islands`, `feature_dimensions`, `population_size`, `random_seed`, …)
plus `num_inspirations`. The built-in feature dimensions are `complexity`,
`diversity` and `score`; any other must be a numeric key the verifier writes
next to `score` (see [Replicate metrics](problems.md#replicate-metrics-optional)).
Each evolved candidate's `climber_meta` records its island, grid cell and
inspirations (`hillclimb show <candidate>` prints it).

### An operator

```python
from hillclimb.sdk import Attempt, Operator, inspiration_filename


class Crossover(Operator):
    name, kind, needs_target = "crossover", "combine", True

    def prepare(self, ctx):
        files = ", ".join(inspiration_filename(i) for i, _ in enumerate(ctx.inspirations, 1))
        return Attempt(prompt="Cross the parent with " + files + ".\n\n{{contract}}", copy_parent=True)
```

An operator turns an action into an `Attempt`: the prompt, and what the
harness should put in the new candidate dir. It never touches disk, the
journal or a coding agent. The problem's contract is the harness's — it fills
`{{contract}}`, and appends it to a prompt that has no such token. See
[operators-and-memory.md](operators-and-memory.md).

`prompts:` in the block names a directory whose templates (`<name>.md`)
shadow the built-in operator templates of the same name (`draft`, `improve`,
`debug`, `ensemble`, the cue snippets). An override may drop `{{tokens}}`
but never add one nothing fills, and the `contract_*` templates cannot be
shadowed — a search refuses to start on such a file.

### A loop

Some optimizers cannot be reduced to "what next?" — they own proposal,
reflection and selection themselves. Those bring a `Loop`: the block says
`loop:` instead of `operator_policy:`, and the loop drives the harness
(`submit` / `wait` / `run`) instead of answering `propose`. A loop class
declares what it needs of its search (`operators = (...)`, `holdout_timing =
"after"`), so `{loop: gepa}` is complete.

- `greedy`: hillclimb chooses the parent and asks a coding agent to mutate;
- `openevolve`: MAP-Elites picks parent and inspirations, hillclimb's coding agent
  still mutates;
- `gepa`: GEPA drives reflective mutation and Pareto search, hillclimb
  evaluates and records.

```bash
uv sync --extra gepa
uv run hillclimb run <problem> --climber gepa --seed-from my_solution.py
```

GEPA's reflective mutation is the `gepa-reflect` operator, run by a routed
hillclimb coding agent (`routing.gepa-reflect`, falling back to `routing.default`
and the global coding agent/model) in an ordinary candidate dir; every evaluation
is a normal journaled `cNNN` candidate, so `watch`, `tree` and `chart` work
unchanged. GEPA checkpoints under `SEARCH_DIR/loop/state/` and `hillclimb
resume` continues both the journal and the optimizer, with a warm evaluation
cache keyed on the solution's hash.

MVP limits: one mutable file (`solution.py`), serial
(`concurrency.parallel_agents: 1`), no merge, and a **required executable
seed** — pass `--seed-from` or ship an executable baseline. Holdout scoring
runs only after the optimizer finishes, and no holdout value ever reaches
GEPA's prompts, feedback or state. When the verifier emits per-instance
scores (see [Per-instance scores](problems.md#per-instance-scores-optional)),
`frontier_type: instance` (the default) tracks GEPA's Pareto frontier per
instance.

## Identity, snapshots, resume

A climber's identity (`climber_sha256` in `search.yaml`) is its block —
without its `name` — plus the bytes of every local file it reaches (files
only reached by a relative import included) and of its `prompts` dir. Two
searches are comparable only when their hashes agree.

A search snapshots its climber into `searches/<id>/climber/`: the block as
`climber.yaml`, its local files under `files/`, its prompts under
`prompts/`. The engine — and `hillclimb resume` — load that copy, so
editing the live files never changes a started search; `resume` says so
when the live climber has changed since.

## Checking one before it costs anything

```bash
hillclimb climber check                       # this folder's block
hillclimb climber check run.yaml              # every entry of a run spec
hillclimb climber check --climber mine.py --set climber.params.k=v --problem P --smoke
```

It resolves every module the block names, then replays every recorded
journal in the store (plus an empty one) through the policy with no coding agent or
verifier, and reports each contract breach: a hold on an empty journal (the
search would never start), two fresh instances disagreeing at some budget
point, a policy that watched the journal grow proposing something else than
one shown the finished journal (a resume would diverge), a target that does
not exist, an operator the climber does not have, a mutated journal or a
file written under a search dir, a prompt override that lints dirty.
`--smoke --problem P` then runs a short `--agent dummy` search; `--json` is
the machine-readable form. Exit 1 on any breach.

## Starting your own

```bash
hillclimb climber list                       # the presets, your one-file climbers, the building blocks
hillclimb climber show greedy                # a preset as a block to paste
hillclimb climber new mine --from greedy     # copy greedy's source into climbers/mine.py, print its block
hillclimb climber check --climber climbers/mine.py
hillclimb run circle-packing --climber climbers/mine.py
```

## Several climbers on one problem: a mixed fleet

Repeat `--climber` and one run holds one search per climber, each its own
engine process on the same problem, tagged as an experiment so they can be
compared afterwards. Per-experiment settings go through
`--experiment-set EXPERIMENT:KEY=VALUE` (applied after `--set`, which is
fleet-wide); experiments are named after their climber (a repeated climber
becomes `greedy-2`):

```bash
uv run hillclimb run circle-packing --budget 30m \
  --climber greedy --climber openevolve --climber gepa \
  --seed-from experiments/seeds/circle-packing.py \
  --experiment-set gepa:concurrency.parallel_agents=1 \
  --experiment-set gepa:climber.params.max_metric_calls=60
uv run hillclimb watch                          # the three searches side by side
uv run hillclimb experiment report <run-id>     # experiments compared
```

The GEPA climber is serial, so its experiment needs
`concurrency.parallel_agents=1`. `--parallel-searches N` repeats every
experiment N times. Searches of one run share live knowledge cards; pass
`--no-learning` for a fair comparison. For repeats across problems with a
committed spec, noise floors and a control, use `hillclimb experiment run`
(see [experiments.md](experiments.md)).

## In Python

The same building blocks, composed as objects, in the sklearn shape: a
`Problem` is what you search, a `Climber` searches it within a `Budget`, and
the result reads off the climber afterwards:

```python
from hillclimb import Budget, Climber, Problem, run_spec
from hillclimb.memory import FilesMemory
from hillclimb.operators import Debug, Draft
from hillclimb.policies import Greedy
from hillclimb.selectors import MapElites

climber = Climber(
    selector_policy=MapElites(num_islands=2, num_drafts=3),   # π_sel: which node, or none
    operator_policy=Greedy(),                                 # π_op: which operator for it
    operators=[Draft(retrieval=False), Debug(), MyCrossover],
    tuner="optuna",
    memory=FilesMemory(max_cards=1),
)

if __name__ == "__main__":
    problem = Problem("heilbronn-11")
    budget = Budget(wall_clock="10m", evaluations=40)
    climber.search(problem, budget=budget)   # one search, here
    print(climber.best.val_score)
    climber.write("climber.yaml")     # the same climber, as the block
    run_spec("run.yaml")              # every entry of a spec, one after the other
```

`hillclimb.run(problem, climber=climber, ...)` is the same search as a function, returning
the `SearchOutcome` that `climber.result` holds. A `Problem` is one that exists
(`Problem("heilbronn-11")`: the folder's, else a bundled one copied in) or one defined
here from a scoring function:

```python
def closeness_to_pi(run_dir):
    return -abs(float((run_dir / "answer.txt").read_text()) - 3.14159265358979)

problem = Problem("closest-to-pi", score=closeness_to_pi, output="answer.txt",
                  description="Write your best approximation of pi to answer.txt.")
```

Its folder is written under `problems/` the first time a climber searches it (a
`verify.py` imports the function, so it must sit at the top level of a .py
file), and it is a problem like any other from then on.

A block may be a name, a class, or an instance — an instance stands for its
class and params, and a search always builds its own. `hc.Climber("openevolve",
params={"num_drafts": 5})` starts from a preset.

Your own classes are written down the most portable way possible: by
registry name, as `module:Class` when the module is importable, else as
`its_file.py:Class`. The file form means a search imports your script again
— keep the code that starts a search under `if __name__ == "__main__":`. A
class that exists only in the running process (a notebook cell) still runs
with `hc.run`, but the climber is not `portable`: `to_spec()`, `resume` and
detached engines refuse it, saying why.

### Reading what a search found

After `search`, the search is readable off the climber:

```python
climber.best                 # the best candidate by validation score (`selected` is the one that ships)
climber.candidates           # every candidate, in creation order
climber.history              # every rise of the best-so-far: (minutes, candidate_id, score)
climber.spend                # evaluations, tokens, cost_usd, seconds
climber.solution             # the source that ships; climber.params its parameter values
climber.to_frame()           # one pandas row per candidate
climber.result               # the whole SearchOutcome: .source("c004"), .journal, .meta, .status
open_search("run-id/search-id")   # an earlier search (no ref: the latest); a running one is followed
```

### One step at a time

`climber.start` opens the same search and leaves the loop to you:

```python
budget = Budget(evaluations=30)
climber.start(Problem("fitness-landscape"), budget=budget, agent="toy", learning=False)

action = climber.propose()       # what the policy would do next; nothing runs
outcome = climber.run(action)    # run it: attempted, scored, committed, observed by the policy
outcome = climber.step()         # both in one call; None when there is nothing to run
climber.run(Action("improve", target_id="c002"))   # a move of your own
climber.state                    # what the policy sees: journal, in flight, budget
climber.finish()                 # the climber's own loop runs the rest
result = climber.close()         # settle it; the same object `search` leaves as climber.result
```

The budget's clock runs only while a step does. An action that cannot run
(an unknown candidate, a spent budget) comes back as a `rejected` outcome
saying why. A search closed with budget left is `stopped`, and
`hillclimb resume` continues it; one a budget closed is `done`. It is an
ordinary search, so `hillclimb watch` shows it while you step. One search
at a time per process, and a `loop:` climber (gepa) owns its control flow:
it can only `finish()`.

The `toy` agent makes all of this free to try: on the bundled
`fitness-landscape` problem a solution is one point on a terrain, so a
search takes seconds and its scores move. [`examples/`](../examples/) has a
script for each of the above and for writing your own policy, operator,
selector and scripted agent.

## From 0.5

- A directory climber (`climbers/mine/climber.yaml`) is no longer something
  to point at: `hillclimb climber show climbers/mine` prints it as the block
  to paste into your run config.
- `climber: {ref: NAME, ...}`, `climber.ref` in `--set` and experiment specs,
  the `operators: {name: {...}}` overlay and the `learning.*` behaviour
  settings still load; they read as the block they meant.
- MAP-Elites' settings are `selector_params`, not the policy's `params`.
- The renamed SDK names are listed in the [changelog](../CHANGELOG.md).
