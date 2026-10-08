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
      operator_policy: climbers/greedy/policy.py:Greedy         # π_op: which operator on what the selector policy chose (or `loop:` — the whole control flow)
      params: {num_drafts: 5}
      selector_policy: climbers/openevolve/policy.py:MapElites  # π_sel: WHICH candidate the next attempt starts from
      selector_params: {num_islands: 3}
      operators: [draft, debug, improve, crossover.py:Crossover]   # HOW an attempt is made
      operator_params: {draft: {retrieval: false}}
      tuner: optuna                 # WHICH parameter values a tunable candidate tries
      tuner_params: {seed: 7}
      memory: files                 # what it knows from other searches, and leaves them
      memory_params: {max_cards: 1}
      prompts: prompts/             # templates that shadow the built-in ones by name
```

Every key but one is optional: a block names its `operator_policy:` or its
`loop:` (the engine ships no default climber — `hillclimb climber get greedy`
fetches one), and everything else defaults to what its class says.

## Where the block goes

The same block in three places, highest first:

| where | what it is |
|---|---|
| an entry of a run spec (`problems: [- {target: ..., climber: {...}}]`) | that search's climber |
| the top of a run spec (`climber:` beside `problems:`) | the default for its entries |
| `runs/config.yaml` | the folder's default (a run default): what `hillclimb run <problem>` uses |

A higher one **replaces** a lower one whole — blocks are never merged, so
one policy's params cannot end up under another policy. Two things then
apply on top of whichever block was chosen:

- `--climber REF` names a climber outright (a `.py` file, or a folder `climber get` wrote).
- `--set climber.params.num_drafts=5` (and `set:` in a spec entry, and an
  experiment's overrides) edits a field of the block.
  `--set climber.operators.draft.retrieval=false` reaches one operator's params.

A run records what it ran: `runs/<run-id>/spec.yaml` carries every search's
full block, and `hillclimb run runs/<run-id>/spec.yaml` runs it again.

## Naming a module

Every slot names its module the same three ways:

| form | example |
|---|---|
| a registry name | `draft`, `optuna`, `files` — the built-in operators, tuners and memories |
| a file | `mine.py` (the one class of that kind it defines) or `mine.py:Class` |
| an importable class | `mypackage.policies:Annealed` |

A whole climber is also named by a **folder** holding `policy.py` (what
`hillclimb climber get` writes, see below: the file builds the whole
`Climber(...)`): `climber: climbers/greedy` names that file. A pre-0.9 folder
holding `climber.yaml` still loads; its file refs and `prompts:` are relative
to the folder.

A file is relative to the file the block is written in (the spec), or to
the hillclimb dir (`runs/config.yaml`). Files may import each other relatively (`from .helpers
import x`). `hillclimb climber list` shows the catalog and what is registered.

A bare string instead of a block is one `.py` file, or a folder holding one.

## The catalog

hillclimb defines no climber of its own. It ships a **catalog** of examples —
the repository's `climbers/` folder, bundled in the wheel — and `hillclimb
climber get <name>` copies one into your `climbers/`, like `problem get`:

| catalog | what it is |
|---|---|
| `greedy` | `Best` over `Greedy`, both in one file: debug the newest failing tip > ensemble in the final budget window > draft until `num_drafts` branches are scored > tune > improve the best |
| `openevolve` | the same schedule, ensemble and tune off, over [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)'s MAP-Elites archive: a population kept diverse over feature dimensions, on islands with migration. `pip install 'hillclimb[openevolve]'` |
| `gepa` | [GEPA](https://github.com/gepa-ai/gepa) owns the whole loop (see below): a folder of several files; its `requirements.txt` names the `gepa` library |

In Python the same files are `hc.catalog.climber("greedy")` (a `Climber`, read
in place) and `hc.catalog.module("greedy")` (its classes, to subclass or
compose with). The names `greedy`, `best`, `openevolve`, `map-elites` and
`gepa` are not something the engine resolves any more: a search recorded
before 0.9 that names them still resumes (its record finds the catalog
file), a new config is told to fetch the climber.

## The catalog climber as a folder you can read

```bash
hillclimb climber get greedy        # -> climbers/greedy/, and `climber: climbers/greedy/policy.py` in runs/config.yaml
```

copies the catalog's greedy out as a folder whose `policy.py` **is** the climber:

```text
climbers/greedy/
├── policy.py           # the whole climber: Best (which candidate next), Greedy (which operator on it),
│                       # and the Climber(...) that wires them to the operators, tuner, memory and prompts
└── prompts/
    ├── README.md       # how a prompt is made: when each template is used, what fills each token
    ├── draft.md        # the templates its operators render — the words the coding agents get
    ├── research_cue.md
    ├── debug.md
    ├── improve.md
    ├── ablation_cue.md
    └── ensemble.md
```

`policy.py` is the package's own file, copied as it is: nothing about the
search is hidden behind it. `Best.schedule` is the order of every step (a
failing tip first, the ensemble window, roots until `num_drafts`, then the
best scored candidate nobody is building on), `Greedy.propose` the operator
for what it chose (draft, debug, ensemble, a tune trial, improve), and every
knob's value is a `DEFAULTS` entry next to the code that reads it. The copy
ends with what makes the file a whole climber — there is no config file:

```python
climber = Climber(
    selector_policy=Best(),
    operator_policy=Greedy(),
    operators=[Draft(), Debug(), Improve(), Ensemble()],
    tuner=RandomSearch(),
    memory=FilesMemory(),
    name='greedy',
)
```

`Best(num_drafts=5)` or `Greedy(tune_budget=0)` there changes a default for
good; `--set climber.params.tune_budget=0` still does for one run. The
`prompts/` beside the file is its prompts dir without being named (any
climber file's is; `Climber(prompts_dir=...)` points elsewhere). The folder
names the file (`climber: climbers/greedy` and `climber:
climbers/greedy/policy.py` are the same climber).

A template is the climber's words plus `{{tokens}}` the harness fills in for
one attempt: the problem, the candidate the attempt builds on, what earlier
attempts tried, what memory knows from other searches. The two policies
decide which operator makes the attempt; the operator renders its template;
the harness adds the contract (`{{contract}}`: how the solution is run and
scored, the same for every climber and never a template of yours).
`prompts/README.md` spells it out for the folder's own templates: the
schedule with its numbers, one row per token with what fills it and when it
is empty.

The copy becomes this folder's climber (`--no-default` leaves
`runs/config.yaml` alone), so change a threshold in `policy.py` or a sentence
in `improve.md` and the next `hillclimb run` climbs with it; `hillclimb
climber check --climber climbers/greedy/policy.py` replays recorded
searches through the edited policy and lints a template's tokens first. The
file is named like any climber — `climber: climbers/greedy/policy.py` in
`runs/config.yaml`, `--climber climbers/greedy/policy.py`, an entry of a run
spec — and has its own identity: a search that ran it is not a run of the
catalog's `greedy`, as it must not be once a line was changed. `--name` copies
under another name (`climber get openevolve --name qd`); `hillclimb climber
new mine --from climbers/greedy` copies the folder as `climbers/mine/`. A
loop (`gepa`) owns its whole control flow and is not copied out this way.

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
| `selector_policy` | which node, or none — its `schedule`: for `best`, a failing tip first, roots until `num_drafts`, the top-k to combine in the final window, else `select` | `SelectorPolicy` | `best`, `map-elites` |
| `operator_policy` | which operator on what the selector policy chose — its `propose`: for `greedy`, draft, debug, ensemble, tune, improve | `OperatorPolicy` | `greedy`, `openevolve` |
| `loop` | the control flow itself (instead of a selector and a policy) | `Loop` | `gepa` |
| `operators` | how one attempt is made: the prompt the coding agent gets | `Operator` | `draft`, `debug`, `improve`, `ensemble` |
| `tuner` | which parameter values a tunable candidate tries | `Tuner` | `random`, `optuna` |
| `memory` | what a search knows from others and leaves for the next | `Memory` | `files`, `none` |

Which coding agent and model run is **not** a climber's: `routing:` in a block is
refused. That is the user's choice, a run default in `runs/config.yaml` (`agent`,
`model`, `routing`; a personal `agent`/`model` may sit in the user config under it).

A climber's own file imports the contracts from `hillclimb.sdk` and, to
build on a prebuilt block, that block from `hillclimb.policies` and its
siblings — nothing deeper.

### A policy

```python
# climbers/drafts_only.py
from hillclimb.policies import Greedy
from hillclimb.sdk import Action


class DraftsOnly(Greedy):
    """Never improves: whatever the selector chose, draft (repair a failing tip first)."""

    DEFAULTS = {"tune_budget": 0}         # its knobs over Greedy's; `params` in the block set them

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

The `OperatorPolicy` base decides nothing — it carries `param(name)` and
`resolved_params()` for the knobs and `self.selector`, the selector the loop
asks first. `Greedy` is the mapping written out (no node → draft, a failing
node → debug, several → ensemble, a scored node → a tune trial while it has
budget, else improve), so a policy of your own subclasses it and overrides
only where it differs; its `draft_action(state)` and `expand_action(state,
selection, operator=...)` build the usual actions. A class with just
`propose` and `observe` and no base runs too.

Three rules the harness relies on:

- A policy only ever sees a holdout-blind `SearchState`. It may read
  candidate dirs; it must never write.
- Every decision must be derivable from the journal — compute it from
  `state`, or rebuild your caches in `observe`. That is what makes `resume`
  work, and what `climber check` verifies.
- A param the block sets must be one the class declares in `DEFAULTS`: a
  typo is an error before anything is spent.

`policy.py` of the greedy climber (`climbers/greedy/policy.py` in the
repository, what `hillclimb climber get greedy` copies) is the reference: both
policies in one file. Subclass one to change one move (`greedy =
hc.catalog.module("greedy")`, then `class Mine(greedy.Greedy)` — a subclass
inherits the selector written beside its base), or copy the file and own it.

### A selector

```python
# oldest.py
from hillclimb.sdk import Selection
from hillclimb.selectors import Best


class Oldest(Best):
    """Build on the oldest scored candidate."""

    def select(self, state, *, busy=frozenset()):
        scored = state.journal.scored_candidates()
        return Selection(scored[0].candidate_id, prompt_context="Oldest first.") if scored else None
```

```yaml
climber: {operator_policy: greedy, selector_policy: oldest.py}
```

A selector implements `schedule` — the whole order of a step: which node(s)
the next attempt starts from, or `None` for a root step (the policy drafts)
— and `select`, the scored node to build on, with whatever rides along:
inspirations (copied in beside the parent), a paragraph for the prompt, a
note journaled on the new candidate. The `SelectorPolicy` base decides
nothing; `Best.schedule` is the order the bundled climbers share (a failing
tip first, the ensemble window, roots until `num_drafts`, then `select`),
so a selector of your own subclasses `Best` to keep it and replace `select`,
or writes its own `schedule`. `Best`'s knobs are its `DEFAULTS`, set as
`selector_params`: `num_drafts` (3), `debug` (True), `max_debug_depth` (3),
`ensemble` (True), `ensemble_reserve_fraction` (0.2), `ensemble_top_k` (3),
`ensemble_max_attempts` (2). A block that still writes those under `params`
(every block before 0.7) loads; they land in `selector_params`. A selector
with state keeps it a function of the journal (`sync(state)`), like a policy.

A one-file climber may hold both: the one `SelectorPolicy` subclass a file
defines (or `SELECTOR = <class>`) is its selector when the block names none,
so `--climber mine.py` needs no `selector_policy:` and a meta-problem's
candidate can change where attempts start.

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
`debug`, `ensemble`, the cue snippets); a climber named by a file needs no
`prompts:` when a `prompts/` directory sits beside the file — that one is
its. An override may drop `{{tokens}}`
but never add one nothing fills, and the `contract_*` templates cannot be
shadowed — a search refuses to start on such a file.

### A loop

Some optimizers cannot be reduced to "what next?" — they own proposal,
reflection and selection themselves. Those bring a `Loop`: the block says
`loop:` instead of `operator_policy:`, and the loop drives the harness
(`submit` / `wait` / `run`) instead of answering `propose`. A loop class
declares what it needs of its search (`operators = (...)`, `holdout_timing =
"after"`), so `{loop: gepa}` is complete.

On a problem with a hidden split (`holdout: true` in its problem.yaml), how
the climber picks the best is its own setting: `holdout: {selection:
rank-blend | holdout | val, top_k: 5, timing: inline | after}` in its block,
or `Climber(holdout={...})` in its policy.py. Unset keys are those defaults; a
climber without `holdout:` keeps the identity it had.

- `greedy`: hillclimb chooses the parent and asks a coding agent to mutate;
- `openevolve`: MAP-Elites picks parent and inspirations, hillclimb's coding agent
  still mutates;
- `gepa`: GEPA drives reflective mutation and Pareto search, hillclimb
  evaluates and records.

```bash
hillclimb climber get gepa
pip install -r climbers/gepa/requirements.txt
hillclimb run <problem> --climber climbers/gepa --seed-from my_solution.py
```

A climber that imports a library beyond hillclimb names it in a `requirements.txt`
beside its file, as a problem names its own: `climber get` prints the install line, and
a missing import fails `climber check` or the run with the same line. Nothing is
installed into your environment behind your back — a climber runs inside the engine's
process, so what it imports is your environment's.

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
hillclimb climber list                       # the catalog, your folders and one-file climbers, the building blocks
hillclimb climber get greedy                 # the catalog's greedy as a folder to read and edit (policy.py, prompts)
hillclimb climber show climbers/greedy       # a climber as the block it builds
hillclimb climber new mine --from greedy     # copy the catalog's greedy as climbers/mine/ (policy.py + prompts/)
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
class and params, and a search always builds its own. `hc.catalog.climber("openevolve")`
is the catalog's, loaded; `hc.catalog.module("openevolve")` its classes to subclass.

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
