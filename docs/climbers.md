# Climbers

A climber is the shareable bundle that decides *how* to hillclimb — a search policy (or a whole loop), the operators it may use, their prompts and a tuner — that the fixed harness runs.

*What to try next* is the climber's, and it is a seam of its own:
`src/hillclimb/modules/policies/base.py` defines the `SearchPolicy` protocol,
`src/hillclimb/modules/policies/` holds the implementations, and
`src/hillclimb/climbers/<name>/climber.yaml` bundles one with its operators,
prompts and tuner. Everything else — candidate dirs, prompts, agent calls,
trials, holdout, journaling, `best/` — is harness, and a climber never
touches it.

| climber | what it does |
|---|---|
| `greedy` | debug the newest failing/buggy tip > ensemble in the final budget window > draft until `num_drafts` branches are scored > improve the best |
| `openevolve` | [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)'s MAP-Elites database decides what to expand: a population kept diverse over feature dimensions, split across islands with migration; parent + inspirations sampled per island (exploration / elite archive / fitness-weighted). Hillclimb's operators do the mutating, the verifier the scoring, and the `debug` rule is kept. `pip install 'hillclimb[openevolve]'` |
| `gepa` | [GEPA](https://github.com/gepa-ai/gepa) owns the whole loop — reflective mutation over evaluation feedback and Pareto selection over the verifier's per-instance scores — while hillclimb evaluates, journals, and holds the private holdout. A climber that brings its own `SearchLoop`, not a policy (see below). `pip install 'hillclimb[gepa]'` |

```yaml
# config.yaml — OpenEvolve's quality-diversity search over hillclimb's operators
climber:
  ref: openevolve
  params:
    num_islands: 3
    feature_dimensions: [complexity, score]   # built-ins: complexity, diversity, score
    num_inspirations: 2                       # copied in as candidate_<i>.py
```

`climber: greedy` is shorthand for `climber: {ref: greedy}`; `--climber REF`
on the command line is the same setting. A ref is a bundled name, a directory
holding `climber.yaml`, or one `.py` file.

## Several optimizers on one problem: a mixed fleet

Repeat `--climber` and one run holds one search per climber, each its own
engine process on the same problem, tagged as an arm so the arms can be
compared afterwards. Per-arm settings go through `--arm-set ARM:KEY=VALUE`
(applied after `--set`, which is fleet-wide); arms are named after their
climber (a repeated climber becomes `greedy-2`):

```bash
uv run hillclimb run circle-packing --budget 30m \
  --climber greedy --climber openevolve --climber gepa \
  --seed-from hillclimb/experiments/seeds/circle-packing.py \
  --arm-set gepa:concurrency.parallel_operators=1 \
  --arm-set gepa:climber.params.max_metric_calls=60
uv run hillclimb watch                          # the three searches side by side, arm in the problem column
uv run hillclimb experiment report <run-id>     # arms compared; --experiment NAME names it instead
```

The GEPA climber is serial, so its arm needs `concurrency.parallel_operators=1`
while the others keep the fleet-wide operator count. `--parallel-searches N`
repeats every arm N times (repeat-major, like `hillclimb experiment run`).
Searches of one run share live knowledge cards; pass `--set
learning.enabled=false` for a fair comparison, or keep it for cooperation.
The same fleet is available to embedders as `hillclimb.api.run_fleet(...,
engines=mixed_fleet([...]))`. For repeats across problems with a committed
spec, noise floors and a control arm, use `hillclimb experiment run` (see
[experiments.md](experiments.md)).

Any other feature dimension must be a numeric key the verifier writes next
to `score` (see [Replicate metrics](problems.md#replicate-metrics-optional)),
e.g. `feature_dimensions: [runtime_s, score]`. Each evolved candidate's
`policy_meta` records its island, grid cell and inspirations in the journal
(`hillclimb show <candidate>` prints it); the watch TUI and knowledge graph
don't surface it yet.

## Writing a search policy

A policy is two methods over a read-only `PolicyInput`:

```python
class SearchPolicy(Protocol):
    name: str
    params: dict   # persisted into SearchMeta, so `resume` restores them

    def propose(self, view: PolicyInput) -> Action | None: ...
    def observe(self, view: PolicyInput, candidate: Candidate) -> None: ...
```

`propose` returns one `Action` — an operator (`draft`/`debug`/`improve`/
`ensemble`), the candidate to target, optional `inspiration_ids`, and
optional per-action routing — or `None` to hold the slot until an in-flight
result lands. `observe` is called after every terminal result, and replayed
over every existing candidate when the policy is constructed, which is what
makes `hillclimb resume` work.

Three rules the harness relies on, spelled out in `modules/policies/base.py`:

- `propose`/`observe` run only on the scheduler thread, under the search's
  state lock. A policy may read candidate dirs; it must never write.
- Every decision must be derivable from replayed journal state — compute it
  from the `PolicyInput`, or rebuild your caches in `observe`.
- Ensemble-style actions must carry their inputs in `inspiration_ids`; the
  harness copies those solutions into the new candidate dir.

To add one: implement the protocol in a file and point `climber.ref` (or
`--climber`) at it, no registry edit needed. Any value ending in `.py` is a
one-file climber, relative to the folder holding the hillclimb dir (like
`paths.runs_dir`):

```python
# hillclimb/climbers/drafts_only.py
from hillclimb.modules.policies.greedy import GreedyPolicy
from hillclimb.sdk import Action


class DraftsOnly(GreedyPolicy):
    name = "drafts-only"

    def propose(self, view):
        tip = self.debuggable_tip(view)
        if tip is not None:
            return Action(operator="debug", target_id=tip.candidate_id)
        return self._draft_action(view)
```

```bash
uv run hillclimb climber check --climber hillclimb/climbers/drafts_only.py   # before spending budget
uv run hillclimb run circle-packing --climber hillclimb/climbers/drafts_only.py
uv run hillclimb run circle-packing --climber greedy --climber hillclimb/climbers/drafts_only.py  # fleet: arm "drafts_only"
```

The file exposes its policy as the one class it defines with `propose`
and `observe`, or as `POLICY = <class or factory>`; the constructor gets
`params` (from `climber.params`) and `complexity_start` when it
accepts them. `search.yaml` records the climber as written (`climber`) and
`climber_sha256`, its hash at search start: the identity of an
edited exploration process, the way `seed_sha256` identifies a seed. A
search snapshots its climber into `searches/<id>/climber/` and `resume`
loads that copy, noting when the live file's hash has changed since — so
editing the live file never changes a started search. Bundled names
(`greedy`, `openevolve`, `gepa`) are the manifests under
`src/hillclimb/climbers/<name>/climber.yaml`.
`modules/policies/greedy.py` is under 300 lines and is the reference. Beam search, MCTS,
evolutionary populations, novelty search and bandits over operators all fit
this shape — greedy is just the one that ships.

The whole exploration process is one dict plus one file. Every knob greedy
reads — `num_drafts`, `max_debug_depth`, the `ensemble*` window and the
`tune_*` budget — comes from the climber's `params` (the manifest's, with
whatever the user set in `climber.params` laid over it), so an experiment
arm (or, later, an agent editing the loop) is handed a single dict;
`GreedyPolicy.resolved_params()` is that dict fully resolved. Before
spending an agent hour on an edited process, run the conformance check:

```bash
uv run hillclimb climber check [--climber REF] [--set climber.params.k=v] [--problem P] [--smoke]
```

It replays every recorded journal in the store (plus an empty one) through
the policy with no agent or verifier and reports each contract breach it
can see: a hold with empty slots on an empty journal (the search would
never start), two fresh instances disagreeing at some budget point (resume
would diverge), a target or inspiration id that does not exist, an
operator that needs a target without one, a `debug` on a non-failing/non-buggy
candidate, a mutated journal or a file written under a search dir, a
factory that hands back the same object, and a prompt override that
lints dirty. `--smoke --problem P` then runs a short `--backend dummy`
search so the whole loop, prompts included, executes once; `--json` is the
machine-readable form. Exit 1 on any breach.

Operator prompts are part of that process too. A climber's manifest may name
a `prompts:` dir next to it; any template in it (`<name>.md`) shadows the
built-in operator template of the same name (`draft`, `improve`, `debug`,
`ensemble`, the cue snippets); an override may drop `{{tokens}}` but never
add one the engine does not fill, and the `contract_*` templates are the
harness's and cannot be shadowed — the engine refuses to start on such a
file. `climber_sha256` covers the prompts, so two searches are comparable
only when their climber hashes agree.

## Starting your own

```bash
hillclimb climber list                          # the bundled climbers and every one under hillclimb/climbers/
hillclimb climber new mine --from greedy        # copy one into hillclimb/climbers/mine/
hillclimb climber check --climber hillclimb/climbers/mine          # replay recorded journals through it
hillclimb climber check --climber hillclimb/climbers/mine --problem circle-packing --smoke   # + one dummy-backend search
hillclimb run circle-packing --climber hillclimb/climbers/mine
```

A local climber is named by its path, relative to the folder holding the
hillclimb dir (`climber new` prints the exact ref); only the bundled ones go
by a bare name.

`climber new` copies the manifest, the policy source and the prompts so every
part is a file you can edit: a bundled `module:Class` policy is copied in as
`<module>.py:Class` — greedy becomes `greedy.py:GreedyPolicy` — and
`climber.yaml` points at that file. Module refs inside a manifest are
`file.py[:Class]` relative to the climber dir, so files may import each
other. The manifest names exactly one of `policy:` or `loop:`, plus `params`,
`operators` (built-in names or `file.py:Class`, each optionally with params
such as `- draft: {retrieval: true}`), `memory`, `tuner`/`tuner_params`,
`similarity` and `prompts`; `routing` is reserved — which backend and model
run is the user's choice in `config.yaml`, never a climber's. The user's
`climber.params`, `climber.operators`, `climber.tuner` and `climber.memory`
lay over the manifest without copying it.

## GEPA: a climber with its own loop (optional extra)

Some optimizers cannot be reduced to "what next?" — they own proposal,
reflection, and selection themselves. Those bring their own **`SearchLoop`**:
the manifest names `loop:` instead of `policy:`, and the loop drives the
harness (`submit`/`wait`/`run`) instead of answering `propose`. The
architecture is `optimizer-host-plan.md`; GEPA is the first such climber:

- greedy: hillclimb chooses the parent and asks an agent to mutate;
- `openevolve`: OpenEvolve supplies selection/inspirations, hillclimb's
  agent still mutates;
- `gepa`: GEPA drives reflective mutation and Pareto search, hillclimb
  evaluates and records.

```bash
uv sync --extra gepa
uv run hillclimb run <problem> --climber gepa --seed-from my_solution.py
```

GEPA's reflective mutation is the `gepa-reflect` operator, run by a routed
hillclimb agent (configure `routing.gepa-reflect`, falling back to
`routing.default` and the global backend/model) in an ordinary candidate
dir; every evaluation becomes a normal journaled `cNNN` candidate, so
`watch`, `tree` and `chart` work unchanged. GEPA checkpoints under
`SEARCH_DIR/loop/state/` and `hillclimb resume` continues both the journal
and the optimizer, with a warm evaluation cache keyed on the solution's
hash so replayed proposals cost nothing.

MVP limits: one mutable file (`solution.py`), serial
(`concurrency.parallel_operators: 1`), no merge, and a **required executable
seed** — pass `--seed-from` or ship an executable baseline. Holdout privacy
is strict and one-way: holdout scoring runs only after the optimizer
finishes, and no holdout value ever reaches GEPA's prompts, feedback, or
state (regression-tested with sentinels). When the verifier emits per-instance
scores (see [Per-instance scores](problems.md#per-instance-scores-optional)),
`frontier_type: instance` (the default) tracks GEPA's Pareto
frontier per instance; without them the frontier degenerates to the aggregate
score.
