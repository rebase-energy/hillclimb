# Concepts

The hierarchy the UI and the on-disk metadata share, coarse to fine.

```text
Run
└── Search
    └── Candidate
        └── Trial
            └── Replicate
```

- **Problem**: reusable definition under `problems/<id>/`. See
  [problems.md](problems.md).
- **Run**: one invocation of hillclimb. A single-problem run contains one
  search; a suite run contains one search per suite entry.
- **Search**: one search worker (engine process) exploring one problem with
  one climber (see [climbers.md](climbers.md)).
- **Candidate**: an immutable code artifact produced by an operator. Any change
  to the code — however small — is a new candidate with a new id.
- **Trial**: one parameter set of a candidate's code (`params`). A candidate
  that declares no tunable parameters has exactly one trial; a tuned
  candidate has several, and its score is the best trial's.
- **Replicate**: one seeded execution of a trial. A trial's score is the
  median of its replicates, so seed variance is measured, never climbed.

## Parallelism

Work runs side by side at four levels, every one a count of how much at
once. The first three are hillclimb's; the last is inside the solution, and
the one most easily forgotten.

| Level | What runs at once | Set with |
| --- | --- | --- |
| search | independent searches on the problem, one engine each | `--parallel-searches N` |
| coding agent | attempts in flight per search (tune jobs share these slots) | `--parallel-agents M` (`concurrency.parallel_agents`); `concurrency.machine_max_agents` caps them across the machine |
| replicate | of a trial's `n_replicates` seeded runs, how many at once | `--parallel-replicates P` (`concurrency.parallel_replicates`): 0 = all (default), 1 = one after another |
| solution | cores one run of `solution.py` may use | `--solution-cpus C` (`concurrency.solution_cpus`, default 1) |

How many replicates a trial gets, `evaluation.n_replicates`, is not a
parallelism setting: it is the median's sample size. `parallel_replicates`
says how many of them overlap.

One run of the solution is one run of the verifier, which starts
`solution.py` and scores what it wrote. Each such run gets `$HILLCLIMB_CPUS`
= C, with the math libraries' thread pools (`OMP_NUM_THREADS` and the rest)
capped to match; the coding agents' own test runs get the same, and the
contract prompt tells the coding agent to size any process pool from it,
never from `os.cpu_count()`.

So a run can keep up to N × M × min(R, P) × C cores busy (R the trial's
replicates), and `hillclimb run` warns when that is more than the machine
has. It matters for any solution that
searches until a deadline: on a busy machine it finds less in the same
seconds, and its score measures the load instead of the code. Nothing stops
a solution from starting more processes than it was given — a run whose CPU
time outpaces its wall time by more than 1.5 × its allotment is marked in
`hillclimb watch` (`cpu 6.9/1`: seven cores busy on a one-core allotment).

`hillclimb watch` opens on the Runs screen. Metadata carries
`schema_version: 3` (v2 folders keep loading); directories from the pre-v2
flat layout are ignored. How a run is laid out on disk is in
[hillclimb-dir.md](hillclimb-dir.md).
