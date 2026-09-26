# Providers

Problems that come from somewhere else — emflow's forecasting registry, MLE-bench's Kaggle competitions, Einstein Arena's construction problems — resolved into the same verifier contract.

## emflow problems (optional extra)

With the `emflow` extra installed (`pip install 'hillclimb[emflow]'`),
targets of the form `emflow://<name>` run problems from
[emflow](https://github.com/rebase-energy/emflow)'s registry — agents author
`Predictor` classes (`solution.py` exposing `get_model()`), and the provider
supplies the verifier command: a generic evaluator fits and scores them on the
problem's validation split, with the hidden holdout as a second run of the
same command. A bare package name is a virtual
suite (one search per variant):

```bash
uv run hillclimb run emflow://gefcom2014:solar --budget 2h   # one track
uv run hillclimb run emflow://gefcom2014 --budget 2h         # all four tracks
```

The baseline candidate (`c000`) is the benchmark's reference model evaluated
for real, and a finished search ends with one official emflow Verifier run
(leaderboard row + rank, with `n_trials` recorded for selection honesty).
Programmatic use: `hillclimb.run_search("emflow://gefcom2014:solar",
budget_s=7200)`.

## MLE-bench problems

Targets of the form `mlebench://<competition-id>` run
[MLE-bench](https://github.com/openai/mle-bench) competitions against a local
mle-bench checkout (located via `paths.mlebench_python`; prepare data first
with `mlebench prepare -c <competition-id>` in that venv). Agents see only the
prepared PUBLIC split and climb on their own validation score; when the search
finishes, the engine runs `mlebench grade-sample` exactly once on the selected
candidate and writes the report (score + medal flags) to
`mlebench-grade.json` — the private test set never influences selection.

A split name is a virtual suite, one search per listed competition
(`lite` is an alias for the 22-competition `low` split):

```bash
uv run hillclimb run mlebench://spaceship-titanic --budget 2h   # one competition
uv run hillclimb run mlebench://lite --budget 4h                # MLE-bench Lite
```

## Einstein Arena problems

`einsteinarena://<slug>` resolves a public
[Einstein Arena](https://einsteinarena.com/) construction problem into a
normal verifier-backed `ProblemSpec`. Hillclimb fetches only the public problem
and leaderboard endpoints, hashes the fields that define evaluation, and runs
the downloaded `evaluate(data) -> float` verifier locally. Candidates write
`submission.json`; Hillclimb never registers an agent, submits a solution,
downloads an incumbent, or posts to a discussion.

```bash
uv run hillclimb run einsteinarena://circle-packing --budget 10m
uv run hillclimb run einsteinarena://smoke --budget 10m  # three-problem pilot suite
```

The first resolution caches a content-addressed snapshot under
`~/.cache/hillclimb/benchmark-problems/einsteinarena/`. `search.yaml` records
the pinned `@sha256:<revision>` target, so resume is reproducible and can run
offline. The verifier source is public but untrusted code: it runs in the
managed local runtime with credentials scrubbed, not in Einstein Arena's E2B
sandbox. Override the API root for a mirror or test deployment with:

```yaml
einsteinarena:
  base_url: https://einsteinarena.com
  request_timeout_s: 30
```

Benchmark integrations use the lazy `BenchmarkProvider` registry rather than
adding target-specific branches to the runner. A provider implements
`load_problem()` and `resolve_target()` (plus optional chart baselines), then
registers a URI scheme with `hillclimb.register_benchmark_provider(...)`.
The harness and every climber consume the resulting `ProblemSpec` unchanged
(the local contract is in [problems.md](problems.md)).
