# GEPA × OpenEvolve × Greedy — three-arm live results

*Paths as written at the time: `search_strategy.py`, `policy.py`, `evaluation.py` and friends moved in the 2026-09 package-layout refactor — see `docs/package-layout-plan.md`.*


The Phase 9 acceptance experiment from `docs/gepa-integration-plan.md`: one
problem, three search engines, one shared seed, three repeats each. The gate is
honest execution and measurement, not a win for any arm.

**Result: no arm is distinguishable from the greedy control.** Both gaps fall
inside the configured noise floor, so this experiment says nothing about which
optimizer is better — only that all three run correctly end to end.

## Environment

| | |
|---|---|
| repository commit | `3066980` (repeats 2–3); `b06ac34` (repeat 1) — see *Deviations* |
| worktree | dirty: 3 untracked files (`docs/einsteinarena-smoke-results.md`, 2 knowledge YAMLs); no tracked modifications |
| hillclimb | 0.2.0 |
| gepa | 0.1.4 |
| openevolve | 0.3.2 |
| Python | 3.12.12 |
| platform | macOS-15.5-arm64 (10 cores) |
| agent / model | `claude-code` / `sonnet` |
| authentication | subscription mode — `agents/claude_code.py:subscription_env` drops `ANTHROPIC_API_KEY` from the child env so calls bill the Max subscription, not the API |
| experiment spec | `hillclimb/experiments/gepa-vs-openevolve-vs-greedy.yaml`, sha256 `f682d9c31378` |
| shared seed | `hillclimb/experiments/seeds/circle-packing.py`, sha256 `217003dfdf96` |
| run id | `20260901-193739-gepa-vs-openevolve-vs-greedy` |
| wall clock | 2026-09-01 19:37 → 22:36 BST |
| total model cost | $12.08 |

Every arm shared: 15-minute budget, `search.n_trials=1`, `search.parallel_agents=1`,
`learning.enabled=false`, the same verifier, the same metric direction (higher is
better), and the same seed file and hash. Only `search.policy` and
`search.policy_params` varied.

## Commands

Repeat 1, launched from a separate session:

```bash
uv sync --extra openevolve --extra gepa
uv run hillclimb experiment run gepa-vs-openevolve-vs-greedy --sequential
```

Repeats 2 and 3, after that launcher was killed (see *Deviations*), tagged into
the same run id. The job list, override serialization and child launch context
come from hillclimb's own `expand()`, `_set_value()` and `_child_launch_context()`,
so the argv is identical to what `experiment run` would have produced:

```bash
uv run python finish_experiment.py   # repeats 2 and 3, --run-id 20260901-193739-…
```

Nine-job expansion verified against the spec:

```bash
uv run hillclimb experiment run gepa-vs-openevolve-vs-greedy --dry-run
# Experiment gepa-vs-openevolve-vs-greedy: 3 arms × 1 problem(s) × 3 repeat(s)
#   = 9 searches, sequential
#   shared seed: …/seeds/circle-packing.py (sha256 217003dfdf96)
```

## Per-search results

All nine reached terminal state `done`. Every search scored the shared seed at
0.5, evaluated at least three post-seed candidates, and selected a non-null
candidate. Circle packing ships no hidden holdout, so selection legitimately
falls back to the validation score (`holdout` is empty by design, not by failure
— the GEPA holdout-leakage tests in `tests/test_gepa_privacy.py` remain the
separate privacy gate, and pass).

| arm | rep | search | state | seed | selected | val score | cands | verifier calls | invalid | wall | tokens | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 1 | `circle-packing` | done | 0.5 | c002 | 2.635896 | 5 | 5 | 0 | 818s | 1.18M | $0.80 |
| openevolve | 1 | `circle-packing-2` | done | 0.5 | c002 | 2.629619 | 6 | 6 | 0 | 881s | 2.25M | $1.26 |
| gepa | 1 | `circle-packing-3` | done | 0.5 | c004 | 2.635941 | 5 | 5 | 0 | 955s | 1.65M | $1.09 |
| greedy | 2 | `circle-packing-4` | done | 0.5 | c005 | 2.630727 | 11 | 11 | 0 | 826s | 3.51M | $2.19 |
| openevolve | 2 | `circle-packing-5` | done | 0.5 | c004 | 2.604008 | 9 | 9 | 0 | 843s | 2.99M | $1.72 |
| gepa | 2 | `circle-packing-6` | done | 0.5 | c003 | 2.624838 | 5 | 5 | 0 | 882s | 1.89M | $0.97 |
| greedy | 3 | `circle-packing-7` | done | 0.5 | c002 | 2.628037 | 5 | 5 | 1 | 869s | 1.83M | $1.14 |
| openevolve | 3 | `circle-packing-8` | done | 0.5 | c003 | 2.635968 | 9 | 9 | 0 | 829s | 2.84M | $1.72 |
| gepa | 3 | `circle-packing-9` | done | 0.5 | c004 | 2.631090 | 5 | 5 | 0 | 860s | 1.92M | $1.19 |

All searches are under `runs/20260901-193739-gepa-vs-openevolve-vs-greedy/searches/<search>/`.

GEPA candidates appear as ordinary journaled candidates with parent lineage and
agent accounting; GEPA's own state, proposals and identity live beside them in
`gepa/{state,proposals,identity.json}`.

## Per-arm comparison

From `uv run hillclimb experiment report gepa-vs-openevolve-vs-greedy`:

| arm | n | mean | median | spread | wins | min→best | tokens |
|---|---|---|---|---|---|---|---|
| greedy (control) | 3 | 2.6316 | 2.6307 | 0.00786 | 1 | 9.23 | 2.2M |
| openevolve | 3 | 2.6232 | 2.6296 | 0.032 | 1 | 7.55 | 2.7M |
| gepa | 3 | 2.6306 | 2.6311 | 0.0111 | 1 | 10.9 | 1.8M |

- openevolve: **−0.008355** vs greedy — within noise (0.01), not a result; wins 1, loses 2, ties 0
- gepa: **−0.0009303** vs greedy — within noise (0.01), not a result; wins 2, loses 1, ties 0

Each arm won exactly one repeat.

## Budget and cap behavior

No arm hit a cost limit. Every search ended on its 15-minute wall clock (818–955s
of a 900s budget), which is the intended binding constraint.

GEPA did not reach `max_metric_calls`. The ceiling is 500 **per instance** and
circle packing exposes 26 instances; at five evaluated candidates per GEPA search
that is 5 metric calls per instance, two orders of magnitude below the cap —
exactly the sizing `b06ac34` set out to guarantee.

## Deviations, failures and retries

Nothing was deleted; every original record is still on disk.

1. **The repeat-1 launcher was killed.** At 20:33:20 the `experiment run`
   process and its live engine both took a SIGTERM in the same second, after
   repeat 1 had completed for all three arms. The engine's status went to
   `stopped` with its journal truncated mid-candidate and no control event —
   the signature of an external process-group reap, not a hillclimb fault. The
   launcher belonged to a different session's background shell. Repeats 2–3
   were relaunched detached via `os.setsid()` so a session reap cannot reach
   them.

2. **A truncated search was excluded from the statistics.** That kill left a
   greedy repeat-2 search stopped at 663s of its 900s budget. `experiment report`
   counted it as a legitimate sample (greedy read `n=2, mean 2.6338`), which
   silently mixes a budget-starved arm into the comparison. It was moved — not
   deleted — to
   `runs/20260901-193739-gepa-vs-openevolve-vs-greedy/aborted/circle-packing-4`,
   outside the `searches/` directory the store scans. **This is a reporting bug
   worth fixing: `experiment report` should exclude searches whose terminal
   state is not `done`.** The greedy repeat 2 in the table above is a full
   re-run, which reused the freed `circle-packing-4` search id.

3. **The code changed mid-experiment.** Repeat 1 ran at `b06ac34`; repeats 2–3
   ran at `3066980`, three commits later. The intervening `e3592a3` touched
   `evaluation.py`, `search.py`, `control.py`, `problem.py` and `baseline.py` to
   thread a problem-declared `output_artifacts` list through the engine. For a
   local problem the list defaults to `["submission.csv"]`, reproducing the
   previous hardcoded literal on every path this experiment exercises, and the
   one genuine behavior change (`resync_best` clearing declared outputs before
   copying) fires only on prune, which no search performed. The scoring path is
   therefore unchanged — but a strictly clean experiment would re-run repeat 1
   at a single commit, and this doc should not be read as if it had.

4. **The completion script continues past a failed job** where `experiment run`
   aborts the whole experiment on a non-zero exit. No job exited non-zero, so
   the deviation had no effect.

## Interpretation

Both gaps to the greedy control are inside the 0.01 noise floor configured in the
spec, so **neither GEPA nor OpenEvolve is distinguishable from greedy here**, and
the direction of each gap (both slightly negative) carries no information. With
n=3 and per-arm spreads of 0.008–0.032 — the openevolve spread being three times
the noise floor on its own — this design can only detect large effects. Nothing
here supports a claim that any engine is better or worse.

What the experiment does establish is the Phase 9 gate: all three engines run to
`done` under identical conditions, GEPA produces canonical journaled candidates
with correct lineage and agent accounting, and its budget behaves as designed.

Two observations that are *not* results but are worth following up:

- GEPA used the fewest tokens per search (1.8M vs greedy 2.2M and openevolve
  2.7M) while landing in the same score band. If that holds up it is a
  cost story rather than a quality one, and it needs its own experiment.
- Greedy repeat 2 evaluated 11 candidates against the 5 that every GEPA search
  managed, at a similar wall clock. Candidate throughput differs sharply between
  engines and is not controlled for here.

A conclusive comparison needs more repeats, more than one problem, and a noise
floor measured with `hillclimb verify circle-packing --repeat 5` rather than
assumed.
