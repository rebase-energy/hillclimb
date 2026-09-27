# Einstein Arena smoke validation

Date: 2026-09-01

## Implementation validation

- Provider registry, pinned snapshot/cache behavior, read-only HTTP behavior,
  local verifier bridge, and all three smoke baselines: passed.
- Generic `submission.json` handling through repeated trials, selection,
  pruning, `best/`, and `summit`: passed.
- Greedy and GEPA runner compatibility with JSON-native problem artifacts:
  passed.
- Broad integration/regression selection with a writable test cache: 220
  passed, 1 skipped; the pre-existing macOS sandbox process-tree test was
  deselected because this sandbox cannot inspect the detached grandchild.
- Full suite at the time of writing: 791 passed, 1 skipped, 21 failed — all
  pre-existing process-tree, Textual/plotui and vocabulary failures, no
  Einstein-provider test among them. Those have since been fixed
  (`91ad35a`, `3066980` and a companion plotui change); the suite now reads
  812 passed, 1 skipped, 0 failed.

## Live three-problem run

Run 2026-09-01 from a machine with outbound HTTPS; all three problems resolved
against the public API and all six searches reached terminal state `done`. Each
was a single 10-minute greedy search, holdout off, learning off, ensembling off,
`n_trials=1`. Two agents, same problems.

| problem | direction | agent | best | candidates | leaderboard #1 | gap |
|---|---|---|---|---|---|---|
| circle-packing | maximize | claude-code / sonnet | 2.62086 | 6 | 2.636 | −0.015 |
| circle-packing | maximize | codex / gpt-5.6-sol | **2.63429** | 4 | 2.636 | −0.002 |
| difference-bases | minimize | claude-code / sonnet | 2.66667 | 8 | 2.639 | +0.028 |
| difference-bases | minimize | codex / gpt-5.6-sol | **2.66587** | 4 | 2.639 | +0.027 |
| heilbronn-triangles | maximize | claude-code / sonnet | 0.02900 | 4 | 0.0365 | −0.0075 |
| heilbronn-triangles | maximize | codex / gpt-5.6-sol | **0.03412** | 3 | 0.0365 | −0.0024 |

Codex was ahead on all three, though on difference-bases only by 0.0008. Each
cell is a single 10-minute search: this is a smoke test of the integration, not
a agent comparison, and no gap here is supported by repeats.

For context on the leaderboards: circle-packing's top ten are all tied at 2.636,
difference-bases' top eight at 2.639 (with #9 at 2.6476 and #10 at 2.6537), and
heilbronn-triangles' top five at 0.0365. Ten minutes of one greedy search lands
just outside the top-ten wall on circle-packing and heilbronn-triangles, and
short of #10 on difference-bases.

### What the runs confirmed

- `search.yaml` records the pinned target
  (`einsteinarena://circle-packing@sha256:c1d749644624…`), `provider_revision`,
  and a `problem_key` with the revision stripped
  (`einsteinarena://circle-packing`), so revisions of one problem group together.
- `submission.json` travelled through evaluation, trial-0 hoisting, `best/` and
  selection with no CSV assumption anywhere.
- Metric name `arena_score`, direction taken from the problem's `scoring` field
  (`difference-bases` correctly resolved as minimize).
- Leaderboard rows became chart reference lines (10, 10 and 5 respectively).
- No registration, submission, incumbent download, or discussion post: the only
  network calls are the public problem and leaderboard endpoints.

### The seed baseline is a floor, not a competitor

Every search's `c000` scores at the degenerate extreme — 0.0 on both maximize
problems, 4.0 on the minimize problem — while any real solution lands near 2.6.
The provider's own summary calls it a "valid-by-construction Einstein Arena
seed", so this is consistent with its intent: it proves the format parses, not
that it solves anything.

The consequence is worth stating: the first draft always looks like an enormous
win, the improvement curve is dominated by that first step, and the search never
has to beat a credible incumbent. The leaderboard chart baselines supply the real
bar, but a seed that scores at the floor makes `min→best` and any
improvement-over-baseline statistic close to meaningless on these problems.
Worth deciding whether the Arena seed should instead be a simple honest
construction (a lattice packing, say) that scores in the right band.
