# Rerun the walkthrough's heilbronn-convex-13 search before launch

Status: open (noted 2026-10-08, launch the week after)

## Why

The docs walkthrough (`hillclimb-web/docs/content/docs/(documentation)/walkthrough.mdx`)
shows the winning `solution.py` of a heilbronn-convex-13 search and puts its score next
to AlphaEvolve's: 0.0309372 against 0.0309369. The code starts from hard-coded
coordinates:

```python
best_pts = polish_iterated(REF_PTS, rounds=3)  # known-good seed
```

The problem gives no coordinates (`baseline.py`, `description.md`), so a reader will ask
where `REF_PTS` came from. A friend who tested hillclimb before launch asked exactly
that. Their own 10-minute run reached 0.029. The answer changes what the result means:

- **built during the search:** an earlier candidate found the points and a later one
  started from them. That is fine, and worth one sentence in the walkthrough.
- **recalled by the coding agent:** the model remembered a published configuration. Then
  the search mostly polished a known answer, and the AlphaEvolve comparison is weak.
- **leaked:** from knowledge/memory of an earlier search, or a file the coding agent could
  read. This repo root is itself a hillclimb dir with `best-known/` next to `problems/`
  and `runs/`, and `best-known/heilbronn-convex-13.csv` holds this very construction. It
  is unknown whether the sandbox stops a coding agent from READING it (it limits writes).

## What we know

- `best-known/index.json` credits the 0.030937207942191215 construction to "a hillclimb
  search (GPT-5.6, 2026-08-24)", run `20260824-121040-heilbronn-convex-13-gpt-5.6-sol`.
- That run folder is not in this checkout's `runs/` or anywhere under `~/Documents/Github`
  (searched 2026-10-08), so the first candidate that wrote `REF_PTS` cannot be traced.

## To do

1. If the run turns up (another machine, a backup): find the first candidate whose
   `solution.py` contains `REF_PTS`, read its parent's notes and submission, and say in
   the walkthrough where the points came from.
2. Otherwise rerun the search:
   - in a fresh folder (`hillclimb init` somewhere other than this repo), so no
     `best-known/`, `knowledge/` or earlier `runs/` is reachable
   - with `--no-learning`, and no `--seed-from`
   - same problem, climber and budget the walkthrough describes (30 min)
   - check the winning code: no hard-coded coordinates that did not come from an
     earlier candidate of the same search
3. Update the walkthrough with that run's code, score and figure. If it does not reach
   AlphaEvolve, drop the "beats AlphaEvolve" framing; a clean ~0.029 is still above
   OpenEvolve (0.0267) and AdaEvolve (0.0290) on the chart.
4. Optional: check whether a sandboxed coding agent can read files outside its candidate
   dir (`hillclimb sandbox check`), and whether `best-known/` should live outside the
   hillclimb dir in this repo.
