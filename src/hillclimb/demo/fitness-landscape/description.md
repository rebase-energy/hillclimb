# Fitness landscape: stand on the highest point

A fixed 2D terrain is defined over the square `[-5, 5] x [-5, 5]`: a dozen
Gaussian peaks of different heights and widths over a gentle cosine ripple.
Your job is to **submit the single point with the highest elevation you can
find**. The score is the terrain height at your point — higher is better.

The terrain is deterministic and open-book: `problem/landscape.py` defines it,
and your script may import it and evaluate `elevation(x, y)` as often as the
time budget allows (`numpy` is available):

```python
import sys
sys.path.insert(0, "problem")
from landscape import elevation, DOMAIN
```

Be warned about its shape: the broad peaks that are easy to find are decoys.
The global maximum is a **narrow needle** — coarse grids step right over it,
and a local optimizer only finds it when started close by. Budget your
evaluations between exploring widely and refining locally.

## Submission format

Write `submission.csv` in the working directory with the header `x,y` and
exactly one row — your chosen point, inside the domain. See
`sample_submission.csv` (the origin, a weak valid baseline).

## Scoring

The orchestrator runs `problem/verify.py` after your script finishes. It
validates the point and prints `val_score: <elevation>` (or 0.0 with a reason
if the submission is invalid). The report also carries the local uphill
gradient at your point, so you can see which way the ground rises from where
you stood.

This is a demo problem: every candidate maps to one point on a known surface,
so the whole search — every candidate and the lineage between them — can be
drawn as a trajectory on the 3D terrain.
