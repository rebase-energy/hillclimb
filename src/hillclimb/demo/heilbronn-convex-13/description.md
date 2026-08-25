# Normalized Heilbronn problem: 13 points

Place **exactly 13 points in the plane** so that the smallest triangle formed
by any three points is as large as possible relative to the convex hull of all
the points. There are C(13,3) = 286 triangles.

The score is

```text
minimum triangle area / convex hull area
```

Higher is better. The normalization makes translation and uniform scaling
irrelevant, so coordinates do not have to lie in a particular box. A
degenerate point set (zero-area convex hull), duplicate points, or three
collinear points scores zero.

This is a continuous, non-smooth max-min optimization problem. Useful
approaches include structured or random starts followed by simulated
annealing, differential evolution, SLSQP on a smooth approximation, and many
restarts. `numpy`, `scipy`, and `pandas` are available. Keep each candidate's
own numerical search comfortably below a minute so the 30-minute Hillclimb run
can test and improve multiple candidates.

## Submission format

Write `submission.csv` in the working directory with this header:

```csv
id,x,y
```

Provide exactly 13 rows, with each id from 0 through 12 appearing once. Both
coordinates must be finite numbers. See `sample_submission.csv` for a weak but
valid starting point.

## Scoring and reference lines

The verifier reports `normalized-min-triangle-area`; higher is better. The
Hillclimb chart includes comparison lines for OpenEvolve and AdaEvolve under a
100-candidate GPT-5 evaluation, plus the published AlphaEvolve reference. The
AlphaEvolve evaluation used a different, undisclosed candidate budget, so its
line is a target rather than a budget-matched comparison.

There is no train/test data and network access is disabled.
