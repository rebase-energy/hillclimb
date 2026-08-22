"""Baseline: one circle filling the square, 25 zero-radius circles along
the edge. Valid by construction, sum of radii = 0.5 — the floor every
agent draft has to beat."""

import csv

N = 26
rows = [(0, 0.5, 0.5, 0.5)]
# zero-radius "circles" on the boundary: inside the square, no overlap
# (dist to the big circle's center is >= 0.5), distinct positions
for i in range(1, N):
    t = i / N
    x, y = (t, 0.0) if i % 2 else (0.0, t)
    rows.append((i, x, y, 0.0))

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "x", "y", "r"])
    writer.writerows(rows)
print("baseline: one circle, sum of radii = 0.5")
