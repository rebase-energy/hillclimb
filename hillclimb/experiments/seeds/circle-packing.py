"""Deterministic circle-packing seed: 26 equal circles on a 6x5 grid in the
unit square. Valid by construction (no randomness, no overlap, all inside —
the radius sits 3.3e-4 under the exact half-cell so 6-decimal rounding can
never push a circle out), fast, and far from optimal — a neutral starting
point every experiment arm improves from. Sum of radii = 26 * 0.083 = 2.158."""

import csv

COLS, ROWS = 6, 5
R = 0.083  # just under half a cell (1/12 ~= 0.08333)

rows = []
for index in range(26):
    col, row = index % COLS, index // COLS
    x = (2 * col + 1) * R
    y = (2 * row + 1) * R
    rows.append((index, round(x, 6), round(y, 6), R))

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "x", "y", "r"])
    writer.writerows(rows)

print(f"seed packing written: 26 circles, sum of radii = {26 * R:.6f}")
