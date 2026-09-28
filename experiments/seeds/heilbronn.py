"""Deterministic Heilbronn seed for any point count: N points on the
parabola y = x², x = i/(N-1). Valid by construction (a line meets a parabola
in at most two points, so no three are collinear), inside the square, fast,
and far from optimal — the smallest triangle is h³ with h = 1/(N-1) — so it
is a neutral start every experiment arm improves from.

One file serves the whole difficulty ladder: N is read from the problem's
own scorer (`problem/` is linked into every candidate dir), so the same
`seed_from` works for heilbronn-11, -14 and -17 alike."""

import csv
import re
from pathlib import Path

match = re.search(r"^N\s*=\s*(\d+)\s*$", Path("problem/verify.py").read_text(), re.MULTILINE)
if match is None:
    raise SystemExit("seed: could not read N from problem/verify.py")
N = int(match.group(1))

rows = [(i, round(i / (N - 1), 6), round((i / (N - 1)) ** 2, 6)) for i in range(N)]

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "x", "y"])
    writer.writerows(rows)

print(f"seed layout written: {N} points on a parabola, smallest triangle ~ {(1 / (N - 1)) ** 3:.6f}")
