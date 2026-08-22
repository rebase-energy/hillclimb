"""Circle packing: maximize sum of radii of 26 non-overlapping circles in the
unit square via monotonic basin hopping: joint SLSQP polish (analytic
Jacobian) from several structured/greedy starts, then repeated
weakest-circle relocation + re-polish, accepting only improving moves."""

import csv
import os
import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = int(os.environ.get("HILLCLIMB_TRIAL_SEED", "0") or "0")
TIME_BUDGET = 240.0  # seconds of solver wall-clock time
START = time.time()

PAIR_I, PAIR_J = np.triu_indices(N, k=1)
BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def time_left():
    return TIME_BUDGET - (time.time() - START)


def unpack(z):
    return z[:N], z[N:2 * N], z[2 * N:3 * N]


def objective(z):
    return -z[2 * N:3 * N].sum()


def objective_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:3 * N] = -1.0
    return g


def constraints_fun(z):
    x, y, r = unpack(z)
    box = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
    pair = dist - r[PAIR_I] - r[PAIR_J]
    return np.concatenate([box, pair])


def constraints_jac(z):
    x, y, r = unpack(z)
    idx = np.arange(N)
    n_box = 4 * N
    n_pair = len(PAIR_I)
    J = np.zeros((n_box + n_pair, 3 * N))
    J[0 * N + idx, idx] = 1.0
    J[0 * N + idx, 2 * N + idx] = -1.0
    J[1 * N + idx, idx] = -1.0
    J[1 * N + idx, 2 * N + idx] = -1.0
    J[2 * N + idx, N + idx] = 1.0
    J[2 * N + idx, 2 * N + idx] = -1.0
    J[3 * N + idx, N + idx] = -1.0
    J[3 * N + idx, 2 * N + idx] = -1.0

    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
    ddx = dx / dist
    ddy = dy / dist
    rows = np.arange(n_pair) + n_box
    J[rows, PAIR_I] += ddx
    J[rows, PAIR_J] -= ddx
    J[rows, N + PAIR_I] += ddy
    J[rows, N + PAIR_J] -= ddy
    J[rows, 2 * N + PAIR_I] -= 1.0
    J[rows, 2 * N + PAIR_J] -= 1.0
    return J


def polish(x0, y0, r0, maxiter=150):
    z0 = np.concatenate([x0, y0, r0])
    res = minimize(
        objective, z0, jac=objective_grad, method="SLSQP",
        bounds=BOUNDS,
        constraints=[{"type": "ineq", "fun": constraints_fun, "jac": constraints_jac}],
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    x, y, r = unpack(res.x)
    return x, y, np.clip(r, 0.0, None)


def repair(x, y, r):
    """Binary-search a uniform radius scale so every constraint holds with
    margin — guards against SLSQP leaving sub-tolerance violations."""
    z = np.concatenate([x, y, r])

    def min_slack(scale):
        zz = z.copy()
        zz[2 * N:3 * N] *= scale
        return constraints_fun(zz).min()

    if min_slack(1.0) >= 0:
        return x, y, r
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if min_slack(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return x, y, r * lo


def row_init(row_counts, rng):
    n_rows = len(row_counts)
    row_h = 1.0 / n_rows
    xs, ys, rs = [], [], []
    for ri, cnt in enumerate(row_counts):
        y = row_h * (ri + 0.5)
        col_w = 1.0 / cnt
        r_row = min(row_h, col_w) / 2.0 * 0.95
        for ci in range(cnt):
            x = col_w * (ci + 0.5)
            xs.append(x)
            ys.append(y)
            rs.append(r_row)
    x = np.array(xs) + rng.normal(0, 0.01, len(xs))
    y = np.array(ys) + rng.normal(0, 0.01, len(ys))
    r = np.array(rs)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return x, y, r


def greedy_init(rng, n_candidates=250):
    xs, ys, rs = [], [], []
    for _ in range(N):
        best = None
        for _ in range(n_candidates):
            cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
            rmax = min(cx, 1 - cx, cy, 1 - cy)
            if xs:
                d = np.hypot(cx - np.array(xs), cy - np.array(ys)) - np.array(rs)
                rmax = min(rmax, d.min())
            if rmax <= 0:
                continue
            if best is None or rmax > best[2]:
                best = (cx, cy, rmax)
        if best is None:
            best = (rng.uniform(0, 1), rng.uniform(0, 1), 1e-4)
        xs.append(best[0])
        ys.append(best[1])
        rs.append(best[2])
    return np.array(xs), np.array(ys), np.array(rs)


def relocate_weakest(x, y, r, k, rng, n_candidates=300):
    """Pull the k smallest circles out and re-place them greedily in the
    largest free gaps left by the rest of the packing."""
    order = np.argsort(r)
    victims = set(order[:k].tolist())
    keep = [i for i in range(N) if i not in victims]
    nx, ny, nr = x[keep].tolist(), y[keep].tolist(), r[keep].tolist()
    for _ in victims:
        best = None
        for _ in range(n_candidates):
            cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
            rmax = min(cx, 1 - cx, cy, 1 - cy)
            if nx:
                d = np.hypot(cx - np.array(nx), cy - np.array(ny)) - np.array(nr)
                rmax = min(rmax, d.min())
            if rmax <= 0:
                continue
            if best is None or rmax > best[2]:
                best = (cx, cy, rmax)
        if best is None:
            best = (rng.uniform(0, 1), rng.uniform(0, 1), 1e-4)
        nx.append(best[0])
        ny.append(best[1])
        nr.append(max(best[2] * 0.5, 1e-4))
    return np.array(nx), np.array(ny), np.array(nr)


def jitter_shake(x, y, r, rng, sigma=0.03):
    """Perturb every center a little and shrink radii to make room, so the
    subsequent SLSQP polish can find a different local optimum."""
    nx = np.clip(x + rng.normal(0, sigma, N), 0.0, 1.0)
    ny = np.clip(y + rng.normal(0, sigma, N), 0.0, 1.0)
    nr = r * 0.85
    return nx, ny, nr


def main():
    rng = np.random.default_rng(SEED)

    starts = []
    for counts in ([6, 7, 6, 7], [5, 5, 5, 5, 6], [4, 5, 4, 5, 4, 4], [6, 5, 4, 5, 6]):
        starts.append(row_init(counts, rng))
    starts.append(greedy_init(rng))
    starts.append(greedy_init(rng))

    best = None
    for x0, y0, r0 in starts:
        if time_left() <= 5:
            break
        x, y, r = polish(x0, y0, r0, maxiter=200)
        x, y, r = repair(x, y, r)
        score = r.sum()
        if best is None or score > best[0]:
            best = (score, x, y, r)

    score, x, y, r = best

    # monotonic basin hopping: alternate relocating the weakest circles and
    # globally shaking the layout, re-polish, accept only if it improves.
    stale = 0
    while time_left() > 8:
        if stale > 0 and stale % 2 == 1:
            rx, ry, rr = jitter_shake(x, y, r, rng)
        else:
            k = int(rng.integers(1, 6))
            rx, ry, rr = relocate_weakest(x, y, r, k, rng)
        px, py, pr = polish(rx, ry, rr, maxiter=150)
        px, py, pr = repair(px, py, pr)
        new_score = pr.sum()
        if new_score > score:
            score, x, y, r = new_score, px, py, pr
            stale = 0
        else:
            stale += 1

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "r"])
        for i in range(N):
            writer.writerow([i, float(x[i]), float(y[i]), float(r[i])])
    print(f"sum of radii = {score:.6f}")


if __name__ == "__main__":
    main()
