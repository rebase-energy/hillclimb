"""Circle packing: maximize sum of radii for 26 circles in the unit square.

Approach: LP-exact-radii-for-fixed-centers as the core primitive (for any set
of centers, the optimal radii given fixed centers is a linear program -
maximize sum(r) s.t. wall clearance and pairwise non-overlap, all linear in
r once centers are fixed), combined with a joint SLSQP polish of all (x, y, r)
variables with analytic Jacobians, wrapped in monotonic basin hopping: perturb
the weakest circles' positions, re-solve, and only accept improvements.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

N = 26
SEED = int(os.environ.get("HILLCLIMB_TRIAL_SEED", 0) or 0)
rng = np.random.default_rng(SEED)

START = time.time()
TIME_BUDGET = 480.0  # seconds of optimization; verifier allows 900s total


def time_left():
    return TIME_BUDGET - (time.time() - START)


def pairwise_dist(centers):
    d = centers[:, None, :] - centers[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def solve_radii_lp(centers):
    """Exact optimal radii for fixed centers (linear program)."""
    x, y = centers[:, 0], centers[:, 1]
    dist = pairwise_dist(centers)
    wall = np.minimum.reduce([x, 1 - x, y, 1 - y])
    wall = np.clip(wall, 0.0, None)
    bounds = [(0.0, float(w)) for w in wall]

    iu = np.triu_indices(N, k=1)
    m = len(iu[0])
    A = np.zeros((m, N))
    A[np.arange(m), iu[0]] = 1.0
    A[np.arange(m), iu[1]] = 1.0
    b = dist[iu]

    c = -np.ones(N)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(N)
    return np.clip(res.x, 0.0, None)


IU = np.triu_indices(N, k=1)


def unpack(z):
    x = z[0:N]
    y = z[N:2 * N]
    r = z[2 * N:3 * N]
    return x, y, r


def pack(x, y, r):
    return np.concatenate([x, y, r])


def neg_sum_r(z):
    return -z[2 * N:3 * N].sum()


def neg_sum_r_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:3 * N] = -1.0
    return g


def constraints_fun(z):
    x, y, r = unpack(z)
    wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    i, j = IU
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-16)
    pair = dist - r[i] - r[j]
    return np.concatenate([wall, pair])


def constraints_jac(z):
    x, y, r = unpack(z)
    m_wall = 4 * N
    i, j = IU
    m_pair = len(i)
    J = np.zeros((m_wall + m_pair, 3 * N))

    idx = np.arange(N)
    # x - r >= 0
    J[0 * N + idx, idx] = 1.0
    J[0 * N + idx, 2 * N + idx] = -1.0
    # 1 - x - r >= 0
    J[1 * N + idx, idx] = -1.0
    J[1 * N + idx, 2 * N + idx] = -1.0
    # y - r >= 0
    J[2 * N + idx, N + idx] = 1.0
    J[2 * N + idx, 2 * N + idx] = -1.0
    # 1 - y - r >= 0
    J[3 * N + idx, N + idx] = -1.0
    J[3 * N + idx, 2 * N + idx] = -1.0

    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-16)
    ux, uy = dx / dist, dy / dist
    rows = np.arange(m_pair) + m_wall
    J[rows, i] = ux
    J[rows, j] = -ux
    J[rows, N + i] = uy
    J[rows, N + j] = -uy
    J[rows, 2 * N + i] = -1.0
    J[rows, 2 * N + j] = -1.0
    return J


def polish(centers, radii, maxiter=150):
    z0 = pack(centers[:, 0], centers[:, 1], radii)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.7)] * N
    cons = [{"type": "ineq", "fun": constraints_fun, "jac": constraints_jac}]
    try:
        res = minimize(
            neg_sum_r, z0, jac=neg_sum_r_grad, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        z = res.x
    except Exception:
        z = z0
    x, y, r = unpack(z)
    centers2 = np.stack([x, y], axis=1)
    centers2 = np.clip(centers2, 0.0, 1.0)
    r2 = solve_radii_lp(centers2)  # re-solve exactly to guarantee feasibility
    return centers2, r2


def repair(centers, radii, iters=60):
    """Cheap projection to remove any residual overlap/wall violation."""
    c = centers.copy()
    r = radii.copy()
    for _ in range(iters):
        moved = False
        r = solve_radii_lp(c)
        d = pairwise_dist(c)
        i, j = IU
        need = r[i] + r[j]
        gap = need - d[i, j]
        bad = gap > 1e-9
        if not np.any(bad):
            break
        moved = True
        for k in np.where(bad)[0]:
            a, b = i[k], j[k]
            dvec = c[a] - c[b]
            dist = np.linalg.norm(dvec)
            if dist < 1e-9:
                dvec = rng.normal(size=2)
                dist = np.linalg.norm(dvec)
            push = (gap[k] / 2 + 1e-6) * dvec / dist
            c[a] = np.clip(c[a] + push, 0.0, 1.0)
            c[b] = np.clip(c[b] - push, 0.0, 1.0)
        if not moved:
            break
    r = solve_radii_lp(c)
    return c, r


def hex_init(row_counts, jitter=0.0):
    rows = len(row_counts)
    ys = (np.arange(rows) + 0.5) / rows
    pts = []
    for ridx, cnt in enumerate(row_counts):
        xs = (np.arange(cnt) + 0.5) / cnt
        y = ys[ridx]
        for xv in xs:
            pts.append((xv, y))
    pts = np.array(pts[:N], dtype=float)
    if jitter > 0:
        pts += rng.normal(scale=jitter, size=pts.shape)
    return np.clip(pts, 0.02, 0.98)


def random_relaxed_init(n_iter=200):
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    step = 0.05
    for _ in range(n_iter):
        d = pairwise_dist(c)
        i, j = IU
        dist = d[i, j]
        target = 1.0 / np.sqrt(N)
        bad = dist < target
        if not np.any(bad):
            break
        for k in np.where(bad)[0]:
            a, b = i[k], j[k]
            dvec = c[a] - c[b]
            nrm = np.linalg.norm(dvec)
            if nrm < 1e-9:
                dvec = rng.normal(size=2)
                nrm = np.linalg.norm(dvec)
            push = step * dvec / nrm
            c[a] = np.clip(c[a] + push, 0.02, 0.98)
            c[b] = np.clip(c[b] - push, 0.02, 0.98)
    return c


def score(radii):
    return float(np.sum(radii))


def build_candidates():
    partitions = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 5, 5, 6],
        [3, 4, 5, 5, 4, 3, 2],
        [6, 6, 6, 5, 3],
    ]
    inits = []
    for p in partitions:
        inits.append(hex_init(p, jitter=0.0))
        inits.append(hex_init(p, jitter=0.02))
    for _ in range(4):
        inits.append(random_relaxed_init())
    return inits


def main():
    inits = build_candidates()
    seeded = []
    per_seed_budget = max(3.0, min(12.0, time_left() / (2 * len(inits) + 2)))
    for centers in inits:
        if time_left() < 20:
            break
        radii = solve_radii_lp(centers)
        c, r = polish(centers, radii, maxiter=100)
        c, r = repair(c, r)
        seeded.append((score(r), c, r))

    seeded.sort(key=lambda t: -t[0])
    best_score, best_c, best_r = seeded[0]

    # a second polish round on the top few seeds
    top_k = seeded[: min(3, len(seeded))]
    for s, c, r in top_k:
        if time_left() < 15:
            break
        c2, r2 = polish(c, r, maxiter=200)
        c2, r2 = repair(c2, r2)
        s2 = score(r2)
        if s2 > best_score:
            best_score, best_c, best_r = s2, c2, r2

    # monotonic basin hopping: perturb the weakest circles, re-optimize,
    # accept only if it strictly improves on the incumbent best.
    cur_c, cur_r, cur_score = best_c.copy(), best_r.copy(), best_score
    n_weak_choices = [1, 2, 3, 4]
    while time_left() > 10:
        k = int(rng.choice(n_weak_choices))
        order = np.argsort(cur_r)
        weak_idx = order[:k]
        trial_c = cur_c.copy()
        for idx in weak_idx:
            if rng.random() < 0.5:
                trial_c[idx] = rng.uniform(0.03, 0.97, size=2)
            else:
                trial_c[idx] = np.clip(
                    trial_c[idx] + rng.normal(scale=0.15, size=2), 0.02, 0.98
                )
        trial_r = solve_radii_lp(trial_c)
        trial_c, trial_r = polish(trial_c, trial_r, maxiter=80)
        trial_c, trial_r = repair(trial_c, trial_r)
        trial_score = score(trial_r)
        if trial_score > cur_score + 1e-9:
            cur_c, cur_r, cur_score = trial_c, trial_r, trial_score
            if cur_score > best_score:
                best_score, best_c, best_r = cur_score, cur_c.copy(), cur_r.copy()
        else:
            cur_c, cur_r, cur_score = best_c.copy(), best_r.copy(), best_score

    # final safety pass: ensure strict feasibility with a small safety margin
    final_c, final_r = repair(best_c, best_r, iters=80)
    if score(final_r) < best_score:
        final_c, final_r = best_c, best_r
    final_r = final_r * (1 - 1e-9)

    df = pd.DataFrame({
        "id": np.arange(N),
        "x": final_c[:, 0],
        "y": final_c[:, 1],
        "r": final_r,
    })
    df.to_csv("submission.csv", index=False)
    print(f"final sum of radii: {final_r.sum():.6f}")


if __name__ == "__main__":
    main()
