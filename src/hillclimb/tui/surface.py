"""The 3D fitness surface of one search — pure data, no Textual.

A problem opts in by shipping `landscape.py` (`elevation(x, y)` and
`grid(n)`, picked up by default like `contract.md`) and having its verifier
journal every candidate's position: the `surface_metrics` keys (default
`x`/`y`) written as extra numeric keys next to `score`, which the engine
stores as `Trial.metrics`. `build_surface` pairs the journal with that
terrain: every positioned candidate becomes an (x, y, z) node carrying the
same fate vocabulary as `tree.py` (the fates come from `build_tree`, so the
two views can never disagree), and the accepted lineage is draped along the
terrain — each hop subdivided and lifted a hair above the ground — so the
climb visibly walks the surface instead of cutting straight through it.
`surfaceview.py` renders this through plotui; both stay replay-only readers
of the journal.

Sibling view: `hillclimb similarity` (`similarity.py`/`similarityview.py`) —
the same fate vocabulary over three candidate-distance coordinates with the
score as a colour scale and no terrain.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hillclimb.harness.candidate import Candidate
from hillclimb.tui.tree import build_tree

GRID_N = 121        # terrain samples per axis for the rendered surface
DRAPE_STEPS = 24    # segments per lineage hop when following the terrain
# Nodes and the lineage float this fraction of the terrain's height range
# above the ground, so they never z-fight with the surface itself.
LIFT = 0.02


class LandscapeError(Exception):
    """The problem's landscape module could not be loaded or is incomplete."""


_LANDSCAPES: dict[tuple[Path, float], object] = {}


def load_landscape(path: Path):
    """Import the problem's landscape module (cached per path + mtime, so an
    edited terrain is picked up on the next refresh). It must expose
    `elevation(x, y)` (broadcasting over numpy arrays) and `grid(n)`."""
    path = path.resolve()
    try:
        key = (path, path.stat().st_mtime)
    except OSError as exc:
        raise LandscapeError(f"cannot read landscape module: {path} ({exc})") from exc
    if key in _LANDSCAPES:
        return _LANDSCAPES[key]
    spec = importlib.util.spec_from_file_location(f"hillclimb_landscape_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise LandscapeError(f"cannot import landscape module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — problem-authored code
        raise LandscapeError(f"landscape module failed to import: {path}: {exc}") from exc
    for name in ("elevation", "grid"):
        if not callable(getattr(module, name, None)):
            raise LandscapeError(f"landscape module must define {name}(): {path}")
    _LANDSCAPES.clear()  # only ever one problem on screen; no need to hoard
    _LANDSCAPES[key] = module
    return module


@dataclass(frozen=True)
class SurfaceNode:
    id: str
    x: float
    y: float
    z: float            # elevation at (x, y), lifted — where the mark is drawn
    score: float | None
    fate: str           # tree.py vocabulary: expanded/best/discontinued/failed/pruned/pending
    on_path: bool       # part of the accepted lineage
    operator: str


@dataclass(frozen=True)
class Polyline:
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    zs: tuple[float, ...]


@dataclass(frozen=True)
class SurfaceView:
    xs: tuple[float, ...]                 # grid axes
    ys: tuple[float, ...]
    zgrid: tuple[tuple[float, ...], ...]  # zgrid[j][i] = height at (xs[i], ys[j])
    nodes: tuple[SurfaceNode, ...]
    lineage: Polyline                     # accepted path draped on the terrain
    peak: tuple[float, float, float]      # grid argmax — the summit marker
    n_unpositioned: int                   # candidates with no position metrics


def build_surface(
    candidates: list[Candidate],
    higher_is_better: bool,
    landscape,
    metric_keys: tuple[str, str] = ("x", "y"),
    grid_n: int = GRID_N,
) -> SurfaceView:
    """Pair one search's journal with the problem's terrain."""
    gxs, gys, gz = landscape.grid(grid_n)
    gz = np.asarray(gz, dtype=float)
    lift = LIFT * float(gz.max() - gz.min() or 1.0)
    kx, ky = metric_keys

    tree = build_tree(candidates, higher_is_better)
    fates = {n.id: n.fate for n in tree.nodes}
    on_path = set(tree.accepted)

    nodes: list[SurfaceNode] = []
    positions: dict[str, tuple[float, float]] = {}
    skipped = 0
    for cand in candidates:
        metrics = cand.metrics
        if kx not in metrics or ky not in metrics:
            skipped += 1
            continue
        x, y = float(metrics[kx]), float(metrics[ky])
        positions[cand.candidate_id] = (x, y)
        nodes.append(
            SurfaceNode(
                id=cand.candidate_id,
                x=x,
                y=y,
                z=float(landscape.elevation(x, y)) + lift,
                score=cand.val_score,
                fate=fates.get(cand.candidate_id, "pending"),
                on_path=cand.candidate_id in on_path,
                operator=cand.operator,
            )
        )

    # Drape the accepted lineage: consecutive accepted candidates that both
    # have positions, each hop subdivided and following the ground.
    dx: list[float] = []
    dy: list[float] = []
    hops = [positions[cid] for cid in tree.accepted if cid in positions]
    for (x0, y0), (x1, y1) in zip(hops[:-1], hops[1:]):
        t = np.linspace(0.0, 1.0, DRAPE_STEPS)
        dx.extend(x0 + t * (x1 - x0))
        dy.extend(y0 + t * (y1 - y0))
    if dx:
        dz = np.asarray(landscape.elevation(np.asarray(dx), np.asarray(dy)), dtype=float) + lift
    else:
        dz = np.empty(0)

    j, i = np.unravel_index(int(gz.argmax()), gz.shape)
    peak = (float(np.asarray(gxs)[i]), float(np.asarray(gys)[j]), float(gz[j, i]) + lift)

    return SurfaceView(
        xs=tuple(float(v) for v in gxs),
        ys=tuple(float(v) for v in gys),
        zgrid=tuple(tuple(float(v) for v in row) for row in gz),
        nodes=tuple(nodes),
        lineage=Polyline(tuple(dx), tuple(dy), tuple(float(v) for v in dz)),
        peak=peak,
        n_unpositioned=skipped,
    )
