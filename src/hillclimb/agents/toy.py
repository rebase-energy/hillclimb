from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from pathlib import Path

from hillclimb.agents.base import AgentRequest, AgentResult

# A scripted stand-in for a coding agent on the fitness-landscape problem:
# every solution is one point on a known terrain, so a search costs nothing,
# finishes in seconds, and its scores MOVE — which a climber needs before a
# different policy, selector or operator can be seen to change anything.
SOLUTION_TEMPLATE = '''\
"""toy: stand on one point of the terrain."""
try:
    from hillclimb import spaces  # the runtime shim makes this importable

    P = spaces.params({{"dx": 0.0, "dy": 0.0}})
except ImportError:
    P = {{"dx": 0.0, "dy": 0.0}}

X, Y = {x!r}, {y!r}
LO, HI = {lo!r}, {hi!r}

x = min(HI, max(LO, X + P["dx"]))
y = min(HI, max(LO, Y + P["dy"]))
with open("submission.csv", "w") as out:
    out.write(f"x,y\\n{{x}},{{y}}\\n")
'''

_POINT = re.compile(r"^X, Y = (?P<x>[-+0-9.e]+), (?P<y>[-+0-9.e]+)$", re.MULTILINE)
# the one thing a prompt can tell the toy: how far to step from the parent
_STEP = re.compile(r"toy:\s*step\s*=\s*([0-9.]+(?:e-?[0-9]+)?)")
_KINDS = ("create", "repair", "refine", "combine")


class ToyAgent:
    """Scripted moves on the fitness-landscape terrain, by the KIND of
    attempt: create = a random point, refine = a step from the parent,
    combine = the mean of the parent and its inspirations, repair = the
    parent pulled back inside the domain. A line `toy: step=<float>` in the
    prompt sets the step length. Deterministic per `seed`."""

    name = "toy"

    def __init__(self, seed: int = 0, step: float = 0.4):
        self.seed = seed
        self.step = step
        self.calls = 0

    def invoke(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        from hillclimb.modules.operators import operator_kind

        kind = request.kind or operator_kind(request.operator)
        if kind not in _KINDS:
            return self._refuse(f"the toy agent only makes solutions; it cannot do {request.operator!r}")
        candidate_dir = request.candidate_dir
        domain = _domain(candidate_dir / "problem" / "landscape.py")
        if domain is None:
            return self._refuse("the toy agent only knows the fitness-landscape problem (no problem/landscape.py)")
        lo, hi = domain
        rng = random.Random(_seed(self.seed, candidate_dir.name, kind))
        found = _STEP.findall(request.prompt)
        step = float(found[-1]) if found else self.step
        parent = _point(candidate_dir / "solution.py", candidate_dir / "params.json")

        if kind == "create" or (parent is None and kind == "refine"):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            note = "a random point"
        elif kind == "refine":
            x, y = parent[0] + rng.gauss(0.0, step), parent[1] + rng.gauss(0.0, step)
            note = f"a step of about {step:g} from the parent at ({parent[0]:.3f}, {parent[1]:.3f})"
        elif kind == "combine":
            from hillclimb.modules.operators.base import inspiration_filename

            points = [parent] if parent is not None else []
            index = 1
            while (candidate_dir / inspiration_filename(index)).exists():
                other = _point(candidate_dir / inspiration_filename(index))
                if other is not None:
                    points.append(other)
                index += 1
            if not points:
                points = [(0.0, 0.0)]
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            note = f"the mean of {len(points)} point(s)"
        else:  # repair
            x, y = parent if parent is not None else (0.0, 0.0)
            note = "the parent, pulled back inside the domain"
        x, y = round(min(hi, max(lo, x)), 6), round(min(hi, max(lo, y)), 6)

        (candidate_dir / "solution.py").write_text(SOLUTION_TEMPLATE.format(x=x, y=y, lo=lo, hi=hi))
        # two declared knobs, so a tuner has something to search around the point
        reach = round(step, 6)
        space = {
            name: {"type": "float", "low": -reach, "high": reach, "default": 0.0} for name in ("dx", "dy")
        }
        (candidate_dir / "params.json").write_text(json.dumps(space) + "\n")
        (candidate_dir / "notes.md").write_text(f"toy: ({x}, {y}), {note}\n")
        return AgentResult(
            ok=True, session_id=f"toy-{self.calls}", cost_usd=0.0, total_tokens=0, duration_s=0.0
        )

    def _refuse(self, reason: str) -> AgentResult:
        return AgentResult(ok=False, error_kind="error", error_message=reason, duration_s=0.0)


def _seed(seed: int, candidate: str, kind: str) -> int:
    """One draw per (seed, candidate, kind): the same search replays the same
    points whatever order its attempts ran in."""
    digest = hashlib.sha256(f"{seed}:{candidate}:{kind}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _domain(landscape: Path) -> tuple[float, float] | None:
    """`DOMAIN = (lo, hi)` read off the landscape's text — the file is the
    problem's, so it is parsed, never imported into the engine."""
    try:
        tree = ast.parse(landscape.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DOMAIN" for target in node.targets
        ):
            try:
                lo, hi = ast.literal_eval(node.value)
                return float(lo), float(hi)
            except (ValueError, TypeError):
                return None
    return None


def _point(solution: Path, params: Path | None = None) -> tuple[float, float] | None:
    """Where a toy solution stands: its `X, Y` line, moved by the offsets a
    tuned parent handed down as this candidate's defaults."""
    try:
        match = _POINT.search(solution.read_text())
    except OSError:
        return None
    if match is None:
        return None
    x, y = float(match["x"]), float(match["y"])
    if params is not None and params.exists():
        try:
            space = json.loads(params.read_text())
            x += float(space.get("dx", {}).get("default", 0.0))
            y += float(space.get("dy", {}).get("default", 0.0))
        except (ValueError, AttributeError, TypeError):
            pass
    return x, y
