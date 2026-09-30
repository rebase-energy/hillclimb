"""Module paths that moved (the package-layout refactor, docs/package-layout-plan.md;
the 0.6 split of integrations/).

Run folders carry `module:Class` refs — a search's climber snapshot, the
the `climber_spec` in search.yaml, a `similarity:` entry in config.yaml —
so a search recorded before the move must still resume. Like
`config.LEGACY_SETTINGS`, the old spelling is mapped at the one place it is
imported and nowhere else.
"""

from __future__ import annotations

MOVED = {
    "hillclimb.policies.": "hillclimb.modules.policies.",
    "hillclimb.operators.": "hillclimb.modules.operators.",
    "hillclimb.tuners.": "hillclimb.modules.tuners.",
    "hillclimb.similarity_scores.": "hillclimb.modules.similarity.",
    # 0.6: integrations/ split into providers/ and the gepa climber library
    "hillclimb.integrations.gepa.": "hillclimb.climbers.gepa.",
}


def modernize(ref: str) -> str:
    """`hillclimb.policies.greedy:GreedyPolicy` -> its current module path;
    anything that did not move comes back unchanged."""
    for old in sorted(MOVED, key=len, reverse=True):
        if ref.startswith(old):
            return MOVED[old] + ref[len(old):]
    return ref
