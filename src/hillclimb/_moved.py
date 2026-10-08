"""Module paths that moved (the package-layout refactor into harness/,
modules/, tui/ and cli/; the 0.6 split of integrations/).

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


# classes that were renamed, or replaced by a composition, in 0.6 — by the
# exact `module:Class` a run folder may hold
RENAMED = {
    # 0.9: the bundled policies live in one file per climber (`climbers/greedy/
    # policy.py` holds Best + Greedy), where `hillclimb climber get` copies them from
    "hillclimb.modules.policies.greedy:GreedyPolicy": "hillclimb.climbers.greedy.policy:Greedy",
    "hillclimb.modules.policies.greedy:Greedy": "hillclimb.climbers.greedy.policy:Greedy",
    "hillclimb.modules.selectors.best:Best": "hillclimb.climbers.greedy.policy:Best",
    "hillclimb.modules.selectors.map_elites:MapElites": "hillclimb.climbers.openevolve.policy:MapElites",
    # the openevolve policy is greedy over the map-elites selector now; a
    # pre-0.6 snapshot keeps MAP-Elites' settings among the policy's params
    "hillclimb.modules.policies.openevolve:OpenEvolvePolicy": "hillclimb.modules.policies.compat:OpenEvolvePolicy",
    # 0.6 dropped the kind from class names (`hillclimb.operators.Draft` says it)
    "hillclimb.modules.operators.builtin:DraftOperator": "hillclimb.modules.operators.builtin:Draft",
    "hillclimb.modules.operators.builtin:DebugOperator": "hillclimb.modules.operators.builtin:Debug",
    "hillclimb.modules.operators.builtin:ImproveOperator": "hillclimb.modules.operators.builtin:Improve",
    "hillclimb.modules.operators.builtin:EnsembleOperator": "hillclimb.modules.operators.builtin:Ensemble",
    "hillclimb.climbers.gepa.operator:GepaReflectOperator": "hillclimb.climbers.gepa.operator:GepaReflect",
    "hillclimb.modules.tuners.random_search:RandomTuner": "hillclimb.modules.tuners.random_search:RandomSearch",
    "hillclimb.modules.tuners.optuna:OptunaTuner": "hillclimb.modules.tuners.optuna:Optuna",
}


def modernize(ref: str) -> str:
    """`hillclimb.policies.greedy:GreedyPolicy` -> where that class lives
    today (its module path, then its name); anything that did not move comes
    back unchanged."""
    for old in sorted(MOVED, key=len, reverse=True):
        if ref.startswith(old):
            ref = MOVED[old] + ref[len(old):]
            break
    return RENAMED.get(ref, ref)
