"""Build a climber, search a problem, read what it found.

    uv run python examples/run_and_read.py          # with Claude Code writing the solutions
    uv run python examples/run_and_read.py toy      # with the free scripted agent, in seconds

The problem is `fitness-landscape`: every solution is one point on a terrain
and the score is the height there. Claude Code writes real code for it, in
the background, one candidate at a time. The scripted `toy` agent stands in
for it when you just want to see the machinery move.
"""

import sys

from hillclimb import Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Debug, Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.selectors import Best
from hillclimb.tuners import RandomSearch


def main(evaluations: int = 10, agent: str = "claude-code") -> SearchOutcome:
    # A problem is what a climber searches. This one ships with hillclimb; it
    # is copied into your problems/ folder the first time you use it.
    problem = Problem("fitness-landscape")

    # A budget is what the search may spend. The first limit reached ends it;
    # Budget(wall_clock="20m", evaluations=30, cost_usd=5) sets several.
    budget = Budget(evaluations=evaluations)

    # A climber is a set of modules. Each slot takes an instance, a class or a
    # name, and anything you leave out keeps its default.
    climber = Climber(
        select=Best(num_drafts=3, ensemble=False),     # which node to build on, or none (π_sel)
        policy=Greedy(),                               # which operator to apply to it (π_op)
        operators=[Draft(), Debug(), Improve()],       # how that attempt is made
        tuner=RandomSearch(),                          # how a tune trial picks its parameter values
    )

    climber.search(
        problem,
        budget=budget,
        agent=agent,            # the coding agent that writes the solutions
        learning=False,         # keep this run out of the folder's knowledge
    )

    # What the search found is on the climber now.
    print(f"\n{climber.result.ref}: {climber.result.state}")
    best = climber.best
    print(f"best: {best.candidate_id} at {best.val_score:.3f}, made by {best.operator!r} from {best.parent_id}")
    print(f"spent: {climber.spend.evaluations} evaluations, {climber.spend.seconds:.1f}s, {climber.spend.tokens} tokens")

    print("\neach time the best score went up:")
    for step in climber.history:
        print(f"  {step.candidate_id}  {step.score:7.3f}  at {step.minutes:5.1f} min")

    print("\nevery candidate:")
    frame = climber.to_frame()
    print(frame[["parent_id", "operator", "status", "val_score", "n_trials", "x", "y"]].to_string())

    print("\nthe solution that ships (best/solution.py), with parameters", climber.params)
    print(climber.solution)
    return climber.result


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
