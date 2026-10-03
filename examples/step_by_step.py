"""Drive one search a step at a time.

    uv run python examples/step_by_step.py          # with Claude Code writing the solutions
    uv run python examples/step_by_step.py toy      # with the free scripted agent, in seconds

`climber.search` hands the whole search to the climber. `climber.start` opens
the same search and leaves the loop in your hands: ask the policy what it
would do, run that or a move of your own, look at what the policy sees, and
let the climber finish when you have seen enough. Paste the body of `main`
into a Python prompt to do it by hand.
"""

import sys

from hillclimb import Action, Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Debug, Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.selectors import Best


def show(action: Action, outcome) -> None:
    move = action.operator + (f" {action.target_id}" if action.target_id else "")
    if outcome.candidate is None:
        print(f"  {move:14} -> {outcome.kind}: {outcome.ticket.rejected}")
    else:
        made = outcome.candidate
        print(f"  {move:14} -> {outcome.kind:9} {made.candidate_id} scores {made.val_score:.3f}")


def main(evaluations: int = 10, agent: str = "claude-code") -> SearchOutcome:
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)
    climber = Climber(
        selector_policy=Best(num_drafts=3, ensemble=False), operator_policy=Greedy(), operators=[Draft(), Debug(), Improve()],
    )

    climber.start(
        problem, budget=budget, agent=agent, learning=False,
        log=lambda *_: None,  # drop this line to see the engine's own log
    )
    print(f"opened {climber.session.ref}; the baseline scores {climber.best.val_score:.3f}")

    # 1. The policy's own moves, one at a time. `propose` runs nothing: it is
    #    the policy's answer to "what next?", and you decide whether to run it.
    print("\nthe policy's moves:")
    for _ in range(4):
        action = climber.propose()
        if action is None:  # the policy holds, or a budget closed the search
            break
        show(action, climber.run(action))

    # 2. What the policy sees when it decides: the journal, what is in
    #    flight, the budget left.
    state = climber.state
    print(f"\nthe policy sees {len(state.journal.candidates)} candidates, "
          f"{state.budget.evaluations_remaining} evaluations left, best {climber.best.candidate_id}")

    # 3. A move of your own. Greedy builds on the best; go back to the worst
    #    draft instead and see what a step from there finds.
    drafts = [c for c in climber.candidates if c.operator == "draft" and c.val_score is not None]
    worst = min(drafts, key=lambda c: c.val_score)
    print(f"\nyour move, from the worst draft ({worst.candidate_id} at {worst.val_score:.3f}):")
    mine = Action("improve", target_id=worst.candidate_id)
    show(mine, climber.run(mine))

    # 4. `step()` is propose + run in one call. The policy observed your move
    #    like any other, so its next proposal already accounts for it.
    print("\none more of the policy's:")
    outcome = climber.step()
    if outcome is not None:  # None: nothing to run (the policy holds, or the search is closed)
        show(outcome.ticket.action, outcome)

    # 5. Enough by hand: the climber's own loop runs what is left of the
    #    budget. (`climber.close()` instead would stop here; a search closed
    #    with budget left can be resumed with `hillclimb resume`.)
    climber.finish()
    print(f"\n{climber.result.state}: best {climber.best.candidate_id} at {climber.best.val_score:.3f} "
          f"after {climber.spend.evaluations} evaluations")
    print(climber.to_frame()[["parent_id", "operator", "val_score", "n_trials"]].to_string())
    return climber.result


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
