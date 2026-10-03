"""Write your own operator_policy: the schedule of a search.

    uv run python examples/custom_policy.py          # with Claude Code writing the solutions
    uv run python examples/custom_policy.py toy      # with the free scripted agent, in seconds

Every step of a search is two decisions, in order. The selector (π_sel)
reads the history and picks the node the next attempt starts from, or none.
Then the policy (π_op) reads the same history and that pick, and names the
operator. A policy of your own is the second decision. Three policies over
the same selector and budget show how much it matters on a terrain with one
narrow peak among broad decoys.
"""

import sys

from hillclimb import Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.selectors import Best
from hillclimb.sdk import Action, OperatorPolicy, SearchState, Selection, improves


class DraftsOnly(OperatorPolicy):
    """Random search: whatever the selector chose, draft again."""

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        return Action("draft")


class GiveUpQuickly(OperatorPolicy):
    """Build on the chosen node, unless its last few children failed to beat
    it. Then it is stuck: draft somewhere new instead."""

    DEFAULTS = {"patience": 2}  # every knob and its default; the base adds its own

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        # `state` is all a policy sees: the journal (every candidate so far),
        # what is in flight, the budget left, the metric's direction.
        # `selection` is what the selector chose: None for a root step.
        if selection is None or selection.combine:
            return super().propose(state, selection)  # the plain mapping: draft, ensemble
        node = state.journal.candidates[selection.target_id]
        if node.status != "passing":
            return super().propose(state, selection)  # a failing tip: debug
        children = [c for c in state.journal.children(node.candidate_id) if c.val_score is not None]
        recent = children[-int(self.param("patience")):]
        stuck = len(recent) >= int(self.param("patience")) and not any(
            improves(c.val_score, node.val_score, higher_is_better=state.higher_is_better, band=state.accept_band)
            for c in recent
        )
        return self.draft_action(state) if stuck else self.expand_action(state, selection)


def main(evaluations: int = 10, agent: str = "claude-code") -> dict[str, SearchOutcome]:
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)   # the same for every climber
    outcomes = {}
    for name, policy in [
        ("drafts only", DraftsOnly),
        ("give up quickly", GiveUpQuickly(patience=2)),
        ("bundled greedy", Greedy()),
    ]:
        climber = Climber(
            selector_policy=Best(num_drafts=4, ensemble=False),  # π_sel, the same for all three
            operator_policy=policy,                              # π_op, what differs
            operators=[Draft(), Improve()],
            name=name,
        )
        climber.search(problem, budget=budget, agent=agent, learning=False, log=lambda *_: None)
        outcomes[name] = climber.result

    print(f"best elevation after {evaluations} evaluations (the peak is 17.11, the best decoy 13.00):")
    for name, outcome in outcomes.items():
        operators = outcome.to_frame()["operator"].value_counts().to_dict()
        print(f"  {name:18} {outcome.best.val_score:7.3f}   {operators}")
    return outcomes


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
