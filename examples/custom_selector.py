"""Write your own selector: which candidate gets expanded.

    uv run python examples/custom_selector.py          # with Claude Code writing the solutions
    uv run python examples/custom_selector.py toy      # with the free scripted agent, in seconds

The selector (π_sel) is the first decision of every step: which node the
next attempt starts from, or none. The base class carries the schedule every
selector shares (a failing tip first, roots until `num_drafts`, the final
ensemble window) and asks a subclass for one thing: `pick`, the scored node
to build on. Swap the pick and the same policy grows a different tree:
`Best` digs one lineage deep, `LeastExpanded` spreads over the top few.
"""

import sys

from hillclimb import Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Debug, Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.sdk import SearchState, Selection, Selector, improvable
from hillclimb.selectors import Best


class LeastExpanded(Selector):
    """Among the top few scored candidates, the one with the fewest children."""

    name = "least-expanded"
    DEFAULTS = {"top": 4}

    def pick(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        journal = state.journal
        scored = [c for c in journal.ranked_candidates(state.higher_is_better, "val") if improvable(c)]
        if not scored:
            return None  # a root step: the policy drafts
        top = scored[: self.param("top")]
        pick = min(top, key=lambda c: len(journal.children(c.candidate_id)))
        return Selection(pick.candidate_id, prompt_context=f"The least explored of the top {len(top)}.")


def main(evaluations: int = 10, agent: str = "claude-code") -> dict[str, SearchOutcome]:
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)   # the same for every climber
    outcomes = {}
    for name, selector in [("best", Best(ensemble=False)), ("least-expanded", LeastExpanded(top=4, ensemble=False))]:
        climber = Climber(
            select=selector,                            # π_sel: which node, or none
            policy=Greedy(tune_budget=0),               # π_op: which operator on it
            operators=[Draft(), Debug(), Improve()],
        )
        climber.search(problem, budget=budget, agent=agent, learning=False, log=lambda *_: None)
        outcomes[name] = climber.result

    for name, outcome in outcomes.items():
        frame = outcome.to_frame()
        children = frame[frame["operator"] == "improve"]["parent_id"].value_counts()
        print(f"{name}: best {outcome.best.val_score:.3f}; improves per parent: {children.to_dict()}")
    return outcomes


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
