"""Write your own selector: which candidate gets expanded.

    uv run python examples/custom_selector.py          # with Claude Code writing the solutions
    uv run python examples/custom_selector.py toy      # with the free scripted agent, in seconds

The selector (π_sel) is the first decision of every step: which node the
next attempt starts from, or none. `Best` writes the whole schedule out (a
failing tip first, roots until `num_drafts`, the final ensemble window, then
`select`: the best scored node). Subclassing it and replacing `select` keeps
that schedule and changes the one choice. Swap it and the same policy grows
a different tree: `Best` digs one lineage deep, `LeastExpanded` spreads over
the top few.
"""

import sys

from hillclimb import Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Debug, Draft, Improve
from hillclimb.sdk import SearchState, Selection, improvable
from hillclimb import catalog

# the catalog's greedy climber (`hillclimb climber get greedy` copies the same file): its classes to compose with
greedy = catalog.module("greedy")
Greedy, Best = greedy.Greedy, greedy.Best


class LeastExpanded(Best):
    """Among the top few scored candidates, the one with the fewest children."""

    name = "least-expanded"
    DEFAULTS = {"top": 4}

    def select(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        journal = state.journal
        scored = [c for c in journal.ranked_candidates(state.higher_is_better, "val") if improvable(c)]
        if not scored:
            return None  # a root step: the policy drafts
        top = scored[: self.param("top")]
        chosen = min(top, key=lambda c: len(journal.children(c.candidate_id)))
        return Selection(chosen.candidate_id, prompt_context=f"The least explored of the top {len(top)}.")


def main(evaluations: int = 10, agent: str = "claude-code") -> dict[str, SearchOutcome]:
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)   # the same for every climber
    outcomes = {}
    for name, selector in [("best", Best(ensemble=False)), ("least-expanded", LeastExpanded(top=4, ensemble=False))]:
        climber = Climber(
            selector_policy=selector,                            # π_sel: which node, or none
            operator_policy=Greedy(tune_budget=0),               # π_op: which operator on it
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
