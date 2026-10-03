"""Write your own operator: how one attempt is made.

    uv run python examples/custom_operator.py          # with Claude Code writing the solutions
    uv run python examples/custom_operator.py toy      # with the free scripted agent, in seconds

The built-in operators (`Draft`, `Debug`, `Improve`, `Ensemble`) are classes
like any you would write. An operator never touches the agent, the disk or
the journal. It returns an `Attempt`: the prompt, and what the harness should
put in the candidate dir (here: a copy of the parent's solution). The prompt
is how an operator talks to the coding agent; the scripted `toy` agent reads
one line of it, `toy: step=<float>`.
"""

import sys

from hillclimb import Budget, Climber, Problem, SearchOutcome
from hillclimb.operators import Draft
from hillclimb.selectors import Best
from hillclimb.sdk import Action, Attempt, Operator, OperatorContext, OperatorPolicy, SearchState, Selection


class Stride(Operator):
    """Move away from the parent by a chosen stride."""

    name = "stride"        # what a policy's Action names
    kind = "refine"        # create | repair | refine | combine
    needs_target = True    # it starts from an existing candidate

    def prepare(self, ctx: OperatorContext) -> Attempt:
        step = self.params.get("step", 0.4)  # this operator's own knob: Stride(step=...)
        prompt = (
            f"Candidate {ctx.target.candidate_id} scores {ctx.target.val_score} "
            f"({ctx.problem.metric_name}). Move its point by about {step} in some direction "
            "and keep the change if the terrain is higher there.\n"
            f"toy: step={step}\n\n"
            "{{contract}}"  # where the harness puts the problem's contract
        )
        return Attempt(prompt=prompt, copy_parent=True, inherit_params=True)


class UseStride(OperatorPolicy):
    """The plain mapping, with `stride` as the operator for a chosen node."""

    def expand_action(self, state: SearchState, selection: Selection, operator: str = "improve") -> Action:
        return super().expand_action(state, selection, operator="stride")


def main(evaluations: int = 10, agent: str = "claude-code") -> dict[float, SearchOutcome]:
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)   # the same for every climber
    outcomes = {}
    for step in (0.02, 0.4, 3.0):
        climber = Climber(selector_policy=Best(num_drafts=3, ensemble=False), operator_policy=UseStride, operators=[Draft(), Stride(step=step)])
        climber.search(problem, budget=budget, agent=agent, learning=False, log=lambda *_: None)
        outcomes[step] = climber.result

    print(f"best elevation after {evaluations} evaluations, by stride:")
    for step, outcome in outcomes.items():
        print(f"  step={step:<5} {outcome.best.val_score:7.3f}")
    last = outcomes[3.0]
    newest = last.candidates[-1].candidate_id
    print(f"\nthe prompt {newest} was made from ({last.search_dir.name}/candidates/{newest}/prompt.md):")
    print((last.search_dir / "candidates" / newest / "prompt.md").read_text()[:300])
    return outcomes


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
