"""Define a problem in Python and search it.

    uv run python examples/define_a_problem.py             # with Claude Code writing the solutions
    uv run python examples/define_a_problem.py bisector    # with a scripted stand-in, in seconds

A problem is a scoring function. The coding agent writes a `solution.py`,
hillclimb runs it, then calls your function in the directory it ran in. The
number your function returns is the score. hillclimb writes the problem's
folder under problems/ for you, so afterwards it is a problem like any
other: `hillclimb verify closest-to-pi`, `hillclimb run closest-to-pi`.
"""

import sys
from pathlib import Path

from hillclimb import Budget, Climber, Problem, SearchOutcome, register_agent
from hillclimb.agents import AgentRequest, AgentResult
from hillclimb.operators import Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.selectors import Best


def closeness_to_pi(run_dir: Path) -> float:
    """How close the number in answer.txt is to pi. Higher is better."""
    guess = float((run_dir / "answer.txt").read_text())
    return -abs(guess - 3.14159265358979)


problem = Problem(
    "closest-to-pi",
    score=closeness_to_pi,            # your verifier, in Python
    higher_is_better=True,
    description="Write your best approximation of pi to answer.txt, computed rather than typed in.",
    output="answer.txt",              # the file a solution must write
    baseline=-3.14159265358979,       # the score of answering 0
)


class Bisector:
    """A scripted stand-in for a coding agent: homes in on pi by bisection."""

    name = "bisector"

    def __init__(self):
        self.low, self.high = 3.0, 3.5

    def invoke(self, request: AgentRequest) -> AgentResult:
        guess = (self.low + self.high) / 2
        if guess < 3.14159265358979:
            self.low = guess
        else:
            self.high = guess
        (request.candidate_dir / "solution.py").write_text(
            f'open("answer.txt", "w").write("{guess!r}")\n'
        )
        return AgentResult(ok=True)


def main(evaluations: int = 8, agent: str = "claude-code") -> SearchOutcome:
    register_agent("bisector", Bisector, replace=True)
    budget = Budget(evaluations=evaluations)
    climber = Climber(
        selector_policy=Best(num_drafts=2, ensemble=False), operator_policy=Greedy(tune_budget=0), operators=[Draft(), Improve()],
    )

    climber.search(problem, budget=budget, agent=agent, learning=False, log=lambda *_: None)
    print(f"best after {evaluations} evaluations: {climber.solution.strip()}  (score {climber.best.val_score:.2e})")
    print(climber.to_frame()[["operator", "status", "val_score"]].to_string())
    return climber.result


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
