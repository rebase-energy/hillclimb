"""Register your own scripted agent, in Python.

    uv run python examples/custom_agent.py

An agent is any object with a `name` and `invoke(request) -> AgentResult`:
it is handed a prompt and a candidate dir, and leaves a `solution.py` there.
The bundled ones (`claude-code`, `codex`, `pi`) run a coding agent's CLI. A
scripted one is handy for trying a climber for free, or for testing your
own modules.

A registered agent lives in this process: `climber.search` and
`climber.start` find it, a detached `hillclimb run` does not.
"""

from hillclimb import Budget, Climber, Problem, SearchOutcome, register_agent
from hillclimb.agents import AgentRequest, AgentResult
from hillclimb.operators import Draft, Improve
from hillclimb.policies import Greedy
from hillclimb.selectors import Best


class GridWalker:
    """Ignores the prompt: each call stands on the next point of a coarse grid."""

    name = "grid"

    def __init__(self, side: int = 6):
        self.side = side
        self.calls = 0  # a search builds its own instance, so this counts one search's calls

    def invoke(self, request: AgentRequest) -> AgentResult:
        row, column = divmod(self.calls % self.side**2, self.side)
        self.calls += 1
        x = -5 + 10 * (column + 0.5) / self.side
        y = -5 + 10 * (row + 0.5) / self.side
        # the solution is a script the problem's verifier runs
        (request.candidate_dir / "solution.py").write_text(
            f'open("submission.csv", "w").write("x,y\\n{x},{y}\\n")\n'
        )
        (request.candidate_dir / "notes.md").write_text(f"grid point ({x:.2f}, {y:.2f})\n")
        return AgentResult(ok=True, cost_usd=0.0, total_tokens=0)


def main(evaluations: int = 36, agent: str = "grid") -> SearchOutcome:
    register_agent("grid", GridWalker, replace=True)
    problem = Problem("fitness-landscape")
    budget = Budget(evaluations=evaluations)
    climber = Climber(select=Best(ensemble=False), policy=Greedy(), operators=[Draft(), Improve()])

    climber.search(problem, budget=budget, agent="grid", learning=False, log=lambda *_: None)
    frame = climber.to_frame()
    print(f"a {evaluations}-point grid finds {climber.best.val_score:.3f} (the needle is 17.11):")
    print(frame[["operator", "val_score", "x", "y"]].sort_values("val_score", ascending=False).head(5).to_string())
    return climber.result


if __name__ == "__main__":
    main()
