"""nano_climb: the smallest complete hill-climbing agent, in one file.

    uv run python examples/nano_climb.py

Everything a climber is, written out in order: a problem (what counts as
better), a selector (which node to build on), a policy (what to do with it),
two operators (what to ask the coding agent), a budget, and the search that
ties them together. Claude Code writes the solutions. Read it top to bottom;
nothing is hidden in a config file.
"""

import sys
from pathlib import Path

from hillclimb import Budget, Climber, Problem
from hillclimb.sdk import Action, Attempt, Operator, OperatorContext, Policy, SearchState, Selection, Selector

# ---------------------------------------------------------------------------
# 1. The problem: what the agent is asked to do, and how an answer is scored.
#
# A solution is a script that writes `pi.txt`. The score is how many leading
# digits of pi it got right. Nothing else about the problem exists: no data,
# no library, just this function and the description the agent reads.
# ---------------------------------------------------------------------------

PI = (
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
    "8628034825342117067982148086513282306647093844609550582231725359408128481117450284"
    "1027019385211055596446229489549303819644288109756659334461284756482337867831652712"
)


def digits_of_pi(run_dir: Path) -> float:
    """How many leading digits of pi.txt match pi. Higher is better."""
    answer = (run_dir / "pi.txt").read_text().strip()
    correct = 0
    for ours, theirs in zip(answer, PI):
        if ours != theirs:
            break
        correct += 1
    return float(correct - 1)  # the decimal point does not count


problem = Problem(
    "pi-digits",
    score=digits_of_pi,
    higher_is_better=True,
    description=(
        "Compute as many correct leading digits of pi as you can, from scratch, in pure Python\n"
        "with the standard library (the `decimal` module is allowed, a hard-coded constant is not).\n"
        "Write them to pi.txt as a plain decimal string like 3.14159... The script must finish\n"
        "within 60 seconds."
    ),
    output="pi.txt",
    baseline=0.0,
)

# ---------------------------------------------------------------------------
# 2. The selector: π_sel. Which node does the next attempt start from?
#
# This is the first decision of every step. It reads the history and
# answers with a node, or None for a root step. The base class handles the
# schedule (a failing attempt is repaired first, no building until
# `num_drafts` roots exist); `pick` is the one choice left: the best so far.
# ---------------------------------------------------------------------------


class BestSoFar(Selector):
    name = "best-so-far"

    def pick(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        scored = state.journal.scored_candidates()
        if not scored:
            return None
        best = max(scored, key=lambda c: c.val_score)
        return Selection(best.candidate_id)

# ---------------------------------------------------------------------------
# 3. The policy: π_op. Which operator is applied to what the selector chose?
#
# The second decision. No node means write a fresh attempt; a node means
# revise it. That is the whole of hill climbing.
# ---------------------------------------------------------------------------


class Climb(Policy):
    name = "climb"

    def propose(self, state: SearchState, selection: Selection | None) -> Action | None:
        if selection is None:
            return Action("write")
        return Action("revise", target_id=selection.target_id)

# ---------------------------------------------------------------------------
# 4. The operators: how one attempt is made. Each one is a prompt.
#
# An operator never runs anything itself. It says what to tell the coding
# agent and what to put in its working directory (`copy_parent` means the
# node's solution is there to edit). The harness appends the problem's
# contract where {{contract}} stands.
# ---------------------------------------------------------------------------


class Write(Operator):
    name, kind = "write", "create"

    def prepare(self, ctx: OperatorContext) -> Attempt:
        prompt = (
            "Write solution.py from scratch for the problem below. Pick a method you know "
            "converges fast and say in notes.md which one you chose.\n\n{{contract}}"
        )
        return Attempt(prompt=prompt)


class Revise(Operator):
    name, kind, needs_target = "revise", "refine", True

    def prepare(self, ctx: OperatorContext) -> Attempt:
        node = ctx.target
        prompt = (
            f"solution.py in this directory scored {node.val_score:g} correct digits "
            f"({ctx.problem.metric_name}). Change it so it computes more digits within the time "
            "limit: a faster-converging series, more precision, or a fix if it failed. "
            "Say in notes.md what you changed.\n\n{{contract}}"
        )
        return Attempt(prompt=prompt, copy_parent=True)

# ---------------------------------------------------------------------------
# 5. The budget, the climber, the search.
# ---------------------------------------------------------------------------


def main(evaluations: int = 8, agent: str = "claude-code"):
    budget = Budget(evaluations=evaluations)
    climber = Climber(
        select=BestSoFar(num_drafts=2),   # π_sel: two fresh attempts, then build on the best
        policy=Climb(),                   # π_op: write, or revise
        operators=[Write(), Revise()],    # the prompts above
    )

    climber.search(problem, budget=budget, agent=agent, learning=False)

    print(f"\n{climber.result.state}: {climber.spend.evaluations} attempts, best {climber.best.val_score:g} digits")
    for step in climber.history:
        print(f"  {step.candidate_id}  {step.score:4g} digits  at {step.minutes:.1f} min")
    print(climber.to_frame()[["parent_id", "operator", "status", "val_score", "summary"]].to_string())
    return climber.result


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
