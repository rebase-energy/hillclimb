"""nano_climb: the smallest complete hill-climbing agent, in one file.

    uv run python examples/nano_climb.py

Everything a climber is, written out in the order a step runs it: a problem
(what counts as better), a selector policy (which node to build on), an
operator policy (what to do with it), two operators (what to ask the coding
agent), a budget, and the search that ties them together. Claude Code writes
the solutions. Read it top to bottom; nothing is hidden in a config file.
"""

import random
import sys
from pathlib import Path

from hillclimb import Budget, Climber, Problem
from hillclimb.sdk import Action, Attempt, Operator, OperatorContext, OperatorPolicy, SearchState, Selection, SelectorPolicy

# ---------------------------------------------------------------------------
# 1. The problem: what the agent is asked to do, and how an answer is scored.
#
# A solution is a script that has ten seconds to find a prime and write it
# to prime.txt. The score is how many digits the prime has. The verifier
# checks it really is prime (Miller-Rabin), so a bigger number is only a
# better score if the search behind it was sound. Nothing else about the
# problem exists: no data, no library, this function and a description.
# ---------------------------------------------------------------------------


def is_probably_prime(n: int, rounds: int = 24) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    rng = random.Random(0)
    for _ in range(rounds):
        x = pow(rng.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_digits(run_dir: Path) -> float:
    """How many digits the prime in prime.txt has; 0 if it is not a prime."""
    text = (run_dir / "prime.txt").read_text().strip()
    if not text.isdigit() or len(text) > 20_000:
        return 0.0
    return float(len(text)) if is_probably_prime(int(text)) else 0.0


problem = Problem(
    "largest-prime",
    score=prime_digits,
    higher_is_better=True,
    description=(
        "Find the largest prime number you can and write it, in decimal, to prime.txt. Pure\n"
        "Python with the standard library only; no hard-coded primes. The verifier checks primality\n"
        "with Miller-Rabin, so the number must really be prime. At most 20000 digits."
    ),
    output="prime.txt",
    time_limit_s=10,      # the verifier stops a solution that runs longer: a failed attempt
    baseline=0.0,
)

# ---------------------------------------------------------------------------
# 2. The selector policy: π_sel. Which node does the next attempt start from?
#
# This is the first decision of every step. It reads the history and
# answers with a node, or None for a root step. The base class carries the
# schedule (a failing attempt is repaired first, nothing is built on until
# `num_drafts` roots exist); `pick` is the one choice left: the best so far.
# ---------------------------------------------------------------------------


class BestSoFar(SelectorPolicy):
    name = "best-so-far"

    def pick(self, state: SearchState, *, busy=frozenset()) -> Selection | None:
        scored = state.journal.scored_candidates()
        if not scored:
            return None
        best = max(scored, key=lambda c: c.val_score)
        return Selection(best.candidate_id)

# ---------------------------------------------------------------------------
# 3. The operator policy: π_op. Which operator is applied to what was chosen?
#
# The second decision. No node means write a fresh attempt; a node means
# revise it. That is the whole of hill climbing.
# ---------------------------------------------------------------------------


class WriteOrRevise(OperatorPolicy):
    name = "write-or-revise"

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
            "Write solution.py from scratch for the problem below. Think about where the time\n"
            "goes: finding a candidate, and proving it prime. Say in notes.md what you chose.\n\n"
            "{{contract}}"
        )
        return Attempt(prompt=prompt)


class Revise(Operator):
    name, kind, needs_target = "revise", "refine", True

    def prepare(self, ctx: OperatorContext) -> Attempt:
        node = ctx.target
        if node.val_score is None:
            what = "It failed, or ran past the time limit. Fix that first, then make it find a bigger prime."
        else:
            what = f"It found a {node.val_score:g}-digit prime. Make it find a bigger one within the time limit."
        prompt = (
            f"solution.py in this directory is your previous attempt. {what}\n"
            "Keep the primality test sound; the verifier re-checks it. Say in notes.md what you changed.\n\n"
            "{{contract}}"
        )
        return Attempt(prompt=prompt, copy_parent=True)

# ---------------------------------------------------------------------------
# 5. The budget, the climber, the search.
# ---------------------------------------------------------------------------


def main(evaluations: int = 8, agent: str = "claude-code"):
    budget = Budget(wall_clock="45m", evaluations=evaluations)
    climber = Climber(
        selector_policy=BestSoFar(num_drafts=2),   # π_sel: two fresh attempts, then build on the best
        operator_policy=WriteOrRevise(),           # π_op: write, or revise
        operators=[Write(), Revise()],             # the two prompts above
    )

    climber.search(problem, budget=budget, agent=agent, learning=False)

    print(f"\n{climber.result.state}: {climber.spend.evaluations} attempts, best {climber.best.val_score:g} digits")
    for step in climber.history:
        print(f"  {step.candidate_id}  {step.score:6g} digits  at {step.minutes:.1f} min")
    print(climber.to_frame()[["parent_id", "operator", "status", "val_score"]].to_string())
    return climber.result


if __name__ == "__main__":
    main(agent=sys.argv[1] if len(sys.argv) > 1 else "claude-code")
