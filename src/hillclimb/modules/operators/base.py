"""The operator seam: HOW one attempt is made.

An `Operator` turns a policy's `Action` into an `Attempt` — the prompt for
the agent plus what the harness should put in the new candidate dir. It never
touches the filesystem, the journal or a agent itself: the harness executes
the preparation, fills in the problem's contract (an operator cannot drop
it), clamps the timeout to the budget, routes the call by the operator's
name, runs the agent and scores the result.

Everything an operator sees is holdout-blind: `OperatorContext.target`,
`.inspirations` and `.journal` are the same masked copies a policy gets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from hillclimb.harness.candidate import Candidate
    from hillclimb.harness.journal import JournalView
    from hillclimb.modules.policies.base import Action, BudgetView

# What an operator's candidates ARE to the rest of the system. Views colour
# by role and the journal walks chains by role, so a climber's own operators
# ("crossover", "reflect") need no change anywhere else.
#   create  — a new solution from the problem alone
#   repair  — a fix of a candidate that failed
#   refine  — a change to a scored candidate
#   combine — one solution out of several
ROLES = ("create", "repair", "refine", "combine")

# operators the harness runs itself (no agent, no prompt): their candidates
# carry the operator's own name as role
RESERVED_ROLES = {"baseline": "baseline", "seed": "seed", "inject": "inject"}

# the token a template marks the contract's place with; the harness fills it
CONTRACT_TOKEN = "{{contract}}"
READ_TEXT_CHARS = 3000
TAIL_CHARS = 2000  # the engine's own log-tail length (evaluation.TAIL_CHARS)


def inspiration_filename(index: int) -> str:
    """The name the harness gives the `index`-th (1-based) inspiration it
    copies into a candidate dir. Prompts and policies that mention those
    files must use this, never a literal."""
    return f"candidate_{index}.py"


@dataclass(frozen=True)
class ProblemInfo:
    """What an operator may know about the problem."""

    problem_id: str
    description: str
    metric_name: str
    higher_is_better: bool
    allow_internet_during_solution: bool = False
    data_listing: str = ""

    @property
    def direction(self) -> str:
        return "higher is better" if self.higher_is_better else "lower is better"

    @property
    def allow_network(self) -> bool:
        """The pre-0.6 name of `allow_internet_during_solution`."""
        return self.allow_internet_during_solution

    @property
    def network_note(self) -> str:
        if self.allow_internet_during_solution:
            return (
                "Internet access IS available at execution time — this problem's rules "
                "permit fetching external data; cache downloads to files in the "
                "working directory so reruns don't refetch."
            )
        return "Assume no internet access at execution time."


@dataclass(frozen=True)
class MemoryContext:
    """What memory retrieved for this search before it started."""

    text: str = ""  # prior experience, ready to paste into a prompt
    reference: Path | None = None  # a proven solution worth scaffolding from
    reference_note: str = ""


class OperatorServices(Protocol):
    """The harness side of an `OperatorContext` — what needs the engine's
    templates, reports or unmasked records to answer."""

    def render(self, template: str, **tokens) -> str: ...
    def live_experience(self) -> str: ...
    def failure_reason(self, candidate_id: str) -> str: ...
    def report_section(self, candidate_id: str) -> str: ...


@dataclass(frozen=True)
class OperatorContext:
    action: Action
    target: Candidate | None
    inspirations: tuple[Candidate, ...]
    journal: JournalView
    problem: ProblemInfo
    budget: BudgetView
    memory: MemoryContext
    services: OperatorServices = field(repr=False)
    # may the agent making this attempt reach the internet (web search, curl)?
    # The user's `allow_internet_for_agents`; False = its shell is jailed and
    # its web tools are off, so a prompt must not ask it to search
    agent_internet: bool = True

    def render(self, template: str, **tokens) -> str:
        """Fill a prompt template's `{{tokens}}`. The climber's own
        `prompts/` shadow the built-in ones by name. Leave `{{contract}}`
        where the contract belongs — the harness fills it, and appends the
        contract when a prompt has no such token."""
        return self.services.render(template, **tokens)

    def live_experience(self) -> str:
        """What concurrent searches on the same problem have found so far
        (polled fresh on every call; empty when there is none)."""
        return self.services.live_experience()

    def failure_reason(self, candidate: Candidate) -> str:
        """Why a candidate is not passing, in the harness's words."""
        return self.services.failure_reason(candidate.candidate_id)

    def report_section(self, candidate: Candidate) -> str:
        """The candidate's validation breakdown as a prompt section (empty
        when the problem reports none)."""
        return self.services.report_section(candidate.candidate_id)

    def read_text(self, candidate: Candidate, name: str, max_chars: int = READ_TEXT_CHARS) -> str:
        """A text file an agent left in a candidate's dir (`ablation.md`,
        `notes.md`, `exec_stderr.log`), confined to that dir; empty when
        missing or unreadable."""
        if not candidate.candidate_dir:
            return ""
        root = Path(candidate.candidate_dir)
        path = root / name
        try:
            if root.resolve() not in path.resolve().parents:
                return ""
            return path.read_text(errors="replace").strip()[:max_chars]
        except OSError:
            return ""

    def tail(self, candidate: Candidate, name: str, max_chars: int = TAIL_CHARS) -> str:
        """The END of a log in a candidate's dir (where a traceback lives)."""
        if not candidate.candidate_dir:
            return ""
        root = Path(candidate.candidate_dir)
        path = root / name
        try:
            if root.resolve() not in path.resolve().parents:
                return ""
            return path.read_text(errors="replace")[-max_chars:]
        except OSError:
            return ""

    @staticmethod
    def summaries(candidates: Sequence[Candidate]) -> str:
        """One line per candidate: id, operator, score-or-status, summary."""
        lines = []
        for candidate in candidates:
            score = (
                f"val_score={candidate.val_score}"
                if candidate.val_score is not None
                else candidate.status
            )
            lines.append(
                f"- {candidate.candidate_id} ({candidate.operator}, {score}): "
                f"{candidate.summary or '(no summary)'}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Attempt:
    """What the harness should set up for one attempt."""

    prompt: str
    copy_parent: bool = False  # start from the target's solution.py
    inherit_params: bool = False  # carry the target's best parameter values as defaults
    copy_inspirations: bool = True  # copy the action's inspirations in (`inspiration_filename`)
    fork_session: bool = False  # ask to continue the target's agent session (granted only where a agent can)
    files: Mapping[str, Path] = field(default_factory=dict)  # extra files: name in the dir -> source
    texts: Mapping[str, str] = field(default_factory=dict)  # extra files written from text: name -> content
    # the attempt only counts if the agent CHANGED the copied parent solution;
    # an untouched one comes back as Outcome `unchanged` and is never scored
    require_change: bool = False


class Operator(ABC):
    """One way of making an attempt. Subclasses set `name` and `role` and
    implement `prepare`; `params` is the operator's own configuration."""

    name: str
    role: str
    needs_target: bool = False

    def __init__(self, params: Mapping | None = None, **knobs):
        self.params = {**dict(params or {}), **knobs}  # `Draft(retrieval=False)`

    def valid_target(self, target: Candidate | None) -> str | None:
        """Why this operator cannot run on `target`, or None when it can.
        Checked by the harness before anything is created or spent."""
        if self.needs_target and target is None:
            return f"{self.name} without a target_id"
        return None

    @abstractmethod
    def prepare(self, ctx: OperatorContext) -> Attempt: ...
