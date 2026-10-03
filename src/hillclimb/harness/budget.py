from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hillclimb.config import Config
    from hillclimb.harness.journal import Journal

# the harness's own floor measurements: their first trial is not the climber's spend
FLOOR_OPERATORS = ("baseline", "seed")


@dataclass(frozen=True)
class Spend:
    """What a search has used so far besides the clock — derived from the
    journal on every read, so a resumed search needs no counter of its own."""

    evaluations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    # wall clock, where a reader has the status record beside the journal
    # (a search's result); the harness's own clock is its BudgetManager
    seconds: float | None = None


def journal_spend(journal: Journal) -> Spend:
    evaluations = tokens = 0
    cost = 0.0
    for candidate in journal.candidates.values():
        trials = len(candidate.trials)
        evaluations += max(0, trials - 1) if candidate.operator in FLOOR_OPERATORS else trials
        tokens += candidate.agent.total_tokens or 0
        cost += candidate.agent.cost_usd or 0.0
    return Spend(evaluations=evaluations, tokens=tokens, cost_usd=cost)


@dataclass(frozen=True)
class Budget:
    """What a search may spend, in every dimension the harness counts. Pass
    it where a budget goes (`climber.search(problem, budget=Budget(...))`):

        Budget(wall_clock="10m")                  # the clock alone
        Budget(evaluations=30)                    # scored attempts, whatever the clock says
        Budget(wall_clock="2h", evaluations=200, tokens=5_000_000, cost_usd=20)

    The search ends when the first limit is reached: the clock and the
    evaluation or token caps end it as `done`, the cost ceiling parks it
    (resumable). A dimension left as None has no limit, except the wall
    clock, which falls back to the problem's own `time_budget_s`. A plain
    "10m" or a number of seconds where a Budget is expected means
    `Budget(wall_clock=...)`."""

    wall_clock: str | int | None = None  # "2h", "30m", "90s" or seconds of wall-clock time
    evaluations: int | None = None     # scored attempts: one per attempt, one per tune trial
    tokens: int | None = None          # tokens the coding agent calls consume, all kinds summed
    cost_usd: float | None = None      # the coding agents' bill, where the agent reports one

    def __post_init__(self) -> None:
        if self.wall_clock is not None:
            parse_budget(self.wall_clock)  # a budget nobody can read fails here, not at the first search
        for name in ("evaluations", "tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"Budget({name}=...) takes a positive whole number, not {value!r}")
        if self.cost_usd is not None and not self.cost_usd > 0:
            raise ValueError(f"Budget(cost_usd=...) takes a positive amount, not {self.cost_usd!r}")

    @classmethod
    def of(cls, value: Any) -> Budget:
        """A Budget from what people pass: a Budget, "10m", seconds, or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return cls(wall_clock=value)
        raise TypeError(f"budget must be a Budget, a duration like '10m', or seconds, not {value!r}")

    @property
    def seconds(self) -> int | None:
        return parse_budget(self.wall_clock) if self.wall_clock is not None else None

    def apply(self, config: Config) -> list[str]:
        """Set the limits on `config.budget` (the clock is the search's own
        argument) and return them as the `set` pairs a run's spec records."""
        pairs = []
        if self.evaluations is not None:
            config.budget.max_evaluations = self.evaluations
            pairs.append(f"budget.max_evaluations={self.evaluations}")
        if self.tokens is not None:
            config.budget.max_tokens = self.tokens
            pairs.append(f"budget.max_tokens={self.tokens}")
        if self.cost_usd is not None:
            config.budget.max_cost_usd = float(self.cost_usd)
            pairs.append(f"budget.max_cost_usd={self.cost_usd}")
        return pairs

    def __repr__(self) -> str:
        fields = {k: v for k, v in vars(self).items() if v is not None}
        return "Budget(" + ", ".join(f"{k}={v!r}" for k, v in fields.items()) + ")"


def parse_budget(value: str | int) -> int:
    """A wall-clock budget as people write it — `2h`, `30m`, `90s`, plain
    seconds — in seconds."""
    import re

    match = re.fullmatch(r"(\d+)\s*([hms]?)", str(value).strip())
    if not match:
        raise ValueError(f"cannot parse budget {value!r} (use e.g. 2h, 30m, 90s)")
    amount, unit = int(match.group(1)), match.group(2)
    return amount * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


def format_remaining(seconds: float) -> str:
    """How the clock reads in prompts: `1h 05m` or `59 minutes` — prose a
    coding agent reads. Logs use `format_clock`."""
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes = rest // 60
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes} minutes"


def format_clock(seconds: float) -> str:
    """How the clock reads in the engine log: `9:42`, `1:05:00` — the same
    width from one line to the next, so the log's clock gutter lines up."""
    whole = max(0, int(round(seconds)))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class BudgetManager:
    """Wall-clock budget for a run. `spent_s` seeds time already used
    (from the journal) when resuming."""

    def __init__(self, total_s: int, stop_margin_s: int = 300, spent_s: float = 0.0):
        self.total_s = total_s
        # the margin is meant as "don't start an operator you can't finish",
        # so it cannot be a fixed 5 minutes when the whole budget is 10:
        # it is capped at a tenth of the budget
        self.stop_margin_s = min(stop_margin_s, total_s // 10)
        self._started = time.monotonic() - spent_s
        self._paused_at: float | None = None

    def elapsed(self) -> float:
        now = self._paused_at if self._paused_at is not None else time.monotonic()
        return now - self._started

    # A search stepped by hand spends its budget only while a step runs: the
    # time a person takes to read an outcome is theirs, not the search's.
    def pause(self) -> None:
        """Stop the clock (idempotent)."""
        if self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        """Start it again where it stopped (idempotent)."""
        if self._paused_at is not None:
            self._started += time.monotonic() - self._paused_at
            self._paused_at = None

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def remaining(self) -> float:
        return max(0.0, self.total_s - self.elapsed())

    def should_stop(self) -> bool:
        return self.remaining() < self.stop_margin_s

    def remaining_str(self) -> str:
        return format_remaining(self.remaining())

    def clock_str(self) -> str:
        return format_clock(self.remaining())
