from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def journal_spend(journal: Journal) -> Spend:
    evaluations = tokens = 0
    cost = 0.0
    for candidate in journal.candidates.values():
        trials = len(candidate.trials)
        evaluations += max(0, trials - 1) if candidate.operator in FLOOR_OPERATORS else trials
        tokens += candidate.agent.total_tokens or 0
        cost += candidate.agent.cost_usd or 0.0
    return Spend(evaluations=evaluations, tokens=tokens, cost_usd=cost)


def format_remaining(seconds: float) -> str:
    """How the clock reads in prompts: `1h 05m` or `59 minutes` — prose an
    agent reads. Logs use `format_clock`."""
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

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def remaining(self) -> float:
        return max(0.0, self.total_s - self.elapsed())

    def should_stop(self) -> bool:
        return self.remaining() < self.stop_margin_s

    def remaining_str(self) -> str:
        return format_remaining(self.remaining())

    def clock_str(self) -> str:
        return format_clock(self.remaining())
