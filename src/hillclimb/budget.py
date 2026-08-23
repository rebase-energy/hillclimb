from __future__ import annotations

import time


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
        seconds = int(round(self.remaining()))
        hours, rest = divmod(seconds, 3600)
        minutes = rest // 60
        return f"{hours}h {minutes:02d}m" if hours else f"{minutes} minutes"
