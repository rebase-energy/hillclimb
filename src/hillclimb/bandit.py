"""UCB1 model routing: pick which model runs an operator, learn from results.

A `routing:` entry with a `models:` pool (2+ entries) turns that operator's
model choice into a bandit arm pull (ShinkaEvolve-style adaptive LLM
selection). Rewards are derived purely from journaled candidate records —
candidate.backend.model names the arm, the parent's val score anchors the
comparison — so bandit state rebuilds from journal replay and `resume` works
without any extra persistence.

Reward scale is [0, 1] (classic UCB1 assumptions), scale-free across metrics:

- improved on a scored parent          -> 1.0   (the move that matters)
- passing, parentless/unscored parent, best -> 1.0 (a draft that took the lead)
- passing but no improvement               -> 0.25 (working code, no progress)
- failing/buggy                             -> 0.0
- parked/abandoned/pending             -> None  (not the model's doing; skip)
"""

from __future__ import annotations

import math
import threading

from hillclimb.candidate import Candidate

REWARD_OK_NO_GAIN = 0.25


def candidate_reward(
    candidate: Candidate,
    parent: Candidate | None,
    higher_is_better: bool,
    band: float = 0.0,
) -> float | None:
    """Journal-derivable reward for the model arm that authored `candidate`;
    None = no update (no arm recorded, or a non-terminal/harness failure)."""
    if not candidate.backend.model:
        return None
    if candidate.status in ("failing", "buggy"):
        return 0.0
    if candidate.status != "passing":
        return None
    val = candidate.val_score
    if val is None:
        return REWARD_OK_NO_GAIN
    parent_val = parent.val_score if parent is not None else None
    if parent_val is not None:
        # `band`: an arm gets full credit only for a gain the search can
        # actually measure, so noise does not train the router
        delta = (val - parent_val) if higher_is_better else (parent_val - val)
        return 1.0 if delta > band else REWARD_OK_NO_GAIN
    return 1.0 if candidate.is_best else REWARD_OK_NO_GAIN


class UCB1:
    """Textbook UCB1 over a fixed arm set. Unpulled arms go first (in pool
    order); afterwards the arm maximizing mean + c*sqrt(ln N / n) wins."""

    def __init__(self, arms: tuple[str, ...], exploration: float = 1.0):
        self.arms = arms
        self.exploration = exploration
        self.pulls = {arm: 0 for arm in arms}
        self.total_reward = {arm: 0.0 for arm in arms}

    def select(self) -> str:
        for arm in self.arms:
            if self.pulls[arm] == 0:
                return arm
        total = sum(self.pulls.values())
        return max(self.arms, key=lambda arm: self._score(arm, total))

    def _score(self, arm: str, total: int) -> float:
        mean = self.total_reward[arm] / self.pulls[arm]
        return mean + self.exploration * math.sqrt(math.log(total) / self.pulls[arm])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.pulls:
            return  # journal from an older pool config; nothing to learn
        self.pulls[arm] += 1
        self.total_reward[arm] += reward


class OperatorBandits:
    """One UCB1 per (operator, pool) — operators have different reward
    structure (a debug 'fix' is cheaper than an improve 'gain'), so their
    statistics must not mix. Selection happens on the scheduler thread under
    the searcher's state lock; updates additionally arrive from journal
    replay at construction. The lock keeps snapshot() safe for observers."""

    def __init__(self, exploration: float = 1.0):
        self.exploration = exploration
        self._bandits: dict[tuple[str, tuple[str, ...]], UCB1] = {}
        self._lock = threading.Lock()

    def _bandit(self, operator: str, pool: tuple[str, ...]) -> UCB1:
        key = (operator, pool)
        if key not in self._bandits:
            self._bandits[key] = UCB1(pool, exploration=self.exploration)
        return self._bandits[key]

    def select(self, operator: str, pool: tuple[str, ...]) -> str:
        with self._lock:
            return self._bandit(operator, pool).select()

    def update(self, operator: str, pool: tuple[str, ...], arm: str, reward: float) -> None:
        with self._lock:
            self._bandit(operator, pool).update(arm, reward)

    def snapshot(self) -> dict:
        """{operator: {model: {pulls, mean_reward}}} for logging/tests."""
        with self._lock:
            out: dict = {}
            for (operator, _pool), bandit in self._bandits.items():
                stats = out.setdefault(operator, {})
                for arm in bandit.arms:
                    pulls = bandit.pulls[arm]
                    stats[arm] = {
                        "pulls": pulls,
                        "mean_reward": (
                            round(bandit.total_reward[arm] / pulls, 4) if pulls else None
                        ),
                    }
            return out
