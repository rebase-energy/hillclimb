"""`solution-card`: compare solutions by method, not by text.

An LLM reads the solution and writes a *solution card* — libraries,
algorithm family, representation, initialization, core procedure,
refinement, constraint handling, budget — in generic vocabulary, with no
identifiers and no problem domain. The cards are embedded, and the
similarity is the cosine between embeddings. Renaming a function leaves the
card (nearly) unchanged; the same method on a different problem lands close.

Both steps go through OpenRouter (`OPENROUTER_API_KEY`). Cards and vectors
are cached in the machine cache separately, so switching the embedding
model reuses every card. The card is an LLM output: two cards of the same
code differ slightly, so treat differences below ~0.02 as noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hillclimb.backends.openrouter as openrouter
from hillclimb.modules.similarity.base import SimilarityScore, SimilarityUnavailable, Solution
from hillclimb.modules.similarity.compute import DiskCache

PROMPT_FILE = Path(__file__).with_name("solution_card.md")
SOURCE_TOKEN = "{{source}}"


class SolutionCard(SimilarityScore):
    name = "solution-card"
    description = "LLM-written method card per solution, embedded; cosine between cards (rename-invariant)"
    version = "1"
    cache = True
    defaults = {
        "card_model": "anthropic/claude-haiku-4.5",
        "embedding_model": "voyageai/voyage-4",
        "prompt": None,             # path to a template with {{source}}; None = the built-in one
        "max_source_chars": 60_000,  # longer sources are truncated before the LLM sees them
        "concurrency": 8,           # parallel card requests
    }

    def __init__(self, params=None):
        super().__init__(params)
        prompt = self.params["prompt"]
        self.template = Path(prompt).expanduser().read_text() if prompt else PROMPT_FILE.read_text()
        if SOURCE_TOKEN not in self.template:
            raise ValueError(f"solution-card prompt must contain {SOURCE_TOKEN}")
        self._cards = DiskCache(f"{self.name}-cards")

    def _card_key(self, solution: Solution) -> str:
        return DiskCache.key(self.params["card_model"], self.template, self.params["max_source_chars"], solution.digest)

    def card(self, solution: Solution) -> str:
        """The solution's card, from the cache or one LLM call."""
        key = self._card_key(solution)
        cached = self._cards.get(key)
        if isinstance(cached, str):
            return cached
        source = solution.source[: int(self.params["max_source_chars"])]
        try:
            text = openrouter.chat(self.params["card_model"], self.template.replace(SOURCE_TOKEN, source)).strip()
        except openrouter.OpenRouterError as exc:
            if "OPENROUTER_API_KEY" in str(exc):
                raise SimilarityUnavailable(str(exc)) from exc
            raise
        self._cards.put(key, text)
        return text

    def represent_many(self, solutions: Sequence[Solution]) -> list[list[float] | None]:
        def safe_card(solution: Solution) -> str | None:
            try:
                return self.card(solution)
            except SimilarityUnavailable:
                raise
            except Exception:  # noqa: BLE001 — unreadable file or one failed call
                return None

        workers = max(1, min(int(self.params["concurrency"]), len(solutions)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cards = list(pool.map(safe_card, solutions))
        texts = [c for c in cards if c]
        if not texts:
            return [None] * len(solutions)
        try:
            vectors = iter(openrouter.embed(self.params["embedding_model"], texts))
        except openrouter.OpenRouterError as exc:
            raise SimilarityUnavailable(str(exc)) from exc
        return [next(vectors) if c else None for c in cards]

    def cache_key(self, solution: Solution) -> str | None:
        """Vectors are keyed by the prompt text too: editing the template
        file must not serve embeddings of cards the old prompt wrote."""
        try:
            return self._card_key(solution)
        except OSError:
            return None

    def represent(self, solution: Solution) -> list[float] | None:
        return self.represent_many([solution])[0]

    def explain(self, solution: Solution) -> str | None:
        return self.card(solution)
