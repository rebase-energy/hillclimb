"""OpenRouter list prices for operator calls.

A lookup over OpenRouter's public model catalogue, cached on the machine.
`cost_usd` is what makes `budget.max_cost_usd` a real ceiling for
provider-billed runs; an unknown model yields None rather than an invented
number, the contract `claude_code.estimate_cost_usd` already follows.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_S = 24 * 3600
FETCH_RETRY_S = 10 * 60
_MEMO: dict | None = None
_LAST_FETCH_FAILURE = 0.0


@dataclass(frozen=True)
class Prices:
    """Dollars per token, as OpenRouter quotes them."""

    prompt: float
    completion: float
    cache_read: float
    cache_write: float


def _cache_path() -> Path:
    return Path.home() / ".cache" / "hillclimb" / "openrouter-models.json"


def _fetch_catalogue() -> dict | None:
    try:
        with urllib.request.urlopen(CATALOGUE_URL, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _catalogue() -> dict | None:
    """Process memo, then the machine cache, then the network. A stale cache
    that cannot be refreshed still beats having no prices, and a failed
    fetch is not retried on every operator call — each one would stall the
    call by the network timeout."""
    global _MEMO, _LAST_FETCH_FAILURE
    if _MEMO is not None:
        return _MEMO
    if time.time() - _LAST_FETCH_FAILURE < FETCH_RETRY_S:
        return None
    path = _cache_path()
    cached: dict | None = None
    fresh = False
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            fresh = time.time() - path.stat().st_mtime < CACHE_TTL_S
        except (OSError, json.JSONDecodeError):
            cached = None
    if not fresh:
        fetched = _fetch_catalogue()
        if fetched is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fetched))
            cached = fetched
        elif cached is None:
            _LAST_FETCH_FAILURE = time.time()
    _MEMO = cached
    return cached


def model_prices(model: str, *, catalogue: dict | None = None) -> Prices | None:
    """List prices for an OpenRouter model id, or None when it is unknown."""
    data = catalogue if catalogue is not None else _catalogue()
    if not data:
        return None
    for entry in data.get("data") or []:
        if entry.get("id") != model:
            continue
        pricing = entry.get("pricing") or {}
        return Prices(
            prompt=float(pricing.get("prompt") or 0.0),
            completion=float(pricing.get("completion") or 0.0),
            cache_read=float(pricing.get("input_cache_read") or 0.0),
            cache_write=float(pricing.get("input_cache_write") or 0.0),
        )
    return None


def cost_usd(
    model: str, token_usage: dict[str, int], *, catalogue: dict | None = None
) -> float | None:
    """What one call's usage costs at list price, over hillclimb's canonical
    token keys. None when the model has no known prices."""
    prices = model_prices(model, catalogue=catalogue)
    if prices is None:
        return None
    return (
        (token_usage.get("input_tokens") or 0) * prices.prompt
        + (token_usage.get("output_tokens") or 0) * prices.completion
        + (token_usage.get("cache_read_input_tokens") or 0) * prices.cache_read
        + (token_usage.get("cache_creation_input_tokens") or 0) * prices.cache_write
    )
