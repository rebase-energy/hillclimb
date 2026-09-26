from __future__ import annotations

import json
import os
import time

import pytest

from hillclimb.harness.pricing import Prices, cost_usd, model_prices

CATALOGUE = {
    "data": [
        {
            "id": "qwen/qwen3-coder",
            "pricing": {
                "prompt": "0.0000002",
                "completion": "0.0000008",
                "input_cache_read": "0.00000002",
                "input_cache_write": "0.00000025",
            },
        },
        {
            "id": "cohere/north-mini-code:free",
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


def test_model_prices_reads_the_catalogue():
    assert model_prices("qwen/qwen3-coder", catalogue=CATALOGUE) == Prices(
        prompt=2e-7, completion=8e-7, cache_read=2e-8, cache_write=2.5e-7
    )


def test_cost_usd_bills_each_token_kind():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
    }

    assert cost_usd("qwen/qwen3-coder", usage, catalogue=CATALOGUE) == pytest.approx(1.27)


def test_a_free_model_costs_nothing():
    assert (
        cost_usd("cohere/north-mini-code:free", {"input_tokens": 50_000}, catalogue=CATALOGUE)
        == 0.0
    )


def test_an_unknown_model_yields_none_not_a_guess():
    assert model_prices("nobody/such-model", catalogue=CATALOGUE) is None
    assert cost_usd("nobody/such-model", {"input_tokens": 1}, catalogue=CATALOGUE) is None


def test_a_model_without_cache_prices_reads_them_as_zero():
    prices = model_prices("cohere/north-mini-code:free", catalogue=CATALOGUE)

    assert prices.cache_read == 0.0
    assert prices.cache_write == 0.0


def test_a_fresh_cache_file_is_used_without_fetching(tmp_path, monkeypatch):
    import hillclimb.harness.pricing as pricing

    cache = tmp_path / "openrouter-models.json"
    cache.write_text(json.dumps(CATALOGUE))
    monkeypatch.setattr(pricing, "_MEMO", None)
    monkeypatch.setattr(pricing, "_cache_path", lambda: cache)
    monkeypatch.setattr(
        pricing, "_fetch_catalogue", lambda: (_ for _ in ()).throw(AssertionError("fetched"))
    )

    assert model_prices("qwen/qwen3-coder") == Prices(
        prompt=2e-7, completion=8e-7, cache_read=2e-8, cache_write=2.5e-7
    )


def test_a_stale_cache_that_cannot_be_refreshed_is_still_used(tmp_path, monkeypatch):
    import hillclimb.harness.pricing as pricing

    cache = tmp_path / "openrouter-models.json"
    cache.write_text(json.dumps(CATALOGUE))
    stale = time.time() - pricing.CACHE_TTL_S - 60
    os.utime(cache, (stale, stale))
    monkeypatch.setattr(pricing, "_MEMO", None)
    monkeypatch.setattr(pricing, "_cache_path", lambda: cache)
    monkeypatch.setattr(pricing, "_fetch_catalogue", lambda: None)

    assert model_prices("qwen/qwen3-coder") is not None


def test_a_failed_fetch_is_not_retried_on_every_call(tmp_path, monkeypatch):
    import hillclimb.harness.pricing as pricing

    calls = []
    monkeypatch.setattr(pricing, "_MEMO", None)
    monkeypatch.setattr(pricing, "_LAST_FETCH_FAILURE", 0.0)
    monkeypatch.setattr(pricing, "_cache_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(pricing, "_fetch_catalogue", lambda: calls.append(1))

    assert model_prices("qwen/qwen3-coder") is None
    assert model_prices("qwen/qwen3-coder") is None
    assert len(calls) == 1
