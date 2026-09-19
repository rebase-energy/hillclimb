"""Minimal OpenRouter client: one chat completion, one embeddings batch.

For one-shot calls that are not operators (no agent, no tools, no
candidate dir) — the solution-card similarity score is the first user. The
key is read from `OPENROUTER_API_KEY` at call time (config loading puts a
`.env` beside config.yaml into the environment), never passed around or
stored. Operators still reach OpenRouter through the codex/pi backends.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1"
EMBED_BATCH = 32  # inputs per embeddings request


class OpenRouterError(RuntimeError):
    """A call could not be made or the provider refused it."""


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set — export it or put it in a .env beside config.yaml"
        )
    return key


def _request(request: urllib.request.Request, path: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise OpenRouterError(f"OpenRouter {path} HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise OpenRouterError(f"OpenRouter {path} failed: {exc}") from exc
    if isinstance(data, dict) and data.get("error"):
        raise OpenRouterError(f"OpenRouter {path} error: {data['error']}")
    return data


def _post(path: str, body: dict, timeout: float) -> dict:
    return _request(
        urllib.request.Request(
            f"{API_URL}/{path}",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        ),
        path,
        timeout,
    )


def key_info(key: str | None = None, *, timeout: float = 10) -> dict:
    """What OpenRouter says about a key: its label, what it has spent, and
    its credit limit (`limit: None` means no cap on the key itself).

    The cheapest possible proof that a key is real — one unbilled GET, no
    model involved. `hillclimb connect openrouter` validates with it before
    storing a key, so a typo surfaces there instead of inside a search.
    `key` overrides the environment, for a key that is not installed yet.
    """
    data = _request(
        urllib.request.Request(
            f"{API_URL}/key",
            headers={"Authorization": f"Bearer {key or api_key()}"},
        ),
        "key",
        timeout,
    )
    info = data.get("data") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        raise OpenRouterError(f"OpenRouter key returned no data: {str(data)[:300]}")
    return info


def chat(
    model: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1200, timeout: float = 300,
) -> str:
    """The text of one single-turn completion."""
    data = _post("chat/completions", {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"OpenRouter chat returned no message: {str(data)[:300]}") from exc
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterError(f"OpenRouter chat returned an empty message ({model})")
    return content


def embed(model: str, texts: list[str], *, timeout: float = 300) -> list[list[float]]:
    """One embedding per text, in input order."""
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        data = _post("embeddings", {"model": model, "input": batch}, timeout)
        rows = sorted(data.get("data") or [], key=lambda row: row["index"])
        if len(rows) != len(batch):
            raise OpenRouterError(f"OpenRouter embeddings returned {len(rows)} vectors for {len(batch)} inputs")
        out.extend([float(x) for x in row["embedding"]] for row in rows)
    return out
