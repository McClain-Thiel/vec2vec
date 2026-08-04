"""A minimal, retrying OpenRouter chat-completions client.

Only the surface the description pipeline needs: one call, honest errors, and
the reported upstream cost so a run can be capped in dollars.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Models that emit hidden reasoning tokens unless explicitly disabled. For a
#: short description that budget is wasted and can consume the whole completion.
REASONING_MODELS = ("z-ai/glm-5.2", "z-ai/glm-4.6")

_MAX_BACKOFF_SECONDS = 30.0


@dataclass(frozen=True)
class Completion:
    """One successful chat completion and its reported cost."""

    text: str
    cost_usd: float


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """Exponential backoff with jitter, honouring a ``Retry-After`` header."""
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(2.0**attempt, _MAX_BACKOFF_SECONDS) + random.uniform(0.0, 0.5)


def _extract(payload: Any) -> Completion:
    """Pull the message text and cost out of an OpenRouter response body."""
    if not isinstance(payload, dict):
        raise TypeError("OpenRouter response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise TypeError("OpenRouter response content must be a string")
    usage = payload.get("usage")
    cost = usage.get("cost", 0.0) if isinstance(usage, dict) else 0.0
    if not isinstance(cost, int | float | str):
        raise TypeError("OpenRouter usage.cost must be numeric")
    return Completion(text=content.strip(), cost_usd=float(cost))


def complete(
    client: httpx.Client,
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    provider: str | None = None,
    max_retries: int = 6,
    timeout: float = 90.0,
) -> Completion:
    """Request one chat completion, retrying rate limits and transient failures.

    Args:
        provider: Pin an upstream provider (for example ``"Groq"``) through
            OpenRouter's provider-routing block; some serve a model far faster.

    Raises:
        httpx.HTTPError: when the request still fails after *max_retries*.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {"include": True},
    }
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if any(model.startswith(prefix) for prefix in REASONING_MODELS):
        body["reasoning"] = {"enabled": False}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "vec2vec plasmid descriptions",
    }

    for attempt in range(max_retries + 1):
        try:
            response = client.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
            if (response.status_code == 429 or response.status_code >= 500) and (
                attempt < max_retries
            ):
                time.sleep(_backoff(attempt, response.headers.get("Retry-After")))
                continue
            response.raise_for_status()
            return _extract(response.json())
        except httpx.TransportError:
            # Retryable statuses are handled above, before raise_for_status, so
            # only transport failures can reach here as worth another attempt.
            if attempt < max_retries:
                time.sleep(_backoff(attempt))
                continue
            raise
    raise RuntimeError("unreachable: retry loop exited without a result")
