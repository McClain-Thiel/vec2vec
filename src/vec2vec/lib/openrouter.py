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
    generation_id: str | None = None
    upstream_model: str | None = None
    upstream_provider: str | None = None


class ResponseExtractionError(ValueError):
    """A charged response that could not be converted into a completion."""

    def __init__(
        self, message: str, *, cost_usd: float = 0.0, generation_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd
        self.generation_id = generation_id
        self.upstream_model: str | None = None
        self.upstream_provider: str | None = None


class APIResponseError(RuntimeError):
    """A non-retryable OpenRouter HTTP response with a bounded error detail."""

    def __init__(self, status_code: int, detail: str, *, request_id: str | None = None) -> None:
        super().__init__(f"OpenRouter HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.request_id = request_id


def _http_error(response: httpx.Response) -> APIResponseError:
    """Preserve an API error message without copying a complete response body."""
    try:
        payload = response.json()
    except ValueError:
        detail = response.text.strip()[:1000] or "empty response body"
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            detail = f"{code}: {message}" if code is not None else str(message)
            metadata = error.get("metadata")
            if isinstance(metadata, dict):
                provider = metadata.get("provider_name")
                error_type = metadata.get("error_type")
                provider_code = metadata.get("provider_code")
                raw = metadata.get("raw")
                fields = [
                    f"provider={provider}" if provider else None,
                    f"error_type={error_type}" if error_type else None,
                    f"provider_code={provider_code}" if provider_code else None,
                    f"provider_detail={str(raw)[:500]}" if raw else None,
                ]
                suffix = "; ".join(field for field in fields if field)
                if suffix:
                    detail = f"{detail}; {suffix}"
        else:
            detail = str(payload)[:1000]
    request_id = response.headers.get("x-request-id")
    return APIResponseError(response.status_code, detail, request_id=request_id)


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
    generation_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    upstream_model = payload.get("model") if isinstance(payload.get("model"), str) else None
    upstream_provider = (
        payload.get("provider") if isinstance(payload.get("provider"), str) else None
    )
    usage = payload.get("usage")
    cost = usage.get("cost", 0.0) if isinstance(usage, dict) else 0.0
    if not isinstance(cost, int | float | str):
        error = ResponseExtractionError(
            "OpenRouter usage.cost must be numeric", generation_id=generation_id
        )
        error.upstream_model = upstream_model
        error.upstream_provider = upstream_provider
        raise error
    cost_usd = float(cost)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        error = ResponseExtractionError(
            "OpenRouter response has no choices",
            cost_usd=cost_usd,
            generation_id=generation_id,
        )
        error.upstream_model = upstream_model
        error.upstream_provider = upstream_provider
        raise error
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        error = ResponseExtractionError(
            "OpenRouter response content must be a string",
            cost_usd=cost_usd,
            generation_id=generation_id,
        )
        error.upstream_model = upstream_model
        error.upstream_provider = upstream_provider
        raise error
    return Completion(
        text=content.strip(),
        cost_usd=cost_usd,
        generation_id=generation_id,
        upstream_model=upstream_model,
        upstream_provider=upstream_provider,
    )


def complete(
    client: httpx.Client,
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float | None = 0.2,
    provider: str | None = None,
    seed: int | None = None,
    reasoning: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    require_parameters: bool = False,
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
        "usage": {"include": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if provider or require_parameters:
        body["provider"] = {
            "allow_fallbacks": False,
            "require_parameters": require_parameters,
        }
        if provider:
            body["provider"]["order"] = [provider]
    if reasoning is not None:
        body["reasoning"] = reasoning
    elif any(model.startswith(prefix) for prefix in REASONING_MODELS):
        body["reasoning"] = {"enabled": False}
    if response_format is not None:
        body["response_format"] = response_format

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
            if response.is_error:
                raise _http_error(response)
            return _extract(response.json())
        except httpx.TransportError:
            # Retryable statuses are handled above, before raise_for_status, so
            # only transport failures can reach here as worth another attempt.
            if attempt < max_retries:
                time.sleep(_backoff(attempt))
                continue
            raise
    raise RuntimeError("unreachable: retry loop exited without a result")
