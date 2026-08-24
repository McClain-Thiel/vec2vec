"""Tests for the small OpenRouter request and response boundary."""

from __future__ import annotations

import json

import httpx
import pytest

from vec2vec.lib import openrouter


def test_completion_sends_explicit_reasoning_and_response_format():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"cost": 0.001},
            },
        )

    response_format = {"type": "json_schema", "json_schema": {"name": "test", "schema": {}}}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        completion = openrouter.complete(
            client,
            [{"role": "user", "content": "test"}],
            model="qwen/test",
            api_key="not-a-real-key",
            reasoning={"enabled": False},
            response_format=response_format,
            seed=17,
            temperature=None,
            provider="Pinned Provider",
            require_parameters=True,
            max_retries=0,
        )

    assert completion.text == '{"ok":true}'
    assert completion.cost_usd == 0.001
    assert observed["reasoning"] == {"enabled": False}
    assert observed["response_format"] == response_format
    assert observed["seed"] == 17
    assert "temperature" not in observed
    assert observed["provider"] == {
        "allow_fallbacks": False,
        "order": ["Pinned Provider"],
        "require_parameters": True,
    }


def test_extraction_error_retains_reported_cost_and_generation_id():
    with pytest.raises(openrouter.ResponseExtractionError) as captured:
        openrouter._extract(
            {
                "id": "generation-123",
                "choices": [{"message": {"content": None}}],
                "usage": {"cost": 0.0123},
            }
        )

    assert captured.value.cost_usd == 0.0123
    assert captured.value.generation_id == "generation-123"


def test_api_error_retains_bounded_provider_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "bad_schema",
                    "message": "Every field must be required",
                    "metadata": {
                        "provider_name": "Test Provider",
                        "error_type": "invalid_request",
                        "raw": "maxItems is not supported",
                    },
                }
            },
            headers={"x-request-id": "request-123"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            openrouter.APIResponseError, match="Every field must be required"
        ) as err:
            openrouter.complete(
                client,
                [{"role": "user", "content": "test"}],
                model="test/model",
                api_key="not-a-real-key",
                max_retries=0,
            )

    assert err.value.status_code == 400
    assert err.value.request_id == "request-123"
    assert "provider=Test Provider" in str(err.value)
    assert "maxItems is not supported" in str(err.value)
