import base64
import json

import httpx
import pytest

from iopaint.openai_compat.client import OpenAICompatClient
from iopaint.openai_compat.config import OpenAIConfig
from iopaint.openai_compat.errors import ErrorStatus, OpenAIError
from iopaint.openai_compat.models import (
    GenerateImageRequest,
    ImageSize,
    RefinePromptRequest,
)


def _make_client(handler) -> OpenAICompatClient:
    config = OpenAIConfig(api_key="test-key", base_url="https://api.test/v1")
    client = OpenAICompatClient(config)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=config.base_url,
        headers={"Authorization": f"Bearer {config.api_key}"},
    )
    return client


def test_list_models_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers.get("Authorization") == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "gpt-image-1",
                        "object": "model",
                        "created": 1,
                        "owned_by": "openai",
                    }
                ]
            },
        )

    client = _make_client(handler)

    models = client.list_models()

    assert len(models) == 1
    assert models[0].id == "gpt-image-1"


def test_generate_image_uses_b64_payload():
    image_bytes = b"image-bytes"
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/generations"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["prompt"] == "A cat"
        assert payload["size"] == "1024x1024"
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client = _make_client(handler)
    request = GenerateImageRequest(prompt="A cat", size=ImageSize.SIZE_1024)

    result = client.generate_image(request)

    assert result == image_bytes


def test_refine_prompt_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Refined prompt."}}]},
        )

    client = _make_client(handler)
    request = RefinePromptRequest(prompt="A cat")

    response = client.refine_prompt(request)

    assert response.refined_prompt == "Refined prompt."


def test_list_models_raises_openai_error():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(429, json={"error": {"message": "Rate limited"}})

    client = _make_client(handler)

    with pytest.raises(OpenAIError) as excinfo:
        client.list_models()

    assert excinfo.value.status == ErrorStatus.RATE_LIMITED
