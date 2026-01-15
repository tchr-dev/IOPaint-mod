import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.api import Api
from iopaint.storage import JobOperation, JobStatus
from iopaint.tests.test_openai_tools_api import _make_api_config, _make_png_bytes


def _make_protocol_client(monkeypatch, tmp_path, handler, edit_handler=None, env=None):
    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AIE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIE_OPENAI_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("AIE_RATE_LIMIT_SECONDS", "0")
    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(Api, "_build_model_manager", lambda self: SimpleNamespace())
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})

    app = FastAPI()
    api = Api(app, _make_api_config())
    api.openai_client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=api.openai_config.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api.openai_config.api_key}"},
        timeout=httpx.Timeout(api.openai_config.timeout_s),
    )
    if edit_handler:
        monkeypatch.setattr(httpx, "post", edit_handler)
    client = TestClient(app)
    return client, api


def test_openai_protocol_generate_creates_history(monkeypatch, tmp_path):
    image_bytes = _make_png_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/v1/images/generations"
        assert payload["prompt"] == "A cat"
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client, api = _make_protocol_client(monkeypatch, tmp_path, handler)

    response = client.post(
        "/v1/images/generations",
        headers={"X-Session-Id": "session-1"},
        json={"prompt": "A cat", "size": "1024x1024", "quality": "standard"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["b64_json"] == encoded

    jobs, total = api.history_storage.list_jobs("session-1", limit=10, offset=0)
    assert total == 1
    assert jobs[0].operation == JobOperation.GENERATE
    assert jobs[0].status == JobStatus.SUCCEEDED
    assert jobs[0].result_image_id
    stored = api.image_storage.get_image(jobs[0].result_image_id)
    assert stored == image_bytes


def test_openai_protocol_generate_dedupe(monkeypatch, tmp_path):
    image_bytes = _make_png_bytes(color=(0, 255, 0))
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client, _api = _make_protocol_client(monkeypatch, tmp_path, handler)

    for _ in range(2):
        response = client.post(
            "/v1/images/generations",
            headers={"X-Session-Id": "session-2"},
            json={"prompt": "A tree", "size": "1024x1024", "quality": "standard"},
        )
        assert response.status_code == 200

    assert calls["count"] == 1


def test_openai_protocol_edit_creates_history(monkeypatch, tmp_path):
    image_bytes = _make_png_bytes()
    mask_bytes = _make_png_bytes(color=(0, 0, 0))
    result_bytes = _make_png_bytes(color=(0, 0, 255))
    encoded = base64.b64encode(result_bytes).decode("utf-8")

    def edit_handler(url, *args, **kwargs):
        assert url.endswith("/images/edits")
        assert "image" in kwargs["files"]
        assert "mask" in kwargs["files"]
        assert kwargs["data"]["prompt"] == "Add clouds"
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client, api = _make_protocol_client(
        monkeypatch, tmp_path, handler, edit_handler=edit_handler
    )

    response = client.post(
        "/v1/images/edits",
        headers={"X-Session-Id": "session-3"},
        data={"prompt": "Add clouds"},
        files={
            "image": ("image.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["b64_json"] == encoded

    jobs, total = api.history_storage.list_jobs("session-3", limit=10, offset=0)
    assert total == 1
    assert jobs[0].operation == JobOperation.EDIT
    assert jobs[0].status == JobStatus.SUCCEEDED


def test_openai_protocol_budget_block(monkeypatch, tmp_path):
    image_bytes = _make_png_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client, api = _make_protocol_client(
        monkeypatch,
        tmp_path,
        handler,
        env={
            "AIE_BUDGET_SESSION_CAP": "0.04",
            "AIE_BUDGET_DAILY_CAP": "0",
            "AIE_BUDGET_MONTHLY_CAP": "0",
            "AIE_DEDUPE_ENABLED": "false",
        },
    )

    ok_response = client.post(
        "/v1/images/generations",
        headers={"X-Session-Id": "budget-session"},
        json={"prompt": "A fox", "size": "1024x1024", "quality": "standard"},
    )
    assert ok_response.status_code == 200

    blocked_response = client.post(
        "/v1/images/generations",
        headers={"X-Session-Id": "budget-session"},
        json={"prompt": "A fox at night", "size": "1024x1024", "quality": "standard"},
    )
    assert blocked_response.status_code == 402
    assert api.budget_storage.get_session_spend("budget-session") == pytest.approx(0.04)
