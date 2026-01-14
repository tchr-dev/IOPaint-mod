from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iopaint.api import Api
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    RealESRGANModel,
    RemoveBGModel,
)


def _make_api_config() -> ApiConfig:
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="lama",
        no_half=True,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=True,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=None,
        quality=95,
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.vit_b,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model=RemoveBGModel.briaai_rmbg_1_4.value,
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
    )


def test_history_snapshots_api(monkeypatch, tmp_path):
    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(Api, "_build_model_manager", lambda self: SimpleNamespace())
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})

    app = FastAPI()
    Api(app, _make_api_config())
    client = TestClient(app)

    headers = {"X-Session-Id": "test-session"}
    payload = {"history": [{"id": "job-1"}], "filter": "all"}

    res = client.post(
        "/api/v1/history/snapshots",
        json={"payload": payload},
        headers=headers,
    )
    assert res.status_code == 200
    snapshot = res.json()
    assert snapshot["payload"] == payload
    snapshot_id = snapshot["id"]

    res = client.get("/api/v1/history/snapshots", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert data["snapshots"][0]["id"] == snapshot_id

    res = client.get(
        f"/api/v1/history/snapshots/{snapshot_id}", headers=headers
    )
    assert res.status_code == 200
    assert res.json()["id"] == snapshot_id

    res = client.delete(
        f"/api/v1/history/snapshots/{snapshot_id}", headers=headers
    )
    assert res.status_code == 200
    assert res.json()["deleted"] is True

    client.post(
        "/api/v1/history/snapshots",
        json={"payload": {"history": [{"id": "job-2"}]}},
        headers=headers,
    )
    client.post(
        "/api/v1/history/snapshots",
        json={"payload": {"history": [{"id": "job-3"}]}},
        headers=headers,
    )
    res = client.delete("/api/v1/history/snapshots/clear", headers=headers)
    assert res.status_code == 200
    assert res.json()["deleted"] == 2

    res = client.get("/api/v1/history/snapshots", headers=headers)
    assert res.status_code == 200
    assert res.json()["total"] == 0


def test_history_snapshots_pagination_and_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(Api, "_build_model_manager", lambda self: SimpleNamespace())
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})

    app = FastAPI()
    Api(app, _make_api_config())
    client = TestClient(app)

    headers = {"X-Session-Id": "test-session"}

    res = client.post("/api/v1/history/snapshots", json={}, headers=headers)
    assert res.status_code == 422

    for idx in range(3):
        res = client.post(
            "/api/v1/history/snapshots",
            json={"payload": {"history": [{"id": f"job-{idx}"}]}},
            headers=headers,
        )
        assert res.status_code == 200

    res = client.get(
        "/api/v1/history/snapshots?limit=2&offset=0",
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["snapshots"]) == 2

    res = client.get(
        "/api/v1/history/snapshots?limit=2&offset=2",
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["snapshots"]) == 1
