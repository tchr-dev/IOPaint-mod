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


class DummyModel:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def model_dump(self):
        return {"id": self.model_id, "object": "model"}


class DummyOpenAIClient:
    def __init__(self):
        self.calls = 0

    def list_models(self):
        self.calls += 1
        return [DummyModel("gpt-image-1"), DummyModel("gpt-image-2")]


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


def test_models_cache_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AIE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIE_OPENAI_MODELS_CACHE_TTL_S", "3600")
    monkeypatch.setattr(Api, "_build_model_manager", lambda self: SimpleNamespace())
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})

    app = FastAPI()
    api = Api(app, _make_api_config())
    api.openai_client = DummyOpenAIClient()
    client = TestClient(app)

    res = client.get("/api/v1/openai/models/cached")
    assert res.status_code == 404

    res = client.post("/api/v1/openai/models/refresh")
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) == 2

    res = client.get("/api/v1/openai/models/cached")
    assert res.status_code == 200
    cached = res.json()
    assert cached["models"][0]["id"] == "gpt-image-1"

    res = client.get("/api/v1/openai/models")
    assert res.status_code == 200
    assert api.openai_client.calls == 1

    res = client.get("/api/v1/openai/models")
    assert res.status_code == 200
    assert api.openai_client.calls == 1
