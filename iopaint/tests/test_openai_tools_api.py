import io
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from iopaint.api import Api
from iopaint.openai_compat.models import ImageSize
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    RealESRGANModel,
    RemoveBGModel,
)


class DummyOpenAIClient:
    def __init__(self):
        self.last_edit_request = None
        self.last_variation_request = None
        self.last_models_request = False

    def list_models(self):
        self.last_models_request = True
        return []

    def edit_image(self, request):
        self.last_edit_request = request
        return b"result"

    def create_variation(self, request):
        self.last_variation_request = request
        return b"variation"


class DummyJobRunner:
    def __init__(self):
        self.submitted = []
        self.cancelled = []

    async def start(self):
        return None

    async def stop(self):
        return None

    async def submit(self, job):
        self.submitted.append(job)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


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


def _make_png_bytes(size=(256, 256), color=(255, 0, 0)):
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_test_client(monkeypatch, tmp_path, with_openai=False):
    monkeypatch.setenv("AIE_DATA_DIR", str(tmp_path / "data"))
    if with_openai:
        monkeypatch.setenv("AIE_OPENAI_API_KEY", "test-key")
    else:
        monkeypatch.delenv("AIE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(Api, "_build_model_manager", lambda self: SimpleNamespace())
    monkeypatch.setattr(Api, "_build_plugins", lambda self: {})

    app = FastAPI()
    api = Api(app, _make_api_config())
    if with_openai:
        api.openai_client = DummyOpenAIClient()
        api.openai_budget_client = None
    client = TestClient(app)
    return client, api


def test_openai_upscale_prompt_uses_edit(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    image_bytes = _make_png_bytes()

    res = client.post(
        "/api/v1/openai/upscale",
        data={"mode": "prompt", "scale": "2"},
        files={"image": ("image.png", image_bytes, "image/png")},
    )

    assert res.status_code == 200
    assert res.content == b"result"
    assert api.openai_client.last_edit_request is not None
    assert api.openai_client.last_edit_request.prompt.startswith("Enhance this image")
    assert api.openai_client.last_edit_request.size == ImageSize.SIZE_512


def test_openai_outpaint_uses_edit(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    image_bytes = _make_png_bytes()
    mask_bytes = _make_png_bytes(color=(0, 0, 0))

    res = client.post(
        "/api/v1/openai/outpaint",
        data={"prompt": "Extend the background"},
        files={
            "image": ("image.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
    )

    assert res.status_code == 200
    assert res.content == b"result"
    assert api.openai_client.last_edit_request is not None
    assert api.openai_client.last_edit_request.prompt == "Extend the background"


def test_openai_upscale_invalid_mode(monkeypatch, tmp_path):
    client, _ = _make_test_client(monkeypatch, tmp_path, with_openai=False)
    image_bytes = _make_png_bytes()

    res = client.post(
        "/api/v1/openai/upscale",
        data={"mode": "bad"},
        files={"image": ("image.png", image_bytes, "image/png")},
    )

    assert res.status_code == 422


def test_openai_background_remove_service_unconfigured(monkeypatch, tmp_path):
    client, _ = _make_test_client(monkeypatch, tmp_path, with_openai=False)
    image_bytes = _make_png_bytes()

    res = client.post(
        "/api/v1/openai/background-remove",
        data={"mode": "service"},
        files={"image": ("image.png", image_bytes, "image/png")},
    )

    assert res.status_code == 503


def test_openai_background_remove_prompt_uses_edit(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    image_bytes = _make_png_bytes()

    res = client.post(
        "/api/v1/openai/background-remove",
        data={"mode": "prompt"},
        files={"image": ("image.png", image_bytes, "image/png")},
    )

    assert res.status_code == 200
    assert res.content == b"result"
    assert api.openai_client.last_edit_request is not None
    assert api.openai_client.last_edit_request.prompt.startswith("Remove the background")


def test_openai_variations_with_dummy_client(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    image_bytes = _make_png_bytes()

    res = client.post(
        "/api/v1/openai/variations",
        files={"image": ("image.png", image_bytes, "image/png")},
    )

    assert res.status_code == 200
    assert res.content == b"variation"
    assert api.openai_client.last_variation_request is not None


def test_openai_invalid_base_url(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)

    res = client.get(
        "/api/v1/openai/models",
        headers={"X-OpenAI-Base-URL": "https://invalid.example.com/v1"},
    )

    assert res.status_code == 422
    assert api.openai_client.last_models_request is False


def test_openai_jobs_submit_and_get(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    api.job_runner = DummyJobRunner()

    res = client.post(
        "/api/v1/openai/jobs",
        json={
            "tool": "generate",
            "prompt": "A serene landscape",
            "model": "gpt-image-1",
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
            "intent": "Landscape",
            "refined_prompt": "A serene landscape with mountains",
            "negative_prompt": "blurry",
            "preset": "draft",
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "queued"
    assert payload["operation"] == "generate"
    assert payload["params"]["tool"] == "generate"
    assert payload["params"]["size"] == "1024x1024"
    assert "prompt" not in payload["params"]

    get_res = client.get(f"/api/v1/openai/jobs/{payload['id']}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == payload["id"]


def test_openai_jobs_cancel(monkeypatch, tmp_path):
    client, api = _make_test_client(monkeypatch, tmp_path, with_openai=True)
    api.job_runner = DummyJobRunner()

    create_res = client.post(
        "/api/v1/openai/jobs",
        json={
            "tool": "generate",
            "prompt": "A night sky",
            "model": "gpt-image-1",
            "n": 1,
        },
    )

    job_id = create_res.json()["id"]
    cancel_res = client.post(f"/api/v1/openai/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"
    assert job_id in api.job_runner.cancelled
