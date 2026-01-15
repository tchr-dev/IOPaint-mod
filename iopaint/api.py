import asyncio
import base64
import binascii
import io
import json
import time
import threading
import traceback
import uuid

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import socketio
import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
except:
    pass

import uvicorn
from PIL import Image
from fastapi import APIRouter, FastAPI, Request, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from socketio import AsyncServer

from iopaint.file_manager import FileManager
from iopaint.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.plugins import build_plugins, RealESRGANUpscaler, InteractiveSeg
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import (
    GenInfoResponse,
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    RunPluginRequest,
    SDSampler,
    PluginInfo,
    AdjustMaskRequest,
    RemoveBGModel,
    SwitchPluginModelRequest,
    ModelInfo,
    InteractiveSegModel,
    RealESRGANModel,
)
from iopaint.services.config import ExternalImageServiceConfig

# OpenAI-compatible API imports
from iopaint.openai_compat.config import OpenAIConfig
from iopaint.openai_compat.client import OpenAICompatClient
from iopaint.openai_compat.models import (
    GenerateImageRequest as OpenAIGenerateRequest,
    GenerateImageResponse as OpenAIGenerateResponse,
    RefinePromptRequest as OpenAIRefineRequest,
    RefinePromptResponse,
    EditImageRequest as OpenAIEditRequest,
    EditImageResponse as OpenAIEditResponse,
    CreateVariationRequest as OpenAIVariationRequest,
    ImageData as OpenAIImageData,
    ImageSize as OpenAIImageSize,
    ResponseFormat as OpenAIResponseFormat,
)
from iopaint.openai_compat.errors import OpenAIError

# Budget safety imports
from iopaint.budget import (
    BudgetConfig,
    BudgetStorage,
    BudgetGuard,
    BudgetStatusResponse,
    BudgetError,
    BudgetExceededError,
    RateLimitedError,
    BudgetAwareOpenAIClient,
    CostEstimator,
    CostEstimateRequest,
    CostEstimateResponse,
)

# Storage imports (Epic 3)
from iopaint.storage import (
    GenerationJob,
    GenerationJobCreate,
    GenerationJobUpdate,
    HistoryListResponse,
    HistorySnapshot,
    HistorySnapshotCreate,
    HistorySnapshotListResponse,
    HistoryStorage,
    ImageStorage,
    ImageUploadResponse,
    InputStorage,
    JobOperation,
    JobStatus,
    ModelCacheStorage,
)


# Runner imports (Epic 5)
from iopaint.runner import JobRunner, JobSubmitRequest, JobTool, QueuedJob

CURRENT_DIR = Path(__file__).parent.absolute().resolve()
WEB_APP_DIR = CURRENT_DIR / "web_app"

# Fallback for running from source repo (where web_app is at repo root)
if not WEB_APP_DIR.is_dir():
    WEB_APP_DIR = CURRENT_DIR.parent / "web_app"   # ../web_app
    
def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console

            console = Console()
            rich_available = True
    except Exception:
        pass

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        return JSONResponse(
            status_code=vars(e).get("status_code", 500), content=jsonable_encoder(err)
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
        "expose_headers": ["X-Seed"],
    }
    app.add_middleware(CORSMiddleware, **cors_options)


global_sio: AsyncServer = None


def diffuser_callback(pipe, step: int, timestep: int, callback_kwargs: Dict = {}):
    # self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict
    # logger.info(f"diffusion callback: step={step}, timestep={timestep}")

    # We use asyncio loos for task processing. Perhaps in the future, we can add a processing queue similar to InvokeAI,
    # but for now let's just start a separate event loop. It shouldn't make a difference for single person use
    asyncio.run(global_sio.emit("diffusion_progress", {"step": step}))
    return {}


class Api:
    def __init__(self, app: FastAPI, config: ApiConfig):
        self.app = app
        self.config = config
        self.router = APIRouter()
        self.queue_lock = threading.Lock()
        api_middleware(self.app)

        self.file_manager = self._build_file_manager()
        self.plugins = self._build_plugins()
        self.model_manager = self._build_model_manager()

        # Initialize OpenAI-compatible client if configured
        self.openai_config = OpenAIConfig()
        self.openai_client: Optional[OpenAICompatClient] = None
        self.openai_budget_client: Optional[BudgetAwareOpenAIClient] = None
        if self.openai_config.is_enabled:
            self.openai_client = OpenAICompatClient(self.openai_config)
            logger.info(f"OpenAI-compatible client enabled: {self.openai_config.base_url}")

        # Initialize budget safety system
        self.budget_config = BudgetConfig()
        self.budget_storage = BudgetStorage(self.budget_config)
        self.budget_guard = BudgetGuard(self.budget_config, self.budget_storage)
        self.cost_estimator = CostEstimator()
        logger.info(f"Budget safety enabled: {self.budget_config}")

        if self.openai_client:
            self.openai_budget_client = BudgetAwareOpenAIClient(
                client=self.openai_client,
                config=self.budget_config,
                storage=self.budget_storage,
                cost_estimator=self.cost_estimator,
            )

        # Initialize history and image storage (Epic 3)
        self.history_storage = HistoryStorage(db_path=self.budget_config.db_path)
        self.image_storage = ImageStorage(
            data_dir=self.budget_config.data_dir,
            db_path=self.budget_config.db_path,
        )
        self.input_storage = InputStorage(self.budget_config.data_dir)
        self.model_cache_storage = ModelCacheStorage(
            db_path=self.budget_config.db_path,
        )
        self.external_services_config = ExternalImageServiceConfig()
        logger.info(f"Storage initialized: {self.budget_config.data_dir}")

        # Initialize job runner (Epic 5)
        self.job_runner: Optional[JobRunner] = None
        if self.openai_client:
            self.job_runner = JobRunner(
                history_storage=self.history_storage,
                image_storage=self.image_storage,
                input_storage=self.input_storage,
                openai_client=self.openai_client,
                budget_client=self.openai_budget_client,
            )
            self.app.add_event_handler("startup", self._start_job_runner)
            self.app.add_event_handler("shutdown", self._stop_job_runner)

        # fmt: off
        self.add_api_route("/api/v1/gen-info", self.api_geninfo, methods=["POST"], response_model=GenInfoResponse)
        self.add_api_route("/api/v1/server-config", self.api_server_config, methods=["GET"],
                           response_model=ServerConfigResponse)
        self.add_api_route("/api/v1/model", self.api_current_model, methods=["GET"], response_model=ModelInfo)
        self.add_api_route("/api/v1/model", self.api_switch_model, methods=["POST"], response_model=ModelInfo)
        self.add_api_route("/api/v1/inputimage", self.api_input_image, methods=["GET"])
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        self.add_api_route("/api/v1/switch_plugin_model", self.api_switch_plugin_model, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_mask", self.api_run_plugin_gen_mask, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_image", self.api_run_plugin_gen_image, methods=["POST"])
        self.add_api_route("/api/v1/samplers", self.api_samplers, methods=["GET"])
        self.add_api_route("/api/v1/adjust_mask", self.api_adjust_mask, methods=["POST"])
        self.add_api_route("/api/v1/save_image", self.api_save_image, methods=["POST"])

        # OpenAI-compatible API routes
        self.add_api_route("/api/v1/openai/models", self.api_openai_list_models, methods=["GET"])
        self.add_api_route("/api/v1/openai/models/cached", self.api_openai_cached_models, methods=["GET"])
        self.add_api_route("/api/v1/openai/models/refresh", self.api_openai_refresh_models, methods=["POST"])
        self.add_api_route("/api/v1/openai/refine", self.api_openai_refine_prompt, methods=["POST"],
                           response_model=RefinePromptResponse)
        self.add_api_route("/api/v1/openai/generate", self.api_openai_generate, methods=["POST"])
        self.add_api_route("/api/v1/openai/edit", self.api_openai_edit, methods=["POST"])
        self.add_api_route("/api/v1/openai/outpaint", self.api_openai_outpaint, methods=["POST"])
        self.add_api_route("/api/v1/openai/variations", self.api_openai_variations, methods=["POST"])
        self.add_api_route("/api/v1/openai/upscale", self.api_openai_upscale, methods=["POST"])
        self.add_api_route("/api/v1/openai/background-remove", self.api_openai_background_remove, methods=["POST"])
        self.add_api_route("/api/v1/openai/jobs", self.api_openai_submit_job, methods=["POST"],
                           response_model=GenerationJob)
        self.add_api_route("/api/v1/openai/jobs/{job_id}", self.api_openai_get_job, methods=["GET"],
                           response_model=GenerationJob)
        self.add_api_route("/api/v1/openai/jobs/{job_id}/cancel", self.api_openai_cancel_job, methods=["POST"],
                           response_model=GenerationJob)
        self.add_api_route(
            "/v1/images/generations",
            self.api_openai_images_generate,
            methods=["POST"],
            response_model=OpenAIGenerateResponse,
        )
        self.add_api_route(
            "/v1/images/edits",
            self.api_openai_images_edit,
            methods=["POST"],
            response_model=OpenAIEditResponse,
        )

        # Budget safety API routes
        self.add_api_route("/api/v1/budget/status", self.api_budget_status, methods=["GET"],
                           response_model=BudgetStatusResponse)
        self.add_api_route("/api/v1/budget/estimate", self.api_budget_estimate, methods=["POST"],
                           response_model=CostEstimateResponse)

        # History API routes (Epic 3)
        self.add_api_route("/api/v1/history", self.api_list_history, methods=["GET"],
                           response_model=HistoryListResponse)
        self.add_api_route("/api/v1/history", self.api_create_history, methods=["POST"],
                           response_model=GenerationJob)
        self.add_api_route("/api/v1/history/snapshots", self.api_list_history_snapshots, methods=["GET"],
                           response_model=HistorySnapshotListResponse)
        self.add_api_route("/api/v1/history/snapshots", self.api_create_history_snapshot, methods=["POST"],
                           response_model=HistorySnapshot)
        self.add_api_route("/api/v1/history/snapshots/clear", self.api_clear_history_snapshots, methods=["DELETE"])
        self.add_api_route("/api/v1/history/snapshots/{snapshot_id}", self.api_get_history_snapshot, methods=["GET"],
                           response_model=HistorySnapshot)
        self.add_api_route("/api/v1/history/snapshots/{snapshot_id}", self.api_delete_history_snapshot, methods=["DELETE"])
        self.add_api_route("/api/v1/history/{job_id}", self.api_get_history, methods=["GET"],
                           response_model=GenerationJob)
        self.add_api_route("/api/v1/history/{job_id}", self.api_update_history, methods=["PATCH"],
                           response_model=GenerationJob)
        self.add_api_route("/api/v1/history/{job_id}", self.api_delete_history, methods=["DELETE"])
        self.add_api_route("/api/v1/history/clear", self.api_clear_history, methods=["DELETE"])

        # Image storage API routes (Epic 3)
        self.add_api_route("/api/v1/storage/images/{image_id}", self.api_get_stored_image, methods=["GET"])
        self.add_api_route("/api/v1/storage/images/{image_id}/thumbnail", self.api_get_thumbnail, methods=["GET"])

        self.app.mount("/", StaticFiles(directory=WEB_APP_DIR, html=True), name="assets")
        # fmt: on

        global global_sio
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.combined_asgi_app = socketio.ASGIApp(self.sio, self.app)
        self.app.mount("/ws", self.combined_asgi_app)
        global_sio = self.sio

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def api_save_image(self, file: UploadFile):
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name  # Get just the filename component

        # Construct the full path within output_dir
        output_path = self.config.output_dir / safe_filename

        # Ensure output directory exists
        if not self.config.output_dir or not self.config.output_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Output directory not configured or doesn't exist",
            )

        # Read and write the file
        origin_image_bytes = file.file.read()
        with open(output_path, "wb") as fw:
            fw.write(origin_image_bytes)

    def api_current_model(self) -> ModelInfo:
        return self.model_manager.current_model

    def api_switch_model(self, req: SwitchModelRequest) -> ModelInfo:
        if req.name == self.model_manager.name:
            return self.model_manager.current_model
        self.model_manager.switch(req.name)
        return self.model_manager.current_model

    def api_switch_plugin_model(self, req: SwitchPluginModelRequest):
        if req.plugin_name in self.plugins:
            self.plugins[req.plugin_name].switch_model(req.model_name)
            if req.plugin_name == RemoveBG.name:
                self.config.remove_bg_model = req.model_name
            if req.plugin_name == RealESRGANUpscaler.name:
                self.config.realesrgan_model = req.model_name
            if req.plugin_name == InteractiveSeg.name:
                self.config.interactive_seg_model = req.model_name
            torch_gc()

    def api_server_config(self) -> ServerConfigResponse:
        plugins = []
        for it in self.plugins.values():
            plugins.append(
                PluginInfo(
                    name=it.name,
                    support_gen_image=it.support_gen_image,
                    support_gen_mask=it.support_gen_mask,
                )
            )

        return ServerConfigResponse(
            plugins=plugins,
            modelInfos=self.model_manager.scan_models(),
            removeBGModel=self.config.remove_bg_model,
            removeBGModels=RemoveBGModel.values(),
            realesrganModel=self.config.realesrgan_model,
            realesrganModels=RealESRGANModel.values(),
            interactiveSegModel=self.config.interactive_seg_model,
            interactiveSegModels=InteractiveSegModel.values(),
            enableFileManager=self.file_manager is not None,
            enableAutoSaving=self.config.output_dir is not None,
            enableControlnet=self.model_manager.enable_controlnet,
            controlnetMethod=self.model_manager.controlnet_method,
            disableModelSwitch=False,
            isDesktop=False,
            samplers=self.api_samplers(),
        )

    def api_input_image(self) -> FileResponse:
        if self.config.input is None:
            raise HTTPException(status_code=200, detail="No input image configured")

        if self.config.input.is_file():
            return FileResponse(self.config.input)
        raise HTTPException(status_code=404, detail="Input image not found")

    def api_geninfo(self, file: UploadFile) -> GenInfoResponse:
        _, _, info = load_img(file.file.read(), return_info=True)
        parts = info.get("parameters", "").split("Negative prompt: ")
        prompt = parts[0].strip()
        negative_prompt = ""
        if len(parts) > 1:
            negative_prompt = parts[1].split("\n")[0].strip()
        return GenInfoResponse(prompt=prompt, negative_prompt=negative_prompt)

    def api_inpaint(self, req: InpaintRequest):
        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        logger.info(f"image ext: {ext}")

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        start = time.time()
        rgb_np_img = self.model_manager(image, mask, req)
        logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
        torch_gc()

        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=self.config.quality,
            infos=infos,
        )

        asyncio.run(self.sio.emit("diffusion_finish"))

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
            headers={"X-Seed": str(req.sd_seed)},
        )

    def api_run_plugin_gen_image(self, req: RunPluginRequest):
        ext = "png"
        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_image:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, alpha_channel, infos, _ = decode_base64_to_image(req.image)
        bgr_or_rgba_np_img = self.plugins[req.name].gen_image(rgb_np_img, req)
        torch_gc()

        if bgr_or_rgba_np_img.shape[2] == 4:
            rgba_np_img = bgr_or_rgba_np_img
        else:
            rgba_np_img = cv2.cvtColor(bgr_or_rgba_np_img, cv2.COLOR_BGR2RGB)
            rgba_np_img = concat_alpha_channel(rgba_np_img, alpha_channel)

        return Response(
            content=pil_to_bytes(
                Image.fromarray(rgba_np_img),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            ),
            media_type=f"image/{ext}",
        )

    def api_run_plugin_gen_mask(self, req: RunPluginRequest):
        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_mask:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, _, _, _ = decode_base64_to_image(req.image)
        bgr_or_gray_mask = self.plugins[req.name].gen_mask(rgb_np_img, req)
        torch_gc()
        res_mask = gen_frontend_mask(bgr_or_gray_mask)
        return Response(
            content=numpy_to_bytes(res_mask, "png"),
            media_type="image/png",
        )

    def api_samplers(self) -> List[str]:
        return [member.value for member in SDSampler.__members__.values()]

    def api_adjust_mask(self, req: AdjustMaskRequest):
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        mask = adjust_mask(mask, req.kernel_size, req.operate)
        return Response(content=numpy_to_bytes(mask, "png"), media_type="image/png")

    # =========================================================================
    # OpenAI-compatible API endpoints
    # =========================================================================

    def api_openai_list_models(self, request: Request):
        """List available models from OpenAI-compatible API."""
        try:
            client, _, config = self._resolve_openai_clients(request)
            provider = self._openai_cache_provider(config)
            cached = self.model_cache_storage.get_cached_models(
                provider,
                max_age_seconds=self.openai_config.models_cache_ttl_s,
            )
            if cached:
                return cached

            models = client.list_models()
            payload = {"models": [m.model_dump() for m in models]}
            self.model_cache_storage.set_cached_models(provider, payload)
            return payload
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_cached_models(self, request: Request):
        """Return cached OpenAI-compatible models if available."""
        _, _, config = self._resolve_openai_clients(request)
        provider = self._openai_cache_provider(config)
        cached = self.model_cache_storage.get_cached_models(
            provider,
            max_age_seconds=self.openai_config.models_cache_ttl_s,
        )
        if not cached:
            raise HTTPException(status_code=404, detail="No cached models available")
        return cached

    def api_openai_refresh_models(self, request: Request):
        """Refresh OpenAI-compatible models cache."""
        try:
            client, _, config = self._resolve_openai_clients(request)
            models = client.list_models()
            payload = {"models": [m.model_dump() for m in models]}
            self.model_cache_storage.set_cached_models(
                self._openai_cache_provider(config),
                payload,
            )
            return payload
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_refine_prompt(
        self, request: Request, req: OpenAIRefineRequest
    ) -> RefinePromptResponse:
        """Refine/expand a prompt using cheap LLM call before expensive image generation."""
        try:
            client, budget_client, _ = self._resolve_openai_clients(request)
            if budget_client:
                session_id = self._get_session_id(request)
                return budget_client.refine_prompt(req, session_id=session_id)
            return client.refine_prompt(req)
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_generate(self, request: Request, req: OpenAIGenerateRequest):
        """Generate image from text prompt using OpenAI-compatible API."""
        try:
            client, budget_client, _ = self._resolve_openai_clients(request)
            if budget_client:
                session_id = self._get_session_id(request)
                image_bytes = budget_client.generate_image(
                    req,
                    session_id=session_id,
                )
            else:
                image_bytes = client.generate_image(req)
            return Response(content=image_bytes, media_type="image/png")
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_images_generate(
        self, request: Request, req: OpenAIGenerateRequest
    ) -> OpenAIGenerateResponse:
        """OpenAI protocol: /v1/images/generations."""
        try:
            client, budget_client, config = self._resolve_openai_clients(request)
            session_id = self._get_session_id(request)
            if budget_client:
                image_bytes = budget_client.generate_image(req, session_id=session_id)
            else:
                image_bytes = client.generate_image(req)

            params = {
                "size": req.size.value,
                "quality": req.quality.value,
                "n": req.n,
            }
            _, image_id = self._record_openai_image_job(
                session_id=session_id,
                operation=JobOperation.GENERATE,
                model=req.model or config.model,
                prompt=req.prompt,
                params=params,
                is_edit=False,
                image_bytes=image_bytes,
            )
            return self._build_openai_image_response(
                request,
                image_bytes,
                req.response_format,
                image_id,
                OpenAIGenerateResponse,
            )
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_edit(
        self,
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(...),
        n: int = Form(1),
        size: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
    ):
        """Edit (inpaint/outpaint) an image using OpenAI-compatible API."""
        image_bytes = self._read_upload_bytes(image, "image")
        mask_bytes = self._read_upload_bytes(mask, "mask")
        edit_request = OpenAIEditRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=n,
            size=self._parse_openai_size(size),
            model=model,
            response_format=self._parse_openai_response_format(response_format),
        )
        try:
            client, budget_client, _ = self._resolve_openai_clients(request)
            if budget_client:
                session_id = self._get_session_id(request)
                result = budget_client.edit_image(
                    edit_request,
                    session_id=session_id,
                    image_bytes=image_bytes,
                    mask_bytes=mask_bytes,
                )
            else:
                result = client.edit_image(edit_request)
            return Response(content=result, media_type="image/png")
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_images_edit(
        self,
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(...),
        n: int = Form(1),
        size: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
    ) -> OpenAIEditResponse:
        """OpenAI protocol: /v1/images/edits."""
        image_bytes = self._read_upload_bytes(image, "image")
        mask_bytes = self._read_upload_bytes(mask, "mask")
        parsed_size = self._parse_openai_size(size)
        parsed_format = self._parse_openai_response_format(response_format)
        edit_request = OpenAIEditRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=n,
            size=parsed_size,
            model=model,
            response_format=parsed_format,
        )
        try:
            client, budget_client, config = self._resolve_openai_clients(request)
            session_id = self._get_session_id(request)
            if budget_client:
                result = budget_client.edit_image(
                    edit_request,
                    session_id=session_id,
                    image_bytes=image_bytes,
                    mask_bytes=mask_bytes,
                )
            else:
                result = client.edit_image(edit_request)

            params = {
                "size": parsed_size.value if parsed_size else None,
                "n": n,
            }
            _, image_id = self._record_openai_image_job(
                session_id=session_id,
                operation=JobOperation.EDIT,
                model=model or config.model,
                prompt=prompt,
                params=params,
                is_edit=True,
                image_bytes=result,
            )
            return self._build_openai_image_response(
                request,
                result,
                parsed_format,
                image_id,
                OpenAIEditResponse,
            )
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_outpaint(
        self,
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(...),
        n: int = Form(1),
        size: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
    ):
        """Outpaint by delegating to the OpenAI image edit API."""
        return self.api_openai_edit(
            request=request,
            image=image,
            mask=mask,
            prompt=prompt,
            n=n,
            size=size,
            model=model,
            response_format=response_format,
        )

    def api_openai_variations(
        self,
        request: Request,
        image: UploadFile = File(...),
        n: int = Form(1),
        size: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
    ):
        """Create image variations using OpenAI-compatible API."""
        image_bytes = self._read_upload_bytes(image, "image")
        variation_request = OpenAIVariationRequest(
            image=image_bytes,
            n=n,
            size=self._parse_openai_size(size),
            model=model,
            response_format=self._parse_openai_response_format(response_format),
        )
        try:
            client, budget_client, _ = self._resolve_openai_clients(request)
            if budget_client:
                session_id = self._get_session_id(request)
                result = budget_client.create_variation(
                    variation_request,
                    session_id=session_id,
                    image_bytes=image_bytes,
                )
            else:
                result = client.create_variation(variation_request)
            return Response(content=result, media_type="image/png")
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def api_openai_upscale(
        self,
        request: Request,
        image: UploadFile = File(...),
        scale: float = Form(2.0),
        size: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
        mode: str = Form("local"),
    ):
        """Upscale an image (local plugin, prompt-based, or external service stub)."""
        if mode not in {"local", "prompt", "service"}:
            raise HTTPException(status_code=422, detail=f"Invalid mode: {mode}")
        if mode == "service":
            if not self.external_services_config.upscale_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Upscale service not configured. Set AIE_UPSCALE_SERVICE_URL and AIE_UPSCALE_SERVICE_API_KEY.",
                )
            raise HTTPException(status_code=501, detail="Upscale service not implemented yet.")

        image_bytes = self._read_upload_bytes(image, "image")

        if mode == "prompt":
            edit_prompt = prompt or (
                "Enhance this image to higher resolution with more detail while preserving original composition."
            )
            size_enum = self._parse_openai_size(size) or self._infer_upscale_size(
                image_bytes, scale
            )
            return self._run_openai_edit_from_bytes(
                request=request,
                image_bytes=image_bytes,
                mask_bytes=self._build_full_edit_mask(image_bytes),
                prompt=edit_prompt,
                n=1,
                size=size_enum,
                model=model,
                response_format=self._parse_openai_response_format(response_format),
            )

        if RealESRGANUpscaler.name not in self.plugins:
            raise HTTPException(status_code=422, detail="RealESRGAN plugin not enabled")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        req = RunPluginRequest(
            name=RealESRGANUpscaler.name,
            image=image_b64,
            scale=scale,
        )
        return self.api_run_plugin_gen_image(req)

    def api_openai_background_remove(
        self,
        request: Request,
        image: UploadFile = File(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        mode: str = Form("local"),
    ):
        """Remove background (local plugin, prompt-based, or external service stub)."""
        if mode not in {"local", "prompt", "service"}:
            raise HTTPException(status_code=422, detail=f"Invalid mode: {mode}")
        if mode == "service":
            if not self.external_services_config.background_remove_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Background removal service not configured. Set AIE_BG_REMOVE_SERVICE_URL and AIE_BG_REMOVE_SERVICE_API_KEY.",
                )
            raise HTTPException(
                status_code=501,
                detail="Background removal service not implemented yet.",
            )

        image_bytes = self._read_upload_bytes(image, "image")
        if mode == "prompt":
            edit_prompt = prompt or (
                "Remove the background around the main object and output a transparent PNG."
            )
            return self._run_openai_edit_from_bytes(
                request=request,
                image_bytes=image_bytes,
                mask_bytes=self._build_full_edit_mask(image_bytes),
                prompt=edit_prompt,
                n=1,
                size=self._parse_openai_size(response_format),
                model=model,
                response_format=self._parse_openai_response_format(response_format),
            )

        if RemoveBG.name not in self.plugins:
            raise HTTPException(status_code=422, detail="RemoveBG plugin not enabled")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        req = RunPluginRequest(name=RemoveBG.name, image=image_b64)
        return self.api_run_plugin_gen_image(req)

    def _record_openai_image_job(
        self,
        session_id: str,
        operation: JobOperation,
        model: str,
        prompt: str,
        params: Dict[str, object],
        is_edit: bool,
        image_bytes: bytes,
    ) -> Tuple[GenerationJob, str]:
        job_params = {key: value for key, value in params.items() if value is not None}
        created_job = GenerationJobCreate(
            operation=operation,
            model=model,
            refined_prompt=prompt,
            params=job_params,
            is_edit=is_edit,
        )
        stored_job = self.history_storage.save_job(session_id=session_id, job=created_job)
        image_record = self.image_storage.save_image(image_bytes, job_id=stored_job.id)
        self.history_storage.update_job(
            stored_job.id,
            GenerationJobUpdate(
                status=JobStatus.SUCCEEDED,
                result_image_id=image_record.id,
            ),
        )
        updated_job = self.history_storage.get_job(stored_job.id)
        if not updated_job:
            raise HTTPException(status_code=404, detail="Job not found after update")
        return updated_job, image_record.id

    def _build_openai_image_response(
        self,
        request: Request,
        image_bytes: bytes,
        response_format: OpenAIResponseFormat,
        image_id: str,
        response_model,
    ):
        if response_format == OpenAIResponseFormat.URL:
            image_url = f"{request.base_url}api/v1/storage/images/{image_id}"
            image_data = OpenAIImageData(url=image_url)
        else:
            image_data = OpenAIImageData(
                b64_json=base64.b64encode(image_bytes).decode("utf-8")
            )
        return response_model(created=int(time.time()), data=[image_data])

    async def api_openai_submit_job(

        self, request: Request, job_request: JobSubmitRequest
    ) -> GenerationJob:
        """Submit an OpenAI-compatible job for async processing."""
        if not self.job_runner:
            raise HTTPException(status_code=503, detail="Job runner not configured")

        session_id = self._get_session_id(request)
        job_id = str(uuid.uuid4())
        operation = JobOperation(job_request.tool.value)
        input_params = self._persist_job_inputs(job_id, job_request)
        params = self._build_job_params(job_request)
        params.update(input_params)
        created_job = GenerationJobCreate(
            operation=operation,
            model=job_request.model or self.openai_config.model,
            intent=job_request.intent,
            refined_prompt=job_request.refined_prompt or job_request.prompt,
            negative_prompt=job_request.negative_prompt,
            preset=job_request.preset,
            params=params,
            estimated_cost_usd=None,
            is_edit=operation != JobOperation.GENERATE,
        )
        stored_job = self.history_storage.save_job(
            session_id=session_id,
            job=created_job,
            job_id=job_id,
        )
        queued_job = QueuedJob(
            job_id=stored_job.id,
            session_id=session_id,
            request=job_request,
            openai_base_url=self._get_openai_base_url_override(request),
        )
        await self.job_runner.submit(queued_job)
        return stored_job

    def api_openai_get_job(self, job_id: str) -> GenerationJob:
        """Get an OpenAI-compatible job record."""
        job = self.history_storage.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def api_openai_cancel_job(self, job_id: str) -> GenerationJob:
        """Cancel an OpenAI-compatible job if possible."""
        if not self.job_runner:
            raise HTTPException(status_code=503, detail="Job runner not configured")

        job = self.history_storage.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.BLOCKED_BUDGET,
            JobStatus.CANCELLED,
        }:
            return job

        self.job_runner.cancel(job_id)
        self.history_storage.update_job(
            job_id,
            GenerationJobUpdate(
                status=JobStatus.CANCELLED,
                error_message="Cancelled by user",
            ),
        )
        updated_job = self.history_storage.get_job(job_id)
        if not updated_job:
            raise HTTPException(status_code=404, detail="Job not found after cancel")
        return updated_job

    async def _start_job_runner(self) -> None:
        if self.job_runner:
            await self.job_runner.start()

    async def _stop_job_runner(self) -> None:
        if self.job_runner:
            await self.job_runner.stop()

    def _build_job_params(self, job_request: JobSubmitRequest) -> Dict[str, object]:
        params = job_request.model_dump(
            exclude={
                "image_b64",
                "mask_b64",
                "prompt",
                "intent",
                "refined_prompt",
                "negative_prompt",
                "preset",
            }
        )
        params["tool"] = job_request.tool.value
        return params

    def _persist_job_inputs(
        self, job_id: str, job_request: JobSubmitRequest
    ) -> Dict[str, object]:
        input_images = []
        mask_images = []

        if job_request.image_b64:
            try:
                image_bytes = base64.b64decode(job_request.image_b64)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail="Invalid image_b64 payload"
                ) from exc
            input_images.append(
                self.input_storage.save_input_bytes(job_id, image_bytes, "image")
            )
            job_request.image_b64 = None

        if job_request.mask_b64:
            try:
                mask_bytes = base64.b64decode(job_request.mask_b64)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(
                    status_code=422, detail="Invalid mask_b64 payload"
                ) from exc
            mask_images.append(
                self.input_storage.save_input_bytes(job_id, mask_bytes, "mask")
            )
            job_request.mask_b64 = None

        params: Dict[str, object] = {}
        if input_images:
            params["input_images"] = input_images
        if mask_images:
            params["mask_images"] = mask_images
        return params

    def _openai_cache_provider(self, config: OpenAIConfig) -> str:
        """Build a provider key for model cache."""
        return f"{config.backend}:{config.base_url.rstrip('/')}"

    def _resolve_openai_clients(
        self, request: Request
    ) -> Tuple[OpenAICompatClient, Optional[BudgetAwareOpenAIClient], OpenAIConfig]:
        if not self.openai_client:
            raise HTTPException(
                status_code=503,
                detail="OpenAI client not configured. Set AIE_OPENAI_API_KEY environment variable.",
            )
        override_base_url = self._get_openai_base_url_override(request)
        if not override_base_url:
            return self.openai_client, self.openai_budget_client, self.openai_config

        normalized_base_url = override_base_url.rstrip("/")
        if normalized_base_url == self.openai_config.base_url.rstrip("/"):
            return self.openai_client, self.openai_budget_client, self.openai_config

        override_config = OpenAIConfig(
            backend=self.openai_config.backend,
            api_key=self.openai_config.api_key,
            base_url=normalized_base_url,
            model=self.openai_config.model,
            timeout_s=self.openai_config.timeout_s,
            refine_model=self.openai_config.refine_model,
            models_cache_ttl_s=self.openai_config.models_cache_ttl_s,
        )
        client = OpenAICompatClient(override_config)
        budget_client = BudgetAwareOpenAIClient(
            client=client,
            config=self.budget_config,
            storage=self.budget_storage,
            cost_estimator=self.cost_estimator,
        )
        return client, budget_client, override_config

    def _get_openai_base_url_override(self, request: Request) -> Optional[str]:
        base_url = request.headers.get("X-OpenAI-Base-URL")
        if not base_url:
            return None
        normalized = base_url.rstrip("/")
        allowed = {
            "https://api.proxyapi.ru/openai/v1",
            "https://openrouter.ai/api/v1",
            "https://api.openai.com/v1",
        }
        if normalized not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported OpenAI base URL: {base_url}",
            )
        return normalized

    def _parse_openai_size(
        self, size: Optional[str]
    ) -> Optional[OpenAIImageSize]:
        if not size:
            return None
        try:
            return OpenAIImageSize(size)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid size: {size}")

    def _parse_openai_response_format(
        self, response_format: Optional[str]
    ) -> OpenAIResponseFormat:
        if not response_format:
            return OpenAIResponseFormat.B64_JSON
        try:
            return OpenAIResponseFormat(response_format)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid response_format: {response_format}",
            )

    def _read_upload_bytes(self, upload: UploadFile, label: str) -> bytes:
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty {label} upload")
        return data

    def _build_full_edit_mask(self, image_bytes: bytes) -> bytes:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        return pil_to_bytes(mask, ext="png", quality=self.config.quality, infos={})

    def _infer_upscale_size(
        self, image_bytes: bytes, scale: float
    ) -> Optional[OpenAIImageSize]:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            target_width = int(image.width * scale)
            target_height = int(image.height * scale)
        except Exception:
            return None
        return OpenAIImageSize.from_dimensions(target_width, target_height)

    def _run_openai_edit_from_bytes(
        self,
        request: Request,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        n: int,
        size: Optional[OpenAIImageSize],
        model: Optional[str],
        response_format: OpenAIResponseFormat,
    ) -> Response:
        edit_request = OpenAIEditRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=n,
            size=size,
            model=model,
            response_format=response_format,
        )
        try:
            client, budget_client, _ = self._resolve_openai_clients(request)
            if budget_client:
                session_id = self._get_session_id(request)
                result = budget_client.edit_image(
                    edit_request,
                    session_id=session_id,
                    image_bytes=image_bytes,
                    mask_bytes=mask_bytes,
                )
            else:
                result = client.edit_image(edit_request)
            return Response(content=result, media_type="image/png")
        except BudgetError as e:
            self._raise_budget_error(e)
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _raise_budget_error(self, error: BudgetError) -> None:
        if isinstance(error, BudgetExceededError):
            raise HTTPException(status_code=402, detail=str(error))
        if isinstance(error, RateLimitedError):
            raise HTTPException(status_code=429, detail=str(error))
        raise HTTPException(status_code=409, detail=str(error))

    # =========================================================================
    # Budget Safety API endpoints
    # =========================================================================

    def _get_session_id(self, request: Request) -> str:
        """Extract session ID from request headers or generate fallback.

        Session ID precedence:
        1. X-Session-Id header
        2. session_id query parameter
        3. Fallback: hash of client IP + User-Agent
        """
        import hashlib

        # Try header first
        session_id = request.headers.get("X-Session-Id")
        if session_id:
            return session_id

        # Try query parameter
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id

        # Fallback: pseudo-session from client fingerprint
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        fingerprint = f"{client_host}:{user_agent}"
        return hashlib.md5(fingerprint.encode()).hexdigest()[:16]

    def api_budget_status(self, request: Request) -> BudgetStatusResponse:
        """Get current budget status including daily/monthly/session usage."""
        session_id = self._get_session_id(request)
        return self.budget_guard.get_status(session_id)

    def api_budget_estimate(self, req: CostEstimateRequest) -> CostEstimateResponse:
        """Estimate cost for an operation before executing it."""
        model = req.model or (self.openai_config.model if self.openai_config else "gpt-image-1")
        cost, tier, warning = self.cost_estimator.estimate_with_tier(
            model=model,
            size=req.size or "1024x1024",
            quality=req.quality or "standard",
            n=req.n or 1,
            operation=req.operation,
        )
        return CostEstimateResponse(
            estimated_cost_usd=cost,
            cost_tier=tier,
            warning=warning,
        )

    # =========================================================================
    # History API endpoints (Epic 3)
    # =========================================================================

    def api_list_history(
        self,
        request: Request,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> HistoryListResponse:
        """List generation history for the current session."""
        session_id = self._get_session_id(request)

        # Validate limit
        if limit > 100:
            limit = 100
        if limit < 1:
            limit = 1

        jobs, total = self.history_storage.list_jobs(
            session_id=session_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return HistoryListResponse(
            jobs=jobs,
            total=total,
            limit=limit,
            offset=offset,
        )

    def api_create_history(
        self,
        request: Request,
        job: GenerationJobCreate,
    ) -> GenerationJob:
        """Create a new history entry."""
        session_id = self._get_session_id(request)
        return self.history_storage.save_job(session_id=session_id, job=job)

    def api_get_history(self, job_id: str) -> GenerationJob:
        """Get a specific history entry."""
        job = self.history_storage.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def api_update_history(
        self,
        job_id: str,
        updates: GenerationJobUpdate,
    ) -> GenerationJob:
        """Update a history entry."""
        # Check job exists
        job = self.history_storage.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        self.history_storage.update_job(job_id, updates)

        # Return updated job
        updated_job = self.history_storage.get_job(job_id)
        if not updated_job:
            raise HTTPException(status_code=404, detail="Job not found after update")
        return updated_job

    def api_delete_history(self, job_id: str) -> dict:
        """Delete a history entry."""
        # Also delete associated images
        self.image_storage.delete_job_images(job_id)

        deleted = self.history_storage.delete_job(job_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"deleted": True, "job_id": job_id}

    def api_clear_history(self, request: Request) -> dict:
        """Clear all history for the current session."""
        session_id = self._get_session_id(request)

        # Get all jobs to delete their images
        jobs, _ = self.history_storage.list_jobs(session_id, limit=10000, offset=0)
        for job in jobs:
            self.image_storage.delete_job_images(job.id)

        count = self.history_storage.clear_history(session_id)
        return {"deleted": count, "session_id": session_id}

    def api_list_history_snapshots(
        self,
        request: Request,
        limit: int = 20,
        offset: int = 0,
    ) -> HistorySnapshotListResponse:
        """List history snapshots for the current session."""
        session_id = self._get_session_id(request)

        if limit > 100:
            limit = 100
        if limit < 1:
            limit = 1

        snapshots, total = self.history_storage.list_snapshots(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
        return HistorySnapshotListResponse(
            snapshots=snapshots,
            total=total,
            limit=limit,
            offset=offset,
        )

    def api_create_history_snapshot(
        self,
        request: Request,
        snapshot: HistorySnapshotCreate,
    ) -> HistorySnapshot:
        """Create a new history snapshot."""
        session_id = self._get_session_id(request)
        return self.history_storage.save_snapshot(
            session_id=session_id,
            payload=snapshot.payload,
        )

    def api_get_history_snapshot(self, snapshot_id: str) -> HistorySnapshot:
        """Get a specific history snapshot."""
        snapshot = self.history_storage.get_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return snapshot

    def api_delete_history_snapshot(self, snapshot_id: str) -> dict:
        """Delete a history snapshot."""
        deleted = self.history_storage.delete_snapshot(snapshot_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return {"deleted": True, "snapshot_id": snapshot_id}

    def api_clear_history_snapshots(self, request: Request) -> dict:
        """Clear all history snapshots for the current session."""
        session_id = self._get_session_id(request)
        count = self.history_storage.clear_snapshots(session_id)
        return {"deleted": count, "session_id": session_id}

    # =========================================================================
    # Image Storage API endpoints (Epic 3)
    # =========================================================================

    def api_get_stored_image(self, image_id: str) -> Response:
        """Get a stored image by ID."""
        image_data = self.image_storage.get_image(image_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        # Determine content type from file extension
        record = self.image_storage.get_image_record(image_id)
        content_type = "image/png"
        if record and record.path.endswith(".jpg"):
            content_type = "image/jpeg"

        return Response(content=image_data, media_type=content_type)

    def api_get_thumbnail(self, image_id: str) -> Response:
        """Get thumbnail for an image (lazy-generated)."""
        thumb_data = self.image_storage.get_thumbnail(image_id)
        if not thumb_data:
            raise HTTPException(status_code=404, detail="Image not found")
        return Response(content=thumb_data, media_type="image/jpeg")

    def launch(self):
        self.app.include_router(self.router)
        uvicorn.run(
            self.combined_asgi_app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=999999999,
        )

    def _build_file_manager(self) -> Optional[FileManager]:
        if self.config.input and self.config.input.is_dir():
            logger.info(
                f"Input is directory, initialize file manager {self.config.input}"
            )

            return FileManager(
                app=self.app,
                input_dir=self.config.input,
                mask_dir=self.config.mask_dir,
                output_dir=self.config.output_dir,
            )
        return None

    def _build_plugins(self) -> Dict[str, BasePlugin]:
        return build_plugins(
            self.config.enable_interactive_seg,
            self.config.interactive_seg_model,
            self.config.interactive_seg_device,
            self.config.enable_remove_bg,
            self.config.remove_bg_device,
            self.config.remove_bg_model,
            self.config.enable_anime_seg,
            self.config.enable_realesrgan,
            self.config.realesrgan_device,
            self.config.realesrgan_model,
            self.config.enable_gfpgan,
            self.config.gfpgan_device,
            self.config.enable_restoreformer,
            self.config.restoreformer_device,
            self.config.no_half,
        )

    def _build_model_manager(self):
        return ModelManager(
            name=self.config.model,
            device=torch.device(self.config.device),
            no_half=self.config.no_half,
            low_mem=self.config.low_mem,
            disable_nsfw=self.config.disable_nsfw_checker,
            sd_cpu_textencoder=self.config.cpu_textencoder,
            local_files_only=self.config.local_files_only,
            cpu_offload=self.config.cpu_offload,
            callback=diffuser_callback,
        )
