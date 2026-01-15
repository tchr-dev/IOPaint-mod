"""Async job runner for queued operations."""

import asyncio
import base64
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from iopaint.budget import BudgetAwareOpenAIClient, BudgetExceededError, RateLimitedError
from iopaint.openai_compat.errors import OpenAIError
from iopaint.openai_compat.models import (
    GenerateImageRequest as OpenAIGenerateRequest,
    EditImageRequest as OpenAIEditRequest,
    CreateVariationRequest as OpenAIVariationRequest,
)
from iopaint.storage import (
    GenerationJobUpdate,
    HistoryStorage,
    ImageStorage,
    InputStorage,
    JobOperation,
    JobStatus,
)

from .models import JobSubmitRequest, JobTool


@dataclass
class QueuedJob:
    """In-memory queued job payload.

    Input metadata is stored in generation_jobs.params while the binary
    inputs themselves live in the input storage directory.
    """

    job_id: str
    session_id: str
    request: JobSubmitRequest
    openai_base_url: Optional[str]


class JobRunner:
    """Async job runner with an in-memory queue.

    The queue backend is intentionally in-memory for the MVP; if we need
    persistence or multi-worker scaling later, swap the queue implementation
    without changing the API contract.
    """

    def __init__(
        self,
        history_storage: HistoryStorage,
        image_storage: ImageStorage,
        input_storage: InputStorage,
        openai_client,
        budget_client: Optional[BudgetAwareOpenAIClient],
    ) -> None:
        self.history_storage = history_storage
        self.image_storage = image_storage
        self.input_storage = input_storage
        self.openai_client = openai_client
        self.budget_client = budget_client
        self.queue: asyncio.Queue[QueuedJob] = asyncio.Queue()
        self._cancelled: set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._stop_event = asyncio.Event()
        self._worker_task = asyncio.create_task(self._run())
        logger.info("JobRunner started")

    async def stop(self) -> None:
        if not self._stop_event:
            return
        self._stop_event.set()
        if self._worker_task:
            await self._worker_task
        logger.info("JobRunner stopped")

    async def submit(self, job: QueuedJob) -> None:
        await self.queue.put(job)

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def _run(self) -> None:
        while True:
            if self._stop_event and self._stop_event.is_set():
                break
            job = await self.queue.get()
            try:
                await self._process_job(job)
            finally:
                self.queue.task_done()

    async def _process_job(self, job: QueuedJob) -> None:
        if job.job_id in self._cancelled:
            self._mark_cancelled(job.job_id, "Cancelled before processing")
            return

        self.history_storage.update_job(
            job.job_id, GenerationJobUpdate(status=JobStatus.RUNNING)
        )

        try:
            image_bytes = await self._dispatch(job)
            if job.job_id in self._cancelled:
                self._mark_cancelled(job.job_id, "Cancelled during processing")
                return

            image_record = self.image_storage.save_image(
                image_bytes,
                job_id=job.job_id,
                format="PNG",
            )
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(
                    status=JobStatus.SUCCEEDED,
                    result_image_id=image_record.id,
                ),
            )
        except BudgetExceededError as exc:
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(
                    status=JobStatus.BLOCKED_BUDGET,
                    error_message=str(exc),
                ),
            )
        except RateLimitedError as exc:
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=str(exc)),
            )
        except OpenAIError as exc:
            error_message = str(exc)
            if exc.retryable and await self._retry(job, error_message):
                return
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=error_message),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=str(exc)),
            )

    async def _retry(self, job: QueuedJob, error_message: str) -> bool:
        max_attempts = 2
        params = job.request
        attempts = getattr(params, "_attempts", 0)
        if attempts >= max_attempts:
            return False

        setattr(params, "_attempts", attempts + 1)
        backoff = 2 ** attempts
        logger.warning(
            "Retrying job %s after %ss (attempt %s/%s): %s",
            job.job_id,
            backoff,
            attempts + 1,
            max_attempts,
            error_message,
        )
        await asyncio.sleep(backoff)
        await self.queue.put(job)
        return True

    async def _dispatch(self, job: QueuedJob) -> bytes:
        tool = job.request.tool
        if tool == JobTool.GENERATE:
            return self._run_openai_generate(job)
        if tool == JobTool.VARIATION:
            return self._run_openai_variation(job)
        if tool in {JobTool.EDIT, JobTool.OUTPAINT}:
            return self._run_openai_edit(job)
        if tool == JobTool.UPSCALE:
            return self._run_openai_upscale(job)
        if tool == JobTool.BACKGROUND_REMOVE:
            return self._run_openai_background_remove(job)

        raise ValueError(f"Unsupported tool: {tool}")

    def _resolve_openai_client(self, job: QueuedJob):
        # For now we only support the configured client. If a base URL override
        # was provided at submission time, a future implementation can construct
        # a per-job client here.
        return self.openai_client

    def _load_input_bytes(
        self, job: QueuedJob, key: str, fallback_b64: Optional[str]
    ) -> Optional[bytes]:
        job_record = self.history_storage.get_job(job.job_id)
        if job_record and job_record.params:
            entries = job_record.params.get(key)
            if isinstance(entries, list) and entries:
                entry = entries[0]
                if isinstance(entry, dict):
                    path = entry.get("path")
                    if path:
                        payload = self.input_storage.load_input_bytes(path)
                        if payload is not None:
                            return payload

        if fallback_b64:
            return base64.b64decode(fallback_b64)

        return None

    def _run_openai_generate(self, job: QueuedJob) -> bytes:
        if not job.request.prompt:
            raise ValueError("Prompt is required for generate jobs")

        request = OpenAIGenerateRequest(
            prompt=job.request.prompt,
            n=job.request.n,
            size=job.request.size or "1024x1024",
            quality=job.request.quality or "standard",
            model=job.request.model,
        )
        if self.budget_client:
            return self.budget_client.generate_image(
                request,
                session_id=job.session_id,
            )
        client = self._resolve_openai_client(job)
        return client.generate_image(request)

    def _run_openai_edit(self, job: QueuedJob) -> bytes:
        if not job.request.prompt:
            raise ValueError("Prompt is required for edit jobs")

        image_bytes = self._load_input_bytes(
            job,
            "input_images",
            job.request.image_b64,
        )
        if not image_bytes:
            raise ValueError("Edit jobs require image input")

        mask_bytes = self._load_input_bytes(
            job,
            "mask_images",
            job.request.mask_b64,
        )
        if not mask_bytes:
            raise ValueError("Edit jobs require mask input")

        request = OpenAIEditRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=job.request.prompt,
            n=job.request.n,
            size=job.request.size,
            model=job.request.model,
        )
        if self.budget_client:
            return self.budget_client.edit_image(
                request,
                session_id=job.session_id,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
            )
        client = self._resolve_openai_client(job)
        return client.edit_image(request)

    def _run_openai_variation(self, job: QueuedJob) -> bytes:
        image_bytes = self._load_input_bytes(
            job,
            "input_images",
            job.request.image_b64,
        )
        if not image_bytes:
            raise ValueError("Variation jobs require image input")

        request = OpenAIVariationRequest(
            image=image_bytes,
            n=job.request.n,
            size=job.request.size,
            model=job.request.model,
        )
        if self.budget_client:
            return self.budget_client.create_variation(
                request,
                session_id=job.session_id,
                image_bytes=image_bytes,
            )
        client = self._resolve_openai_client(job)
        return client.create_variation(request)

    def _run_openai_upscale(self, job: QueuedJob) -> bytes:
        image_bytes = self._load_input_bytes(
            job,
            "input_images",
            job.request.image_b64,
        )
        if not image_bytes:
            raise ValueError("Upscale jobs require image input")

        mask_bytes = self._full_mask(image_bytes)
        request = OpenAIEditRequest(
            image=image_bytes,
            mask=mask_bytes,
            prompt=job.request.prompt
            or "Enhance this image to higher resolution with more detail while preserving original composition.",
            n=1,
            size=job.request.size,
            model=job.request.model,
        )
        if self.budget_client:
            return self.budget_client.edit_image(
                request,
                session_id=job.session_id,
                image_bytes=image_bytes,
                mask_bytes=mask_bytes,
            )
        client = self._resolve_openai_client(job)
        return client.edit_image(request)

    def _run_openai_background_remove(self, job: QueuedJob) -> bytes:
        image_bytes = self._load_input_bytes(
            job,
            "input_images",
            job.request.image_b64,
        )
        if not image_bytes:
            raise ValueError("Background remove jobs require image input")

        request = OpenAIEditRequest(
            image=image_bytes,
            mask=self._full_mask(image_bytes),
            prompt=job.request.prompt
            or "Remove the background around the main object and output a transparent PNG.",
            n=1,
            size=job.request.size,
            model=job.request.model,
        )
        client = self._resolve_openai_client(job)
        return client.edit_image(request)

    def _full_mask(self, image_bytes: bytes) -> bytes:
        # For prompt-based tools we reuse the edit endpoint with a full mask.
        # This mirrors existing API behavior and can be replaced later by
        # specialized services without changing the queue contract.
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))
        mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
        buffer = BytesIO()
        mask.save(buffer, format="PNG")
        return buffer.getvalue()

    def _mark_cancelled(self, job_id: str, message: str) -> None:
        self.history_storage.update_job(
            job_id,
            GenerationJobUpdate(status=JobStatus.CANCELLED, error_message=message),
        )
