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
    ImageQuality,
    ImageSize,
)
from iopaint.storage import (
    GenerationJobUpdate,
    HistoryStorage,
    ImageStorage,
    InputStorage,
    JobOperation,
    JobStatus,
    OpenAIJobFilesStorage,
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
        openai_job_storage: OpenAIJobFilesStorage,
        openai_client,
        budget_client: Optional[BudgetAwareOpenAIClient],
    ) -> None:
        self.history_storage = history_storage
        self.image_storage = image_storage
        self.input_storage = input_storage
        self.openai_job_storage = openai_job_storage
        self.openai_client = openai_client
        self.budget_client = budget_client
        self.queue: asyncio.Queue[QueuedJob] = asyncio.Queue()
        self._cancelled: set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            logger.info("JobRunner already started, skipping")
            return
        logger.info("Starting JobRunner...")
        self._stop_event = asyncio.Event()
        self._worker_task = asyncio.create_task(self._run())
        logger.info("JobRunner started successfully (queue_size=%d)", self.queue.qsize())

    async def stop(self) -> None:
        if not self._stop_event:
            return
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("JobRunner stopped")

    async def submit(self, job: QueuedJob) -> None:
        logger.info(f"Submitting job {job.job_id} to queue (queue_size={self.queue.qsize()})")
        await self.queue.put(job)
        logger.info(f"Job {job.job_id} submitted successfully (queue_size={self.queue.qsize()})")

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def _run(self) -> None:
        logger.info("JobRunner worker loop started")
        iteration = 0
        try:
            while True:
                if self._stop_event and self._stop_event.is_set():
                    logger.info("Stop event received, exiting worker loop")
                    break
                iteration += 1
                logger.debug(f"Worker loop iteration {iteration}, waiting for job (queue_size={self.queue.qsize()})")
                job = await self.queue.get()
                logger.info(f"Dequeued job {job.job_id} (queue_size={self.queue.qsize()}, tool={job.request.tool})")
                try:
                    await self._process_job(job)
                finally:
                    self.queue.task_done()
                    logger.debug(f"Job {job.job_id} task_done signaled")
        except asyncio.CancelledError:
            logger.info("JobRunner worker loop cancelled")
            raise
        logger.info("JobRunner worker loop exited")

    async def _process_job(self, job: QueuedJob) -> None:
        logger.info(f"Processing job {job.job_id} (tool={job.request.tool})")

        if job.job_id in self._cancelled:
            logger.info(f"Job {job.job_id} was cancelled before processing")
            self._mark_cancelled(job.job_id, "Cancelled before processing")
            return

        logger.debug(f"Updating job {job.job_id} status to RUNNING")
        self.history_storage.update_job(
            job.job_id, GenerationJobUpdate(status=JobStatus.RUNNING)
        )

        try:
            logger.debug(f"Dispatching job {job.job_id} to handler")
            image_bytes = await self._dispatch(job)
            logger.info(f"Job {job.job_id} dispatch completed, received {len(image_bytes) if image_bytes else 0} bytes")

            if job.job_id in self._cancelled:
                logger.info(f"Job {job.job_id} was cancelled during processing")
                self._mark_cancelled(job.job_id, "Cancelled during processing")
                return

            logger.debug(f"Saving result image for job {job.job_id}")
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
            self._persist_job_files(
                job,
                status=JobStatus.SUCCEEDED,
                output_bytes=image_bytes,
                error_message=None,
            )
            logger.info(f"Job {job.job_id} completed successfully")
        except BudgetExceededError as exc:
            logger.warning(f"Job {job.job_id} blocked by budget: {exc}")
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(
                    status=JobStatus.BLOCKED_BUDGET,
                    error_message=str(exc),
                ),
            )
            self._persist_job_files(
                job,
                status=JobStatus.BLOCKED_BUDGET,
                output_bytes=None,
                error_message=str(exc),
            )
        except RateLimitedError as exc:
            logger.warning(f"Job {job.job_id} rate limited: {exc}")
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=str(exc)),
            )
            self._persist_job_files(
                job,
                status=JobStatus.FAILED,
                output_bytes=None,
                error_message=str(exc),
            )
        except OpenAIError as exc:
            error_message = str(exc)
            logger.warning(
                f"Job {job.job_id} OpenAI error (retryable={exc.retryable}): {error_message}"
            )
            if exc.retryable and await self._retry(job, error_message):
                return
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=error_message),
            )
            self._persist_job_files(
                job,
                status=JobStatus.FAILED,
                output_bytes=None,
                error_message=error_message,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"Job {job.job_id} failed with unexpected error: {exc}")
            self.history_storage.update_job(
                job.job_id,
                GenerationJobUpdate(status=JobStatus.FAILED, error_message=str(exc)),
            )
            self._persist_job_files(
                job,
                status=JobStatus.FAILED,
                output_bytes=None,
                error_message=str(exc),
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

    def _persist_job_files(
        self,
        job: QueuedJob,
        *,
        status: JobStatus,
        output_bytes: Optional[bytes],
        error_message: Optional[str],
    ) -> None:
        job_record = self.history_storage.get_job(job.job_id)
        if not job_record:
            return
        input_bytes = self._load_input_bytes(
            job,
            "input_images",
            job.request.image_b64,
        )
        mask_bytes = self._load_input_bytes(
            job,
            "mask_images",
            job.request.mask_b64,
        )
        params = job_record.params or {}
        prompt = (
            job_record.refined_prompt
            or job_record.intent
            or job.request.prompt
            or ""
        )
        metadata = {
            "session_id": job_record.session_id,
            "job_id": job_record.id,
            "backend": "openai",
            "model": job_record.model,
            "state": status.value,
            "created_at": job_record.created_at.isoformat(),
            "operation": job_record.operation.value,
            "prompt": prompt,
            "size": params.get("size"),
            "quality": params.get("quality"),
            "cost_estimate": {
                "usd": job_record.estimated_cost_usd,
            },
            "actual_cost_usd": job_record.actual_cost_usd,
            "retry_count": getattr(job.request, "_attempts", 0),
            "error": error_message,
        }
        self.openai_job_storage.persist_job_files(
            session_id=job_record.session_id,
            job_id=job_record.id,
            created_at=job_record.created_at,
            metadata=metadata,
            input_bytes=input_bytes,
            mask_bytes=mask_bytes,
            output_bytes=output_bytes,
            error_message=error_message,
        )

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

    def _parse_size(self, size: Optional[str]) -> Optional[ImageSize]:
        if not size:
            return None
        try:
            return ImageSize(size)
        except ValueError:
            return None

    def _parse_quality(self, quality: Optional[str]) -> ImageQuality:
        if not quality:
            return ImageQuality.STANDARD
        try:
            return ImageQuality(quality)
        except ValueError:
            return ImageQuality.STANDARD

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

        size = self._parse_size(job.request.size) or ImageSize.SIZE_1024
        quality = self._parse_quality(job.request.quality)
        request = OpenAIGenerateRequest(
            prompt=job.request.prompt,
            n=job.request.n,
            size=size,
            quality=quality,
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
            prompt=job.request.prompt or "Describe the edit",
            n=job.request.n,
            size=self._parse_size(job.request.size),
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
            size=self._parse_size(job.request.size),
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
            size=self._parse_size(job.request.size),
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
            size=self._parse_size(job.request.size),
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
        job_record = self.history_storage.get_job(job_id)
        if not job_record:
            return
        metadata = {
            "session_id": job_record.session_id,
            "job_id": job_record.id,
            "backend": "openai",
            "model": job_record.model,
            "state": JobStatus.CANCELLED.value,
            "created_at": job_record.created_at.isoformat(),
            "operation": job_record.operation.value,
            "prompt": job_record.refined_prompt or job_record.intent or "",
            "size": (job_record.params or {}).get("size"),
            "quality": (job_record.params or {}).get("quality"),
            "cost_estimate": {
                "usd": job_record.estimated_cost_usd,
            },
            "actual_cost_usd": job_record.actual_cost_usd,
            "retry_count": 0,
            "error": message,
        }
        self.openai_job_storage.persist_job_files(
            session_id=job_record.session_id,
            job_id=job_record.id,
            created_at=job_record.created_at,
            metadata=metadata,
            input_bytes=None,
            mask_bytes=None,
            output_bytes=None,
            error_message=message,
        )
