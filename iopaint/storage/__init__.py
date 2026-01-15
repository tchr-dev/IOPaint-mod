"""Storage module for generation history and images.

This module provides persistent storage for:
- Generation job history (SQLite)
- Generated images (file-based with SQLite metadata)
- Thumbnails (lazy-generated)

Usage:
    from iopaint.storage import HistoryStorage, ImageStorage
    from iopaint.storage.models import GenerationJob, GenerationJobCreate

    # Initialize storage
    history = HistoryStorage(db_path=Path("~/.iopaint/data/budget.db"))
    images = ImageStorage(
        data_dir=Path("~/.iopaint/data"),
        db_path=Path("~/.iopaint/data/budget.db")
    )

    # Create a job
    job = history.save_job(
        session_id="abc123",
        job=GenerationJobCreate(
            operation=JobOperation.GENERATE,
            model="gpt-image-1",
            refined_prompt="A beautiful sunset..."
        )
    )

    # Save image
    image = images.save_image(image_bytes, job_id=job.id)
"""

from .history import HistoryStorage
from .images import ImageStorage
from .model_cache import ModelCacheStorage
from .models import (
    GenerationJob,
    GenerationJobCreate,
    GenerationJobUpdate,
    ImageRecord,
    HistorySnapshot,
    HistorySnapshotCreate,
    HistorySnapshotListResponse,
    JobStatus,
    JobOperation,
    HistoryListResponse,
    ImageUploadResponse,
)

__all__ = [
    "HistoryStorage",
    "ImageStorage",
    "ModelCacheStorage",
    "GenerationJob",
    "GenerationJobCreate",
    "GenerationJobUpdate",
    "ImageRecord",
    "HistorySnapshot",
    "HistorySnapshotCreate",
    "HistorySnapshotListResponse",
    "JobStatus",
    "JobOperation",
    "HistoryListResponse",
    "ImageUploadResponse",
]
