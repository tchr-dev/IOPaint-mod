"""Pydantic models for storage API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Status of a generation job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED_BUDGET = "blocked_budget"


class JobOperation(str, Enum):
    """Type of generation operation."""

    GENERATE = "generate"
    EDIT = "edit"
    REFINE = "refine"


class GenerationJobCreate(BaseModel):
    """Request model for creating a generation job."""

    operation: JobOperation
    model: str
    intent: Optional[str] = None
    refined_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    preset: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    is_edit: bool = False


class GenerationJobUpdate(BaseModel):
    """Request model for updating a generation job."""

    status: Optional[JobStatus] = None
    actual_cost_usd: Optional[float] = None
    error_message: Optional[str] = None
    result_image_id: Optional[str] = None
    thumbnail_image_id: Optional[str] = None


class GenerationJob(BaseModel):
    """Complete generation job record."""

    id: str
    session_id: str
    status: JobStatus
    operation: JobOperation
    model: str
    intent: Optional[str] = None
    refined_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    preset: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    is_edit: bool = False
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Image references
    result_image_id: Optional[str] = None
    thumbnail_image_id: Optional[str] = None

    class Config:
        from_attributes = True


class ImageRecord(BaseModel):
    """Record for a stored image."""

    id: str
    job_id: str
    path: str
    thumbnail_path: Optional[str] = None
    width: int
    height: int
    created_at: datetime

    class Config:
        from_attributes = True


class HistorySnapshotCreate(BaseModel):
    """Request model for creating a history snapshot."""

    payload: Dict[str, Any]


class HistorySnapshot(BaseModel):
    """Snapshot record for history cache/restore."""

    id: str
    session_id: str
    payload: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class HistorySnapshotListResponse(BaseModel):
    """Response model for listing history snapshots."""

    snapshots: List[HistorySnapshot]
    total: int
    limit: int
    offset: int


class HistoryListResponse(BaseModel):
    """Response model for listing history."""

    jobs: List[GenerationJob]
    total: int
    limit: int
    offset: int


class ImageUploadResponse(BaseModel):
    """Response model for image upload."""

    id: str
    path: str
    width: int
    height: int
