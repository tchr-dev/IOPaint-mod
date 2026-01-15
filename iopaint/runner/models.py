"""Models for queued job submission."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobTool(str, Enum):
    """Supported job tool types for the runner."""

    GENERATE = "generate"
    EDIT = "edit"
    OUTPAINT = "outpaint"
    VARIATION = "variation"
    UPSCALE = "upscale"
    BACKGROUND_REMOVE = "background_remove"


class JobSubmitRequest(BaseModel):
    """Client-submitted job request payload.

    Keep inputs minimal and flexible to avoid frequent schema migrations.
    Large binary inputs should remain in-memory during queue processing and
    should not be stored in SQLite. If persistent inputs become necessary,
    migrate to file-backed storage and store only references here.
    """

    tool: JobTool = Field(..., description="Requested tool/action")
    prompt: Optional[str] = Field(default=None, description="Prompt or intent")
    model: Optional[str] = Field(default=None, description="Model identifier")
    size: Optional[str] = Field(default=None, description="Output size (e.g. 1024x1024)")
    quality: Optional[str] = Field(default=None, description="Quality preset")
    n: int = Field(default=1, ge=1, le=10, description="Number of images")
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG image (kept in memory, not persisted)",
    )
    mask_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG mask (kept in memory, not persisted)",
    )
    scale: Optional[float] = Field(default=None, description="Upscale factor")
    mode: Optional[str] = Field(default=None, description="Tool mode (local/prompt/service)")
    intent: Optional[str] = Field(default=None, description="User intent for history")
    refined_prompt: Optional[str] = Field(default=None, description="Refined prompt")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt")
    preset: Optional[str] = Field(default=None, description="Preset name")
