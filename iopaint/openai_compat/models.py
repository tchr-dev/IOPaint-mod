"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import Optional, List, Literal
from enum import Enum

from pydantic import BaseModel, Field


class ImageSize(str, Enum):
    """Supported image sizes for OpenAI image generation/editing."""

    SIZE_256 = "256x256"
    SIZE_512 = "512x512"
    SIZE_1024 = "1024x1024"
    SIZE_1792_1024 = "1792x1024"
    SIZE_1024_1792 = "1024x1792"

    @classmethod
    def from_dimensions(cls, width: int, height: int) -> Optional["ImageSize"]:
        """Get ImageSize enum from width and height."""
        size_str = f"{width}x{height}"
        for size in cls:
            if size.value == size_str:
                return size
        return None


class ImageQuality(str, Enum):
    """Image quality settings for generation."""

    STANDARD = "standard"
    HD = "hd"


class ResponseFormat(str, Enum):
    """Response format for image API."""

    URL = "url"
    B64_JSON = "b64_json"


# ============================================================================
# Model Information
# ============================================================================


class OpenAIModelInfo(BaseModel):
    """Model information returned by list_models endpoint."""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    owned_by: str = Field(..., description="Owner organization")


class ListModelsResponse(BaseModel):
    """Response from /models endpoint."""

    object: str = Field(default="list")
    data: List[OpenAIModelInfo] = Field(default_factory=list)


# ============================================================================
# Image Edit (Inpaint) Requests/Responses
# ============================================================================


class EditImageRequest(BaseModel):
    """Request for image editing (inpaint/outpaint).

    The image and mask should be PNG format.
    For the mask: transparent areas (alpha=0) indicate where to edit.
    """

    image: bytes = Field(..., description="PNG image bytes")
    mask: bytes = Field(..., description="PNG mask bytes (transparent = edit area)")
    prompt: str = Field(..., description="Text description of the edit")
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: Optional[ImageSize] = Field(
        default=None,
        description="Output image size. If None, matches input size.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use. If None, uses default from config.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.B64_JSON,
        description="Format of response data",
    )


class ImageData(BaseModel):
    """Single image data in response."""

    url: Optional[str] = Field(default=None, description="URL to image (if url format)")
    b64_json: Optional[str] = Field(
        default=None, description="Base64 encoded image (if b64_json format)"
    )
    revised_prompt: Optional[str] = Field(
        default=None, description="Revised prompt if applicable"
    )


class EditImageResponse(BaseModel):
    """Response from image editing endpoint."""

    created: int = Field(..., description="Unix timestamp of creation")
    data: List[ImageData] = Field(default_factory=list)


# ============================================================================
# Image Generation Requests/Responses
# ============================================================================


class GenerateImageRequest(BaseModel):
    """Request for text-to-image generation."""

    prompt: str = Field(..., min_length=1, description="Text description of image")
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: ImageSize = Field(
        default=ImageSize.SIZE_1024,
        description="Output image size",
    )
    quality: ImageQuality = Field(
        default=ImageQuality.STANDARD,
        description="Image quality (standard or hd)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use. If None, uses default from config.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.B64_JSON,
        description="Format of response data",
    )
    style: Optional[Literal["vivid", "natural"]] = Field(
        default=None,
        description="Style of generated images (dall-e-3 only)",
    )


class GenerateImageResponse(BaseModel):
    """Response from image generation endpoint."""

    created: int = Field(..., description="Unix timestamp of creation")
    data: List[ImageData] = Field(default_factory=list)


# ============================================================================
# Prompt Refinement Requests/Responses
# ============================================================================


class RefinePromptRequest(BaseModel):
    """Request for prompt refinement using cheap LLM call.

    Uses a small/cheap model to expand and improve image generation prompts
    before sending to expensive image generation API.
    """

    prompt: str = Field(..., min_length=1, description="Original prompt to refine")
    context: Optional[str] = Field(
        default=None,
        description="Additional context for refinement (e.g., style, mood)",
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model for refinement. If None, uses gpt-4o-mini.",
    )
    max_tokens: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Maximum tokens for refined prompt",
    )


class RefinePromptResponse(BaseModel):
    """Response from prompt refinement."""

    original_prompt: str = Field(..., description="Original input prompt")
    refined_prompt: str = Field(..., description="Refined/expanded prompt")
    model_used: str = Field(..., description="Model used for refinement")


# ============================================================================
# Image Variations Requests/Responses
# ============================================================================


class CreateVariationRequest(BaseModel):
    """Request for creating image variations."""

    image: bytes = Field(..., description="PNG image bytes")
    n: int = Field(default=1, ge=1, le=10, description="Number of variations")
    size: Optional[ImageSize] = Field(
        default=None,
        description="Output image size",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.B64_JSON,
        description="Format of response data",
    )


class CreateVariationResponse(BaseModel):
    """Response from create variations endpoint."""

    created: int = Field(..., description="Unix timestamp of creation")
    data: List[ImageData] = Field(default_factory=list)
