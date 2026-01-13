"""OpenAI-compatible API client for IOPaint.

This module provides a unified client for OpenAI-compatible image APIs,
supporting providers like OpenAI, ProxyAPI, OpenRouter, and local endpoints.

Note: OpenAICompatModel is imported separately to avoid circular imports.
Use: from iopaint.openai_compat.model_adapter import OpenAICompatModel
"""

from .config import OpenAIConfig
from .errors import OpenAIError, ErrorStatus, classify_error
from .client import OpenAICompatClient
from .models import (
    OpenAIModelInfo,
    EditImageRequest,
    EditImageResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    RefinePromptRequest,
    RefinePromptResponse,
    ImageSize,
)

__all__ = [
    # Config
    "OpenAIConfig",
    # Errors
    "OpenAIError",
    "ErrorStatus",
    "classify_error",
    # Client
    "OpenAICompatClient",
    # Schemas
    "OpenAIModelInfo",
    "EditImageRequest",
    "EditImageResponse",
    "GenerateImageRequest",
    "GenerateImageResponse",
    "RefinePromptRequest",
    "RefinePromptResponse",
    "ImageSize",
]
