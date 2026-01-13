"""Unified error handling for OpenAI-compatible API."""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from enum import Enum


class ErrorStatus(str, Enum):
    """Unified error status codes for OpenAI-compatible API errors."""

    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONTENT_POLICY = "content_policy"
    INSUFFICIENT_QUOTA = "insufficient_quota"
    MODEL_NOT_FOUND = "model_not_found"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class OpenAIError(Exception):
    """Unified error structure for OpenAI-compatible API errors.

    Attributes:
        status: Classified error status.
        retryable: Whether the request can be retried.
        detail: Human-readable error message.
        original_error: Original exception if available.
        http_status: HTTP status code if available.
        error_code: Provider-specific error code if available.
    """

    status: ErrorStatus
    retryable: bool
    detail: str
    original_error: Optional[Exception] = field(default=None, repr=False)
    http_status: Optional[int] = None
    error_code: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.status.value}]"]
        if self.http_status:
            parts.append(f"HTTP {self.http_status}")
        parts.append(self.detail)
        if self.retryable:
            parts.append("(retryable)")
        return " ".join(parts)

    def __post_init__(self):
        # Ensure we call Exception.__init__ properly
        super().__init__(str(self))


def classify_error(
    status_code: int,
    error_body: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
) -> OpenAIError:
    """Classify HTTP errors into unified error types.

    Args:
        status_code: HTTP status code from response.
        error_body: Parsed JSON error body from response.
        exception: Original exception if available.

    Returns:
        OpenAIError with classified status and retryable flag.
    """
    error_body = error_body or {}
    error_obj = error_body.get("error", {})

    # Extract error details
    error_type = error_obj.get("type", "")
    error_code = error_obj.get("code", "")
    error_message = error_obj.get("message", "") or str(error_body)

    # Rate limiting
    if status_code == 429:
        return OpenAIError(
            status=ErrorStatus.RATE_LIMITED,
            retryable=True,
            detail=error_message or "Rate limit exceeded. Please retry after a delay.",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Authentication errors
    if status_code == 401:
        return OpenAIError(
            status=ErrorStatus.AUTHENTICATION_ERROR,
            retryable=False,
            detail=error_message or "Invalid API key or authentication failed.",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Forbidden - often quota issues
    if status_code == 403:
        if "quota" in error_message.lower() or "billing" in error_message.lower():
            return OpenAIError(
                status=ErrorStatus.INSUFFICIENT_QUOTA,
                retryable=False,
                detail=error_message or "Insufficient quota or billing issue.",
                original_error=exception,
                http_status=status_code,
                error_code=error_code,
            )
        return OpenAIError(
            status=ErrorStatus.AUTHENTICATION_ERROR,
            retryable=False,
            detail=error_message or "Access forbidden.",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Not found - model doesn't exist
    if status_code == 404:
        return OpenAIError(
            status=ErrorStatus.MODEL_NOT_FOUND,
            retryable=False,
            detail=error_message or "Model or endpoint not found.",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Bad request
    if status_code == 400:
        # Check for content policy violation
        if "content_policy" in error_type or "safety" in error_message.lower():
            return OpenAIError(
                status=ErrorStatus.CONTENT_POLICY,
                retryable=False,
                detail=error_message or "Content policy violation.",
                original_error=exception,
                http_status=status_code,
                error_code=error_code,
            )
        return OpenAIError(
            status=ErrorStatus.INVALID_REQUEST,
            retryable=False,
            detail=error_message or "Invalid request parameters.",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Server errors (5xx) - generally retryable
    if 500 <= status_code < 600:
        return OpenAIError(
            status=ErrorStatus.SERVER_ERROR,
            retryable=True,
            detail=error_message or f"Server error (HTTP {status_code}).",
            original_error=exception,
            http_status=status_code,
            error_code=error_code,
        )

    # Unknown error
    return OpenAIError(
        status=ErrorStatus.UNKNOWN,
        retryable=False,
        detail=error_message or f"Unknown error (HTTP {status_code}).",
        original_error=exception,
        http_status=status_code,
        error_code=error_code,
    )


def create_timeout_error(exception: Exception) -> OpenAIError:
    """Create an error for timeout exceptions."""
    return OpenAIError(
        status=ErrorStatus.TIMEOUT,
        retryable=True,
        detail=f"Request timed out: {exception}",
        original_error=exception,
    )


def create_network_error(exception: Exception) -> OpenAIError:
    """Create an error for network/connection exceptions."""
    return OpenAIError(
        status=ErrorStatus.NETWORK_ERROR,
        retryable=True,
        detail=f"Network error: {exception}",
        original_error=exception,
    )
