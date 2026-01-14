"""Budget-specific error types."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BudgetErrorStatus(str, Enum):
    """Error status codes for budget-related errors."""

    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "budget_rate_limited"
    DUPLICATE_REQUEST = "duplicate_request"


@dataclass
class BudgetError(Exception):
    """Base class for budget-related errors.

    Attributes:
        status: Classified error status.
        retryable: Whether the operation can be retried.
        detail: Human-readable error message.
        retry_after_seconds: Seconds to wait before retry (if retryable).
    """

    status: BudgetErrorStatus
    retryable: bool
    detail: str
    retry_after_seconds: Optional[int] = None

    def __str__(self) -> str:
        parts = [f"[{self.status.value}]", self.detail]
        if self.retryable and self.retry_after_seconds:
            parts.append(f"(retry after {self.retry_after_seconds}s)")
        return " ".join(parts)

    def __post_init__(self):
        super().__init__(str(self))


@dataclass
class BudgetExceededError(BudgetError):
    """Raised when a budget cap is exceeded."""

    cap_type: str = ""  # "daily", "monthly", "session"
    spent_usd: float = 0.0
    cap_usd: float = 0.0

    def __post_init__(self):
        self.status = BudgetErrorStatus.BUDGET_EXCEEDED
        self.retryable = False
        if not self.detail:
            self.detail = (
                f"{self.cap_type.capitalize()} budget exceeded: "
                f"${self.spent_usd:.2f} spent of ${self.cap_usd:.2f} cap"
            )
        super().__post_init__()


@dataclass
class RateLimitedError(BudgetError):
    """Raised when rate limit is hit."""

    def __post_init__(self):
        self.status = BudgetErrorStatus.RATE_LIMITED
        self.retryable = True
        if not self.detail:
            self.detail = (
                f"Rate limited. Please wait {self.retry_after_seconds or 'a moment'} "
                "before making another request."
            )
        super().__post_init__()


@dataclass
class DuplicateRequestError(BudgetError):
    """Raised when a duplicate request is detected with cached result available."""

    fingerprint: str = ""
    cached_result_path: Optional[str] = None

    def __post_init__(self):
        self.status = BudgetErrorStatus.DUPLICATE_REQUEST
        self.retryable = False
        if not self.detail:
            self.detail = f"Duplicate request detected (fingerprint: {self.fingerprint[:16]}...)"
        super().__post_init__()
