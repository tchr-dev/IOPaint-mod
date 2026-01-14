"""Budget safety features for IOPaint.

This package provides:
- BudgetGuard: Enforces daily/monthly/session spending caps
- BudgetStorage: SQLite-based storage for budget ledger
- BudgetConfig: Configuration via environment variables
- Error types: BudgetExceededError, RateLimitedError, DuplicateRequestError
"""

from .config import BudgetConfig
from .storage import BudgetStorage
from .guard import BudgetGuard, BudgetCheckResult
from .errors import (
    BudgetError,
    BudgetErrorStatus,
    BudgetExceededError,
    RateLimitedError,
    DuplicateRequestError,
)
from .models import (
    BudgetUsage,
    BudgetStatusResponse,
    CostEstimateRequest,
    CostEstimateResponse,
    LedgerEntry,
)
from .dedupe import (
    DedupeCache,
    calculate_fingerprint,
    normalize_prompt,
    hash_image_bytes,
)
from .rate_limiter import RateLimiter
from .session import (
    generate_session_id,
    get_session_id_from_request,
    get_session_id,
)
from .cost_estimator import CostEstimator, PricingRule, DEFAULT_PRICING
from .client_wrapper import BudgetAwareOpenAIClient

__all__ = [
    # Config
    "BudgetConfig",
    # Storage
    "BudgetStorage",
    # Guard
    "BudgetGuard",
    "BudgetCheckResult",
    # Errors
    "BudgetError",
    "BudgetErrorStatus",
    "BudgetExceededError",
    "RateLimitedError",
    "DuplicateRequestError",
    # Models
    "BudgetUsage",
    "BudgetStatusResponse",
    "CostEstimateRequest",
    "CostEstimateResponse",
    "LedgerEntry",
    # Dedupe
    "DedupeCache",
    "calculate_fingerprint",
    "normalize_prompt",
    "hash_image_bytes",
    # Rate Limiter
    "RateLimiter",
    # Session
    "generate_session_id",
    "get_session_id_from_request",
    "get_session_id",
    # Cost Estimator
    "CostEstimator",
    "PricingRule",
    "DEFAULT_PRICING",
    # Client Wrapper
    "BudgetAwareOpenAIClient",
]
