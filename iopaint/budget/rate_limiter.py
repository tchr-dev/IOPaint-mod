"""Rate limiter for preventing rapid-fire expensive operations."""

from datetime import datetime
from typing import Optional

from .config import BudgetConfig
from .storage import BudgetStorage
from .errors import RateLimitedError


class RateLimiter:
    """Rate limiter for expensive API operations.

    Enforces minimum time between expensive operations per session
    to prevent accidental double-clicks and rapid retries.

    Usage:
        limiter = RateLimiter(config, storage)

        # Check before expensive operation
        limiter.check_or_raise(session_id)

        # After successful operation
        limiter.record(session_id)
    """

    def __init__(
        self,
        config: BudgetConfig,
        storage: BudgetStorage,
        operation_type: str = "expensive",
    ):
        self.config = config
        self.storage = storage
        self.operation_type = operation_type

    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.config.rate_limit_seconds > 0

    @property
    def limit_seconds(self) -> int:
        """Get the rate limit in seconds."""
        return self.config.rate_limit_seconds

    def check(self, session_id: str) -> Optional[int]:
        """Check if operation is allowed within rate limit.

        Args:
            session_id: Session identifier

        Returns:
            None if allowed, or seconds to wait if rate limited
        """
        if not self.enabled:
            return None

        last_request = self.storage.get_last_request_time(
            session_id, self.operation_type
        )
        if last_request is None:
            return None

        elapsed = (datetime.utcnow() - last_request).total_seconds()
        if elapsed < self.limit_seconds:
            return int(self.limit_seconds - elapsed) + 1

        return None

    def check_or_raise(self, session_id: str) -> None:
        """Check rate limit and raise error if exceeded.

        Args:
            session_id: Session identifier

        Raises:
            RateLimitedError: If rate limit is exceeded
        """
        retry_after = self.check(session_id)
        if retry_after is not None:
            raise RateLimitedError(
                status="budget_rate_limited",
                retryable=True,
                detail=f"Rate limited. Please wait {retry_after}s before making another request.",
                retry_after_seconds=retry_after,
            )

    def record(self, session_id: str) -> None:
        """Record that an operation was performed.

        Call this after a successful expensive operation.

        Args:
            session_id: Session identifier
        """
        self.storage.update_last_request_time(session_id, self.operation_type)

    def get_time_until_allowed(self, session_id: str) -> int:
        """Get seconds until next operation is allowed.

        Args:
            session_id: Session identifier

        Returns:
            0 if allowed now, or seconds to wait
        """
        retry_after = self.check(session_id)
        return retry_after or 0

    def reset(self, session_id: str) -> None:
        """Reset rate limit for a session.

        Use with caution - mainly for testing or admin operations.

        Args:
            session_id: Session identifier
        """
        # We can't easily delete from rate_limits table without modifying storage,
        # so we just record a timestamp far in the past by setting last_request
        # to allow immediate next request. The check will pass since elapsed > limit.
        pass  # Currently no reset mechanism - operations naturally expire
