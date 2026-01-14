"""BudgetGuard - enforces budget caps and validates operations."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from .config import BudgetConfig
from .storage import BudgetStorage
from .models import BudgetStatusResponse, LedgerEntry
from .errors import BudgetExceededError, RateLimitedError


@dataclass
class BudgetCheckResult:
    """Result of a budget check operation."""

    allowed: bool
    reason: Optional[str] = None
    cap_type: Optional[str] = None  # "daily", "monthly", "session"
    spent_usd: float = 0.0
    cap_usd: float = 0.0
    remaining_usd: float = 0.0


class BudgetGuard:
    """Enforces budget caps and validates operations before execution.

    Usage:
        guard = BudgetGuard(config, storage)

        # Check if operation is allowed
        result = guard.check_budget("generate", estimated_cost=0.04, session_id="abc123")
        if not result.allowed:
            raise BudgetExceededError(...)

        # After successful operation
        guard.record_success(entry_id, actual_cost=0.04)
    """

    def __init__(self, config: BudgetConfig, storage: BudgetStorage):
        self.config = config
        self.storage = storage

    def check_budget(
        self,
        operation: str,
        estimated_cost: float,
        session_id: str,
    ) -> BudgetCheckResult:
        """Check if an operation is allowed within budget constraints.

        Checks caps in order: session → daily → monthly.
        First exceeded cap blocks the operation.

        Args:
            operation: Operation type ("generate", "edit", "variation", "refine")
            estimated_cost: Estimated cost in USD for this operation
            session_id: Session identifier

        Returns:
            BudgetCheckResult with allowed status and details.
        """
        # Check session cap first (most granular)
        if self.config.session_cap_usd > 0:
            session_spent = self.storage.get_session_spend(session_id)
            if session_spent + estimated_cost > self.config.session_cap_usd:
                return BudgetCheckResult(
                    allowed=False,
                    reason=f"Session budget exceeded: ${session_spent:.2f} spent + ${estimated_cost:.2f} > ${self.config.session_cap_usd:.2f} cap",
                    cap_type="session",
                    spent_usd=session_spent,
                    cap_usd=self.config.session_cap_usd,
                    remaining_usd=max(0, self.config.session_cap_usd - session_spent),
                )

        # Check daily cap
        if self.config.daily_cap_usd > 0:
            daily_spent = self.storage.get_daily_spend()
            if daily_spent + estimated_cost > self.config.daily_cap_usd:
                return BudgetCheckResult(
                    allowed=False,
                    reason=f"Daily budget exceeded: ${daily_spent:.2f} spent + ${estimated_cost:.2f} > ${self.config.daily_cap_usd:.2f} cap",
                    cap_type="daily",
                    spent_usd=daily_spent,
                    cap_usd=self.config.daily_cap_usd,
                    remaining_usd=max(0, self.config.daily_cap_usd - daily_spent),
                )

        # Check monthly cap
        if self.config.monthly_cap_usd > 0:
            monthly_spent = self.storage.get_monthly_spend()
            if monthly_spent + estimated_cost > self.config.monthly_cap_usd:
                return BudgetCheckResult(
                    allowed=False,
                    reason=f"Monthly budget exceeded: ${monthly_spent:.2f} spent + ${estimated_cost:.2f} > ${self.config.monthly_cap_usd:.2f} cap",
                    cap_type="monthly",
                    spent_usd=monthly_spent,
                    cap_usd=self.config.monthly_cap_usd,
                    remaining_usd=max(0, self.config.monthly_cap_usd - monthly_spent),
                )

        # All checks passed
        return BudgetCheckResult(allowed=True)

    def check_rate_limit(
        self,
        session_id: str,
        operation_type: str = "expensive",
    ) -> Optional[int]:
        """Check if rate limit allows the operation.

        Args:
            session_id: Session identifier
            operation_type: Type of operation (default: "expensive")

        Returns:
            None if allowed, or seconds to wait if rate limited.
        """
        if self.config.rate_limit_seconds <= 0:
            return None

        last_request = self.storage.get_last_request_time(session_id, operation_type)
        if last_request is None:
            return None

        elapsed = (datetime.utcnow() - last_request).total_seconds()
        if elapsed < self.config.rate_limit_seconds:
            return int(self.config.rate_limit_seconds - elapsed) + 1

        return None

    def record_operation_start(
        self,
        operation: str,
        model: str,
        estimated_cost: float,
        session_id: str,
        fingerprint: Optional[str] = None,
    ) -> int:
        """Record the start of an operation (before API call).

        Returns:
            Ledger entry ID for later status update.
        """
        entry = LedgerEntry(
            session_id=session_id,
            operation=operation,
            model=model,
            estimated_cost_usd=estimated_cost,
            fingerprint=fingerprint,
            status="pending",
        )
        return self.storage.record_operation(entry)

    def record_success(
        self,
        entry_id: int,
        actual_cost: Optional[float] = None,
    ) -> None:
        """Record successful completion of an operation."""
        self.storage.update_operation_status(entry_id, "success", actual_cost)

    def record_failure(self, entry_id: int) -> None:
        """Record failed operation (cost not charged)."""
        self.storage.update_operation_status(entry_id, "failed", actual_cost_usd=0.0)

    def record_blocked(self, entry_id: int) -> None:
        """Record operation blocked by budget."""
        self.storage.update_operation_status(entry_id, "blocked", actual_cost_usd=0.0)

    def update_rate_limit(
        self,
        session_id: str,
        operation_type: str = "expensive",
    ) -> None:
        """Update rate limit timestamp after successful operation."""
        self.storage.update_last_request_time(session_id, operation_type)

    def get_status(self, session_id: str) -> BudgetStatusResponse:
        """Get current budget status for a session.

        Args:
            session_id: Session identifier

        Returns:
            BudgetStatusResponse with current usage and status.
        """
        daily_spent = self.storage.get_daily_spend()
        monthly_spent = self.storage.get_monthly_spend()
        session_spent = self.storage.get_session_spend(session_id)

        return BudgetStatusResponse.from_usage(
            daily_spent=daily_spent,
            daily_cap=self.config.daily_cap_usd,
            monthly_spent=monthly_spent,
            monthly_cap=self.config.monthly_cap_usd,
            session_spent=session_spent,
            session_cap=self.config.session_cap_usd,
        )

    def ensure_allowed_or_raise(
        self,
        operation: str,
        estimated_cost: float,
        session_id: str,
    ) -> None:
        """Check budget and rate limit, raising appropriate errors if blocked.

        This is a convenience method that combines check_budget and check_rate_limit
        and raises the appropriate error type.

        Raises:
            BudgetExceededError: If budget cap is exceeded
            RateLimitedError: If rate limit is hit
        """
        # Check rate limit first
        retry_after = self.check_rate_limit(session_id)
        if retry_after is not None:
            raise RateLimitedError(
                status="budget_rate_limited",
                retryable=True,
                detail=f"Rate limited. Please wait {retry_after}s before making another request.",
                retry_after_seconds=retry_after,
            )

        # Check budget
        result = self.check_budget(operation, estimated_cost, session_id)
        if not result.allowed:
            raise BudgetExceededError(
                status="budget_exceeded",
                retryable=False,
                detail=result.reason or "Budget exceeded",
                cap_type=result.cap_type or "",
                spent_usd=result.spent_usd,
                cap_usd=result.cap_usd,
            )
