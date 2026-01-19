"""Pydantic models for budget API responses."""

from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


class BudgetUsage(BaseModel):
    """Usage statistics for a single budget period."""

    spent_usd: float
    remaining_usd: float
    cap_usd: float
    is_unlimited: bool = False

    @property
    def percentage_used(self) -> float:
        """Percentage of budget used (0-100)."""
        if self.is_unlimited or self.cap_usd <= 0:
            return 0.0
        return min(100.0, (self.spent_usd / self.cap_usd) * 100)


class BudgetStatusResponse(BaseModel):
    """Response model for /api/v1/budget/status endpoint."""

    daily: BudgetUsage
    monthly: BudgetUsage
    session: BudgetUsage
    status: Literal["ok", "warning", "blocked"]
    message: Optional[str] = None

    @classmethod
    def from_usage(
        cls,
        daily_spent: float,
        daily_cap: float,
        monthly_spent: float,
        monthly_cap: float,
        session_spent: float,
        session_cap: float,
    ) -> "BudgetStatusResponse":
        """Create response from raw usage values."""
        daily = BudgetUsage(
            spent_usd=daily_spent,
            remaining_usd=max(0, daily_cap - daily_spent) if daily_cap > 0 else 999999.99,
            cap_usd=daily_cap,
            is_unlimited=daily_cap <= 0,
        )
        monthly = BudgetUsage(
            spent_usd=monthly_spent,
            remaining_usd=max(0, monthly_cap - monthly_spent) if monthly_cap > 0 else 999999.99,
            cap_usd=monthly_cap,
            is_unlimited=monthly_cap <= 0,
        )
        session = BudgetUsage(
            spent_usd=session_spent,
            remaining_usd=max(0, session_cap - session_spent) if session_cap > 0 else 999999.99,
            cap_usd=session_cap,
            is_unlimited=session_cap <= 0,
        )

        # Determine overall status
        status: Literal["ok", "warning", "blocked"] = "ok"
        message = None

        # Check for blocked status (any cap exceeded)
        for name, usage in [("Daily", daily), ("Monthly", monthly), ("Session", session)]:
            if not usage.is_unlimited and usage.spent_usd >= usage.cap_usd:
                status = "blocked"
                message = f"{name} budget cap exceeded"
                break

        # Check for warning status (>80% of any cap used)
        if status == "ok":
            for name, usage in [("Daily", daily), ("Monthly", monthly), ("Session", session)]:
                if not usage.is_unlimited and usage.percentage_used >= 80:
                    status = "warning"
                    message = f"{name} budget is {usage.percentage_used:.0f}% used"
                    break

        return cls(
            daily=daily,
            monthly=monthly,
            session=session,
            status=status,
            message=message,
        )


class BudgetLimits(BaseModel):
    """Persisted budget limits overrides."""

    daily_cap_usd: float
    monthly_cap_usd: float
    session_cap_usd: float


class BudgetLimitsUpdate(BaseModel):
    """Request model for updating budget limits."""

    daily_cap_usd: Optional[float] = None
    monthly_cap_usd: Optional[float] = None
    session_cap_usd: Optional[float] = None


class CostEstimateRequest(BaseModel):
    """Request model for cost estimation."""

    operation: Literal["generate", "edit", "variation", "refine"]
    model: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1


class CostEstimateResponse(BaseModel):
    """Response model for /api/v1/budget/estimate endpoint."""

    estimated_cost_usd: float
    cost_tier: Literal["low", "medium", "high"]
    warning: Optional[str] = None

    @classmethod
    def from_cost(cls, cost: float, threshold_low: float = 0.02, threshold_high: float = 0.10) -> "CostEstimateResponse":
        """Create response from estimated cost."""
        if cost <= threshold_low:
            tier: Literal["low", "medium", "high"] = "low"
        elif cost <= threshold_high:
            tier = "medium"
        else:
            tier = "high"

        warning = None
        if tier == "high":
            warning = f"This operation costs ${cost:.2f}. Consider using Draft preset for lower cost."

        return cls(
            estimated_cost_usd=cost,
            cost_tier=tier,
            warning=warning,
        )


class LedgerEntry(BaseModel):
    """Model for budget ledger entries."""

    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    session_id: str
    operation: str
    model: str
    estimated_cost_usd: float
    actual_cost_usd: Optional[float] = None
    fingerprint: Optional[str] = None
    status: Literal["pending", "success", "failed", "blocked"] = "pending"
