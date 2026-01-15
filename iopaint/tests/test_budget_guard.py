import pytest

from iopaint.budget.config import BudgetConfig
from iopaint.budget.errors import BudgetExceededError, RateLimitedError
from iopaint.budget.guard import BudgetGuard
from iopaint.budget.models import LedgerEntry
from iopaint.budget.storage import BudgetStorage


def _make_guard(tmp_path, **config_overrides):
    config = BudgetConfig(data_dir=tmp_path / "data", **config_overrides)
    storage = BudgetStorage(config)
    return BudgetGuard(config, storage), storage


def test_budget_guard_blocks_session_cap(tmp_path):
    guard, storage = _make_guard(
        tmp_path,
        session_cap_usd=0.5,
        daily_cap_usd=10.0,
        monthly_cap_usd=100.0,
        rate_limit_seconds=0,
    )
    storage.record_operation(
        LedgerEntry(
            session_id="session-1",
            operation="generate",
            model="gpt-image-1",
            estimated_cost_usd=0.4,
            status="success",
        )
    )

    result = guard.check_budget("generate", estimated_cost=0.2, session_id="session-1")

    assert result.allowed is False
    assert result.cap_type == "session"
    assert result.spent_usd == pytest.approx(0.4)
    assert result.remaining_usd == pytest.approx(0.1)


def test_budget_guard_raises_on_cap_exceeded(tmp_path):
    guard, storage = _make_guard(
        tmp_path,
        session_cap_usd=0.2,
        daily_cap_usd=0.0,
        monthly_cap_usd=0.0,
        rate_limit_seconds=0,
    )
    storage.record_operation(
        LedgerEntry(
            session_id="session-2",
            operation="generate",
            model="gpt-image-1",
            estimated_cost_usd=0.2,
            status="success",
        )
    )

    with pytest.raises(BudgetExceededError) as excinfo:
        guard.ensure_allowed_or_raise(
            operation="generate",
            estimated_cost=0.01,
            session_id="session-2",
        )

    assert excinfo.value.cap_type == "session"
    assert excinfo.value.retryable is False


def test_budget_guard_rate_limit(tmp_path):
    guard, _storage = _make_guard(
        tmp_path,
        session_cap_usd=0.0,
        daily_cap_usd=0.0,
        monthly_cap_usd=0.0,
        rate_limit_seconds=10,
    )
    guard.update_rate_limit(session_id="session-3", operation_type="expensive")

    with pytest.raises(RateLimitedError) as excinfo:
        guard.ensure_allowed_or_raise(
            operation="generate",
            estimated_cost=0.01,
            session_id="session-3",
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.retry_after_seconds is not None
