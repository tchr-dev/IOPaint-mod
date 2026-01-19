import json

from iopaint.budget.config import BudgetConfig
from iopaint.budget.guard import BudgetGuard
from iopaint.budget.models import BudgetStatusResponse
from iopaint.budget.storage import BudgetStorage


def test_budget_limits_override(tmp_path):
    config = BudgetConfig(
        data_dir=tmp_path / "data",
        daily_cap_usd=10.0,
        monthly_cap_usd=100.0,
        session_cap_usd=5.0,
        rate_limit_seconds=0,
    )
    storage = BudgetStorage(config)
    guard = BudgetGuard(config, storage)

    storage.set_budget_limits(daily_cap_usd=2.0, monthly_cap_usd=20.0, session_cap_usd=1.0)

    status = guard.get_status("session-1")
    assert status.daily.cap_usd == 2.0
    assert status.monthly.cap_usd == 20.0
    assert status.session.cap_usd == 1.0

    result = guard.check_budget("generate", estimated_cost=1.2, session_id="session-1")
    assert result.allowed is False
    assert result.cap_type == "session"


def test_budget_status_json_serializable():
    """Test that budget status with unlimited caps is JSON serializable (no infinity values)."""
    # Test unlimited budgets (cap = 0) should return 999999.99 instead of infinity
    status = BudgetStatusResponse.from_usage(
        daily_spent=5.0,
        daily_cap=0,  # unlimited
        monthly_spent=50.0,
        monthly_cap=0,  # unlimited
        session_spent=2.0,
        session_cap=0,  # unlimited
    )

    # Verify unlimited budgets have finite remaining_usd values
    assert status.daily.remaining_usd == 999999.99
    assert status.monthly.remaining_usd == 999999.99
    assert status.session.remaining_usd == 999999.99

    # Verify unlimited flags are set correctly
    assert status.daily.is_unlimited is True
    assert status.monthly.is_unlimited is True
    assert status.session.is_unlimited is True

    # Test JSON serialization works (should not raise exception)
    json_str = json.dumps(status.model_dump())
    assert json_str is not None

    # Verify no 'inf' appears in JSON
    assert "inf" not in json_str.lower()


def test_budget_status_mixed_limits():
    """Test budget status with mix of limited and unlimited budgets."""
    status = BudgetStatusResponse.from_usage(
        daily_spent=5.0,
        daily_cap=10.0,  # limited
        monthly_spent=50.0,
        monthly_cap=0,  # unlimited
        session_spent=2.0,
        session_cap=5.0,  # limited
    )

    # Limited budgets should have normal calculations
    assert status.daily.remaining_usd == 5.0
    assert status.daily.is_unlimited is False

    # Unlimited budgets should have 999999.99
    assert status.monthly.remaining_usd == 999999.99
    assert status.monthly.is_unlimited is True

    # Another limited budget
    assert status.session.remaining_usd == 3.0
    assert status.session.is_unlimited is False

    # Test JSON serialization
    json_str = json.dumps(status.model_dump())
    assert "inf" not in json_str.lower()
