from iopaint.budget.config import BudgetConfig
from iopaint.budget.guard import BudgetGuard
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
