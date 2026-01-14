"""Configuration for budget safety features."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class BudgetConfig:
    """Configuration for budget safety features.

    Configuration is loaded from environment variables with the following precedence:
    1. Constructor arguments (highest)
    2. Environment variables
    3. Default values (lowest)

    Environment Variables:
        AIE_BUDGET_DAILY_CAP: Daily spending cap in USD (default: 10.0, 0 = unlimited)
        AIE_BUDGET_MONTHLY_CAP: Monthly spending cap in USD (default: 100.0, 0 = unlimited)
        AIE_BUDGET_SESSION_CAP: Per-session spending cap in USD (default: 5.0, 0 = unlimited)
        AIE_RATE_LIMIT_SECONDS: Min seconds between expensive operations (default: 10)
        AIE_DEDUPE_ENABLED: Enable request deduplication (default: true)
        AIE_DEDUPE_TTL_SECONDS: Dedupe cache TTL in seconds (default: 3600)
        AIE_DATA_DIR: Data directory for SQLite and cache (default: ~/.iopaint/data)
    """

    daily_cap_usd: float = field(
        default_factory=lambda: float(os.getenv("AIE_BUDGET_DAILY_CAP", "10.0"))
    )
    monthly_cap_usd: float = field(
        default_factory=lambda: float(os.getenv("AIE_BUDGET_MONTHLY_CAP", "100.0"))
    )
    session_cap_usd: float = field(
        default_factory=lambda: float(os.getenv("AIE_BUDGET_SESSION_CAP", "5.0"))
    )
    rate_limit_seconds: int = field(
        default_factory=lambda: int(os.getenv("AIE_RATE_LIMIT_SECONDS", "10"))
    )
    dedupe_enabled: bool = field(
        default_factory=lambda: os.getenv("AIE_DEDUPE_ENABLED", "true").lower() == "true"
    )
    dedupe_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("AIE_DEDUPE_TTL_SECONDS", "3600"))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("AIE_DATA_DIR", "~/.iopaint/data")
        ).expanduser()
    )

    @property
    def db_path(self) -> Path:
        """Path to SQLite database file."""
        return self.data_dir / "budget.db"

    @property
    def cache_dir(self) -> Path:
        """Path to dedupe cache directory for result images."""
        return self.data_dir / "cache"

    @property
    def is_enabled(self) -> bool:
        """Check if any budget cap is enabled (non-zero)."""
        return (
            self.daily_cap_usd > 0
            or self.monthly_cap_usd > 0
            or self.session_cap_usd > 0
        )

    def ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"BudgetConfig("
            f"daily_cap=${self.daily_cap_usd:.2f}, "
            f"monthly_cap=${self.monthly_cap_usd:.2f}, "
            f"session_cap=${self.session_cap_usd:.2f}, "
            f"rate_limit={self.rate_limit_seconds}s, "
            f"dedupe={self.dedupe_enabled})"
        )
