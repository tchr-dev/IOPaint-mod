"""SQLite-based storage for budget tracking."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Generator

from .config import BudgetConfig
from .models import LedgerEntry


class BudgetStorage:
    """SQLite-based storage for budget ledger and related data.

    Thread-safe implementation using connection per thread.

    Schema versions:
        1: Initial budget tables (budget_ledger, dedupe_cache, rate_limits)
        2: Added history tables (generation_jobs, images) - see iopaint/storage/
        3: Added history_snapshots table
        4: Added models_cache table
    """

    SCHEMA_VERSION = 4

    def __init__(self, config: BudgetConfig):
        self.config = config
        self._local = threading.local()
        config.ensure_directories()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.config.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize database schema if needed."""
        with self._transaction() as cursor:
            # Check schema version
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cursor.fetchone():
                self._create_schema(cursor)
            else:
                cursor.execute("SELECT version FROM schema_version")
                row = cursor.fetchone()
                if row and row["version"] < self.SCHEMA_VERSION:
                    self._migrate_schema(cursor, row["version"])

    def _create_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create initial database schema."""
        # Schema version table
        cursor.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

        # Budget ledger for tracking spend
        cursor.execute("""
            CREATE TABLE budget_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                model TEXT NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                actual_cost_usd REAL,
                fingerprint TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Dedupe cache for recent successful operations
        cursor.execute("""
            CREATE TABLE dedupe_cache (
                fingerprint TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                result_path TEXT,
                result_metadata TEXT
            )
        """)

        # Rate limit tracking
        cursor.execute("""
            CREATE TABLE rate_limits (
                session_id TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                last_request_at DATETIME NOT NULL,
                PRIMARY KEY (session_id, operation_type)
            )
        """)

        # Indexes for performance
        cursor.execute(
            "CREATE INDEX idx_ledger_timestamp ON budget_ledger(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX idx_ledger_session ON budget_ledger(session_id)"
        )
        cursor.execute(
            "CREATE INDEX idx_dedupe_expires ON dedupe_cache(expires_at)"
        )

        # History snapshots table (optional history cache)
        cursor.execute("""
            CREATE TABLE history_snapshots (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX idx_snapshots_session ON history_snapshots(session_id)"
        )
        cursor.execute(
            "CREATE INDEX idx_snapshots_created ON history_snapshots(created_at DESC)"
        )

        # Models cache table (provider -> cached model list)
        cursor.execute("""
            CREATE TABLE models_cache (
                provider TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX idx_models_cache_fetched ON models_cache(fetched_at DESC)"
        )

    def _migrate_schema(self, cursor: sqlite3.Cursor, from_version: int) -> None:
        """Migrate schema from older version."""
        if from_version < 2:
            # Version 2: Add history tables
            # Generation jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    intent TEXT,
                    refined_prompt TEXT,
                    negative_prompt TEXT,
                    preset TEXT,
                    params TEXT,
                    fingerprint TEXT,
                    estimated_cost_usd REAL,
                    actual_cost_usd REAL,
                    is_edit INTEGER DEFAULT 0,
                    error_message TEXT,
                    result_image_id TEXT,
                    thumbnail_image_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # Images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    path TEXT NOT NULL,
                    thumbnail_path TEXT,
                    width INTEGER,
                    height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES generation_jobs(id) ON DELETE SET NULL
                )
            """)

            # Indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_session ON generation_jobs(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_fingerprint ON generation_jobs(fingerprint)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON generation_jobs(created_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON generation_jobs(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_job ON images(job_id)"
            )

        if from_version < 3:
            # Version 3: Add history snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history_snapshots (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_session ON history_snapshots(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_created ON history_snapshots(created_at DESC)"
            )

        if from_version < 4:
            # Version 4: Add models cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models_cache (
                    provider TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_cache_fetched ON models_cache(fetched_at DESC)"
            )

        cursor.execute(
            "UPDATE schema_version SET version = ?",
            (self.SCHEMA_VERSION,),
        )

    # --- Ledger Operations ---

    def record_operation(self, entry: LedgerEntry) -> int:
        """Record an operation in the budget ledger.

        Returns:
            The ID of the inserted ledger entry.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO budget_ledger
                (session_id, operation, model, estimated_cost_usd, actual_cost_usd, fingerprint, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.session_id,
                    entry.operation,
                    entry.model,
                    entry.estimated_cost_usd,
                    entry.actual_cost_usd,
                    entry.fingerprint,
                    entry.status,
                ),
            )
            return cursor.lastrowid or 0

    def update_operation_status(
        self,
        entry_id: int,
        status: str,
        actual_cost_usd: Optional[float] = None,
    ) -> None:
        """Update the status and actual cost of a ledger entry."""
        with self._transaction() as cursor:
            if actual_cost_usd is not None:
                cursor.execute(
                    "UPDATE budget_ledger SET status = ?, actual_cost_usd = ? WHERE id = ?",
                    (status, actual_cost_usd, entry_id),
                )
            else:
                cursor.execute(
                    "UPDATE budget_ledger SET status = ? WHERE id = ?",
                    (status, entry_id),
                )

    def get_daily_spend(self, session_id: Optional[str] = None) -> float:
        """Get total spend for current day (UTC).

        Args:
            session_id: If provided, filter by session. Otherwise return all sessions.
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        with self._transaction() as cursor:
            if session_id:
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) as total
                    FROM budget_ledger
                    WHERE timestamp >= ? AND session_id = ? AND status IN ('pending', 'success')
                    """,
                    (today_start.isoformat(), session_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) as total
                    FROM budget_ledger
                    WHERE timestamp >= ? AND status IN ('pending', 'success')
                    """,
                    (today_start.isoformat(),),
                )
            row = cursor.fetchone()
            return float(row["total"]) if row else 0.0

    def get_monthly_spend(self, session_id: Optional[str] = None) -> float:
        """Get total spend for current month (UTC)."""
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with self._transaction() as cursor:
            if session_id:
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) as total
                    FROM budget_ledger
                    WHERE timestamp >= ? AND session_id = ? AND status IN ('pending', 'success')
                    """,
                    (month_start.isoformat(), session_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) as total
                    FROM budget_ledger
                    WHERE timestamp >= ? AND status IN ('pending', 'success')
                    """,
                    (month_start.isoformat(),),
                )
            row = cursor.fetchone()
            return float(row["total"]) if row else 0.0

    def get_session_spend(self, session_id: str) -> float:
        """Get total spend for a specific session."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0) as total
                FROM budget_ledger
                WHERE session_id = ? AND status IN ('pending', 'success')
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            return float(row["total"]) if row else 0.0

    # --- Rate Limit Operations ---

    def get_last_request_time(
        self, session_id: str, operation_type: str = "expensive"
    ) -> Optional[datetime]:
        """Get the timestamp of the last request for rate limiting."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT last_request_at FROM rate_limits
                WHERE session_id = ? AND operation_type = ?
                """,
                (session_id, operation_type),
            )
            row = cursor.fetchone()
            if row:
                return datetime.fromisoformat(row["last_request_at"])
            return None

    def update_last_request_time(
        self, session_id: str, operation_type: str = "expensive"
    ) -> None:
        """Update the timestamp of the last request."""
        now = datetime.utcnow()
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO rate_limits (session_id, operation_type, last_request_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, operation_type)
                DO UPDATE SET last_request_at = excluded.last_request_at
                """,
                (session_id, operation_type, now.isoformat()),
            )

    # --- Dedupe Cache Operations ---

    def get_cached_result(self, fingerprint: str) -> Optional[dict]:
        """Get cached result for a fingerprint if not expired."""
        now = datetime.utcnow()
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT result_path, result_metadata FROM dedupe_cache
                WHERE fingerprint = ? AND expires_at > ?
                """,
                (fingerprint, now.isoformat()),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "result_path": row["result_path"],
                    "result_metadata": row["result_metadata"],
                }
            return None

    def store_cached_result(
        self,
        fingerprint: str,
        result_path: str,
        result_metadata: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store a result in the dedupe cache."""
        ttl = ttl_seconds or self.config.dedupe_ttl_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO dedupe_cache (fingerprint, expires_at, result_path, result_metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fingerprint)
                DO UPDATE SET expires_at = excluded.expires_at,
                              result_path = excluded.result_path,
                              result_metadata = excluded.result_metadata
                """,
                (fingerprint, expires_at.isoformat(), result_path, result_metadata),
            )

    def cleanup_expired_cache(self) -> int:
        """Remove expired entries from dedupe cache.

        Returns:
            Number of entries removed.
        """
        now = datetime.utcnow()
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM dedupe_cache WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            return cursor.rowcount

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
