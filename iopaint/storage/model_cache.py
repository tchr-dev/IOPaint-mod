"""SQLite-based storage for cached model lists."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Generator, Dict, Any


class ModelCacheStorage:
    """Storage for cached model lists by provider/backend."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_table()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _ensure_table(self) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='models_cache'"
            )
            if cursor.fetchone():
                return
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

    def get_cached_models(
        self,
        provider: str,
        max_age_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        """Return cached payload if present and not expired."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT payload, fetched_at FROM models_cache WHERE provider = ?",
                (provider,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            fetched_at = datetime.fromisoformat(row["fetched_at"])
            if datetime.utcnow() - fetched_at > timedelta(seconds=max_age_seconds):
                return None

            try:
                return json.loads(row["payload"])
            except json.JSONDecodeError:
                return None

    def set_cached_models(self, provider: str, payload: Dict[str, Any]) -> None:
        """Upsert cached payload."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO models_cache (provider, payload, fetched_at)
                VALUES (?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    payload = excluded.payload,
                    fetched_at = excluded.fetched_at
                """,
                (provider, json.dumps(payload), datetime.utcnow().isoformat()),
            )

    def clear_cache(self, provider: Optional[str] = None) -> int:
        """Clear cached payload(s)."""
        with self._transaction() as cursor:
            if provider:
                cursor.execute(
                    "DELETE FROM models_cache WHERE provider = ?",
                    (provider,),
                )
            else:
                cursor.execute("DELETE FROM models_cache")
            return cursor.rowcount

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
