"""SQLite-based storage for generation history."""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Generator, Dict, Any

from .models import (
    GenerationJob,
    GenerationJobCreate,
    GenerationJobUpdate,
    JobStatus,
    JobOperation,
    HistorySnapshot,
)


class HistoryStorage:
    """SQLite-based storage for generation job history.

    Thread-safe implementation using connection per thread.
    Shares the database with BudgetStorage.
    """

    SCHEMA_VERSION = 3  # Bump when adding history tables

    def __init__(self, db_path: Path):
        """Initialize history storage.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_tables()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
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

    def _ensure_tables(self) -> None:
        """Ensure history tables exist."""
        with self._transaction() as cursor:
            # Check if generation_jobs table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='generation_jobs'"
            )
            if not cursor.fetchone():
                self._create_tables(cursor)
            else:
                # Check schema version and migrate if needed
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT version FROM schema_version")
                    row = cursor.fetchone()
                    if row and row["version"] < self.SCHEMA_VERSION:
                        self._migrate_tables(cursor, row["version"])
                        cursor.execute(
                            "UPDATE schema_version SET version = ?",
                            (self.SCHEMA_VERSION,),
                        )

    def _create_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create history tables."""
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

        # Indexes for performance
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

        # History snapshots table
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

        # Update schema version if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone():
            cursor.execute(
                "UPDATE schema_version SET version = ?",
                (self.SCHEMA_VERSION,),
            )

    def _migrate_tables(self, cursor: sqlite3.Cursor, from_version: int) -> None:
        """Migrate tables from older schema version."""
        if from_version < 2:
            # Add history tables if migrating from version 1
            self._create_tables(cursor)
        elif from_version < 3:
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

    # --- Job Operations ---

    def save_job(self, session_id: str, job: GenerationJobCreate) -> GenerationJob:
        """Save a new generation job.

        Args:
            session_id: Session identifier.
            job: Job creation data.

        Returns:
            The created job with ID.
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO generation_jobs (
                    id, session_id, status, operation, model,
                    intent, refined_prompt, negative_prompt, preset, params,
                    fingerprint, estimated_cost_usd, is_edit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    JobStatus.QUEUED.value,
                    job.operation.value,
                    job.model,
                    job.intent,
                    job.refined_prompt,
                    job.negative_prompt,
                    job.preset,
                    json.dumps(job.params) if job.params else None,
                    job.fingerprint,
                    job.estimated_cost_usd,
                    1 if job.is_edit else 0,
                    now.isoformat(),
                ),
            )

        return GenerationJob(
            id=job_id,
            session_id=session_id,
            status=JobStatus.QUEUED,
            operation=job.operation,
            model=job.model,
            intent=job.intent,
            refined_prompt=job.refined_prompt,
            negative_prompt=job.negative_prompt,
            preset=job.preset,
            params=job.params,
            fingerprint=job.fingerprint,
            estimated_cost_usd=job.estimated_cost_usd,
            is_edit=job.is_edit,
            created_at=now,
        )

    def update_job(self, job_id: str, updates: GenerationJobUpdate) -> None:
        """Update a generation job.

        Args:
            job_id: Job identifier.
            updates: Fields to update.
        """
        set_clauses = []
        params = []

        if updates.status is not None:
            set_clauses.append("status = ?")
            params.append(updates.status.value)
            if updates.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.BLOCKED_BUDGET,
                JobStatus.CANCELLED,
            ):
                set_clauses.append("completed_at = ?")
                params.append(datetime.utcnow().isoformat())

        if updates.actual_cost_usd is not None:
            set_clauses.append("actual_cost_usd = ?")
            params.append(updates.actual_cost_usd)

        if updates.error_message is not None:
            set_clauses.append("error_message = ?")
            params.append(updates.error_message)

        if updates.result_image_id is not None:
            set_clauses.append("result_image_id = ?")
            params.append(updates.result_image_id)

        if updates.thumbnail_image_id is not None:
            set_clauses.append("thumbnail_image_id = ?")
            params.append(updates.thumbnail_image_id)

        if not set_clauses:
            return

        params.append(job_id)

        with self._transaction() as cursor:
            cursor.execute(
                f"UPDATE generation_jobs SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get a job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            The job or None if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM generation_jobs WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
            return None

    def list_jobs(
        self,
        session_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[GenerationJob], int]:
        """List jobs for a session.

        Args:
            session_id: Session identifier.
            status: Optional status filter.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.

        Returns:
            Tuple of (jobs list, total count).
        """
        with self._transaction() as cursor:
            # Build query
            where_clauses = ["session_id = ?"]
            params: list = [session_id]

            if status:
                where_clauses.append("status = ?")
                params.append(status)

            where_sql = " AND ".join(where_clauses)

            # Get total count
            cursor.execute(
                f"SELECT COUNT(*) as count FROM generation_jobs WHERE {where_sql}",
                params,
            )
            total = cursor.fetchone()["count"]

            # Get jobs
            params.extend([limit, offset])
            cursor.execute(
                f"""
                SELECT * FROM generation_jobs
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )

            jobs = [self._row_to_job(row) for row in cursor.fetchall()]
            return jobs, total

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM generation_jobs WHERE id = ?",
                (job_id,),
            )
            return cursor.rowcount > 0

    def clear_history(self, session_id: str) -> int:
        """Clear all history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of jobs deleted.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM generation_jobs WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount

    def get_job_by_fingerprint(
        self, fingerprint: str, session_id: Optional[str] = None
    ) -> Optional[GenerationJob]:
        """Find a recent successful job with the same fingerprint.

        Used for deduplication.

        Args:
            fingerprint: Operation fingerprint.
            session_id: Optional session filter.

        Returns:
            Most recent matching job or None.
        """
        with self._transaction() as cursor:
            if session_id:
                cursor.execute(
                    """
                    SELECT * FROM generation_jobs
                    WHERE fingerprint = ? AND session_id = ? AND status = 'succeeded'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (fingerprint, session_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM generation_jobs
                    WHERE fingerprint = ? AND status = 'succeeded'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (fingerprint,),
                )

            row = cursor.fetchone()
            if row:
                return self._row_to_job(row)
            return None

    def _row_to_job(self, row: sqlite3.Row) -> GenerationJob:
        """Convert a database row to a GenerationJob model."""
        params = None
        if row["params"]:
            try:
                params = json.loads(row["params"])
            except json.JSONDecodeError:
                params = None

        return GenerationJob(
            id=row["id"],
            session_id=row["session_id"],
            status=JobStatus(row["status"]),
            operation=JobOperation(row["operation"]),
            model=row["model"],
            intent=row["intent"],
            refined_prompt=row["refined_prompt"],
            negative_prompt=row["negative_prompt"],
            preset=row["preset"],
            params=params,
            fingerprint=row["fingerprint"],
            estimated_cost_usd=row["estimated_cost_usd"],
            actual_cost_usd=row["actual_cost_usd"],
            is_edit=bool(row["is_edit"]),
            error_message=row["error_message"],
            result_image_id=row["result_image_id"],
            thumbnail_image_id=row["thumbnail_image_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    # --- Snapshot Operations ---

    def save_snapshot(self, session_id: str, payload: Dict[str, Any]) -> HistorySnapshot:
        """Save a history snapshot for a session."""
        snapshot_id = str(uuid.uuid4())
        now = datetime.utcnow()

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO history_snapshots (id, session_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (snapshot_id, session_id, json.dumps(payload), now.isoformat()),
            )

        return HistorySnapshot(
            id=snapshot_id,
            session_id=session_id,
            payload=payload,
            created_at=now,
        )

    def get_snapshot(self, snapshot_id: str) -> Optional[HistorySnapshot]:
        """Get a history snapshot by ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM history_snapshots WHERE id = ?",
                (snapshot_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_snapshot(row)
            return None

    def list_snapshots(
        self,
        session_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[HistorySnapshot], int]:
        """List snapshots for a session."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM history_snapshots WHERE session_id = ?",
                (session_id,),
            )
            total = cursor.fetchone()["count"]

            cursor.execute(
                """
                SELECT * FROM history_snapshots
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            )
            snapshots = [self._row_to_snapshot(row) for row in cursor.fetchall()]
            return snapshots, total

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot by ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM history_snapshots WHERE id = ?",
                (snapshot_id,),
            )
            return cursor.rowcount > 0

    def clear_snapshots(self, session_id: str) -> int:
        """Delete all snapshots for a session."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM history_snapshots WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount

    def _row_to_snapshot(self, row: sqlite3.Row) -> HistorySnapshot:
        """Convert a database row to a HistorySnapshot model."""
        payload = {}
        if row["payload"]:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = {}

        return HistorySnapshot(
            id=row["id"],
            session_id=row["session_id"],
            payload=payload,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
