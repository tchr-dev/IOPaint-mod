"""File-based storage for generated images."""

import hashlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Generator, Tuple

from PIL import Image

from .models import ImageRecord


class ImageStorage:
    """File-based storage for generated images with SQLite metadata.

    Images are stored in:
    - data/images/{job_id}/{image_id}.png - Full resolution images
    - data/thumbnails/{image_id}.jpg - 256x256 thumbnails (lazy generated)

    Metadata is stored in SQLite for fast lookups.
    """

    THUMBNAIL_SIZE = (256, 256)
    THUMBNAIL_QUALITY = 85

    def __init__(self, data_dir: Path, db_path: Path):
        """Initialize image storage.

        Args:
            data_dir: Base directory for image files.
            db_path: Path to SQLite database.
        """
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.thumbnails_dir = data_dir / "thumbnails"
        self.db_path = db_path
        self._local = threading.local()

        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

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

    def save_image(
        self,
        image_data: bytes,
        job_id: Optional[str] = None,
        format: str = "PNG",
    ) -> ImageRecord:
        """Save an image to storage.

        Args:
            image_data: Raw image bytes.
            job_id: Optional job ID to associate with.
            format: Image format (PNG, JPEG, etc).

        Returns:
            ImageRecord with path and metadata.
        """
        image_id = str(uuid.uuid4())

        # Parse image to get dimensions
        img = Image.open(BytesIO(image_data))
        width, height = img.size

        # Determine file path
        if job_id:
            job_dir = self.images_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            ext = format.lower()
            if ext == "jpeg":
                ext = "jpg"
            path = job_dir / f"{image_id}.{ext}"
        else:
            ext = format.lower()
            if ext == "jpeg":
                ext = "jpg"
            path = self.images_dir / f"{image_id}.{ext}"

        # Save image file
        with open(path, "wb") as f:
            f.write(image_data)

        # Store metadata in database
        now = datetime.utcnow()
        relative_path = str(path.relative_to(self.data_dir))

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO images (id, job_id, path, width, height, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (image_id, job_id, relative_path, width, height, now.isoformat()),
            )

        return ImageRecord(
            id=image_id,
            job_id=job_id or "",
            path=relative_path,
            width=width,
            height=height,
            created_at=now,
        )

    def save_image_from_pil(
        self,
        img: Image.Image,
        job_id: Optional[str] = None,
        format: str = "PNG",
    ) -> ImageRecord:
        """Save a PIL Image to storage.

        Args:
            img: PIL Image object.
            job_id: Optional job ID to associate with.
            format: Image format.

        Returns:
            ImageRecord with path and metadata.
        """
        buffer = BytesIO()
        img.save(buffer, format=format)
        return self.save_image(buffer.getvalue(), job_id, format)

    def get_image(self, image_id: str) -> Optional[bytes]:
        """Get image data by ID.

        Args:
            image_id: Image identifier.

        Returns:
            Image bytes or None if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT path FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            path = self.data_dir / row["path"]
            if not path.exists():
                return None

            with open(path, "rb") as f:
                return f.read()

    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the file path for an image.

        Args:
            image_id: Image identifier.

        Returns:
            Path object or None if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT path FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return self.data_dir / row["path"]

    def get_image_record(self, image_id: str) -> Optional[ImageRecord]:
        """Get image metadata by ID.

        Args:
            image_id: Image identifier.

        Returns:
            ImageRecord or None if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return ImageRecord(
                id=row["id"],
                job_id=row["job_id"] or "",
                path=row["path"],
                thumbnail_path=row["thumbnail_path"],
                width=row["width"],
                height=row["height"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def get_thumbnail(self, image_id: str) -> Optional[bytes]:
        """Get or create thumbnail for an image.

        Thumbnails are created lazily on first request.

        Args:
            image_id: Image identifier.

        Returns:
            Thumbnail JPEG bytes or None if image not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT path, thumbnail_path FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Check if thumbnail exists
            if row["thumbnail_path"]:
                thumb_path = self.data_dir / row["thumbnail_path"]
                if thumb_path.exists():
                    with open(thumb_path, "rb") as f:
                        return f.read()

            # Generate thumbnail
            image_path = self.data_dir / row["path"]
            if not image_path.exists():
                return None

            thumb_data, thumb_path = self._create_thumbnail(image_id, image_path)

            # Update database with thumbnail path
            relative_thumb_path = str(thumb_path.relative_to(self.data_dir))
            cursor.execute(
                "UPDATE images SET thumbnail_path = ? WHERE id = ?",
                (relative_thumb_path, image_id),
            )
            self._conn.commit()

            return thumb_data

    def _create_thumbnail(
        self, image_id: str, image_path: Path
    ) -> Tuple[bytes, Path]:
        """Create a thumbnail for an image.

        Args:
            image_id: Image identifier.
            image_path: Path to source image.

        Returns:
            Tuple of (thumbnail bytes, thumbnail path).
        """
        img = Image.open(image_path)

        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Create thumbnail maintaining aspect ratio
        img.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

        # Save thumbnail
        thumb_path = self.thumbnails_dir / f"{image_id}.jpg"
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self.THUMBNAIL_QUALITY)
        thumb_data = buffer.getvalue()

        with open(thumb_path, "wb") as f:
            f.write(thumb_data)

        return thumb_data, thumb_path

    def delete_image(self, image_id: str) -> bool:
        """Delete an image and its thumbnail.

        Args:
            image_id: Image identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT path, thumbnail_path FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            # Delete files
            image_path = self.data_dir / row["path"]
            if image_path.exists():
                image_path.unlink()

            if row["thumbnail_path"]:
                thumb_path = self.data_dir / row["thumbnail_path"]
                if thumb_path.exists():
                    thumb_path.unlink()

            # Delete database record
            cursor.execute(
                "DELETE FROM images WHERE id = ?",
                (image_id,),
            )

            return True

    def delete_job_images(self, job_id: str) -> int:
        """Delete all images for a job.

        Args:
            job_id: Job identifier.

        Returns:
            Number of images deleted.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT id, path, thumbnail_path FROM images WHERE job_id = ?",
                (job_id,),
            )
            rows = cursor.fetchall()

            for row in rows:
                # Delete files
                image_path = self.data_dir / row["path"]
                if image_path.exists():
                    image_path.unlink()

                if row["thumbnail_path"]:
                    thumb_path = self.data_dir / row["thumbnail_path"]
                    if thumb_path.exists():
                        thumb_path.unlink()

            # Delete database records
            cursor.execute(
                "DELETE FROM images WHERE job_id = ?",
                (job_id,),
            )

            # Try to remove job directory if empty
            job_dir = self.images_dir / job_id
            if job_dir.exists():
                try:
                    job_dir.rmdir()
                except OSError:
                    pass  # Directory not empty

            return len(rows)

    def cleanup_orphaned(self) -> int:
        """Remove orphaned images (no associated job).

        Returns:
            Number of images cleaned up.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT id, path, thumbnail_path FROM images
                WHERE job_id IS NULL OR job_id NOT IN (
                    SELECT id FROM generation_jobs
                )
                """
            )
            rows = cursor.fetchall()

            for row in rows:
                # Delete files
                image_path = self.data_dir / row["path"]
                if image_path.exists():
                    image_path.unlink()

                if row["thumbnail_path"]:
                    thumb_path = self.data_dir / row["thumbnail_path"]
                    if thumb_path.exists():
                        thumb_path.unlink()

            # Delete database records
            cursor.execute(
                """
                DELETE FROM images
                WHERE job_id IS NULL OR job_id NOT IN (
                    SELECT id FROM generation_jobs
                )
                """
            )

            return len(rows)

    def get_storage_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dict with storage stats (total_images, total_size_bytes, etc).
        """
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM images")
            total_images = cursor.fetchone()["count"]

            cursor.execute(
                "SELECT COUNT(*) as count FROM images WHERE thumbnail_path IS NOT NULL"
            )
            total_thumbnails = cursor.fetchone()["count"]

        # Calculate disk usage
        total_size = 0
        for path in self.images_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        thumb_size = 0
        for path in self.thumbnails_dir.rglob("*"):
            if path.is_file():
                thumb_size += path.stat().st_size

        return {
            "total_images": total_images,
            "total_thumbnails": total_thumbnails,
            "images_size_bytes": total_size,
            "thumbnails_size_bytes": thumb_size,
            "total_size_bytes": total_size + thumb_size,
        }

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
