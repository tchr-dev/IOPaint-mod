import json
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image


class OpenAIJobFilesStorage:
    """Store OpenAI job artifacts in a human-friendly structure."""

    THUMBNAIL_SIZE = (256, 256)
    THUMBNAIL_QUALITY = 85

    def __init__(self, data_dir: Path) -> None:
        self.base_dir = data_dir / "openai"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def persist_job_files(
        self,
        *,
        session_id: str,
        job_id: str,
        created_at: datetime,
        metadata: Dict[str, Any],
        input_bytes: Optional[bytes] = None,
        mask_bytes: Optional[bytes] = None,
        output_bytes: Optional[bytes] = None,
        error_message: Optional[str] = None,
    ) -> Path:
        job_dir = self._job_dir(session_id, job_id, created_at)
        job_dir.mkdir(parents=True, exist_ok=True)

        if input_bytes:
            self._write_png(job_dir / "input.png", input_bytes)
        if mask_bytes:
            self._write_png(job_dir / "mask.png", mask_bytes)
        if output_bytes:
            self._write_png(job_dir / "output.png", output_bytes)
            self._write_thumbnail(job_dir / "output_thumb.webp", output_bytes)

        self._write_json(job_dir / "meta.json", metadata)

        if error_message:
            self._atomic_write(job_dir / "logs.txt", error_message.encode("utf-8"))

        return job_dir

    def _job_dir(self, session_id: str, job_id: str, created_at: datetime) -> Path:
        date_path = Path(
            f"{created_at.year:04d}",
            f"{created_at.month:02d}",
            f"{created_at.day:02d}",
        )
        return (
            self.base_dir
            / date_path
            / f"session_{session_id}"
            / f"job_{job_id}"
        )

    def _write_png(self, path: Path, payload: bytes) -> None:
        try:
            with Image.open(BytesIO(payload)) as image:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                self._atomic_write(path, buffer.getvalue())
        except Exception:
            self._atomic_write(path, payload)

    def _write_thumbnail(self, path: Path, payload: bytes) -> None:
        try:
            with Image.open(BytesIO(payload)) as image:
                image = image.convert("RGB")
                image.thumbnail(self.THUMBNAIL_SIZE)
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="WEBP",
                    quality=self.THUMBNAIL_QUALITY,
                )
                self._atomic_write(path, buffer.getvalue())
        except Exception:
            return

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(payload, indent=2).encode("utf-8"))

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
