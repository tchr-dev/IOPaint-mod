"""File-based storage for job input assets."""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image


class InputStorage:
    """File-based storage for job input images and masks.

    Inputs are stored under:
    - data/inputs/{job_id}/{sha256}.{ext}
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.inputs_dir = data_dir / "inputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)

    def save_input_bytes(
        self,
        job_id: str,
        payload: bytes,
        label: str,
    ) -> Dict[str, str]:
        """Persist input bytes and return metadata for params storage."""
        sha256 = hashlib.sha256(payload).hexdigest()
        ext, mime = self._detect_format(payload)

        job_dir = self.inputs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / f"{sha256}.{ext}"
        if not path.exists():
            with open(path, "wb") as handle:
                handle.write(payload)

        return {
            "type": label,
            "path": str(path.relative_to(self.data_dir)),
            "sha256": sha256,
            "mime": mime,
        }

    def load_input_bytes(self, relative_path: str) -> Optional[bytes]:
        """Load stored input bytes from a relative path."""
        path = self.data_dir / relative_path
        if not path.exists():
            return None
        return path.read_bytes()

    def get_input_path(self, relative_path: str) -> Path:
        """Return absolute path for a stored input."""
        return self.data_dir / relative_path

    def _detect_format(self, payload: bytes) -> Tuple[str, str]:
        """Detect file extension and MIME type for an input payload."""
        try:
            with Image.open(BytesIO(payload)) as image:
                fmt = (image.format or "PNG").upper()
        except Exception:
            fmt = "BIN"

        ext = fmt.lower()
        if ext == "jpeg":
            ext = "jpg"
        mime = Image.MIME.get(fmt, "application/octet-stream")
        return ext, mime
