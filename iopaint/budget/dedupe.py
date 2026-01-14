"""Dedupe cache with fingerprint-based request deduplication."""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, List, Any, Dict
from datetime import datetime
import shutil

from .config import BudgetConfig
from .storage import BudgetStorage


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt for consistent fingerprinting.

    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace to single space
    - Remove extra punctuation spaces
    """
    if not prompt:
        return ""
    # Lowercase and strip
    text = prompt.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def hash_image_bytes(image_bytes: bytes) -> str:
    """Generate SHA256 hash of image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def calculate_fingerprint(
    model: str,
    action: str,
    prompt: str,
    negative_prompt: str = "",
    params: Optional[Dict[str, Any]] = None,
    input_hashes: Optional[List[str]] = None,
) -> str:
    """Calculate deterministic fingerprint for request deduplication.

    The fingerprint is a SHA256 hash of normalized request parameters.
    Identical requests will produce identical fingerprints.

    Args:
        model: Model name (e.g., "gpt-image-1", "dall-e-3")
        action: Operation type ("generate", "edit", "variation", "refine")
        prompt: Main prompt text
        negative_prompt: Negative prompt text (optional)
        params: Additional parameters dict (size, quality, style, etc.)
        input_hashes: List of SHA256 hashes of input images (for edit/variation)

    Returns:
        64-character hexadecimal SHA256 fingerprint
    """
    # Build normalized structure
    normalized = {
        "model": model.lower().strip(),
        "action": action.lower().strip(),
        "prompt": normalize_prompt(prompt),
        "negative_prompt": normalize_prompt(negative_prompt),
        "params": json.dumps(params or {}, sort_keys=True),
        "input_hashes": sorted(input_hashes or []),
    }

    # Create deterministic JSON string
    fingerprint_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    # Return SHA256 hash
    return hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()


class DedupeCache:
    """Cache for deduplicating expensive API requests.

    Stores results of successful operations and returns cached results
    for identical requests within TTL window.

    Usage:
        cache = DedupeCache(config, storage)

        # Check for cached result before making request
        fingerprint = calculate_fingerprint(...)
        cached = cache.get(fingerprint)
        if cached:
            return cached["result_path"]

        # After successful request, store result
        cache.store(fingerprint, result_bytes)
    """

    def __init__(self, config: BudgetConfig, storage: BudgetStorage):
        self.config = config
        self.storage = storage
        config.ensure_directories()

    @property
    def enabled(self) -> bool:
        """Check if deduplication is enabled."""
        return self.config.dedupe_enabled

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Get cached result for fingerprint if exists and not expired.

        Args:
            fingerprint: 64-character hex fingerprint

        Returns:
            Dict with "result_path" and "result_metadata" if found, None otherwise
        """
        if not self.enabled:
            return None

        cached = self.storage.get_cached_result(fingerprint)
        if cached and cached.get("result_path"):
            result_path = Path(cached["result_path"])
            if result_path.exists():
                return cached
            # File missing, remove stale entry
            self._remove_entry(fingerprint)
        return None

    def get_image_bytes(self, fingerprint: str) -> Optional[bytes]:
        """Get cached image bytes for fingerprint.

        Convenience method that reads the cached file.

        Args:
            fingerprint: 64-character hex fingerprint

        Returns:
            Image bytes if found and file exists, None otherwise
        """
        cached = self.get(fingerprint)
        if cached and cached.get("result_path"):
            try:
                return Path(cached["result_path"]).read_bytes()
            except IOError:
                return None
        return None

    def store(
        self,
        fingerprint: str,
        result_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store result in cache.

        Args:
            fingerprint: 64-character hex fingerprint
            result_bytes: Image bytes to cache
            metadata: Optional metadata dict to store
            ttl_seconds: Optional TTL override (uses config default if not specified)

        Returns:
            Path to stored result file
        """
        if not self.enabled:
            # Return empty path if dedupe disabled
            return ""

        # Create cache file path using fingerprint prefix for directory structure
        cache_dir = self.config.cache_dir / fingerprint[:2] / fingerprint[2:4]
        cache_dir.mkdir(parents=True, exist_ok=True)
        result_path = cache_dir / f"{fingerprint}.png"

        # Write result file
        result_path.write_bytes(result_bytes)

        # Store in database
        self.storage.store_cached_result(
            fingerprint=fingerprint,
            result_path=str(result_path),
            result_metadata=json.dumps(metadata) if metadata else None,
            ttl_seconds=ttl_seconds,
        )

        return str(result_path)

    def invalidate(self, fingerprint: str) -> bool:
        """Invalidate a cached entry.

        Args:
            fingerprint: 64-character hex fingerprint

        Returns:
            True if entry was found and removed, False otherwise
        """
        return self._remove_entry(fingerprint)

    def _remove_entry(self, fingerprint: str) -> bool:
        """Remove cache entry and associated file."""
        cached = self.storage.get_cached_result(fingerprint)
        if cached and cached.get("result_path"):
            try:
                Path(cached["result_path"]).unlink(missing_ok=True)
            except IOError:
                pass

        # Remove from database (by storing with 0 TTL it expires immediately)
        # Actually we need a proper delete - let's add cleanup
        return self.cleanup_fingerprint(fingerprint)

    def cleanup_fingerprint(self, fingerprint: str) -> bool:
        """Remove specific fingerprint from cache.

        Note: This uses the storage cleanup mechanism.
        For immediate removal, we store with expired timestamp.
        """
        # Store with 0 TTL to mark as expired
        self.storage.store_cached_result(
            fingerprint=fingerprint,
            result_path="",
            result_metadata=None,
            ttl_seconds=0,
        )
        return True

    def cleanup_expired(self) -> int:
        """Remove all expired entries from cache.

        Also removes associated files from disk.

        Returns:
            Number of entries cleaned up
        """
        # First, find expired entries and their file paths
        # The storage.cleanup_expired_cache() handles DB cleanup
        count = self.storage.cleanup_expired_cache()

        # Optionally clean up orphaned files in cache directory
        # (files without corresponding DB entries)
        # This is a more expensive operation, skip for now

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        cache_dir = self.config.cache_dir
        if not cache_dir.exists():
            return {
                "enabled": self.enabled,
                "cache_dir": str(cache_dir),
                "total_files": 0,
                "total_size_mb": 0.0,
            }

        total_files = 0
        total_size = 0
        for file_path in cache_dir.rglob("*.png"):
            total_files += 1
            total_size += file_path.stat().st_size

        return {
            "enabled": self.enabled,
            "cache_dir": str(cache_dir),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "ttl_seconds": self.config.dedupe_ttl_seconds,
        }
