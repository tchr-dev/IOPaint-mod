"""Environment file loader with hot-reload support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import os
import signal
import threading

from loguru import logger

_BASE_ENV_KEYS = set(os.environ.keys())
_LOADED_KEYS: set[str] = set()
_LOCK = threading.Lock()
_SIGHUP_INSTALLED = False

_PUBLIC_KEYS = {
    "AIE_BACKEND",
    "AIE_OPENAI_BASE_URL",
    "AIE_OPENAI_MODEL",
    "AIE_OPENAI_TIMEOUT_S",
    "AIE_OPENAI_REFINE_MODEL",
    "AIE_OPENAI_MODELS_CACHE_TTL_S",
    "AIE_BUDGET_DAILY_CAP",
    "AIE_BUDGET_MONTHLY_CAP",
    "AIE_BUDGET_SESSION_CAP",
    "AIE_RATE_LIMIT_SECONDS",
    "AIE_DEDUPE_ENABLED",
    "AIE_DEDUPE_TTL_SECONDS",
}


def default_config_path() -> Path:
    return Path(os.getenv("AIE_CONFIG_FILE", "config/secret.env")).expanduser()


def register_env_override(keys: Iterable[str]) -> None:
    _BASE_ENV_KEYS.update(keys)


def _parse_env_line(raw_line: str) -> Optional[tuple[str, str]]:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if value and value[0] in {'"', "'"} and value[-1] == value[0]:
        value = value[1:-1]
    return key, value


def parse_env_file(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            key, value = parsed
            entries[key] = value
    return entries


def load_env_file(path: Optional[Path] = None) -> Dict[str, object]:
    config_path = (path or default_config_path()).expanduser()
    with _LOCK:
        if not config_path.exists():
            logger.info("Config file not found: %s", config_path)
            return {"path": str(config_path), "loaded": False, "added": [], "updated": [], "removed": []}

        entries = parse_env_file(config_path)
        new_keys = set(entries.keys())
        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []

        for key, value in entries.items():
            if key in _BASE_ENV_KEYS:
                continue
            previous = os.environ.get(key)
            if previous is None:
                added.append(key)
            elif previous != value:
                updated.append(key)
            os.environ[key] = value

        for key in list(_LOADED_KEYS):
            if key not in new_keys and key not in _BASE_ENV_KEYS:
                if key in os.environ:
                    del os.environ[key]
                removed.append(key)

        _LOADED_KEYS.clear()
        _LOADED_KEYS.update({key for key in new_keys if key not in _BASE_ENV_KEYS})

        if added or updated or removed:
            logger.info(
                "Config reloaded from %s (added=%s, updated=%s, removed=%s)",
                config_path,
                added,
                updated,
                removed,
            )
        else:
            logger.info("Config reloaded from %s (no changes)", config_path)

        return {
            "path": str(config_path),
            "loaded": True,
            "added": added,
            "updated": updated,
            "removed": removed,
        }


def install_sighup_handler(reload_callback) -> None:
    global _SIGHUP_INSTALLED
    if _SIGHUP_INSTALLED:
        return
    if not hasattr(signal, "SIGHUP"):
        logger.warning("SIGHUP not available; config reload disabled")
        return

    def _handler(signum, frame):
        logger.info("Received SIGHUP, reloading config")
        reload_callback()

    signal.signal(signal.SIGHUP, _handler)
    _SIGHUP_INSTALLED = True


def public_config() -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for key in sorted(_PUBLIC_KEYS):
        value = os.getenv(key)
        if value is not None and value != "":
            payload[key] = value
    return payload
